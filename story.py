"""Top-issue presentation layer.

The engine detects every finding and the JSON report keeps all of them. For a
short demo, this module picks the single most important problem and builds a
simple story around it:

    overview -> main problem -> what changed -> is it isolated? -> possible causes
             -> one primary action (+ one secondary action)

Every number here comes from engine.compare_periods, and each number the UI
shows gets an evidence item (the same format as the engine's). Sentences are
chosen by explicit conditions on those numbers, never hardcoded to one dataset.
"""

from __future__ import annotations

import pandas as pd

from analyzer import calculate_kpis
from engine import Period, Thresholds, _evidence, compare_periods

STORY_METRICS = ["revenue", "profit", "units", "return_rate"]

# Short, non-factual cause hypotheses per engine rule (always shown as "not proven"),
# each with the measured signal that makes it plausible and the direction that signal
# must have moved. A cause is only shown if its signal actually moved that way.
POSSIBLE_CAUSES = {
    "decline_with_rising_returns": [
        ("Product quality or defect issues", "return_rate", "up"),
        ("Fulfillment / delivery problems", "return_rate", "up"),
        ("Demand, pricing, or competition changes", "units", "down"),
    ],
    "decline_with_stable_returns": [
        ("Demand or competition changes", "units", "down"),
        ("Pricing changes", "revenue", "down"),
        ("Stock availability problems", "units", "down"),
    ],
    "rising_returns_without_decline": [
        ("Product quality or defect issues", "return_rate", "up"),
        ("Fulfillment / delivery problems", "return_rate", "up"),
        ("Product listings not matching what customers receive", "return_rate", "up"),
    ],
    "profit_decline_with_stable_revenue": [
        ("Higher costs", "profit_margin", "down"),
        ("Deeper discounting", "profit_margin", "down"),
        ("A shift toward lower-margin sales", "profit_margin", "down"),
    ],
}
SIGNAL_NAMES = {"return_rate": "Return rate", "units": "Units", "revenue": "Revenue",
                "profit_margin": "Profit margin"}


def linked_possible_causes(rule: str | None, metrics: dict) -> list[dict]:
    """The rule's possible causes, each tied to the measured signal that supports it."""
    causes = []
    for cause, metric, direction in POSSIBLE_CAUSES.get(rule, []):
        m = metrics[metric]
        if (m["change"] > 0) != (direction == "up") or m["change"] == 0:
            continue  # the supporting signal did not move this way, so don't suggest it
        value = fmt_value(metric, m["previous"]) + " → " + fmt_value(metric, m["current"])
        causes.append({"cause": cause, "metric": metric,
                       "signal": f"{SIGNAL_NAMES[metric]} {value} ({fmt_pct(m['change_percent'])})"})
    return causes

PRIMARY_ACTION = {
    "decline_with_rising_returns": (
        "Investigate {segment} product quality, fulfillment, and customer experience "
        "before increasing inventory or marketing spend."
    ),
    "decline_with_stable_returns": (
        "Investigate {segment} demand, pricing, and availability before changing inventory plans."
    ),
    "rising_returns_without_decline": (
        "Audit {segment} returns (defects, delivery damage, listing accuracy) before the rise affects sales."
    ),
    "profit_decline_with_stable_revenue": "Review {segment} costs, discount depth, and pricing.",
}


# ---------------------------------------------------------------------------
# Small text helpers (formatting only; numbers are computed elsewhere)
# ---------------------------------------------------------------------------

def fmt_money(value: float) -> str:
    sign, a = ("-" if value < 0 else ""), abs(value)
    if a >= 1e6:
        return f"{sign}${a / 1e6:.2f}M"
    if a >= 1e3:
        return f"{sign}${a / 1e3:.1f}K"
    return f"{sign}${a:,.2f}"


def fmt_rate(value: float) -> str:
    return f"{value * 100:.1f}%"


def fmt_pct(value) -> str:
    return "n/a" if value is None else f"{value:+.1f}%"


def fmt_value(metric: str, value) -> str:
    if metric in ("revenue", "profit", "cost"):
        return fmt_money(value)
    if metric in ("return_rate", "profit_margin"):
        return fmt_rate(value)
    return f"{int(value):,}"


def _join(words: list[str]) -> str:
    if len(words) <= 1:
        return "".join(words)
    return ", ".join(words[:-1]) + (", and " if len(words) > 2 else " and ") + words[-1]


# ---------------------------------------------------------------------------
# Picking the top issue
# ---------------------------------------------------------------------------

def pick_top_issue(report: dict) -> dict | None:
    """The single most important problem.

    Preference: the highest-priority recommendation whose segment has a
    concentrated (region + product) decline, then the highest-priority
    recommendation, then the most severe insight.
    """
    insights = report.get("key_insights", [])
    recs = report.get("recommendations", [])
    if not insights:
        return None
    concentrated = {i["dimension_value"] for i in insights if i["type"] == "concentrated_decline"}
    top_priority = [r for r in recs if r["priority"] == recs[0]["priority"]] if recs else []
    rec = next((r for r in top_priority if r["segment"] in concentrated), recs[0] if recs else None)
    if rec:
        segment, filters = rec["segment"], rec["filters"]
    else:
        segment, filters = insights[0]["dimension_value"], insights[0]["filters"]
    by_id = {i["id"]: i for i in insights}
    related_ids = set(rec["based_on_insights"]) if rec else set()
    related_ids |= {i["id"] for i in insights if i["dimension_value"] == segment}
    return {
        "segment": segment,
        "filters": filters,
        "recommendation": rec,
        "insights": [i for i in insights if i["id"] in related_ids],
        "severity": min((by_id[i]["severity"] for i in related_ids),
                        key={"high": 0, "medium": 1, "low": 2}.get),
    }


def _peer_groups(df: pd.DataFrame, filters: dict) -> list[tuple[str, dict]]:
    """For each dimension, the same segment with that dimension set to 'everything else'."""
    peers = []
    for dim, values in filters.items():
        others = sorted(str(v) for v in df[dim].unique() if v not in values)
        if not others:
            continue
        fixed = " / ".join(v[0] for d, v in filters.items() if d != dim)
        if not fixed:
            label = f"Other {dim}s"
        elif dim == "product":
            label = f"Other products in {fixed}"
        else:
            label = f"{fixed} in other {dim}s"
        peers.append((label, {d: (others if d == dim else v) for d, v in filters.items()}))
    return peers


# ---------------------------------------------------------------------------
# The story
# ---------------------------------------------------------------------------

def build_story(df: pd.DataFrame, report: dict, thresholds: Thresholds | None = None) -> dict:
    """Everything the concise UI needs, as plain data."""
    t = thresholds or Thresholds()
    meta = report["metadata"]
    previous, current = Period(**meta["period_previous"]), Period(**meta["period_current"])
    kpis = calculate_kpis(df)
    growth = report["overview"]["metrics"]["revenue"]["change_percent"]

    issue = pick_top_issue(report)
    story = {
        "period_previous": previous.to_dict(),
        "period_current": current.to_dict(),
        "overview": {
            "revenue": kpis["revenue"], "profit": kpis["profit"], "units": kpis["units"],
            "return_rate": kpis["return_rate"], "revenue_growth_percent": growth,
        },
        "problem": None,
        "focused_report": report,
        "question_placeholder": "Why did revenue decline?",
    }
    if issue is None:
        story["overview"]["sentence"] = (
            "No significant problems were detected: no segment shows a significant decline "
            "or return-rate increase between the two periods."
        )
        return story

    segment, filters = issue["segment"], issue["filters"]
    direction = "declined" if (growth or 0) < 0 else "grew"
    story["overview"]["sentence"] = (
        f"Overall business performance {direction}, but the largest concentrated problem is {segment}."
    )

    # Step 1: what changed in the segment.
    comparison = compare_periods(df, previous, current, filters)
    m = comparison["metrics"]
    evidence = [_evidence(comparison, metric, "segment", f"{segment} {metric.replace('_', ' ')}")
                for metric in STORY_METRICS]

    # Step 2: is it isolated? Compare with the peer groups over the same periods.
    peers = []
    for label, peer_filters in _peer_groups(df, filters):
        peer = compare_periods(df, previous, current, peer_filters)
        evidence.append(_evidence(peer, "revenue", "baseline", f"{label} revenue"))
        peers.append({"label": label, "filters": peer_filters,
                      "change_percent": peer["metrics"]["revenue"]["change_percent"]})

    seg_pct = m["revenue"]["change_percent"]
    declined = seg_pct is not None and seg_pct <= -t.decline_pct
    worse_than_peers = bool(peers) and seg_pct is not None and all(
        p["change_percent"] is not None and seg_pct <= p["change_percent"] - t.gap_pp for p in peers)
    if declined and worse_than_peers:
        explanation = (f"{segment} is performing significantly worse than both its historical "
                       f"performance and its peer groups.")
    elif worse_than_peers:
        explanation = f"{segment} is performing significantly worse than its peer groups."
    elif declined:
        explanation = (f"{segment} is performing significantly worse than its historical performance, "
                       f"but its peer groups show a similar trend.")
    else:
        explanation = f"{segment} shows the strongest warning signs in the data."
    if worse_than_peers:
        emphasis = f"{segment} is declining much faster than its peers."
    elif seg_pct is not None and seg_pct < 0:
        emphasis = f"{segment} is declining at a similar pace to its peers."
    else:
        emphasis = f"{segment} revenue is not falling faster than its peers."

    # Step 3 and the action come from the engine rule for this segment.
    rec = issue["recommendation"]
    rules = rec["rules_triggered"] if rec else []
    main_rule = next((r for r in rules if r in PRIMARY_ACTION), None)
    causes = linked_possible_causes(main_rule, m)
    if "margin_erosion" in rules or any(c["metric"] == "profit_margin" for c in causes):
        evidence.append(_evidence(comparison, "profit_margin", "segment", f"{segment} profit margin"))

    story["problem"] = {
        "segment": segment,
        "filters": filters,
        "severity": issue["severity"],
        "metrics": {k: m[k] for k in STORY_METRICS},
        "explanation": explanation,
        "peers": peers,
        "segment_change_percent": seg_pct,
        "worse_than_peers": worse_than_peers,
        "emphasis": emphasis,
        "possible_causes": causes,
        "evidence": evidence,
        "action": _action(segment, filters, m, peers, rules, main_rule, t),
        "rule": main_rule,
        "insight_ids": [i["id"] for i in issue["insights"]],
    }
    story["focused_report"] = focused_report(report, issue, evidence, _summary(story, previous, current))
    story["question_placeholder"] = (
        f"Why did {segment} revenue decline?" if seg_pct is not None and seg_pct < 0
        else f"What changed for {segment}?"
    )
    return story


def _action(segment, filters, m, peers, rules, main_rule, t: Thresholds) -> dict:
    if main_rule is None:
        return {"primary": None, "why": None, "secondary": None, "secondary_why": None}

    fell = [name for name in ("revenue", "profit", "units")
            if m[name]["change_percent"] is not None and m[name]["change_percent"] <= -t.decline_pct]
    rr = m["return_rate"]
    ratio = rr["current"] / rr["previous"] if rr["previous"] else None
    rr_range = f"{fmt_rate(rr['previous'])} → {fmt_rate(rr['current'])}"
    if ratio and ratio >= 2:
        returns_text = f"the return rate more than doubled ({rr_range})"
    elif ratio and ratio >= 1.5:
        returns_text = f"the return rate rose sharply ({rr_range})"
    elif rr["change"] > 0:
        returns_text = f"the return rate rose ({rr_range})"
    else:
        returns_text = f"the return rate did not rise ({rr_range})"
    if fell:
        why = f"Because {_join(fell)} declined sharply while {returns_text}."
    else:
        why = f"Because {returns_text}."
    why = why[0].upper() + why[1:]

    # Secondary action: the region itself is weak, or margins are being squeezed.
    secondary = secondary_why = None
    region = filters.get("region", [None])[0]
    region_peer = next((p for p in peers if "region" in p["filters"]
                        and p["filters"]["region"] == [region] and "product" in filters), None)
    if region_peer and region_peer["change_percent"] is not None \
            and region_peer["change_percent"] <= -t.decline_pct:
        secondary = f"Review pricing, demand, and availability in the {region} region."
        secondary_why = f"{region_peer['label']} also fell ({fmt_pct(region_peer['change_percent'])} revenue)."
    elif "margin_erosion" in rules:
        pm = m["profit_margin"]
        secondary = f"Review discount depth and pricing for {segment}."
        secondary_why = f"Profit margin also fell ({fmt_rate(pm['previous'])} → {fmt_rate(pm['current'])})."
    return {
        "primary": PRIMARY_ACTION[main_rule].format(segment=segment),
        "why": why,
        "secondary": secondary,
        "secondary_why": secondary_why,
    }


def _summary(story: dict, previous: Period, current: Period) -> str:
    """2-3 sentence summary from the story's own numbers (used as the fallback AI summary)."""
    p = story["problem"]
    m = p["metrics"]
    rev, rr = m["revenue"], m["return_rate"]
    first = (f"Between {previous.label} and {current.label}, {p['segment']} revenue changed "
             f"{fmt_pct(rev['change_percent'])} ({fmt_money(rev['previous'])} to {fmt_money(rev['current'])})")
    if rr["change"] > 0:
        first += f" while its return rate rose from {fmt_rate(rr['previous'])} to {fmt_rate(rr['current'])}."
    else:
        first += "."
    sentences = [first]
    if p["peers"]:
        def label(x, i):  # lower-case "Other ..." mid-sentence, keep proper nouns
            return "o" + x["label"][1:] if i and x["label"].startswith("Other") else x["label"]
        peers = " and ".join(f"{label(x, i)} changed {fmt_pct(x['change_percent'])}"
                             for i, x in enumerate(p["peers"]))
        verdict = ("so the problem is concentrated in this segment" if p["worse_than_peers"]
                   else "so the decline is not unique to this segment")
        sentences.append(f"Over the same periods, {peers}, {verdict}.")
    sentences.append("The data does not show the cause, so the possible causes need investigation.")
    return " ".join(sentences)


def focused_report(report: dict, issue: dict, story_evidence: list[dict], summary: str) -> dict:
    """A copy of the report restricted to the top issue, for the LLM and the concise UI.

    The full report (all findings) is left untouched.
    """
    insights = issue["insights"]
    evidence = {}
    for insight in insights:
        for item in insight["evidence"]:
            evidence.setdefault(item["id"], item)
    known = {(json_key(e)): e["id"] for e in report["evidence"]}
    for n, item in enumerate(story_evidence, start=1):
        existing = known.get(json_key(item))
        item["id"] = existing or f"S{n}"
        evidence.setdefault(item["id"], item)
    meta = dict(report["metadata"])
    meta["insight_count"] = len(insights)
    meta["focus"] = issue["segment"]
    return {
        "metadata": meta,
        "executive_summary": summary,
        "overview": report["overview"],
        "key_insights": insights,
        "evidence": list(evidence.values()),
        "recommendations": [issue["recommendation"]] if issue["recommendation"] else [],
    }


def json_key(item: dict) -> tuple:
    """Identity of an evidence item: what was measured, where, and when."""
    return (item["metric"], str(sorted((k, tuple(v)) for k, v in item["filters"].items())),
            tuple(item["period_previous_range"]), tuple(item["period_current_range"]))
