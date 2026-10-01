# """Insight and evidence engine.

# Pipeline:  CSV -> analyzer -> period comparisons -> detected insights
#            -> traceable evidence -> rule-based recommendations -> report

# Design rules:
# * Every number is computed here, deterministically, from the data. Nothing is
#   estimated or invented, and there is no LLM in this module.
# * Every insight carries evidence items. Each item records the exact filters and
#   date range it was computed from, so it can be recomputed from the CSV.
# * Recommendations are produced by explicit rules that only look at detected
#   insights, and they cite the insights and evidence they are based on.

# Usage:
#     python engine.py                               # report for data/sales.csv
#     python engine.py path/to.csv --out report.json
# """

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from analyzer import calculate_kpis, load_data, validate_data

METRICS = ["revenue", "cost", "profit", "profit_margin", "units", "returns", "return_rate"]
DECLINE_METRICS = ["revenue", "profit", "units"]
SEGMENT_DIMENSIONS = ["region", "product"]
MONEY_METRICS = {"revenue", "cost", "profit"}
RATE_METRICS = {"return_rate", "profit_margin"}
SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}

DECLINE_TYPES = {
    "revenue_decline",
    "units_decline",
    "region_specific_decline",
    "product_specific_decline",
    "concentrated_decline",
}
SPECIFIC_DECLINE_TYPE = {
    ("region",): "region_specific_decline",
    ("product",): "product_specific_decline",
    ("region", "product"): "concentrated_decline",
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Period:
    """An inclusive date range with a human-readable label."""

    label: str
    start: str  # YYYY-MM-DD, inclusive
    end: str    # YYYY-MM-DD, inclusive

    def mask(self, df: pd.DataFrame) -> pd.Series:
        return (df["date"] >= pd.Timestamp(self.start)) & (df["date"] <= pd.Timestamp(self.end))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Thresholds:
    """What counts as significant. Stored in the report so results are reproducible."""

    decline_pct: float = 10.0               # A/B/C: period-over-period drop, in %
    gap_pp: float = 5.0                     # E/F/G: underperformance vs. baseline, in pp
    return_rate_increase_pct: float = 20.0  # D: relative increase, in %
    return_rate_increase_pp: float = 0.5    # D: absolute increase, in pp
    min_z_score: float = 3.0                # D: two-proportion z-test, filters out noise
    margin_drop_pp: float = 1.0             # profit margin fall that counts as erosion
    min_units: int = 100                    # skip segments too small to judge


# ---------------------------------------------------------------------------
# Periods and comparisons
# ---------------------------------------------------------------------------

def _period_label(first: pd.Period, last: pd.Period) -> str:
    if first.year == last.year and (first.month, last.month) == (1, 6):
        return f"{first.year} H1"
    if first.year == last.year and (first.month, last.month) == (7, 12):
        return f"{first.year} H2"
    return f"{first} to {last}"


def split_periods(df: pd.DataFrame) -> tuple[Period, Period]:
    """Split the data's months into an earlier and a later half."""
    months = df["date"].dt.to_period("M").drop_duplicates().sort_values().reset_index(drop=True)
    if len(months) < 2:
        raise ValueError("Need at least two months of data to compare periods.")
    half = len(months) // 2

    def make(ms: pd.Series) -> Period:
        first, last = ms.iloc[0], ms.iloc[-1]
        return Period(
            _period_label(first, last),
            first.start_time.strftime("%Y-%m-%d"),
            last.end_time.strftime("%Y-%m-%d"),
        )

    return make(months.iloc[:half]), make(months.iloc[half:])


def _resolve_periods(df, previous, current) -> tuple[Period, Period]:
    if previous is None or current is None:
        return split_periods(df)
    return previous, current


def apply_filters(df: pd.DataFrame, filters: dict[str, list[str]] | None) -> pd.DataFrame:
    """Rows whose columns take one of the listed values, e.g. {"region": ["South"]}."""
    mask = pd.Series(True, index=df.index)
    for column, values in (filters or {}).items():
        mask &= df[column].isin(values)
    return df[mask]


def _round(metric: str, value: float):
    if metric in ("units", "returns"):
        return int(value)
    return round(float(value), 6)


def _change(metric: str, previous: float, current: float) -> dict:
    change = current - previous
    return {
        "previous": _round(metric, previous),
        "current": _round(metric, current),
        "change": _round(metric, change),
        "change_percent": round(change / abs(previous) * 100, 2) if previous else None,
    }


def _two_proportion_z(x1: int, n1: int, x2: int, n2: int) -> float:
    """z-score for the change in a rate (returns / units) between two periods."""
    if not n1 or not n2:
        return 0.0
    pooled = (x1 + x2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    return round((x2 / n2 - x1 / n1) / se, 2) if se else 0.0


def compare_periods(
    df: pd.DataFrame,
    previous: Period,
    current: Period,
    filters: dict[str, list[str]] | None = None,
) -> dict:
    """KPIs for two periods (optionally for a filtered segment) and how they changed.

    For each metric: previous value, current value, absolute change, % change.
    """
    subset = apply_filters(df, filters)
    prev_df, cur_df = subset[previous.mask(subset)], subset[current.mask(subset)]
    prev_k, cur_k = calculate_kpis(prev_df), calculate_kpis(cur_df)

    metrics = {m: _change(m, prev_k[m], cur_k[m]) for m in METRICS}
    metrics["return_rate"]["z_score"] = _two_proportion_z(
        prev_k["returns"], prev_k["units"], cur_k["returns"], cur_k["units"]
    )
    return {
        "filters": filters or {},
        "period_previous": previous.to_dict(),
        "period_current": current.to_dict(),
        "rows_previous": int(len(prev_df)),
        "rows_current": int(len(cur_df)),
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Segment analysis
# ---------------------------------------------------------------------------

def _segment_name(filters: dict[str, list[str]]) -> str:
    return " / ".join(values[0] for values in filters.values()) if filters else "All business"


def _baselines(df: pd.DataFrame, filters: dict[str, list[str]]) -> dict[str, dict]:
    """Comparable peer groups: for each dimension, the same segment but 'everything else'.

    Region "South"                 -> other regions
    Region "South" + "Laptop Pro"  -> Laptop Pro in other regions,
                                      and other products in the South
    """
    baselines = {}
    for dim, values in filters.items():
        others = sorted(str(v) for v in df[dim].unique() if v not in values)
        if not others:
            continue
        base_filters = {d: (others if d == dim else v) for d, v in filters.items()}
        fixed = " / ".join(v[0] for d, v in filters.items() if d != dim)
        if not fixed:
            description = f"all other {dim}s"
        elif dim == "product":
            description = f"other products in {fixed}"
        else:
            description = f"{fixed} in other {dim}s"
        baselines[f"other_{dim}s"] = {"description": description, "filters": base_filters}
    return baselines


def _is_meaningful(pct, gaps: dict, t: Thresholds) -> bool:
    if pct is not None and abs(pct) >= t.decline_pct:
        return True
    return bool(gaps) and all(g is not None and abs(g) >= t.gap_pp for g in gaps.values())


def _analyze_segments(df, previous, current, dims, thresholds) -> list[dict]:
    previous, current = _resolve_periods(df, previous, current)
    t = thresholds or Thresholds()
    results = []

    for values in df[dims].drop_duplicates().sort_values(dims).itertuples(index=False):
        filters = {d: [str(v)] for d, v in zip(dims, values)}
        comparison = compare_periods(df, previous, current, filters)

        baselines = {}
        for name, base in _baselines(df, filters).items():
            base["comparison"] = compare_periods(df, previous, current, base["filters"])
            baselines[name] = base

        gaps_pp = {}
        for metric in DECLINE_METRICS:
            seg_pct = comparison["metrics"][metric]["change_percent"]
            gaps_pp[metric] = {
                name: (
                    round(seg_pct - b_pct, 2)
                    if seg_pct is not None and (b_pct := b["comparison"]["metrics"][metric]["change_percent"]) is not None
                    else None
                )
                for name, b in baselines.items()
            }

        # Drill-down one level: which sub-segments moved revenue the most?
        drivers = []
        for child_dim in (d for d in SEGMENT_DIMENSIONS if d not in dims):
            child_values = apply_filters(df, filters)[child_dim].unique()
            for child in sorted(str(v) for v in child_values):
                child_filters = {d: filters.get(d, [child]) for d in SEGMENT_DIMENSIONS}
                drivers.append(compare_periods(df, previous, current, child_filters))
        drivers.sort(key=lambda c: c["metrics"]["revenue"]["change"])

        results.append({
            "dimension": "+".join(dims),
            "dimension_value": _segment_name(filters),
            "filters": filters,
            "comparison": comparison,
            "baselines": baselines,
            "gaps_pp": gaps_pp,
            "drivers": drivers,
            "meaningful_changes": [
                m for m in ("revenue", "profit")
                if _is_meaningful(comparison["metrics"][m]["change_percent"], gaps_pp[m], t)
            ],
        })
    return results


def analyze_region_performance(df, previous=None, current=None, thresholds=None) -> list[dict]:
    """Each region vs. its own previous period and vs. all other regions."""
    return _analyze_segments(df, previous, current, ["region"], thresholds)


def analyze_product_performance(df, previous=None, current=None, thresholds=None) -> list[dict]:
    """Each product vs. its own previous period and vs. all other products."""
    return _analyze_segments(df, previous, current, ["product"], thresholds)


def analyze_region_product(df, previous=None, current=None, thresholds=None) -> list[dict]:
    """Each region+product combination vs. the same product elsewhere and the
    same region's other products, which shows whether a problem is concentrated."""
    return _analyze_segments(df, previous, current, ["region", "product"], thresholds)


def _overall_segment(df, previous, current) -> dict:
    return {
        "dimension": "overall",
        "dimension_value": "All business",
        "filters": {},
        "comparison": compare_periods(df, previous, current),
        "baselines": {},
        "gaps_pp": {},
        "drivers": [],
    }


# ---------------------------------------------------------------------------
# Formatting (used only for human-readable text; data stays numeric)
# ---------------------------------------------------------------------------

def _fmt_money(value: float) -> str:
    sign, a = ("-" if value < 0 else ""), abs(value)
    if a >= 1e6:
        return f"{sign}${a / 1e6:.2f}M"
    if a >= 1e3:
        return f"{sign}${a / 1e3:.1f}K"
    return f"{sign}${a:.2f}"


def _fmt_value(metric: str, value: float) -> str:
    if metric in MONEY_METRICS:
        return _fmt_money(value)
    if metric in RATE_METRICS:
        return f"{value * 100:.1f}%"
    return f"{int(value):,}"


def _fmt_pct(pct) -> str:
    return "n/a" if pct is None else f"{pct:+.1f}%"


def _fmt_pp(pp: float) -> str:
    return f"{pp:+.1f} pp"


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

def _evidence(comparison: dict, metric: str, role: str, description: str) -> dict:
    """One traceable fact: a metric for a filtered segment over two date ranges."""
    m = comparison["metrics"][metric]
    prev, cur = comparison["period_previous"], comparison["period_current"]
    item = {
        "role": role,
        "description": description,
        "metric": metric,
        "filters": comparison["filters"],
        "period_previous": prev["label"],
        "period_previous_range": [prev["start"], prev["end"]],
        "period_current": cur["label"],
        "period_current_range": [cur["start"], cur["end"]],
        "previous_value": m["previous"],
        "current_value": m["current"],
        "change": m["change"],
        "change_percent": m["change_percent"],
        "rows_previous": comparison["rows_previous"],
        "rows_current": comparison["rows_current"],
    }
    if metric == "return_rate":
        c = comparison["metrics"]
        item["details"] = {
            "returns_previous": c["returns"]["previous"],
            "units_previous": c["units"]["previous"],
            "returns_current": c["returns"]["current"],
            "units_current": c["units"]["current"],
            "z_score": m["z_score"],
        }
    return item


def _driver_evidence(driver: dict, parent: dict, parent_name: str) -> dict:
    item = _evidence(driver, "revenue", "driver",
                     f"Largest negative contributor to {parent_name} revenue change")
    parent_change = parent["metrics"]["revenue"]["change"]
    item["parent_filters"] = parent["filters"]
    item["parent_change"] = parent_change
    item["share_of_parent_change_percent"] = (
        round(driver["metrics"]["revenue"]["change"] / parent_change * 100, 2) if parent_change else None
    )
    return item


def _assign_ids(insights: list[dict]) -> None:
    """Give insights I1..In and unique evidence items E1..En (shared items share an id)."""
    seen: dict[str, str] = {}
    for i, insight in enumerate(insights, start=1):
        insight["id"] = f"I{i}"
        for item in insight["evidence"]:
            key = json.dumps({k: v for k, v in item.items() if k != "id"}, sort_keys=True)
            if key not in seen:
                seen[key] = f"E{len(seen) + 1}"
            item["id"] = seen[key]


# ---------------------------------------------------------------------------
# Insight detection
# ---------------------------------------------------------------------------

def _severity(magnitude: float, medium: float, high: float) -> str:
    return "high" if magnitude >= high else "medium" if magnitude >= medium else "low"


def _insight(type_, title, summary, severity, metric, segment, evidence, **extra) -> dict:
    m = segment["comparison"]["metrics"][metric]
    comparison = segment["comparison"]
    return {
        "id": None,
        "type": type_,
        "title": title,
        "summary": summary,
        "severity": severity,
        "metric": metric,
        "previous_value": m["previous"],
        "current_value": m["current"],
        "change_percent": m["change_percent"],
        "dimension": segment["dimension"],
        "dimension_value": segment["dimension_value"],
        "filters": segment["filters"],
        "period_previous": comparison["period_previous"]["label"],
        "period_current": comparison["period_current"]["label"],
        **extra,
        "evidence": evidence,
    }


def _segment_insights(segment: dict, t: Thresholds) -> list[dict]:
    comparison = segment["comparison"]
    metrics = comparison["metrics"]
    name = segment["dimension_value"]
    p_label = comparison["period_previous"]["label"]
    c_label = comparison["period_current"]["label"]
    if metrics["units"]["previous"] < t.min_units:
        return []

    insights = []
    context_rr = _evidence(comparison, "return_rate", "context", f"{name} return rate")

    # A/B/C: significant revenue, profit or unit declines.
    for metric in DECLINE_METRICS:
        m = metrics[metric]
        pct = m["change_percent"]
        if pct is None or pct > -t.decline_pct:
            continue
        evidence = [_evidence(comparison, metric, "segment", f"{name} {metric}"), context_rr]
        if metric == "profit":
            evidence.append(_evidence(comparison, "profit_margin", "context", f"{name} profit margin"))
        insights.append(_insight(
            f"{metric}_decline",
            f"{metric.capitalize()} decline: {name}",
            f"{name} {metric} fell {abs(pct):.1f}%, from {_fmt_value(metric, m['previous'])} "
            f"({p_label}) to {_fmt_value(metric, m['current'])} ({c_label}).",
            _severity(abs(pct), 15, 20), metric, segment, evidence,
        ))

    # D: significant return-rate increase (relative, absolute, and statistically).
    rr = metrics["return_rate"]
    rr_pct, rr_pp = rr["change_percent"], rr["change"] * 100
    if (
        rr_pct is not None
        and rr_pct >= t.return_rate_increase_pct
        and rr_pp >= t.return_rate_increase_pp
        and rr["z_score"] >= t.min_z_score
    ):
        evidence = [_evidence(comparison, "return_rate", "segment", f"{name} return rate")]
        for base in segment["baselines"].values():
            evidence.append(_evidence(base["comparison"], "return_rate", "baseline",
                                      f"{base['description']} return rate"))
        insights.append(_insight(
            "return_rate_increase",
            f"Return rate increase: {name}",
            f"{name} return rate rose from {_fmt_value('return_rate', rr['previous'])} ({p_label}) "
            f"to {_fmt_value('return_rate', rr['current'])} ({c_label}), {_fmt_pp(rr_pp)} "
            f"({_fmt_pct(rr_pct)} relative, z = {rr['z_score']:.1f}).",
            _severity(rr_pct, 30, 50), "return_rate", segment, evidence,
            change_pp=round(rr_pp, 2),
        ))

    # E/F/G: decline that is specific to this segment rather than business-wide.
    gaps = segment["gaps_pp"].get("revenue", {})
    rev = metrics["revenue"]
    if (
        gaps
        and rev["change_percent"] is not None
        and rev["change_percent"] < 0
        and all(g is not None and g <= -t.gap_pp for g in gaps.values())
    ):
        evidence = [_evidence(comparison, "revenue", "segment", f"{name} revenue")]
        comparisons_text = []
        for key, base in segment["baselines"].items():
            b_pct = base["comparison"]["metrics"]["revenue"]["change_percent"]
            evidence.append(_evidence(base["comparison"], "revenue", "baseline",
                                      f"{base['description']} revenue"))
            comparisons_text.append(
                f"{base['description']} changed {_fmt_pct(b_pct)} (gap {_fmt_pp(gaps[key])})"
            )
        evidence.append(context_rr)
        summary = (
            f"{name} revenue changed {_fmt_pct(rev['change_percent'])} "
            f"({_fmt_value('revenue', rev['previous'])} to {_fmt_value('revenue', rev['current'])}), "
            f"while {' and '.join(comparisons_text)}."
        )
        top = segment["drivers"][0] if segment["drivers"] else None
        if top and top["metrics"]["revenue"]["change"] < 0:
            evidence.append(_driver_evidence(top, comparison, name))
            summary += (
                f" Largest negative contributor: {_segment_name(top['filters'])} "
                f"({_fmt_money(top['metrics']['revenue']['change'])}, against a net change of "
                f"{_fmt_money(rev['change'])} for {name})."
            )
        type_ = SPECIFIC_DECLINE_TYPE[tuple(segment["filters"])]
        insights.append(_insight(
            type_,
            f"{type_.replace('_', ' ').capitalize()}: {name}",
            summary,
            _severity(min(abs(g) for g in gaps.values()), 10, 20), "revenue", segment, evidence,
            gaps_pp=gaps,
        ))
    return insights


def detect_insights(
    df: pd.DataFrame,
    previous: Period | None = None,
    current: Period | None = None,
    thresholds: Thresholds | None = None,
) -> list[dict]:
    """Scan the business overall, by region, by product and by region+product,
    and return every significant finding, most severe first."""
    previous, current = _resolve_periods(df, previous, current)
    t = thresholds or Thresholds()
    segments = (
        [_overall_segment(df, previous, current)]
        + analyze_region_performance(df, previous, current, t)
        + analyze_product_performance(df, previous, current, t)
        + analyze_region_product(df, previous, current, t)
    )
    insights = [i for segment in segments for i in _segment_insights(segment, t)]
    insights.sort(key=lambda i: (SEVERITY_RANK[i["severity"]], -abs(i["change_percent"] or 0)))
    _assign_ids(insights)
    return insights


# ---------------------------------------------------------------------------
# Recommendations (rule-based, evidence-only)
# ---------------------------------------------------------------------------

RULES = {
    "decline_with_rising_returns": (
        "Investigate product quality, fulfillment, or customer experience for {segment} "
        "before increasing inventory or marketing spend."
    ),
    "decline_with_stable_returns": (
        "Investigate demand, pricing, availability, or regional sales performance for {segment}."
    ),
    "rising_returns_without_decline": (
        "Audit returns for {segment} (defect reasons, delivery damage, listing accuracy) "
        "before the rise starts to affect sales."
    ),
    "profit_decline_with_stable_revenue": (
        "Review costs, discount depth, and pricing for {segment}."
    ),
    "margin_erosion": (
        "Review discount depth and pricing for {segment}; margin is shrinking on top of the volume loss."
    ),
}


def _segment_key(filters: dict) -> tuple:
    return tuple(sorted((k, tuple(v)) for k, v in filters.items()))


def _find_evidence(insight: dict, role: str, metric: str) -> dict | None:
    return next((e for e in insight["evidence"] if e["role"] == role and e["metric"] == metric), None)


def generate_recommendation(
    segment_insights: list[dict],
    context_insights: list[dict] = (),
    thresholds: Thresholds | None = None,
) -> dict | None:
    """Turn the insights for ONE segment into a recommendation.

    Which rules fire depends only on which insight types were detected. The
    rationale quotes numbers from those insights' evidence.
    context_insights are parent-level findings this segment explains (for example,
    the South region decline explained by South / Laptop Pro). They add
    rationale but don't change which rules fire.
    """
    if not segment_insights:
        return None
    t = thresholds or Thresholds()
    by_type = {i["type"]: i for i in segment_insights}
    first = segment_insights[0]
    segment = first["dimension_value"]
    p_label, c_label = first["period_previous"], first["period_current"]

    declining = next((by_type[t] for t in (
        "concentrated_decline", "region_specific_decline", "product_specific_decline",
        "revenue_decline", "units_decline") if t in by_type), None)
    returns_up = by_type.get("return_rate_increase")
    profit_down = by_type.get("profit_decline")

    rules, rationale = [], []
    if declining and returns_up:
        rules.append("decline_with_rising_returns")
    elif declining:
        rules.append("decline_with_stable_returns")
    elif returns_up:
        rules.append("rising_returns_without_decline")
    elif profit_down:
        rules.append("profit_decline_with_stable_revenue")

    if declining:
        rationale.append(declining["summary"])
        if not returns_up:
            rr = _find_evidence(declining, "context", "return_rate")
            if rr:
                rationale.append(
                    f"Return rate is not a detected issue: {_fmt_value('return_rate', rr['previous_value'])} "
                    f"({p_label}) vs {_fmt_value('return_rate', rr['current_value'])} ({c_label})."
                )
    if returns_up:
        rationale.append(returns_up["summary"])
    if profit_down:
        margin = _find_evidence(profit_down, "context", "profit_margin")
        rationale.append(profit_down["summary"])
        if margin and margin["change"] * 100 <= -t.margin_drop_pp:
            rules.append("margin_erosion")
            rationale.append(
                f"Profit margin fell from {_fmt_value('profit_margin', margin['previous_value'])} to "
                f"{_fmt_value('profit_margin', margin['current_value'])} ({_fmt_pp(margin['change'] * 100)})."
            )
    for parent in context_insights:
        rationale.append(f"This explains a wider finding: {parent['summary']}")

    if not rules:
        return None
    all_insights = list(segment_insights) + list(context_insights)
    evidence_ids = list(dict.fromkeys(e["id"] for i in all_insights for e in i["evidence"]))
    return {
        "id": None,
        "segment": segment,
        "dimension": first["dimension"],
        "filters": first["filters"],
        "priority": min((i["severity"] for i in segment_insights), key=SEVERITY_RANK.get),
        "rules_triggered": rules,
        "actions": [RULES[r].format(segment=segment) for r in rules],
        "rationale": rationale,
        "based_on_insights": [i["id"] for i in all_insights],
        "evidence_ids": evidence_ids,
    }


def generate_recommendations(insights: list[dict], thresholds: Thresholds | None = None) -> list[dict]:
    """One recommendation per affected segment.

    A region-level or product-level decline whose largest driver is a
    region+product segment with its own findings is folded into that
    segment's recommendation as context, so the same problem isn't recommended
    on three times.
    """
    groups: dict[tuple, list[dict]] = {}
    for insight in insights:
        groups.setdefault(_segment_key(insight["filters"]), []).append(insight)

    context: dict[tuple, list[dict]] = {}
    absorbed: set[str] = set()
    for insight in insights:
        driver = _find_evidence(insight, "driver", "revenue")
        if driver and _segment_key(driver["filters"]) in groups:
            context.setdefault(_segment_key(driver["filters"]), []).append(insight)
            absorbed.add(insight["id"])

    recommendations = []
    for key, group in groups.items():
        own = [i for i in group if i["id"] not in absorbed]
        if not own:
            continue
        rec = generate_recommendation(own, context.get(key, []), thresholds)
        if rec:
            recommendations.append(rec)

    recommendations.sort(key=lambda r: SEVERITY_RANK[r["priority"]])
    for i, rec in enumerate(recommendations, start=1):
        rec["id"] = f"R{i}"
    return recommendations


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _executive_summary(overview: dict, insights: list[dict], recommendations: list[dict]) -> str:
    m = overview["metrics"]
    p_label = overview["period_previous"]["label"]
    c_label = overview["period_current"]["label"]
    text = (
        f"From {p_label} to {c_label}, total revenue changed {_fmt_pct(m['revenue']['change_percent'])} "
        f"({_fmt_money(m['revenue']['previous'])} to {_fmt_money(m['revenue']['current'])}) and profit "
        f"changed {_fmt_pct(m['profit']['change_percent'])}."
    )
    if not insights:
        return text + " No significant declines or return-rate increases were detected."
    counts = {s: sum(i["severity"] == s for i in insights) for s in SEVERITY_RANK}
    text += (
        f" {len(insights)} issues detected ({counts['high']} high, {counts['medium']} medium, "
        f"{counts['low']} low severity). Most severe: {insights[0]['summary']}"
    )
    if recommendations:
        text += f" Top recommendation: {recommendations[0]['actions'][0]}"
    return text


def generate_decision_report(
    data: pd.DataFrame | str | Path,
    previous: Period | None = None,
    current: Period | None = None,
    thresholds: Thresholds | None = None,
) -> dict:
    """Full pipeline from a CSV path (or DataFrame) to a JSON-serializable report."""
    if isinstance(data, (str, Path)):
        source, df = str(data), load_data(data)
    else:
        source, df = "DataFrame", validate_data(data)
    previous, current = _resolve_periods(df, previous, current)
    t = thresholds or Thresholds()

    overview = compare_periods(df, previous, current)
    insights = detect_insights(df, previous, current, t)
    recommendations = generate_recommendations(insights, t)

    evidence = {}
    for insight in insights:
        for item in insight["evidence"]:
            evidence.setdefault(item["id"], item)

    return {
        "metadata": {
            "source": source,
            "rows": int(len(df)),
            "period_previous": previous.to_dict(),
            "period_current": current.to_dict(),
            "thresholds": asdict(t),
            "insight_count": len(insights),
            "severity_counts": {s: sum(i["severity"] == s for i in insights) for s in SEVERITY_RANK},
        },
        "executive_summary": _executive_summary(overview, insights, recommendations),
        "overview": overview,
        "key_insights": insights,
        "evidence": list(evidence.values()),
        "recommendations": recommendations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a decision report from sales data.")
    parser.add_argument("csv", nargs="?", default=str(Path(__file__).parent / "data" / "sales.csv"))
    parser.add_argument("--out", help="write the JSON report here instead of printing it")
    args = parser.parse_args()

    report = json.dumps(generate_decision_report(args.csv), indent=2)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"Report written to {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
