"""Tests for story.py, the top-issue presentation layer."""

import json

import numpy as np
import pandas as pd
import pytest

import llm
from analyzer import validate_data
from data_generator import generate_sales_data
from engine import generate_decision_report
from story import build_story, pick_top_issue


def _scale_h2(df, mask, factor):
    """Shrink volume for the masked H2 rows, keeping the data valid."""
    rows = mask & (df["date"] >= "2025-07-01")
    for col in ("units", "revenue", "cost"):
        df.loc[rows, col] = df.loc[rows, col] * factor
    df.loc[rows, "units"] = df.loc[rows, "units"].round()
    df.loc[rows, "returns"] = np.minimum((df.loc[rows, "returns"] * factor).round(), df.loc[rows, "units"])
    return df


@pytest.fixture(scope="module")
def noisy():
    """Like the dataset from the demo test: everything softer in H2, the South much weaker,
    and South / Laptop Pro worst of all. Produces many overlapping findings."""
    df = validate_data(generate_sales_data()).astype({"units": float, "returns": float})
    df = _scale_h2(df, df["region"].notna(), 0.93)
    df = _scale_h2(df, df["region"] == "South", 0.85)
    df = df.astype({"units": int, "returns": int})
    df = validate_data(df)
    report = generate_decision_report(df)
    return df, report, build_story(df, report)


@pytest.fixture(scope="module")
def demo():
    df = validate_data(generate_sales_data())
    report = generate_decision_report(df)
    return df, report, build_story(df, report)


def _recompute(df, filters, date_range, metric):
    subset = df
    for col, values in filters.items():
        subset = subset[subset[col].isin(values)]
    subset = subset[(subset["date"] >= date_range[0]) & (subset["date"] <= date_range[1])]
    revenue, cost = subset["revenue"].sum(), subset["cost"].sum()
    return {
        "revenue": revenue, "profit": revenue - cost, "units": subset["units"].sum(),
        "return_rate": subset["returns"].sum() / subset["units"].sum(),
        "profit_margin": (revenue - cost) / revenue,
    }[metric]


def test_noisy_data_has_many_findings_but_one_story(noisy):
    _, report, story = noisy
    assert report["metadata"]["insight_count"] > 10       # the engine still finds everything
    assert len(report["recommendations"]) > 3
    assert story["problem"]["segment"] == "South / Laptop Pro"


def test_full_report_is_not_modified(noisy):
    df, report, _ = noisy
    fresh = generate_decision_report(df)
    assert json.loads(json.dumps(report)) == json.loads(json.dumps(fresh))


def test_overview_sentence_follows_the_data(noisy, demo):
    assert noisy[2]["overview"]["sentence"] == (
        "Overall business performance declined, but the largest concentrated problem is South / Laptop Pro.")
    assert noisy[2]["overview"]["revenue_growth_percent"] < 0
    assert demo[2]["overview"]["sentence"] == (
        "Overall business performance grew, but the largest concentrated problem is South / Laptop Pro.")


def test_problem_cards(noisy):
    p = noisy[2]["problem"]
    m = p["metrics"]
    assert set(m) == {"revenue", "profit", "units", "return_rate"}
    assert m["revenue"]["change_percent"] < -30
    assert m["return_rate"]["current"] > m["return_rate"]["previous"]
    assert p["explanation"] == ("South / Laptop Pro is performing significantly worse than both its "
                                "historical performance and its peer groups.")


def test_peers_and_emphasis(noisy):
    p = noisy[2]["problem"]
    assert [x["label"] for x in p["peers"]] == ["Laptop Pro in other regions", "Other products in South"]
    assert all(x["change_percent"] < 0 for x in p["peers"])  # everything declined in this data
    assert p["worse_than_peers"]
    assert p["emphasis"] == "South / Laptop Pro is declining much faster than its peers."


def test_exactly_three_possible_causes_each_linked_to_a_signal(noisy):
    causes = noisy[2]["problem"]["possible_causes"]
    assert [c["cause"] for c in causes] == [
        "Product quality or defect issues",
        "Fulfillment / delivery problems",
        "Demand, pricing, or competition changes",
    ]
    m = noisy[2]["problem"]["metrics"]
    assert [c["metric"] for c in causes] == ["return_rate", "return_rate", "revenue"]
    assert causes[0]["signal"].startswith(
        f"Return rate {m['return_rate']['previous'] * 100:.1f}% → {m['return_rate']['current'] * 100:.1f}%")
    assert causes[2]["signal"].startswith("Revenue ")
    for c in causes:
        assert c["statement"] == f"{c['cause']} may be contributing. This requires investigation."
    rr = m["return_rate"]
    assert causes[0]["evidence_text"] == (
        f"Return rate increased from {rr['previous'] * 100:.1f}% to {rr['current'] * 100:.1f}%.")
    assert causes[2]["evidence_text"].startswith(
        f"Revenue declined by {abs(m['revenue']['change_percent']):.1f}% (")


def test_cause_is_dropped_when_its_signal_did_not_move():
    from story import linked_possible_causes

    flat = {"change": 0.0, "previous": 0.05, "current": 0.05, "change_percent": 0.0}
    down = {"change": -10, "previous": 100, "current": 90, "change_percent": -10.0}
    causes = linked_possible_causes("decline_with_rising_returns",
                                    {"return_rate": flat, "revenue": down})
    assert [c["cause"] for c in causes] == ["Demand, pricing, or competition changes"]


def test_one_primary_and_one_secondary_action(noisy, demo):
    action = noisy[2]["problem"]["action"]
    assert action["primary"] == ("Investigate South / Laptop Pro product quality, fulfillment, and "
                                 "customer experience before increasing inventory or marketing spend.")
    assert action["why"].startswith("Because revenue, profit, and units declined sharply while the return rate")
    # Other South products also fell, so the secondary action is about the region.
    assert action["secondary"] == "Review pricing, demand, and availability in the South region."
    assert "Other products in South also fell" in action["secondary_why"]
    # In the demo data other South products grew, but margins shrank.
    demo_action = demo[2]["problem"]["action"]
    assert demo_action["secondary"] == "Review discount depth and pricing for South / Laptop Pro."


@pytest.mark.parametrize("previous,current,phrase", [
    (0.044, 0.098, "more than doubled"),
    (0.047, 0.085, "rose sharply"),
    (0.040, 0.045, "rose ("),
    (0.050, 0.040, "did not rise"),
])
def test_why_wording_follows_the_return_rate(noisy, previous, current, phrase):
    from story import _action

    m = json.loads(json.dumps(noisy[2]["problem"]["metrics"]))
    m["return_rate"].update(previous=previous, current=current, change=current - previous)
    m["profit_margin"] = {"previous": 0.3, "current": 0.3, "change": 0.0, "change_percent": 0.0}
    action = _action("S / P", {"region": ["S"], "product": ["P"]}, m, [],
                     ["decline_with_rising_returns"], "decline_with_rising_returns",
                     __import__("engine").Thresholds())
    assert phrase in action["why"]


def test_every_story_number_is_traceable_to_the_csv(noisy):
    df, _, story = noisy
    evidence = story["problem"]["evidence"]
    shown = {(tuple(sorted((k, tuple(v)) for k, v in e["filters"].items())), e["metric"]) for e in evidence}
    for metric in ("revenue", "profit", "units", "return_rate"):
        assert ((("product", ("Laptop Pro",)), ("region", ("South",))), metric) in shown
    for item in evidence:
        for side in ("previous", "current"):
            expected = _recompute(df, item["filters"], item[f"period_{side}_range"], item["metric"])
            assert item[f"{side}_value"] == pytest.approx(expected, abs=1e-5), item["description"]
    # Step 2 numbers are the peers' evidence
    for peer in story["problem"]["peers"]:
        match = next(e for e in evidence if e["filters"] == peer["filters"])
        assert match["change_percent"] == peer["change_percent"]


def test_focused_report_is_a_subset(noisy):
    _, report, story = noisy
    focused = story["focused_report"]
    all_ids = {i["id"] for i in report["key_insights"]}
    focused_ids = {i["id"] for i in focused["key_insights"]}
    assert focused_ids < all_ids
    assert len(focused["recommendations"]) == 1
    assert focused["recommendations"][0]["segment"] == "South / Laptop Pro"
    evidence_ids = [e["id"] for e in focused["evidence"]]
    assert len(evidence_ids) == len(set(evidence_ids))
    report_ids = {e["id"] for e in report["evidence"]}
    for e in focused["evidence"]:  # reused engine IDs where the same fact exists
        assert e["id"] in report_ids or e["id"].startswith("S")


def test_focused_summary_is_short_and_grounded(noisy):
    focused = noisy[2]["focused_report"]
    summary = focused["executive_summary"]
    assert 2 <= summary.count(". ") + 1 <= 3
    assert llm.unsupported_numbers(summary, llm.allowed_numbers(focused)) == []
    assert "does not show the cause" in summary


def test_fallback_explanation_on_focused_report_is_valid(noisy):
    focused = noisy[2]["focused_report"]
    result = llm.explain_decision_report(focused, use_llm=False)
    assert llm.validate_explanation(result, focused) == []
    assert result["summary"] == focused["executive_summary"]


def test_llm_gets_only_the_focused_context(noisy):
    _, report, story = noisy
    full = llm.build_messages(report)[1]["content"]
    focused = llm.build_messages(story["focused_report"])[1]["content"]
    assert len(focused) < len(full) / 2


def test_story_is_json_serializable(noisy, demo):
    for _, _, story in (noisy, demo):
        assert json.loads(json.dumps(story, allow_nan=False)) == story


def test_healthy_data_has_no_problem():
    df = validate_data(generate_sales_data(anomaly_start="2030-01-01"))
    report = generate_decision_report(df)
    story = build_story(df, report)
    assert story["problem"] is None
    assert pick_top_issue(report) is None
    assert story["overview"]["sentence"].startswith("No significant problems")
    assert story["question_placeholder"] == "Why did revenue decline?"


def test_question_placeholder_names_the_segment(noisy):
    assert noisy[2]["question_placeholder"] == "Why did South / Laptop Pro revenue decline?"


DEMO_QUESTIONS = [
    "Why did South / Laptop Pro revenue decline?",
    "How much did profit fall for South / Laptop Pro?",
    "Is the problem only in the South region?",
    "What should we do first?",
    "What will revenue be next quarter?",
]


@pytest.mark.parametrize("question", DEMO_QUESTIONS)
def test_demo_questions_never_invent_numbers(noisy, question):
    """The 5 demo questions, answered from the focused report without an LLM."""
    focused = noisy[2]["focused_report"]
    answer = llm.answer_question(focused, question, use_llm=False)
    assert llm.validate_answer(answer, focused) == []
    allowed = llm.allowed_numbers(focused)
    causes = answer["possible_explanations"]
    for text in [answer["answer"]] + [c["cause"] for c in causes] + [c["evidence"] for c in causes]:
        assert llm.unsupported_numbers(text, allowed) == [], text
    for c in causes:
        assert c["linked_insight_ids"]
    assert answer["evidence_references"]


def test_forecast_question_gets_no_forecast(noisy):
    answer = llm.answer_question(noisy[2]["focused_report"], "What will revenue be next quarter?",
                                 use_llm=False)
    assert answer["answer"].startswith("The report cannot forecast future results; it only compares "
                                      "2025 H1 and 2025 H2.")


@pytest.mark.parametrize("question,first_metric", [
    ("Why did South / Laptop Pro revenue decline?", "revenue"),
    ("How much did profit fall for South / Laptop Pro?", "profit"),
    ("What happened to South / Laptop Pro returns?", "return_rate"),
])
def test_fallback_answer_leads_with_the_asked_metric_for_the_named_segment(noisy, question, first_metric):
    focused = noisy[2]["focused_report"]
    answer = llm.answer_question(focused, question, use_llm=False)
    by_id = {i["id"]: i for i in focused["key_insights"]}
    first = by_id[answer["insight_ids"][0]]
    assert first["metric"] == first_metric
    assert first["dimension_value"] == "South / Laptop Pro"



def test_significance_checks_use_the_engine_thresholds(noisy):
    from engine import Thresholds

    t = Thresholds()
    checks = {c["metric"]: c for c in noisy[2]["problem"]["significance"]}
    assert set(checks) == {"revenue", "profit", "units", "return_rate"}
    m = noisy[2]["problem"]["metrics"]
    for metric in ("revenue", "profit", "units"):
        assert checks[metric]["significant"] is (m[metric]["change_percent"] <= -t.decline_pct)
        assert f"{t.decline_pct:.0f}%" in checks[metric]["criterion"]
    rr = checks["return_rate"]
    assert rr["significant"] and "z = " in rr["value"] and f"z ≥ {t.min_z_score:g}" in rr["criterion"]


def test_significance_marks_small_changes_as_not_significant():
    from engine import Thresholds
    from story import significance_checks

    small = {"change_percent": -2.0}
    rr = {"change_percent": 5.0, "change": 0.001, "z_score": 0.4}
    checks = significance_checks({"revenue": small, "profit": small, "units": small, "return_rate": rr},
                                 [], "S / P", Thresholds())
    assert not any(c["significant"] for c in checks)


def test_supporting_findings_list_the_engine_insights(noisy):
    p = noisy[2]["problem"]
    focused = noisy[2]["focused_report"]
    assert [f["id"] for f in p["findings"]] == [i["id"] for i in focused["key_insights"]]
    evidence_ids = {e["id"] for e in focused["evidence"]}
    for f in p["findings"]:
        assert f["evidence_ids"] and set(f["evidence_ids"]) <= evidence_ids
        assert f["label"] and f["severity"] in {"high", "medium", "low"}


def test_action_explains_its_rule(noisy):
    action = noisy[2]["problem"]["action"]
    assert noisy[2]["problem"]["rule"] == "decline_with_rising_returns"
    assert action["rule_description"].startswith("Sales fell significantly while the return rate rose")


def test_cause_statements_never_claim_causation(noisy):
    for c in noisy[2]["problem"]["possible_causes"]:
        assert "may be contributing" in c["statement"]
        assert not llm.states_cause(c["statement"])
