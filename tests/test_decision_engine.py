"""Tests for engine.py.

The key checks: the planted problem is detected, every evidence number can be
recomputed from the raw CSV, and every number in generated text comes from
the structured evidence.
"""

import json
import re

import numpy as np
import pandas as pd
import pytest

from analyzer import load_data, validate_data
from data_generator import (
    ANOMALY_PRODUCT,
    ANOMALY_REGION,
    generate_sales_data,
    save_dataset,
)
from engine import (
    Period,
    Thresholds,
    analyze_product_performance,
    analyze_region_performance,
    analyze_region_product,
    apply_filters,
    compare_periods,
    detect_insights,
    generate_decision_report,
    generate_recommendation,
    generate_recommendations,
    split_periods,
)

H1 = Period("2025 H1", "2025-01-01", "2025-06-30")
H2 = Period("2025 H2", "2025-07-01", "2025-12-31")
TARGET = f"{ANOMALY_REGION} / {ANOMALY_PRODUCT}"
INSIGHT_KEYS = {
    "title", "severity", "metric", "current_value", "previous_value", "change_percent",
    "dimension", "dimension_value", "evidence",
}


@pytest.fixture(scope="module")
def csv_path(tmp_path_factory):
    return save_dataset(generate_sales_data(), tmp_path_factory.mktemp("data") / "sales.csv")


@pytest.fixture(scope="module")
def raw(csv_path):
    """The CSV read with plain pandas, independent of analyzer/engine code."""
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture(scope="module")
def df(csv_path):
    return load_data(csv_path)


@pytest.fixture(scope="module")
def report(csv_path):
    return generate_decision_report(csv_path)


@pytest.fixture(scope="module")
def insights(report):
    return report["key_insights"]


def _find(insights, type_, dimension_value):
    return [i for i in insights if i["type"] == type_ and i["dimension_value"] == dimension_value]


def _control_df():
    """Same generator but the planted problem never starts: a healthy business."""
    return validate_data(generate_sales_data(anomaly_start="2030-01-01"))


# --- Periods & compare_periods ----------------------------------------------

def test_split_periods_gives_calendar_halves(df):
    previous, current = split_periods(df)
    assert previous == H1
    assert current == H2


def test_compare_periods_matches_pandas(df, raw):
    result = compare_periods(df, H1, H2, {"region": ["North"]})
    north = raw[raw["region"] == "North"]
    prev = north[north["date"] <= "2025-06-30"]
    cur = north[north["date"] >= "2025-07-01"]

    rev = result["metrics"]["revenue"]
    assert rev["previous"] == pytest.approx(prev["revenue"].sum())
    assert rev["current"] == pytest.approx(cur["revenue"].sum())
    assert rev["change"] == pytest.approx(cur["revenue"].sum() - prev["revenue"].sum())
    assert rev["change_percent"] == pytest.approx(
        (cur["revenue"].sum() / prev["revenue"].sum() - 1) * 100, abs=0.01
    )
    profit_prev = prev["revenue"].sum() - prev["cost"].sum()
    assert result["metrics"]["profit"]["previous"] == pytest.approx(profit_prev)
    assert result["metrics"]["units"]["current"] == cur["units"].sum()
    assert result["metrics"]["returns"]["current"] == cur["returns"].sum()
    assert result["metrics"]["return_rate"]["current"] == pytest.approx(
        cur["returns"].sum() / cur["units"].sum(), abs=1e-6
    )
    assert result["rows_previous"] == len(prev)
    assert result["rows_current"] == len(cur)


def test_compare_periods_zero_previous_has_no_percent():
    frame = validate_data(pd.DataFrame({
        "date": ["2025-01-05", "2025-07-05"], "region": ["North"] * 2, "product": ["Tablet"] * 2,
        "units": [0, 10], "revenue": [0.0, 500.0], "cost": [0.0, 300.0], "returns": [0, 1],
    }))
    result = compare_periods(frame, H1, H2)
    assert result["metrics"]["revenue"]["change"] == 500.0
    assert result["metrics"]["revenue"]["change_percent"] is None
    assert result["metrics"]["return_rate"]["previous"] == 0.0


def test_apply_filters(df):
    subset = apply_filters(df, {"region": ["South"], "product": ["Tablet", "Laptop Pro"]})
    assert set(subset["region"]) == {"South"}
    assert set(subset["product"]) == {"Tablet", "Laptop Pro"}
    assert len(apply_filters(df, None)) == len(df)


# --- Segment analysis --------------------------------------------------------

def test_region_analysis_flags_south(df):
    rows = {r["dimension_value"]: r for r in analyze_region_performance(df)}
    assert set(rows) == {"North", "South", "East", "West"}
    south = rows["South"]
    assert south["comparison"]["metrics"]["revenue"]["change_percent"] < 0
    assert south["gaps_pp"]["revenue"]["other_regions"] <= -5
    assert "revenue" in south["meaningful_changes"]
    for name in ("North", "East", "West"):
        assert rows[name]["comparison"]["metrics"]["revenue"]["change_percent"] > 0


def test_product_analysis_laptop_pro_is_worst(df):
    rows = analyze_product_performance(df)
    worst = min(rows, key=lambda r: r["comparison"]["metrics"]["revenue"]["change_percent"])
    assert worst["dimension_value"] == ANOMALY_PRODUCT
    # Its biggest drag is the South.
    assert worst["drivers"][0]["filters"] == {"region": ["South"], "product": ["Laptop Pro"]}


def test_region_product_isolates_combination(df):
    rows = analyze_region_product(df)
    assert len(rows) == 20
    worst = min(rows, key=lambda r: r["comparison"]["metrics"]["revenue"]["change_percent"])
    assert worst["dimension_value"] == TARGET
    gaps = worst["gaps_pp"]["revenue"]
    assert set(gaps) == {"other_regions", "other_products"}
    assert all(g <= -20 for g in gaps.values())


# --- Planted problem detection (required checks 1-3) -------------------------

def test_detects_south_region_decline(insights):
    found = _find(insights, "region_specific_decline", "South")
    assert len(found) == 1
    assert found[0]["change_percent"] < 0
    driver = next(e for e in found[0]["evidence"] if e["role"] == "driver")
    assert driver["filters"] == {"region": ["South"], "product": ["Laptop Pro"]}


def test_detects_south_laptop_pro_decline(insights):
    for type_ in ("revenue_decline", "units_decline", "profit_decline", "concentrated_decline"):
        found = _find(insights, type_, TARGET)
        assert len(found) == 1, type_
        assert found[0]["change_percent"] <= -10
    assert _find(insights, "concentrated_decline", TARGET)[0]["severity"] == "high"


def test_detects_south_laptop_pro_return_rate_increase(insights):
    [found] = _find(insights, "return_rate_increase", TARGET)
    assert found["severity"] == "high"
    assert found["current_value"] > found["previous_value"]
    assert found["change_pp"] >= 2


def test_detects_laptop_pro_product_decline(insights):
    assert len(_find(insights, "product_specific_decline", ANOMALY_PRODUCT)) == 1


def test_no_findings_outside_planted_problem(insights):
    allowed = {ANOMALY_REGION, ANOMALY_PRODUCT, TARGET}
    assert {i["dimension_value"] for i in insights} <= allowed


def test_healthy_business_has_no_findings():
    control = _control_df()
    assert detect_insights(control) == []
    report = generate_decision_report(control)
    assert report["recommendations"] == []
    assert "No significant declines" in report["executive_summary"]


def test_insight_schema(insights):
    assert insights
    for insight in insights:
        assert INSIGHT_KEYS <= insight.keys()
        assert insight["severity"] in {"high", "medium", "low"}
        assert insight["evidence"]


def test_insights_sorted_by_severity(insights):
    ranks = [{"high": 0, "medium": 1, "low": 2}[i["severity"]] for i in insights]
    assert ranks == sorted(ranks)


# --- Evidence traceability (required check 4) --------------------------------

def _recompute(raw, filters, date_range):
    subset = raw
    for col, values in filters.items():
        subset = subset[subset[col].isin(values)]
    subset = subset[(subset["date"] >= date_range[0]) & (subset["date"] <= date_range[1])]
    revenue, cost = subset["revenue"].sum(), subset["cost"].sum()
    units, returns = subset["units"].sum(), subset["returns"].sum()
    values = {
        "revenue": revenue, "cost": cost, "profit": revenue - cost,
        "profit_margin": (revenue - cost) / revenue if revenue else 0.0,
        "units": units, "returns": returns,
        "return_rate": returns / units if units else 0.0,
    }
    return values, len(subset)


def test_every_evidence_item_matches_csv(report, raw):
    assert report["evidence"]
    for item in report["evidence"]:
        prev, n_prev = _recompute(raw, item["filters"], item["period_previous_range"])
        cur, n_cur = _recompute(raw, item["filters"], item["period_current_range"])
        metric = item["metric"]
        assert item["previous_value"] == pytest.approx(prev[metric], abs=1e-5), item["id"]
        assert item["current_value"] == pytest.approx(cur[metric], abs=1e-5), item["id"]
        assert item["change"] == pytest.approx(cur[metric] - prev[metric], abs=1e-5)
        assert item["change_percent"] == pytest.approx(
            (cur[metric] - prev[metric]) / abs(prev[metric]) * 100, abs=0.01
        )
        assert (item["rows_previous"], item["rows_current"]) == (n_prev, n_cur)

        if metric == "return_rate":
            d = item["details"]
            assert (d["returns_previous"], d["units_previous"]) == (prev["returns"], prev["units"])
            assert (d["returns_current"], d["units_current"]) == (cur["returns"], cur["units"])
        if item["role"] == "driver":
            parent_prev, _ = _recompute(raw, item["parent_filters"], item["period_previous_range"])
            parent_cur, _ = _recompute(raw, item["parent_filters"], item["period_current_range"])
            parent_change = parent_cur["revenue"] - parent_prev["revenue"]
            assert item["parent_change"] == pytest.approx(parent_change, abs=1e-5)
            assert item["share_of_parent_change_percent"] == pytest.approx(
                item["change"] / parent_change * 100, abs=0.01
            )


def test_planted_numbers_match_known_values(insights, raw):
    """The headline example from the brief: South Laptop Pro revenue H1 -> H2."""
    [rev] = _find(insights, "revenue_decline", TARGET)
    target = raw[(raw["region"] == "South") & (raw["product"] == "Laptop Pro")]
    h1 = target[target["date"] < "2025-07-01"]["revenue"].sum()
    h2 = target[target["date"] >= "2025-07-01"]["revenue"].sum()
    assert rev["previous_value"] == pytest.approx(h1)
    assert rev["current_value"] == pytest.approx(h2)
    assert rev["change_percent"] == pytest.approx((h2 / h1 - 1) * 100, abs=0.01)
    assert rev["evidence"][0]["period_previous"] == "2025 H1"
    assert rev["evidence"][0]["period_current"] == "2025 H2"


def test_insight_headline_values_come_from_its_evidence(insights):
    for insight in insights:
        segment = next(e for e in insight["evidence"] if e["role"] == "segment")
        assert segment["metric"] == insight["metric"]
        assert segment["filters"] == insight["filters"]
        assert segment["previous_value"] == insight["previous_value"]
        assert segment["current_value"] == insight["current_value"]
        assert segment["change_percent"] == insight["change_percent"]


def test_gaps_are_derived_from_baseline_evidence(insights):
    for insight in insights:
        if "gaps_pp" not in insight:
            continue
        baselines = [e for e in insight["evidence"] if e["role"] == "baseline"]
        assert len(baselines) == len(insight["gaps_pp"])
        expected = sorted(round(insight["change_percent"] - b["change_percent"], 2) for b in baselines)
        assert sorted(insight["gaps_pp"].values()) == pytest.approx(expected, abs=0.01)


def test_evidence_ids_are_unique_and_resolve(report):
    ids = [e["id"] for e in report["evidence"]]
    assert len(ids) == len(set(ids))
    for insight in report["key_insights"]:
        for item in insight["evidence"]:
            assert item["id"] in ids


# --- Recommendations (required check 5) --------------------------------------

def test_planted_problem_gets_one_quality_recommendation(report):
    recs = report["recommendations"]
    assert len(recs) == 1
    [rec] = recs
    assert rec["segment"] == TARGET
    assert rec["priority"] == "high"
    assert "decline_with_rising_returns" in rec["rules_triggered"]
    assert "product quality" in rec["actions"][0]
    assert "before increasing inventory" in rec["actions"][0]
    # The regional and product-level findings are folded in as context.
    types = {i["type"] for i in report["key_insights"] if i["id"] in rec["based_on_insights"]}
    assert {"region_specific_decline", "product_specific_decline"} <= types


def test_margin_erosion_rule_matches_evidence(report):
    [rec] = report["recommendations"]
    [profit] = _find(report["key_insights"], "profit_decline", TARGET)
    margin = next(e for e in profit["evidence"] if e["metric"] == "profit_margin")
    assert margin["change"] * 100 <= -Thresholds().margin_drop_pp
    assert "margin_erosion" in rec["rules_triggered"]


def test_recommendations_only_cite_real_insights_and_evidence(report):
    insight_ids = {i["id"] for i in report["key_insights"]}
    evidence_ids = {e["id"] for e in report["evidence"]}
    by_id = {i["id"]: i for i in report["key_insights"]}
    for rec in report["recommendations"]:
        assert rec["based_on_insights"] and set(rec["based_on_insights"]) <= insight_ids
        assert rec["evidence_ids"] and set(rec["evidence_ids"]) <= evidence_ids
        own_types = {by_id[i]["type"] for i in rec["based_on_insights"]
                     if by_id[i]["dimension_value"] == rec["segment"]}
        if "decline_with_rising_returns" in rec["rules_triggered"]:
            assert "return_rate_increase" in own_types
            assert own_types & {"revenue_decline", "units_decline", "concentrated_decline",
                                "region_specific_decline", "product_specific_decline"}
        if "decline_with_stable_returns" in rec["rules_triggered"]:
            assert "return_rate_increase" not in own_types


def _shift_segment(frame, region, product, units_factor=1.0, returns_factor=1.0):
    """Change a segment's H2 volume and/or returns while keeping the data valid."""
    frame = frame.copy()
    mask = (frame["region"] == region) & (frame["product"] == product) & (frame["date"] >= "2025-07-01")
    for col in ("units", "returns"):
        frame[col] = frame[col].astype(float)
    for col in ("units", "revenue", "cost", "returns"):
        frame.loc[mask, col] = frame.loc[mask, col] * units_factor
    frame.loc[mask, "units"] = frame.loc[mask, "units"].round()
    frame.loc[mask, "returns"] = np.minimum(
        (frame.loc[mask, "returns"] * returns_factor).round(), frame.loc[mask, "units"]
    )
    return frame.astype({"units": int, "returns": int})


def test_decline_with_stable_returns_recommends_demand_review():
    frame = _shift_segment(_control_df(), "North", "Tablet", units_factor=0.7)
    recs = generate_recommendations(detect_insights(frame))
    [rec] = [r for r in recs if r["segment"] == "North / Tablet"]
    assert rec["rules_triggered"] == ["decline_with_stable_returns"]
    assert "demand, pricing, availability" in rec["actions"][0]
    assert any("Return rate is not a detected issue" in line for line in rec["rationale"])


def test_rising_returns_without_decline_recommends_returns_audit():
    frame = _shift_segment(_control_df(), "East", "Laptop Air", returns_factor=2.5)
    insights = detect_insights(frame)
    assert {i["type"] for i in insights} == {"return_rate_increase"}
    [rec] = generate_recommendations(insights)
    assert rec["rules_triggered"] == ["rising_returns_without_decline"]
    assert "Audit returns" in rec["actions"][0]


def test_no_insights_no_recommendation():
    assert generate_recommendation([]) is None
    assert generate_recommendations([]) == []


# --- No fabricated numbers (required check 6) --------------------------------

NUMBER = re.compile(r"(?<![\w.])([-+]?)\$?(\d[\d,]*(?:\.\d+)?)\s?(M|K|%|pp)?")


def _numeric_leaves(obj):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield float(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _numeric_leaves(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _numeric_leaves(v)


def _text_fields(report):
    yield report["executive_summary"]
    for insight in report["key_insights"]:
        yield insight["title"]
        yield insight["summary"]
    for rec in report["recommendations"]:
        yield from rec["actions"]
        yield from rec["rationale"]


def test_every_number_in_text_is_backed_by_report_data(report):
    known = [abs(v) for v in _numeric_leaves(report)]
    labels = [report["metadata"]["period_previous"]["label"], report["metadata"]["period_current"]["label"]]
    checked = 0
    for text in _text_fields(report):
        for label in labels:
            text = text.replace(label, "")
        for _, digits, suffix in NUMBER.findall(text):
            value = float(digits.replace(",", ""))
            decimals = len(digits.split(".")[1]) if "." in digits else 0
            tol = 0.5 * 10 ** -decimals + 1e-9
            scales = {"M": [1e-6], "K": [1e-3], "%": [1, 100], "pp": [1, 100]}.get(suffix, [1])
            assert any(abs(k * s - value) <= tol for k in known for s in scales), (value, suffix, text)
            checked += 1
    assert checked > 20  # the check actually ran over real numbers


# --- JSON (required check 7) -------------------------------------------------

def test_report_is_json_serializable(report):
    text = json.dumps(report, allow_nan=False)
    assert json.loads(text) == report


def test_report_structure(report):
    assert {"executive_summary", "key_insights", "evidence", "recommendations"} <= report.keys()
    assert isinstance(report["executive_summary"], str) and report["executive_summary"]
    assert report["metadata"]["insight_count"] == len(report["key_insights"])


def test_report_is_deterministic(csv_path, report):
    assert generate_decision_report(csv_path) == report


def test_report_accepts_dataframe(raw):
    frame = raw.assign(date=raw["date"].dt.strftime("%Y-%m-%d"))
    from_df = generate_decision_report(frame)
    assert [i["type"] for i in from_df["key_insights"]]
    assert from_df["metadata"]["source"] == "DataFrame"
