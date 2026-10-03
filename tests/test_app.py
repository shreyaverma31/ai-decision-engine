"""Tests for app.py: data loading errors and the full UI flow via Streamlit's AppTest.

Ollama is never required: tests turn the AI toggle off, point OLLAMA_URL at a
closed port, or mock llm.call_ollama.
"""

import json
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import app
import llm
from data_generator import generate_sales_data

DEMO_BYTES = app.DEMO_PATH.read_bytes()


def csv_bytes(frame) -> bytes:
    return frame.to_csv(index=False).encode()


@pytest.fixture(autouse=True)
def clear_cache():
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def new_app() -> AppTest:
    return AppTest.from_file(str(Path(app.__file__).resolve()), default_timeout=120)


def run_demo(use_llm=False) -> AppTest:
    at = new_app().run()
    if not use_llm:
        at.sidebar.toggle[0].set_value(False).run()
    at.sidebar.button[0].click().run()   # Use Demo Dataset
    at.sidebar.button[1].click().run()   # Analyze Business Data
    return at


def run_with_data(data: bytes) -> AppTest:
    """Simulates an analyzed upload by putting the bytes where the Analyze button stores them."""
    at = new_app()
    at.session_state["analyzed"] = data
    at.run()
    at.sidebar.toggle[0].set_value(False).run()
    return at


def texts(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += [e.value for kind in (at.info, at.success, at.warning, at.error) for e in kind]
    return "\n".join(parts).replace(r"\$", "$")  # undo app.md() escaping


def assert_no_crash(at: AppTest):
    assert not at.exception, [e.value for e in at.exception]
    assert "Something went wrong" not in texts(at)


# --- Loading & friendly errors -----------------------------------------------

def test_load_demo_csv():
    assert len(app.load_csv(DEMO_BYTES)) == 7300


@pytest.mark.parametrize("data,message", [
    (b"", "empty"),
    (b"   \n  ", "empty"),
    (b"date,region,product\n2025-01-01,North,Tablet\n", "Missing required columns"),
    (b"just some text without the right columns", "Missing required columns"),
    (b"\xff\xfe\x00\x01garbage\x00", "could not be read"),
    (b"date,region,product,units,revenue,cost,returns\n", "no rows"),
    (b"date,region,product,units,revenue,cost,returns\n2025-01-01,North,Tablet,-5,10,5,0\n", "negative"),
    (b"date,region,product,units,revenue,cost,returns\nnot-a-date,North,Tablet,5,10,5,0\n", "unparseable"),
])
def test_load_csv_friendly_errors(data, message):
    with pytest.raises(app.DataError, match=message):
        app.load_csv(data)


def test_single_month_is_a_friendly_error():
    one_month = generate_sales_data(start_date="2025-03-01", end_date="2025-03-31")
    with pytest.raises(app.DataError, match="two months"):
        app.analyze(csv_bytes(one_month))


def test_other_datasets_are_analyzed():
    other = generate_sales_data(start_date="2024-01-01", end_date="2024-12-31", seed=7)
    df, report, story = app.analyze(csv_bytes(other))
    assert story["problem"] is None  # the planted problem starts in 2025, so 2024 data is healthy
    assert len(df) == len(other)
    assert report["metadata"]["period_previous"]["label"] == "2024 H1"


# --- UI flow -----------------------------------------------------------------

def test_start_page():
    at = new_app().run()
    assert_no_crash(at)
    assert at.title[0].value == "AI Decision Engine"
    assert "traceable insights and actionable decisions" in at.markdown[0].value
    assert "Use Demo Dataset" in at.info[0].value
    assert at.sidebar.button[1].label == "Analyze Business Data"
    assert at.sidebar.button[1].disabled  # nothing selected yet


def test_demo_flow_shows_all_sections():
    at = run_demo()
    assert_no_crash(at)
    headers = [h.value for h in at.header if h.value != "Data"]
    assert headers == ["Business Overview", "🚨 Main Business Problem", "Why did the system flag it?",
                       "Possible causes — NOT proven by the data", "🤖 AI Explanation",
                       "Recommended Action", "Ask the Business Data"]
    steps = [h.value for h in at.subheader][1:]
    assert steps == ["Step 1 — What changed?", "Step 2 — How significant is the change?",
                     "Step 3 — Is the problem concentrated in this segment?",
                     "Step 4 — What evidence supports the finding?"]
    assert app.FOOTER in texts(at)


def test_demo_overview_metrics():
    metrics = {m.label: m.value for m in run_demo().metric[:5]}
    assert metrics == {"Revenue": "$97.18M", "Profit": "$29.62M", "Units": "199,214",
                       "Return Rate": "2.7%", "Overall Revenue Growth": "+3.3%"}


def test_demo_key_issue():
    at = run_demo()
    assert at.subheader[0].value == "South / Laptop Pro"
    metrics = [(m.label, m.value) for m in at.metric[5:]]
    assert metrics == [("Revenue", "↓ 25.2%"), ("Profit", "↓ 29.9%"), ("Units", "↓ 23.5%"),
                       ("Return Rate", "↑ 4.7% → 8.5%")]
    assert ("South / Laptop Pro is performing significantly worse than both its historical "
            "performance and its peer groups.") in texts(at)


def test_demo_four_evidence_steps():
    body = texts(run_demo())
    # Step 1 - what changed
    for line in ("**Revenue:** $3.19M → $2.39M", "**Profit:** $859.8K → $603.0K",
                 "**Units:** 2,217 → 1,696", "**Return rate:** 4.7% → 8.5%"):
        assert line in body
    # Step 2 - how significant (the engine's own thresholds)
    assert "✅ **Revenue: -25.2%** — significant (significant if it falls by 10% or more)" in body
    assert "✅ **Return rate: +80.5% relative (+3.8 pp), z = 4.8** — significant" in body
    # Step 3 - is it concentrated?
    assert "Laptop Pro in other regions: **+4.6%**" in body
    assert "Other products in South: **+3.1%**" in body
    assert "South / Laptop Pro: **-25.2%**" in body
    assert "South / Laptop Pro is declining much faster than its peers." in body
    # Step 4 - which detected findings support it, with evidence IDs
    assert "**Return-rate increase** · South / Laptop Pro · high severity · evidence E1" in body
    assert "**Decline concentrated in this region + product** · South / Laptop Pro" in body


def test_demo_possible_causes_are_possibilities_with_evidence():
    at = run_demo()
    body = texts(at)
    assert "Possible causes — NOT proven by the data" in [h.value for h in at.header]
    for cause, evidence in (
        ("Product quality or defect issues", "Return rate increased from 4.7% to 8.5%."),
        ("Fulfillment / delivery problems", "Return rate increased from 4.7% to 8.5%."),
        ("Demand, pricing, or competition changes", "Revenue declined by 25.2% ($3.19M → $2.39M)."),
    ):
        assert f"💭 **{cause} may be contributing. This requires investigation.**" in body
        assert f"*Evidence that led to this hypothesis:* {evidence}" in body
    # Never stated as fact anywhere on the page.
    assert "caused the decline" not in body
    assert "caused by" not in body


def test_demo_reading_guide_labels_each_kind_of_statement():
    body = texts(run_demo())
    for label in ("Data-backed finding", "Possible explanation — not proven", "Recommended investigation"):
        assert label in body
    for title in ("**What the data shows**", "**What the system suspects**",
                  "**What should be investigated**"):
        assert title in body
    assert "It does not, by itself, prove why it happened." in body


def test_demo_traceability_is_collapsed_and_traceable():
    at = run_demo()
    [expander] = [e for e in at.expander if "Show how every number was calculated" in e.label]
    assert expander.label == "🔎 Show how every number was calculated"
    assert not expander.proto.expanded
    table = at.dataframe[0].value
    assert list(table.columns) == ["Evidence", "Metric", "Segment", "Previous (2025 H1)",
                                   "Current (2025 H2)", "Absolute change", "% change",
                                   "Source filter", "Rows used"]
    assert "region = South, product = Laptop Pro" in table["Source filter"].tolist()
    assert table["Evidence"].is_unique
    pairs = list(zip(table["Metric"], table["Segment"]))
    assert len(pairs) == len(set(pairs))  # the same fact is shown once, with all its IDs
    first = table.iloc[0]  # the page's own numbers come first
    assert (first["Metric"], first["Segment"]) == ("Revenue", "South / Laptop Pro")
    assert (first["Previous (2025 H1)"], first["Current (2025 H2)"]) == ("$3.19M", "$2.39M")
    assert (first["Absolute change"], first["% change"]) == ("-$805.7K", "-25.2%")
    rr = table[(table["Metric"] == "Return rate") & (table["Segment"] == "South / Laptop Pro")].iloc[0]
    assert rr["Absolute change"] == "+3.8 pp"
    assert "Laptop Pro in other regions" in table["Segment"].tolist()
    assert "Other products in South" in table["Segment"].tolist()


def test_demo_explanation_and_recommendation_with_ai_off():
    at = run_demo()
    assert [i.value for i in at.info] == ["Using evidence-based fallback"]
    [ai_causes] = [w.value for w in at.warning if "Possible causes — not facts" in w.value]
    assert ai_causes.count("\n- ") == 3
    body = texts(at)
    assert body.count("**Key findings**") == 1
    assert ("Investigate South / Laptop Pro product quality, fulfillment, and customer experience "
            "before increasing inventory or marketing spend.") in at.success[-1].value
    assert ("**Why?** Because revenue, profit, and units declined sharply while the return rate "
            "rose sharply (4.7% → 8.5%).") in body
    assert "**Secondary action:** Review discount depth and pricing for South / Laptop Pro." in body
    assert ("**Rule:** `decline_with_rising_returns` — Sales fell significantly while the return rate "
            "rose significantly in the same segment") in body
    assert "Recommended investigation" in body
    assert "not a conclusion about the cause" in body
    assert len([s for s in at.success if "🎯" in (s.icon or "")]) == 1  # one primary action


def test_ollama_unavailable_uses_fallback(monkeypatch):
    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:9")
    at = run_demo(use_llm=True)
    assert_no_crash(at)
    assert "Using evidence-based fallback" in [i.value for i in at.info]
    assert "not reachable" in texts(at)


def test_llm_success_shows_verified_message(monkeypatch):
    from test_llm import good_response

    monkeypatch.setattr(llm, "call_ollama", lambda *a, **k: json.dumps(good_response()))
    at = run_demo(use_llm=True)
    assert_no_crash(at)
    assert any("AI explanation generated from verified evidence" in s.value for s in at.success)
    assert good_response()["summary"] in texts(at)


def test_llm_bad_response_falls_back(monkeypatch):
    monkeypatch.setattr(llm, "call_ollama", lambda *a, **k: '{"summary": "Revenue fell 99.9%"}')
    at = run_demo(use_llm=True)
    assert_no_crash(at)
    assert "Using evidence-based fallback" in [i.value for i in at.info]
    assert "did not pass the evidence checks" in texts(at)


def test_ask_question():
    at = run_demo()
    at.text_input[0].input("Why did revenue decline?")
    next(b for b in at.button if b.label == "Ask").click().run()
    assert_no_crash(at)
    assert "South / Laptop Pro revenue fell 25.2%, from $3.19M" in texts(at)
    assert "Evidence used:" in texts(at)


def test_ask_empty_question():
    at = run_demo()
    next(b for b in at.button if b.label == "Ask").click().run()
    assert_no_crash(at)
    assert any("type a question" in w.value for w in at.warning)


def test_question_placeholder():
    assert run_demo().text_input[0].placeholder == "Why did South / Laptop Pro revenue decline?"


# --- Uploaded data -----------------------------------------------------------

def test_uploaded_invalid_csv_shows_friendly_error():
    at = run_with_data(b"name,value\nfoo,1\n")
    assert_no_crash(at)
    assert "Missing required columns" in at.error[0].value


def test_uploaded_empty_csv_shows_friendly_error():
    at = run_with_data(b"")
    assert_no_crash(at)
    assert "empty" in at.error[0].value


def test_uploaded_healthy_data_shows_no_issue():
    healthy = generate_sales_data(anomaly_start="2030-01-01")
    at = run_with_data(csv_bytes(healthy))
    assert_no_crash(at)
    assert any("No significant problems detected" in s.value for s in at.success)
    assert any("No action needed" in s.value for s in at.success)


def test_uploaded_other_period_data():
    other = generate_sales_data(start_date="2024-01-01", end_date="2024-12-31", seed=7)
    at = run_with_data(csv_bytes(other))
    assert_no_crash(at)
    assert "2024 H1" in texts(at)


def test_money_is_escaped_for_markdown():
    """'$3.19M to $2.39M' must not be rendered as LaTeX math."""
    assert app.md("from $3.19M to $2.39M") == r"from \$3.19M to \$2.39M"
    at = run_demo()
    raw = "\n".join(m.value for m in at.markdown)
    assert r"(\$3.19M to \$2.39M)" in raw
    assert "($3.19M" not in raw


def test_rejected_ai_answer_shows_what_the_checks_caught(monkeypatch):
    monkeypatch.setattr(llm, "call_ollama", lambda *a, **k: '{"summary": "Revenue fell 99.9%"}')
    at = run_demo(use_llm=True)
    assert_no_crash(at)
    assert any("What the evidence checks caught" in e.label for e in at.expander)


def test_traceability_evidence_ids_are_sorted():
    table = run_demo().dataframe[0].value
    for ids in table["Evidence"]:
        parts = ids.split(", ")
        assert parts == sorted(parts, key=lambda x: (x[0], int(x[1:])))
