"""Tests for llm.py. Ollama is always mocked; no server is needed."""

import copy
import json
import re

import httpx
import pytest

import llm
from analyzer import validate_data
from data_generator import generate_sales_data, save_dataset
from engine import generate_decision_report


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    path = save_dataset(generate_sales_data(), tmp_path_factory.mktemp("data") / "sales.csv")
    return generate_decision_report(path)


@pytest.fixture(scope="module")
def healthy_report():
    return generate_decision_report(validate_data(generate_sales_data(anomaly_start="2030-01-01")))


def good_response():
    """What a well-behaved model returns: facts, numbers and IDs straight from the report."""
    return {
        "summary": (
            "Overall revenue grew 3.3% from 2025 H1 to 2025 H2, but South / Laptop Pro declined sharply: "
            "revenue fell 25.2% from $3.19M to $2.39M while its return rate rose from 4.7% to 8.5%."
        ),
        "key_findings": [
            {"statement": "South / Laptop Pro revenue fell 25.2%, from $3.19M to $2.39M.",
             "insight_ids": ["I3"]},
            {"statement": "South / Laptop Pro return rate rose from 4.7% to 8.5% (+3.8 pp).",
             "insight_ids": ["I1"]},
            {"statement": "South revenue changed -4.8% while all other regions grew 5.8%.",
             "insight_ids": ["I6"]},
        ],
        "reasoning": [{
            "observed_facts": [
                "Sales and profit fell in the same segment where returns rose.",
                "Laptop Pro in other regions grew 4.6%, so the decline is concentrated in the South.",
            ],
            "possible_explanations": [
                {"cause": "A product quality issue may be affecting South / Laptop Pro.",
                 "evidence": "The return rate rose from 4.7% to 8.5%.",
                 "linked_insight_ids": ["I1"]},
                {"cause": "Fulfillment problems in the South could be causing both returns and lost sales.",
                 "evidence": "Returns rose to 8.5% while revenue fell 25.2%.",
                 "linked_insight_ids": ["I1", "I3"]},
            ],
            "needs_further_investigation": True,
            "insight_ids": ["I1", "I3", "I4"],
        }],
        "recommendations": [{
            "action": "Investigate product quality and fulfillment for South / Laptop Pro before adding inventory.",
            "rationale": "Revenue fell 25.2% while the return rate rose to 8.5%.",
            "recommendation_id": "R1",
            "insight_ids": ["I1", "I3"],
        }],
    }


class FakeOllama:
    """Stands in for llm.call_ollama; returns queued responses and records calls."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, messages, model=None, base_url=None, timeout=None):
        self.calls.append({"messages": messages, "model": model})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response if isinstance(response, str) else json.dumps(response)


def _mutated(**changes):
    response = good_response()
    for path, value in changes.items():
        target = response
        *keys, last = path.split("__")
        for key in keys:
            target = target[int(key)] if key.isdigit() else target[key]
        target[int(last) if last.isdigit() else last] = value
    return response


def all_text(result):
    facts, hypotheses = llm.explanation_texts(result)
    notes = [r.get("note", "") for r in result["reasoning"]]
    return facts + hypotheses + notes


# --- Ollama unavailable -> fallback ------------------------------------------

def test_ollama_unavailable_falls_back(report, monkeypatch):
    def refuse(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(llm.httpx, "post", refuse)
    result = llm.explain_decision_report(report)
    assert result["metadata"]["source"] == "fallback"
    assert "Ollama unavailable" in result["metadata"]["fallback_reason"]
    assert result["summary"] == report["executive_summary"]
    assert len(result["key_findings"]) == len(report["key_insights"])
    assert llm.validate_explanation(result, report) == []


def test_unreachable_server_falls_back(report):
    """A real HTTP call to a closed port must not raise."""
    result = llm.explain_decision_report(report, base_url="http://127.0.0.1:9")
    assert result["metadata"]["source"] == "fallback"


def test_http_error_status_falls_back(report, monkeypatch):
    def not_found(url, json, timeout):
        return httpx.Response(404, json={"error": "model not found"}, request=httpx.Request("POST", url))

    monkeypatch.setattr(llm.httpx, "post", not_found)
    assert llm.explain_decision_report(report)["metadata"]["source"] == "fallback"


def test_fallback_covers_every_recommendation(report):
    result = llm.explain_decision_report(report, use_llm=False)
    actions = [r["action"] for r in result["recommendations"]]
    assert actions == [a for r in report["recommendations"] for a in r["actions"]]
    for reasoning in result["reasoning"]:
        assert reasoning["needs_further_investigation"] is True
        assert "does not establish their cause" in reasoning["note"]
        for c in reasoning["possible_explanations"]:
            assert re.search(r"\b(may|might|could)\b", c["cause"])
            assert c["linked_insight_ids"] and c["evidence"]


def test_fallback_margin_action_has_margin_rationale(report):
    result = llm.fallback_explanation(report)
    margin = next(r for r in result["recommendations"] if "discount depth" in r["action"])
    assert "Profit margin fell" in margin["rationale"]
    assert "return rate" not in margin["rationale"]


# --- Empty reports -----------------------------------------------------------

@pytest.mark.parametrize("empty", [{}, None, {"key_insights": [], "evidence": []}])
def test_empty_report_is_safe(empty, monkeypatch):
    fake = FakeOllama()
    monkeypatch.setattr(llm, "call_ollama", fake)
    result = llm.explain_decision_report(empty)
    assert fake.calls == []  # nothing to explain, so the LLM is not called
    assert result["summary"]
    assert result["key_findings"] == result["recommendations"] == result["reasoning"] == []
    assert result["evidence_references"] == []
    json.dumps(result)


def test_report_without_findings_is_safe(healthy_report, monkeypatch):
    fake = FakeOllama()
    monkeypatch.setattr(llm, "call_ollama", fake)
    result = llm.explain_decision_report(healthy_report)
    assert fake.calls == []
    assert "No significant declines" in result["summary"]
    assert result["metadata"]["source"] == "fallback"


# --- Valid LLM response -----------------------------------------------------

def test_valid_response_is_used(report, monkeypatch):
    fake = FakeOllama(good_response())
    monkeypatch.setattr(llm, "call_ollama", fake)
    result = llm.explain_decision_report(report, model="test-model")
    assert result["metadata"] == {"source": "llm", "model": "test-model",
                                  "fallback_reason": None, "validation_issues": []}
    assert result["summary"] == good_response()["summary"]
    assert len(fake.calls) == 1


def test_evidence_is_attached_from_cited_insights(report, monkeypatch):
    monkeypatch.setattr(llm, "call_ollama", FakeOllama(good_response()))
    result = llm.explain_decision_report(report)
    insights = {i["id"]: i for i in report["key_insights"]}
    evidence = {e["id"]: e for e in report["evidence"]}
    for finding in result["key_findings"]:
        expected = {e["id"] for i in finding["insight_ids"] for e in insights[i]["evidence"]}
        assert set(finding["evidence_ids"]) == expected
    assert result["evidence_references"]
    for ref in result["evidence_references"]:
        assert ref["previous_value"] == evidence[ref["id"]]["previous_value"]
        assert ref["current_value"] == evidence[ref["id"]]["current_value"]
        assert ref["filters"] == evidence[ref["id"]]["filters"]


def test_retry_then_success(report, monkeypatch):
    fake = FakeOllama("not json at all", good_response())
    monkeypatch.setattr(llm, "call_ollama", fake)
    result = llm.explain_decision_report(report)
    assert result["metadata"]["source"] == "llm"
    assert len(fake.calls) == 2
    assert "not valid JSON" in fake.calls[1]["messages"][-1]["content"]


def test_fenced_json_is_accepted(report, monkeypatch):
    fenced = "```json\n" + json.dumps(good_response()) + "\n```"
    monkeypatch.setattr(llm, "call_ollama", FakeOllama(fenced))
    assert llm.explain_decision_report(report)["metadata"]["source"] == "llm"


# --- Invalid LLM responses -> fallback ---------------------------------------

INVALID_RESPONSES = {
    "invalid_json": "{summary: oops",
    "not_an_object": json.dumps(["a", "list"]),
    "missing_field": {k: v for k, v in good_response().items() if k != "reasoning"},
    "fabricated_percent": _mutated(summary="Revenue in the South fell 31.7% in 2025 H2."),
    "fabricated_money": _mutated(key_findings__0__statement="South / Laptop Pro lost $1.42M of revenue."),
    "fabricated_count": _mutated(recommendations__0__rationale="Returns rose to 412 units."),
    "fabricated_metric": _mutated(key_findings__1__statement="Conversion rate dropped for South / Laptop Pro."),
    "causal_certainty": _mutated(reasoning__0__observed_facts=["The decline was caused by defective batteries."]),
    "attributed_cause": _mutated(summary="The South decline is attributed to poor fulfillment."),
    "certainty_in_hypothesis": _mutated(reasoning__0__possible_explanations=[
        {"cause": "This definitely is a quality problem.", "evidence": "Returns rose to 8.5%.",
         "linked_insight_ids": ["I1"]}]),
    "no_investigation_flag": _mutated(reasoning__0__needs_further_investigation=False),
    "unknown_insight": _mutated(key_findings__0__insight_ids=["I99"]),
    "finding_without_insight": _mutated(key_findings__0__insight_ids=[]),
    "unknown_recommendation": _mutated(recommendations__0__recommendation_id="R7"),
    "recommendation_without_findings": _mutated(recommendations__0__insight_ids=[]),
    "no_recommendations": _mutated(recommendations=[]),
    "no_findings": _mutated(key_findings=[]),
    "wrong_type": _mutated(reasoning__0__needs_further_investigation="yes"),
}


@pytest.mark.parametrize("name", INVALID_RESPONSES)
def test_invalid_response_falls_back(name, report, monkeypatch):
    bad = INVALID_RESPONSES[name]
    fake = FakeOllama(bad, bad)
    monkeypatch.setattr(llm, "call_ollama", fake)
    result = llm.explain_decision_report(report)
    assert result["metadata"]["source"] == "fallback", name
    assert result["metadata"]["validation_issues"], name
    assert len(fake.calls) == llm.MAX_ATTEMPTS
    assert result["summary"] == report["executive_summary"]


def test_hedged_cause_in_hypothesis_is_allowed(report):
    hedged = {"cause": "Returns may be caused by a defective batch; this is not established by the data.",
              "evidence": "The return rate rose from 4.7% to 8.5%.", "linked_insight_ids": ["I1"]}
    response = _mutated(reasoning__0__possible_explanations=[
        hedged, good_response()["reasoning"][0]["possible_explanations"][1]])  # keep decline coverage
    assert llm.validate_explanation(response, report) == []


def test_good_response_validates(report):
    assert llm.validate_explanation(good_response(), report) == []


# --- Number grounding --------------------------------------------------------

@pytest.mark.parametrize("text,ok", [
    ("revenue fell 25.2%", True),
    ("revenue fell 25%", True),             # rounded display of 25.25
    ("from $3.19M to $2.39M", True),
    ("from $3.2 million", True),
    ("units fell from 2,217 to 1,696", True),
    ("return rate 8.5%", True),             # stored as 0.085495
    ("+3.8 pp", True),
    ("z = 4.8", True),
    ("in 2025 H1", True),
    ("three actions, 3 segments", True),    # small counts
    ("revenue fell 31.7%", False),
    ("$9.99M", False),
    ("412 units", False),
    ("return rate hit 12.4%", False),
])
def test_unsupported_numbers(report, text, ok):
    assert (llm.unsupported_numbers(text, llm.allowed_numbers(report)) == []) is ok


@pytest.mark.parametrize("use_llm", [True, False])
def test_final_numbers_come_from_report(report, monkeypatch, use_llm):
    monkeypatch.setattr(llm, "call_ollama", FakeOllama(good_response()))
    result = llm.explain_decision_report(report, use_llm=use_llm)
    allowed = llm.allowed_numbers(report)
    for text in all_text(result):
        assert llm.unsupported_numbers(text, allowed) == [], text
    assert len(all_text(result)) > 5


def test_no_fabricated_metrics_in_output(report, monkeypatch):
    for use_llm in (True, False):
        monkeypatch.setattr(llm, "call_ollama", FakeOllama(good_response()))
        result = llm.explain_decision_report(report, use_llm=use_llm)
        facts, _ = llm.explanation_texts(result)
        assert not any(llm.UNAVAILABLE_METRICS.search(t) for t in facts)
        for ref in result["evidence_references"]:
            assert ref["metric"] in {"revenue", "cost", "profit", "profit_margin", "units",
                                     "returns", "return_rate"}


# --- Recommendations reference real findings --------------------------------

@pytest.mark.parametrize("use_llm", [True, False])
def test_recommendations_reference_detected_findings(report, monkeypatch, use_llm):
    monkeypatch.setattr(llm, "call_ollama", FakeOllama(good_response()))
    result = llm.explain_decision_report(report, use_llm=use_llm)
    insight_ids = {i["id"] for i in report["key_insights"]}
    rec_ids = {r["id"] for r in report["recommendations"]}
    assert result["recommendations"]
    for rec in result["recommendations"]:
        assert rec["insight_ids"] and set(rec["insight_ids"]) <= insight_ids
        assert rec["recommendation_id"] in rec_ids
    assert {r["recommendation_id"] for r in result["recommendations"]} == rec_ids


# --- What the model is sent --------------------------------------------------

def test_prompt_states_grounding_rules():
    prompt = llm.SYSTEM_PROMPT.lower()
    for rule in ("evidence is authoritative", "do not invent numbers", "do not invent causes",
                 "observed facts", "possible explanations", "grounded in the supplied evidence",
                 "further investigation is needed", "correlation"):
        assert rule in prompt, rule


def test_context_contains_report_not_raw_data(report):
    messages = llm.build_messages(report)
    content = messages[1]["content"]
    assert not re.search(r"\d{4}-\d{2}-\d{2}", content)  # no row-level dates
    assert len(content) < 6000
    context = llm.build_llm_context(report)
    assert [i["id"] for i in context["insights"]] == [i["id"] for i in report["key_insights"]]
    assert "rows" not in json.dumps(context)


def test_call_ollama_request(monkeypatch):
    sent = {}

    def fake_post(url, json, timeout):
        sent.update(url=url, payload=json, timeout=timeout)
        return httpx.Response(200, json={"message": {"content": "{}"}},
                              request=httpx.Request("POST", url))

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    assert llm.call_ollama([{"role": "user", "content": "hi"}]) == "{}"
    assert sent["url"] == "http://localhost:11434/api/chat"
    assert sent["payload"]["model"] == "llama3.2:3b"
    assert sent["payload"]["stream"] is False
    assert sent["payload"]["format"] == llm.RESPONSE_SCHEMA
    assert sent["payload"]["options"]["temperature"] == 0

    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:3b")
    llm.call_ollama([])
    assert sent["payload"]["model"] == "qwen2.5:3b"


# --- JSON --------------------------------------------------------------------

@pytest.mark.parametrize("use_llm", [True, False])
def test_result_is_json_serializable(report, monkeypatch, use_llm):
    monkeypatch.setattr(llm, "call_ollama", FakeOllama(good_response()))
    result = llm.explain_decision_report(report, use_llm=use_llm)
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert {"summary", "key_findings", "reasoning", "recommendations",
            "evidence_references", "metadata"} <= result.keys()


def test_input_report_is_not_modified(report, monkeypatch):
    before = copy.deepcopy(report)
    monkeypatch.setattr(llm, "call_ollama", FakeOllama(good_response()))
    llm.explain_decision_report(report)
    llm.explain_decision_report(report, use_llm=False)
    assert report == before


# --- answer_question -----------------------------------------------------------

def good_answer():
    return {
        "answer": "South / Laptop Pro revenue fell 25.2%, from $3.19M to $2.39M, while its return rate "
                  "rose from 4.7% to 8.5%. Laptop Pro in other regions grew 4.6%.",
        "possible_explanations": [
            {"cause": "A product quality issue may be affecting South / Laptop Pro.",
             "evidence": "Its return rate rose from 4.7% to 8.5%.", "linked_insight_ids": ["I1"]}],
        "insight_ids": ["I3", "I1"],
    }


class FakeAnswerOllama(FakeOllama):
    def __call__(self, messages, model=None, base_url=None, timeout=None, schema=None):
        self.calls.append({"messages": messages, "schema": schema})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response if isinstance(response, str) else json.dumps(response)


def test_answer_uses_valid_llm_response(report, monkeypatch):
    fake = FakeAnswerOllama(good_answer())
    monkeypatch.setattr(llm, "call_ollama", fake)
    result = llm.answer_question(report, "Why did revenue decline?")
    assert result["metadata"]["source"] == "llm"
    assert result["answer"] == good_answer()["answer"]
    assert fake.calls[0]["schema"] == llm.ANSWER_SCHEMA
    user_message = fake.calls[0]["messages"][-1]["content"]
    assert "QUESTION: Why did revenue decline?" in user_message
    assert not re.search(r"\d{4}-\d{2}-\d{2}", user_message)  # report context only, no rows
    assert {e["id"] for e in result["evidence_references"]} >= {"E1", "E7"}


@pytest.mark.parametrize("bad", [
    None,
    "not json",
    {"answer": "Revenue fell 44.4% in the South.", "possible_explanations": [], "insight_ids": ["I3"]},
    {"answer": "The decline was caused by a bad battery batch.", "possible_explanations": [],
     "insight_ids": ["I3"]},
    {"answer": "Conversion rate dropped in the South.", "possible_explanations": [], "insight_ids": []},
    {"answer": "Revenue fell 25.2%.", "possible_explanations": [], "insight_ids": ["I42"]},
    {"answer": "", "possible_explanations": [], "insight_ids": []},
])
def test_answer_invalid_response_falls_back(report, monkeypatch, bad):
    bad = bad if bad is not None else ["not", "an", "object"]
    monkeypatch.setattr(llm, "call_ollama", FakeAnswerOllama(bad, bad))
    result = llm.answer_question(report, "Why did revenue decline?")
    assert result["metadata"]["source"] == "fallback"
    assert result["metadata"]["validation_issues"]


def test_answer_ollama_unavailable(report, monkeypatch):
    monkeypatch.setattr(llm, "call_ollama", FakeAnswerOllama(llm.OllamaUnavailable("down")))
    result = llm.answer_question(report, "What happened to returns?")
    assert result["metadata"]["source"] == "fallback"
    assert "Ollama unavailable" in result["metadata"]["fallback_reason"]
    assert result["insight_ids"] == ["I1"]
    assert "return rate rose from 4.7%" in result["answer"]


@pytest.mark.parametrize("question,expected_metric", [
    ("Why did revenue decline?", "revenue"),
    ("How did profit change?", "profit"),
    ("Why are units down?", "units"),
    ("What happened to returns?", "return_rate"),
])
def test_fallback_answer_picks_relevant_insights(report, question, expected_metric):
    result = llm.answer_question(report, question, use_llm=False)
    by_id = {i["id"]: i for i in report["key_insights"]}
    assert by_id[result["insight_ids"][0]]["metric"] == expected_metric
    assert llm.validate_answer(result, report) == []


def test_fallback_answer_off_topic(report):
    result = llm.answer_question(report, "What is the weather tomorrow?", use_llm=False)
    assert result["answer"].startswith("The report does not directly address this question.")
    assert llm.validate_answer(result, report) == []


@pytest.mark.parametrize("question", ["", "   ", None])
def test_answer_empty_question(report, monkeypatch, question):
    fake = FakeAnswerOllama()
    monkeypatch.setattr(llm, "call_ollama", fake)
    result = llm.answer_question(report, question)
    assert fake.calls == []
    assert "enter a question" in result["answer"]


def test_answer_empty_report(monkeypatch):
    fake = FakeAnswerOllama()
    monkeypatch.setattr(llm, "call_ollama", fake)
    result = llm.answer_question({}, "Why did revenue decline?")
    assert fake.calls == []
    assert result["answer"] and result["insight_ids"] == []


@pytest.mark.parametrize("use_llm", [True, False])
def test_answer_is_json_serializable_and_grounded(report, monkeypatch, use_llm):
    monkeypatch.setattr(llm, "call_ollama", FakeAnswerOllama(good_answer()))
    result = llm.answer_question(report, "Why did revenue decline?", use_llm=use_llm)
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    allowed = llm.allowed_numbers(report)
    causes = result["possible_explanations"]
    for text in [result["answer"]] + [c["cause"] for c in causes] + [c["evidence"] for c in causes]:
        assert llm.unsupported_numbers(text, allowed) == []


# --- Evidence-linked causes ----------------------------------------------------

def _cause(cause, evidence="The return rate rose from 4.7% to 8.5%.", ids=("I1",)):
    return {"cause": cause, "evidence": evidence, "linked_insight_ids": list(ids)}


BAD_CAUSES = {
    "plain_string": "A product quality issue may be affecting sales.",
    "not_linked": _cause("A product quality issue may be affecting sales.", ids=()),
    "unknown_link": _cause("A product quality issue may be affecting sales.", ids=("I77",)),
    "not_hedged": _cause("A product quality issue is affecting sales."),
    "quality_linked_to_revenue_only": _cause(
        "A product quality issue may be affecting sales.",
        evidence="Revenue fell 25.2%.", ids=("I3",)),
    "pricing_linked_to_returns_only": _cause("Deeper discounting may explain this."),
    "demand_linked_to_returns_only": _cause("Weaker demand could be reducing sales."),
    "invented_number_in_evidence": _cause("A product quality issue may be affecting sales.",
                                          evidence="The return rate rose to 14.2%."),
    "invented_metric_in_evidence": _cause("A product quality issue may be affecting sales.",
                                          evidence="Customer satisfaction fell sharply."),
    "missing_evidence": _cause("A product quality issue may be affecting sales.", evidence=""),
}


@pytest.mark.parametrize("name", BAD_CAUSES)
def test_unlinked_or_mislinked_cause_is_rejected(report, monkeypatch, name):
    bad = _mutated(reasoning__0__possible_explanations=[BAD_CAUSES[name]])
    assert llm.validate_explanation(bad, report), name
    monkeypatch.setattr(llm, "call_ollama", FakeOllama(bad, bad))
    assert llm.explain_decision_report(report)["metadata"]["source"] == "fallback"


@pytest.mark.parametrize("cause,ids", [
    ("A product quality or defect issue may be affecting this segment.", ["I1"]),
    ("Deeper discounting may have been used to support sales volume.", ["I2"]),
    ("Demand, pricing, or competition changes could be reducing sales.", ["I3", "I5"]),
])
def test_correctly_linked_causes_are_accepted(report, cause, ids):
    by_id = {i["id"]: i for i in report["key_insights"]}
    # Cause-level checks only; coverage of all required findings is tested separately.
    assert llm.check_causes([_cause(cause, evidence=by_id[ids[0]]["summary"], ids=ids)], report, "c") == []


@pytest.mark.parametrize("bad_cause", [
    _cause("Weaker demand could be reducing sales."),            # demand linked to returns only
    _cause("A product quality issue is the problem."),           # not hedged
    _cause("A defect may be involved.", ids=()),                  # not linked
])
def test_answer_rejects_unlinked_causes(report, monkeypatch, bad_cause):
    bad = dict(good_answer(), possible_explanations=[bad_cause])
    assert llm.validate_answer(bad, report)
    monkeypatch.setattr(llm, "call_ollama", FakeAnswerOllama(bad, bad))
    assert llm.answer_question(report, "Why?")["metadata"]["source"] == "fallback"


def test_fallback_only_suggests_causes_with_detected_support(report):
    """With no return-rate finding, quality/fulfillment causes must not be suggested."""
    no_returns = copy.deepcopy(report)
    no_returns["key_insights"] = [i for i in report["key_insights"] if i["type"] != "return_rate_increase"]
    for rec in no_returns["recommendations"]:
        rec["based_on_insights"] = [i for i in rec["based_on_insights"] if i != "I1"]
    result = llm.fallback_explanation(no_returns)
    causes = [c["cause"] for r in result["reasoning"] for c in r["possible_explanations"]]
    assert causes and not any("quality" in c or "Fulfillment" in c for c in causes)
    assert llm.validate_explanation(result, no_returns) == []


def test_prompt_requires_linked_causes():
    prompt = llm.SYSTEM_PROMPT
    assert "linked to evidence" in prompt
    assert "must cite a return-rate finding" in prompt
    assert '"linked_insight_ids"' in prompt
    assert '"linked_insight_ids"' in llm.QA_SYSTEM_PROMPT


@pytest.mark.parametrize("circular", [
    "Revenue may have declined because units fell.",
    "South / Laptop Pro revenue decline may be due to a decline in profit.",
])
def test_circular_cause_is_rejected(report, circular):
    """A 'cause' that just restates another metric is not a business hypothesis."""
    bad = _mutated(reasoning__0__possible_explanations=[
        _cause(circular, evidence="Revenue fell 25.2%.", ids=("I3",))])
    assert any("restates a metric" in i for i in llm.validate_explanation(bad, report))


@pytest.mark.parametrize("text,ok", [
    ("South / Laptop Pro revenue fell 25.2%.", True),
    ("South revenue changed -4.8%.", True),
    ("The South region's revenue declined by 25.2%.", False),       # 25.2% is South / Laptop Pro
    ("Laptop Pro revenue fell 25.2%.", False),                       # also South / Laptop Pro's figure
    ("South / Laptop Pro revenue fell 25.2% while Laptop Pro in other regions grew 4.6%.", True),
    ("Overall revenue grew 3.3% but South / Laptop Pro fell 25.2%.", True),
])
def test_numbers_must_belong_to_the_segment_they_describe(report, text, ok):
    assert (llm.misattributed_numbers(text, report) == []) is ok


def test_misattributed_answer_falls_back(report, monkeypatch):
    bad = dict(good_answer(), answer="The South region's revenue declined by 25.2%.")
    monkeypatch.setattr(llm, "call_ollama", FakeAnswerOllama(bad, bad))
    result = llm.answer_question(report, "Is the problem only in the South region?")
    assert result["metadata"]["source"] == "fallback"
    assert any("not a South figure" in i for i in result["metadata"]["validation_issues"])


@pytest.mark.parametrize("cause,ok", [
    ("Product quality issues", True),                                    # bare hypothesis
    ("A product quality issue may be affecting sales.", True),           # hedged sentence
    ("Product quality issues are causing the decline.", False),          # asserted as fact
    ("The decline was caused by defects.", False),
])
def test_bare_hypotheses_allowed_but_assertions_must_be_hedged(report, cause, ok):
    issues = llm.check_causes([_cause(cause)], report, "c")
    assert (not any("without may/might/could" in i for i in issues)) is ok


def test_context_lists_supportable_cause_links(report):
    links = llm.build_llm_context(report)["cause_links"]
    by_id = {i["id"]: i for i in report["key_insights"]}
    return_ids = [i["id"] for i in report["key_insights"] if i["type"] == "return_rate_increase"]
    assert links["Product quality or defect issues"] == return_ids
    assert links["Fulfillment or delivery problems"] == return_ids
    for hypothesis, ids in links.items():
        assert ids and all(x in by_id for x in ids)
        # Every listed hypothesis is itself a valid business hypothesis.
        assert llm.BUSINESS_HYPOTHESIS.search(hypothesis), hypothesis
    assert "cause_links" in llm.SYSTEM_PROMPT


def test_misattribution_feedback_names_the_right_segment(report):
    issues = llm.misattributed_numbers("The South region's revenue declined by 25.2%.", report)
    assert any('It is the South / Laptop Pro figure' in i for i in issues)


def test_retry_feedback_lists_each_issue_once():
    message = llm._retry_message(["a", "b", "a", "a"])
    assert message.count("- a") == 1 and message.count("- b") == 1


@pytest.mark.parametrize("sentence,claims", [
    ("We should investigate the root cause of the decline.", False),
    ("Based on the return rate increase, identify the reason for the decline.", False),
    ("The root cause is a defective batch.", True),
    ("Investigate it, because the decline is due to quality problems.", True),
    ("The decline is due to quality problems.", True),
])
def test_recommending_an_investigation_is_not_a_causal_claim(sentence, claims):
    assert llm.states_cause(sentence) is claims


def test_pp_figures_must_be_real_percentage_point_values(report):
    """A 'pp' figure must be a gap or a rate change, not any number that appears elsewhere."""
    allowed, allowed_pp = llm.allowed_numbers(report), llm.allowed_pp_numbers(report)
    assert llm.unsupported_numbers("return rate +3.8 pp", allowed, allowed_pp) == []   # rate change
    assert llm.unsupported_numbers("gap -29.8 pp", allowed, allowed_pp) == []          # peer gap
    # 4.8 is South's revenue % change, not a percentage-point value.
    assert llm.unsupported_numbers("revenue fell 4.8%", allowed, allowed_pp) == []
    assert llm.unsupported_numbers("a gap of +4.8 pp", allowed, allowed_pp) == ["+4.8 pp"]
    bad = _mutated(reasoning__0__observed_facts=["Profit fell faster than revenue by +4.8 pp."])
    assert any("+4.8 pp" in i for i in llm.validate_explanation(bad, report))


def test_prompt_says_explain_only_the_supplied_evidence():
    prompt = llm.SYSTEM_PROMPT
    for phrase in ("Your job is to EXPLAIN the", "Only use the supplied evidence",
                   "does not by itself prove WHY", "cause is not established by the data",
                   "Never present a hypothesis as a fact"):
        assert phrase in prompt, phrase
    assert "Only use the supplied evidence" in llm.QA_SYSTEM_PROMPT


@pytest.mark.parametrize("restated,ids", [
    ("South / Laptop Pro return rate increase may be contributing to the decline in revenue and profit.", ["I1"]),
    ("The higher return rate may be contributing to lower sales.", ["I1"]),
    ("The revenue decline may be contributing to the profit decline.", ["I3"]),
    ("Margin compression may be contributing to the profit decline.", ["I2"]),
    ("Fewer units sold may be contributing to the decline.", ["I5"]),
])
def test_cause_that_restates_a_finding_is_rejected(report, restated, ids):
    """A measured change is evidence, not a cause: the cause must name a business hypothesis."""
    issues = llm.check_causes([_cause(restated, ids=ids)], report, "c")
    assert any("instead of naming a business hypothesis" in i for i in issues), issues


def test_restated_cause_feedback_suggests_a_hypothesis_from_the_linked_finding(report):
    [issue] = [i for i in llm.check_causes(
        [_cause("The return rate increase may be contributing to lower revenue.", ids=["I1"])], report, "c")
        if "business hypothesis" in i]
    assert 'e.g. "Product quality or defect issues may be contributing"' in issue
    [issue] = [i for i in llm.check_causes(
        [_cause("The profit decline may be contributing.", evidence="Profit fell 29.9%.", ids=["I2"])],
        report, "c") if "business hypothesis" in i]
    assert 'e.g. "Pricing or discount changes may be contributing"' in issue


@pytest.mark.parametrize("hypothesis,ids", [
    ("Product quality or defect issues may be contributing to the decline in revenue and units.", ["I1"]),
    ("Fulfillment or delivery problems may be contributing to the higher return rate.", ["I1"]),
    ("Pricing or discount changes may be contributing to the profit decline.", ["I2"]),
    ("Weaker demand or stronger competition may be contributing to the revenue decline.", ["I3", "I5"]),
])
def test_business_hypothesis_with_linked_evidence_is_accepted(report, hypothesis, ids):
    by_id = {i["id"]: i for i in report["key_insights"]}
    cause = _cause(hypothesis, evidence=by_id[ids[0]]["summary"], ids=ids)
    assert llm.check_causes([cause], report, "c") == []


def test_retry_is_sent_restated_cause_feedback_then_fallback_keeps_hypotheses(report, monkeypatch):
    bad = _mutated(reasoning__0__possible_explanations=[
        _cause("South / Laptop Pro return rate increase may be contributing to the decline.")])
    fake = FakeOllama(bad, bad)
    monkeypatch.setattr(llm, "call_ollama", fake)
    result = llm.explain_decision_report(report)
    assert "Product quality or defect issues may be contributing" in fake.calls[1]["messages"][-1]["content"]
    assert result["metadata"]["source"] == "fallback"
    for c in [c for r in result["reasoning"] for c in r["possible_explanations"]]:
        assert llm.BUSINESS_HYPOTHESIS.search(c["cause"]) and c["linked_insight_ids"]


def test_prompt_requires_causes_to_name_a_business_hypothesis():
    for prompt in (llm.SYSTEM_PROMPT, llm.QA_SYSTEM_PROMPT):
        assert "NAME A BUSINESS HYPOTHESIS" in prompt
        assert "a measured change is evidence, not a cause" in prompt


def test_required_coverage_follows_the_recommendation_rule(report):
    """decline_with_rising_returns is driven by the return-rate rise AND the decline."""
    required = llm.required_cause_coverage(report)
    by_id = {i["id"]: i for i in report["key_insights"]}
    kinds = [{by_id[i]["type"] for i in r["insight_ids"]} for r in required]
    assert {"return_rate_increase"} in kinds
    assert any(k & llm.DECLINE_TYPES for k in kinds)
    returns = next(r for r in required if r["insight_ids"] == ["I1"])
    assert returns["suggested_hypotheses"] == ["Product quality or defect issues",
                                              "Fulfillment or delivery problems"]
    assert llm.build_llm_context(report)["causes_to_cover"] == required


def test_explanation_missing_the_returns_hypothesis_is_rejected(report, monkeypatch):
    """Valid causes that skip the return-rate finding behind the recommendation are not enough."""
    by_id = {i["id"]: i for i in report["key_insights"]}
    only_pricing = _mutated(reasoning__0__possible_explanations=[
        _cause("Pricing or discount changes may be contributing to the decline in revenue.",
               evidence=by_id["I3"]["summary"], ids=["I3", "I4"])])
    issues = llm.validate_explanation(only_pricing, report)
    assert any("No possible cause is linked to I1" in i and "Product quality or defect issues" in i
               for i in issues)
    fake = FakeOllama(only_pricing, only_pricing)
    monkeypatch.setattr(llm, "call_ollama", fake)
    result = llm.explain_decision_report(report)
    assert "No possible cause is linked to I1" in fake.calls[1]["messages"][-1]["content"]
    assert result["metadata"]["source"] == "fallback"


def test_explanation_with_quality_hypothesis_on_returns_evidence_passes(report):
    by_id = {i["id"]: i for i in report["key_insights"]}
    covered = _mutated(reasoning__0__possible_explanations=[
        _cause("Product quality or defect issues may be contributing to the decline in revenue and units.",
               evidence="South / Laptop Pro return rate rose from 4.7% to 8.5% (+3.8 pp).", ids=["I1"]),
        _cause("Weaker demand or stronger competition may be contributing to the revenue decline.",
               evidence=by_id["I3"]["summary"], ids=["I3", "I5"]),
    ])
    assert llm.validate_explanation(covered, report) == []


def test_fallback_always_covers_the_required_findings(report):
    fallback = llm.fallback_explanation(report)
    causes = [c for r in fallback["reasoning"] for c in r["possible_explanations"]]
    assert llm.coverage_issues(causes, report) == []
    assert "causes_to_cover" in llm.SYSTEM_PROMPT
