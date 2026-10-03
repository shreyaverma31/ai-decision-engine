"""LLM explanation layer.

Raw CSV -> deterministic analysis (engine.py) -> structured evidence -> LLM explanation

The LLM gets a compact version of the decision report, never the CSV, and
only explains it. Its answer is validated before use:
* it must be valid JSON in the expected shape,
* every insight, recommendation and evidence ID it cites must exist,
* every number it writes must appear in the report,
* observed facts must not claim causes or metrics that the data doesn't contain.

If Ollama is unreachable or the answer fails validation, a deterministic
explanation built from the same report is returned instead, so callers always
get a usable, grounded response.

Usage:
    python llm.py data/decision_report.json
    OLLAMA_MODEL=qwen2.5:3b python llm.py data/decision_report.json --fallback-only
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import httpx

DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = 180.0  # seconds; small local models on CPU are slow
MAX_ATTEMPTS = 2         # one retry with the validation errors fed back
MAX_OUTPUT_TOKENS = 900

GROUNDING_RULES = """The report was produced by a deterministic analysis engine. Your job is to EXPLAIN the
evidence it already produced, not to analyse the business yourself or come up with your own
explanation. The data shows WHAT changed and WHERE; it does not by itself prove WHY.
Rules:
0. Only use the supplied evidence. Do not add facts, numbers, segments or causes from outside it.
1. The evidence is authoritative. Treat every number in it as correct and final.
2. Do not invent numbers. Only use numbers that appear in the report, copied as written.
   Do not calculate new numbers (no new percentages, sums, differences or forecasts).
3. Do not invent causes that are not supported by the evidence.
4. Clearly separate observed facts (what the data shows) from possible explanations
   (hypotheses that would need to be checked).
5. Recommendations must be grounded in the supplied evidence, must cite the
   insight IDs they are based on, and must use a recommendation_id from the report.
6. If the evidence is insufficient to establish a cause, say that further investigation is needed.
   The data in this report cannot establish a cause on its own, so say explicitly that the
   cause is not established by the data.
7. Never claim certainty about causation when the data only shows correlation.
   Never present a hypothesis as a fact: write "X may be contributing", never "X caused the decline".
   Phrase possible explanations with "may", "might" or "could".
8. Only mention metrics that are in the report (revenue, cost, profit, profit margin,
   units, returns, return rate). Do not mention metrics such as traffic, conversion,
   ratings or customer satisfaction as facts.
9. Every possible explanation must be linked to evidence: cite the insight IDs whose
   evidence makes it plausible, and quote that evidence (with its numbers) in "evidence".
   Quality, defect, fulfillment, delivery or returns causes must cite a return-rate finding.
   Pricing, discount or cost causes must cite a profit or revenue finding.
   Demand, competition or availability causes must cite a revenue or units decline.
   If no finding supports a cause, do not list it. The report's "cause_links" lists business
   hypotheses and, for each, the insight IDs that can support it: only propose those hypotheses,
   and cite IDs from that list.
10. A possible cause must NAME A BUSINESS HYPOTHESIS (what might be happening in the business),
   not restate the finding. Write it as "<hypothesis from cause_links> may be contributing to
   <the observed change>", and put the finding itself in "evidence". Never write
   "<metric> increase/decline may be contributing" - a measured change is evidence, not a cause.
11. The report's "causes_to_cover" lists the findings that drove the recommendation. Give at
   least one possible cause for EACH entry, linked to that entry's insight IDs and named as one
   of its suggested hypotheses."""

SYSTEM_PROMPT = """You are a business analyst explaining a decision report to a manager.

""" + GROUNDING_RULES + """

Be concise. Respond with JSON only, in exactly this shape:
{
  "summary": "2-3 sentence overview for a manager",
  "key_findings": [
    {"statement": "one observed fact (at most 4 findings)", "insight_ids": ["I1"]}
  ],
  "reasoning": [
    {
      "observed_facts": ["short fact from the report (at most 3)"],
      "possible_explanations": [
        {"cause": "hypothesis phrased with may/might/could (at most 3)",
         "evidence": "the observed finding that makes it plausible, quoted from the report",
         "linked_insight_ids": ["I1"]}
      ],
      "needs_further_investigation": true,
      "insight_ids": ["I1"]
    }
  ],
  "recommendations": [
    {"action": "what to do", "rationale": "one sentence citing the evidence", "recommendation_id": "R1", "insight_ids": ["I1"]}
  ]
}"""

CAUSE_SCHEMA = {
    "type": "object",
    "properties": {
        "cause": {"type": "string"},
        "evidence": {"type": "string"},
        "linked_insight_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["cause", "evidence", "linked_insight_ids"],
}

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "key_findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "statement": {"type": "string"},
                    "insight_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["statement", "insight_ids"],
            },
        },
        "reasoning": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "observed_facts": {"type": "array", "items": {"type": "string"}},
                    "possible_explanations": {"type": "array", "items": CAUSE_SCHEMA},
                    "needs_further_investigation": {"type": "boolean"},
                    "insight_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["observed_facts", "possible_explanations",
                             "needs_further_investigation", "insight_ids"],
            },
        },
        "recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "rationale": {"type": "string"},
                    "recommendation_id": {"type": "string"},
                    "insight_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["action", "rationale", "recommendation_id", "insight_ids"],
            },
        },
    },
    "required": ["summary", "key_findings", "reasoning", "recommendations"],
}

ASSERTIVE = re.compile(r"\b(is|are|was|were|has|have|had|caus\w*|led|leads?|result\w*|due to|because|"
                       r"explains?|drives?|driven)\b", re.I)

# Finding types that can support each kind of cause.
RETURN_TYPES = {"return_rate_increase"}
DECLINE_TYPES = {"revenue_decline", "units_decline", "concentrated_decline",
                 "region_specific_decline", "product_specific_decline"}
PRICE_TYPES = DECLINE_TYPES | {"profit_decline"}

# A cause that mentions one of these topics must cite at least one finding of a matching type.
CAUSE_TOPICS = [
    (re.compile(r"quality|defect|faulty|fulfil|deliver|shipping|damage|listing|expectation|return", re.I),
     RETURN_TYPES, "a return-rate finding"),
    (re.compile(r"pric|discount|margin|cost", re.I), PRICE_TYPES, "a profit or revenue finding"),
    (re.compile(r"demand|competit|market|preference|availab|stock|inventory|seasonal", re.I),
     DECLINE_TYPES, "a revenue or units decline"),
]

# Words that name something that might be happening in the business. Metric words such as
# "return rate", "revenue" or "margin" are deliberately absent: a measured change is evidence,
# not a cause. (They still decide, via CAUSE_TOPICS, which finding a cause must cite.)
BUSINESS_HYPOTHESIS = re.compile(
    r"quality|defect|faulty|fulfil|deliver|shipping|damage|packag|listing|expectation|"
    r"pric|discount|cost|supplier|demand|competit|market|preference|availab|stock|inventory|"
    r"seasonal|promotion|customer experience|customer service", re.I)

# Named business hypotheses and the finding types that can make each plausible.
# Used for the model's "cause_links" and to suggest a proper hypothesis in retry feedback.
CAUSE_HYPOTHESES = [
    ("Product quality or defect issues", RETURN_TYPES),
    ("Fulfillment or delivery problems", RETURN_TYPES),
    ("Pricing or discount changes", PRICE_TYPES),
    ("Weaker demand or stronger competition", DECLINE_TYPES),
]


def suggested_hypotheses(linked_types: set[str]) -> list[str]:
    """Business hypotheses the linked findings can support, most specific first."""
    return [name for name, types in CAUSE_HYPOTHESES if linked_types & types]


# The findings each engine rule is based on. The explanation must offer a possible cause for each,
# so a key signal (e.g. the return-rate rise behind "decline with rising returns") is never skipped.
RULE_SIGNALS = {
    "decline_with_rising_returns": [RETURN_TYPES, DECLINE_TYPES],
    "decline_with_stable_returns": [DECLINE_TYPES],
    "rising_returns_without_decline": [RETURN_TYPES],
    "profit_decline_with_stable_revenue": [{"profit_decline"}],
}


def required_cause_coverage(report: dict) -> list[dict]:
    """For each recommendation's main rule: the findings that drove it, and hypotheses that fit them."""
    by_id = {i["id"]: i for i in report.get("key_insights", [])}
    required, seen = [], set()
    for rec in report.get("recommendations", []):
        main_rule = next((r for r in rec.get("rules_triggered", []) if r in RULE_SIGNALS), None)
        if main_rule is None:
            continue
        own = [by_id[i] for i in rec["based_on_insights"]
               if i in by_id and by_id[i]["dimension_value"] == rec["segment"]]
        for types in RULE_SIGNALS[main_rule]:
            ids = [i["id"] for i in own if i["type"] in types]
            if not ids or frozenset(ids) in seen:
                continue
            seen.add(frozenset(ids))
            required.append({"insight_ids": ids,
                             "suggested_hypotheses": suggested_hypotheses({by_id[i]["type"] for i in ids})})
    return required


def coverage_issues(causes: list[dict], report: dict) -> list[str]:
    """Findings that drove a recommendation but have no possible cause linked to them."""
    linked = {x for c in causes for x in c.get("linked_insight_ids", [])}
    issues = []
    for need in required_cause_coverage(report):
        if not linked & set(need["insight_ids"]):
            examples = " or ".join(f"\"{h} may be contributing\"" for h in need["suggested_hypotheses"][:2])
            issues.append(f"No possible cause is linked to {', '.join(need['insight_ids'])}, which drove the "
                          f"recommendation. Add one, e.g. {examples}, with that finding as evidence.")
    return issues

# Hypotheses the data is consistent with but cannot confirm, one set per engine rule,
# each with the finding types that make it plausible. Used by the deterministic fallback.
RULE_HYPOTHESES = {
    "decline_with_rising_returns": [
        ("A product quality or defect issue may be affecting this segment.", RETURN_TYPES),
        ("Fulfillment or delivery problems could be causing both returns and lost sales.", RETURN_TYPES),
        ("Demand, pricing, or competition changes could be reducing sales.", DECLINE_TYPES),
    ],
    "decline_with_stable_returns": [
        ("Demand may have weakened in this segment.", DECLINE_TYPES),
        ("Pricing, competition, or stock availability could be limiting sales.", DECLINE_TYPES),
    ],
    "rising_returns_without_decline": [
        ("A product quality or delivery issue may be emerging before it affects sales.", RETURN_TYPES),
    ],
    "profit_decline_with_stable_revenue": [
        ("Costs or discounting may have increased relative to revenue.", {"profit_decline"}),
    ],
    "margin_erosion": [
        ("Deeper discounting may have been used to support sales volume.", {"profit_decline"}),
    ],
}


def linked_causes(rules: list[str], insights: list[dict]) -> list[dict]:
    """Fallback causes for the triggered rules, each linked to the findings that support it.

    A hypothesis is only listed if a supporting finding was actually detected.
    """
    causes, seen = [], set()
    for rule in rules:
        for text, types in RULE_HYPOTHESES.get(rule, []):
            support = [i for i in insights if i["type"] in types]
            if not support or text in seen:
                continue
            seen.add(text)
            causes.append({"cause": text, "evidence": support[0]["summary"],
                           "linked_insight_ids": [i["id"] for i in support[:2]]})
    return causes

CORRELATION_NOTE = (
    "The data shows these changes happening together; it does not establish their cause. "
    "Further investigation is needed to confirm any explanation."
)


class OllamaUnavailable(RuntimeError):
    """Ollama could not be reached or returned an unusable HTTP response."""


# ---------------------------------------------------------------------------
# Context sent to the model (compact; never raw data)
# ---------------------------------------------------------------------------

def _compact_number(value):
    return round(value, 4) if isinstance(value, float) else value


def build_llm_context(report: dict) -> dict:
    """What the model sees: detected insights (with their evidence numbers) and the
    engine's recommendations. Never raw rows."""
    meta = report.get("metadata", {})
    overview = report.get("overview", {}).get("metrics", {})
    return {
        "period_previous": meta.get("period_previous", {}).get("label"),
        "period_current": meta.get("period_current", {}).get("label"),
        "overall_change_percent": {
            m: overview[m]["change_percent"] for m in ("revenue", "profit") if m in overview
        },
        "insights": [
            {
                "id": i["id"],
                "type": i.get("type"),
                "severity": i.get("severity"),
                "segment": i.get("dimension_value"),
                "metric": i.get("metric"),
                "previous_value": _compact_number(i.get("previous_value")),
                "current_value": _compact_number(i.get("current_value")),
                "change_percent": i.get("change_percent"),
                "evidence": i.get("summary"),
            }
            for i in report.get("key_insights", [])
        ],
        "recommendations": [
            {k: r.get(k) for k in ("id", "segment", "priority", "actions", "based_on_insights")}
            for r in report.get("recommendations", [])
        ],
        "cause_links": cause_links(report),
        "causes_to_cover": required_cause_coverage(report),
    }


def cause_links(report: dict) -> dict[str, list[str]]:
    """Which detected findings can support each kind of cause (computed, not guessed).

    Given to the model so it only proposes causes the evidence can support.
    """
    insights = report.get("key_insights", [])
    out = {name: [i["id"] for i in insights if i["type"] in types] for name, types in CAUSE_HYPOTHESES}
    return {name: ids for name, ids in out.items() if ids}


def build_messages(report: dict) -> list[dict]:
    context = json.dumps(build_llm_context(report), indent=1)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Explain this decision report.\n\nREPORT:\n{context}"},
    ]


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

def call_ollama(
    messages: list[dict],
    model: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    schema: dict | None = None,
) -> str:
    """Send a chat request to Ollama's HTTP API and return the message content.

    `schema` constrains the output format (defaults to the explanation schema)."""
    model = model or os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
    base_url = (base_url or os.environ.get("OLLAMA_URL", DEFAULT_URL)).rstrip("/")
    timeout = timeout or float(os.environ.get("OLLAMA_TIMEOUT", DEFAULT_TIMEOUT))
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": schema or RESPONSE_SCHEMA,
        "options": {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": MAX_OUTPUT_TOKENS},
    }
    try:
        response = httpx.post(f"{base_url}/api/chat", json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()["message"]["content"]
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        raise OllamaUnavailable(f"{type(exc).__name__}: {exc}") from exc


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

NUMBER = re.compile(
    r"(?<![\w.])[-+]?\$?(\d[\d,]*(?:\.\d+)?)\s?"
    r"(percentage points?|percent|million|thousand|billion|pp|%|M|K|B)?(?![\w])"
)
SCALES = {
    "M": [1e-6], "million": [1e-6], "K": [1e-3], "thousand": [1e-3], "B": [1e-9], "billion": [1e-9],
    "%": [1, 100], "percent": [1, 100], "pp": [1, 100],
    "percentage point": [1, 100], "percentage points": [1, 100],
}
SMALL_COUNT_LIMIT = 10  # bare integers up to this ("two segments", "3 actions") are allowed

CAUSAL = re.compile(
    r"\b(caused by|because of|due to|as a result of|resulted from|results from|"
    r"root cause|the reason (?:is|was|for)|led to|leads to|attributable to|attributed to|"
    r"driven by|stems? from|triggered by)\b", re.I)
HEDGE = re.compile(
    r"\b(may|might|could|possibl[ey]|potential(?:ly)?|suggests?|hypothes[ie]s|unclear|"
    r"not (?:yet )?(?:established|confirmed|known))\b", re.I)
CERTAINTY = re.compile(
    r"\b(definitely|certainly|undoubtedly|clearly caused|proves?|proven|without doubt)\b", re.I)
UNAVAILABLE_METRICS = re.compile(
    r"\b(conversion(?: rate)?|traffic|site visits|page views|NPS|net promoter|"
    r"satisfaction score|customer satisfaction|churn|market share|customer count|"
    r"inventory levels?|stock levels?|star ratings?|review scores?|complaint (?:rate|volume)|"
    r"delivery times?|lead times?)\b", re.I)


def _numeric_leaves(obj):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield abs(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _numeric_leaves(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _numeric_leaves(v)


def allowed_numbers(report: dict) -> list[float]:
    """Every number the report contains, plus the digits in its period labels (e.g. 2025)."""
    numbers = set(_numeric_leaves(report))
    meta = report.get("metadata", {})
    for key in ("period_previous", "period_current"):
        period = meta.get(key, {})
        for text in (period.get("label", ""), period.get("start", ""), period.get("end", "")):
            numbers.update(float(n) for n in re.findall(r"\d+", text or ""))
    return sorted(numbers)


def allowed_pp_numbers(report: dict) -> list[float]:
    """Percentage-point values the report actually contains: peer gaps, return-rate and margin changes.

    A figure written with "pp" must be one of these, not just any number that happens to be in
    the report (e.g. "+4.8 pp" when 4.8 is a revenue % change elsewhere).
    """
    values = set()

    def walk(obj):
        if isinstance(obj, dict):
            for key, v in obj.items():
                if key in ("gaps_pp", "change_pp"):
                    values.update(_numeric_leaves(v))
                walk(v)
            if obj.get("metric") in ("return_rate", "profit_margin") and isinstance(obj.get("change"), (int, float)):
                values.add(abs(obj["change"]) * 100)
            if isinstance(obj.get("return_rate"), dict) and isinstance(obj["return_rate"].get("change"), (int, float)):
                values.add(abs(obj["return_rate"]["change"]) * 100)
            if isinstance(obj.get("profit_margin"), dict) and isinstance(obj["profit_margin"].get("change"), (int, float)):
                values.add(abs(obj["profit_margin"]["change"]) * 100)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(report)
    return sorted(values)


def unsupported_numbers(text: str, allowed: list[float], allowed_pp: list[float] | None = None) -> list[str]:
    """Numbers in `text` that don't match (within display rounding) a report number.

    If `allowed_pp` is given, figures written with "pp" must match one of those values.
    """
    bad = []
    for match in NUMBER.finditer(text):
        digits, unit = match.group(1), match.group(2)
        value = float(digits.replace(",", ""))
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        if unit is None and decimals == 0 and value <= SMALL_COUNT_LIMIT:
            continue
        # A number rounded for display may be up to half a unit of its last digit off.
        tol = 0.5 * 10 ** -decimals + 1e-9
        if allowed_pp is not None and unit and unit.lower() in ("pp", "percentage point", "percentage points"):
            if not any(abs(a - value) <= tol for a in allowed_pp):
                bad.append(match.group(0).strip())
            continue
        scales = SCALES.get(unit.lower() if unit and unit not in ("M", "K", "B") else unit, [1])
        if not any(abs(a * s - value) <= tol for a in allowed for s in scales):
            bad.append(match.group(0).strip())
    return bad


# Sentences mentioning these talk about more than one segment, so attribution isn't checked.
MULTI_SEGMENT = re.compile(r"\b(overall|total|business|company|other regions|other products|peers?|"
                           r"while|whereas|compared|than)\b", re.I)


def _segment_numbers(report: dict) -> dict[str, list[float]]:
    """Numbers that belong to each named segment (e.g. 'South', 'South / Laptop Pro')."""
    numbers: dict[str, set] = {}
    for e in report.get("evidence", []):
        if e.get("filters") and all(len(v) == 1 for v in e["filters"].values()):
            name = " / ".join(v[0] for v in e["filters"].values())
            numbers.setdefault(name, set()).update(_numeric_leaves(e))
    for i in report.get("key_insights", []):
        numbers.setdefault(i["dimension_value"], set()).update(
            _numeric_leaves({k: v for k, v in i.items() if k != "evidence"}))
    return {k: sorted(v) for k, v in numbers.items()}


def misattributed_numbers(text: str, report: dict) -> list[str]:
    """Percentages and money figures attributed to a segment they don't belong to.

    Only checks sentences that name exactly one segment and don't compare with others,
    e.g. "The South region's revenue fell 54.5%" when 54.5% is South / Laptop Pro's figure.
    """
    by_segment = _segment_numbers(report)
    names = sorted(by_segment, key=len, reverse=True)
    issues = []
    for sentence in _sentences(text):
        if MULTI_SEGMENT.search(sentence):
            continue
        rest, mentioned = sentence, []
        for name in names:
            pattern = re.compile(rf"(?<![\w/]){re.escape(name)}(?![\w])")
            if pattern.search(rest):
                mentioned.append(name)
                rest = pattern.sub(" ", rest)
        if len(mentioned) != 1:
            continue
        own = by_segment[mentioned[0]]
        for match in NUMBER.finditer(sentence):
            digits, unit = match.group(1), match.group(2)
            if unit is None and "$" not in match.group(0):
                continue  # counts, years and IDs are not attributed
            value = float(digits.replace(",", ""))
            decimals = len(digits.split(".")[1]) if "." in digits else 0
            tol = 0.5 * 10 ** -decimals + 1e-9
            scales = SCALES.get(unit.lower() if unit and unit not in ("M", "K", "B") else unit, [1])
            if not any(abs(a * s - value) <= tol for a in own for s in scales):
                owner = next((name for name, nums in by_segment.items() if name != mentioned[0]
                              and any(abs(a * s - value) <= tol for a in nums for s in scales)), None)
                hint = f" It is the {owner} figure: write \"{owner}\"." if owner else ""
                issues.append(f"'{match.group(0).strip()}' is not a {mentioned[0]} figure: "
                              f"\"{sentence[:80]}\".{hint}")
    return issues


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def explanation_texts(explanation: dict) -> tuple[list[str], list[str]]:
    """(fact texts, hypothesis texts) from an explanation."""
    facts = [explanation.get("summary", "")]
    facts += [f.get("statement", "") for f in explanation.get("key_findings", [])]
    hypotheses = []
    for r in explanation.get("reasoning", []):
        facts += r.get("observed_facts", [])
        for c in r.get("possible_explanations", []):
            if isinstance(c, dict):
                hypotheses.append(c.get("cause", ""))
                facts.append(c.get("evidence", ""))  # the quoted evidence is a factual claim
    for r in explanation.get("recommendations", []):
        facts += [r.get("action", ""), r.get("rationale", "")]
    return facts, hypotheses


def _check_shape(obj) -> list[str]:
    if not isinstance(obj, dict):
        return ["Response is not a JSON object."]
    issues = []
    for key in RESPONSE_SCHEMA["required"]:
        expected = RESPONSE_SCHEMA["properties"][key]["type"]
        value = obj.get(key)
        if value is None:
            issues.append(f"Missing field '{key}'.")
        elif expected == "string" and not isinstance(value, str):
            issues.append(f"'{key}' must be a string.")
        elif expected == "array" and not isinstance(value, list):
            issues.append(f"'{key}' must be a list.")
    if issues:
        return issues

    for key in ("key_findings", "reasoning", "recommendations"):
        item_schema = RESPONSE_SCHEMA["properties"][key]["items"]
        for n, item in enumerate(obj[key]):
            if not isinstance(item, dict):
                issues.append(f"{key}[{n}] must be an object.")
                continue
            for field in item_schema["required"]:
                fschema = item_schema["properties"][field]
                value = item.get(field)
                if fschema.get("items") is CAUSE_SCHEMA:
                    issues += _cause_shape_issues(value, f"{key}[{n}].{field}")
                    continue
                ok = {
                    "string": isinstance(value, str) and value.strip() != "",
                    "array": isinstance(value, list) and all(isinstance(v, str) for v in value),
                    "boolean": isinstance(value, bool),
                }[fschema["type"]]
                if not ok:
                    issues.append(f"{key}[{n}].{field} is missing or has the wrong type.")
    return issues


def _cause_shape_issues(causes, where: str) -> list[str]:
    if not isinstance(causes, list):
        return [f"{where} must be a list."]
    issues = []
    for n, c in enumerate(causes):
        if not isinstance(c, dict):
            issues.append(f"{where}[{n}] must be an object with cause, evidence and linked_insight_ids.")
            continue
        for field in ("cause", "evidence"):
            if not isinstance(c.get(field), str) or not c[field].strip():
                issues.append(f"{where}[{n}].{field} must be a non-empty string.")
        ids = c.get("linked_insight_ids")
        if not isinstance(ids, list) or not all(isinstance(v, str) for v in ids):
            issues.append(f"{where}[{n}].linked_insight_ids must be a list of IDs.")
    return issues


def check_causes(causes: list[dict], report: dict, where: str) -> list[str]:
    """Each possible cause must be hedged, cite real findings, and cite the right kind of finding."""
    by_id = {i["id"]: i for i in report.get("key_insights", [])}
    issues = []
    for n, c in enumerate(causes):
        label = f"{where}[{n}]"
        ids = c["linked_insight_ids"]
        if not ids:
            issues.append(f"{label} is not linked to any finding.")
        unknown = [x for x in ids if x not in by_id]
        issues += [f"{label} cites unknown ID '{x}'." for x in unknown]
        # A bare hypothesis ("quality issues") is fine in a field labelled "possible cause";
        # a sentence that asserts something ("quality issues are causing it") must be hedged.
        if ASSERTIVE.search(c["cause"]) and not HEDGE.search(c["cause"]):
            issues.append(f"{label} states a cause without may/might/could: \"{c['cause'][:80]}\".")
        linked_types = {by_id[x]["type"] for x in ids if x in by_id}
        if not BUSINESS_HYPOTHESIS.search(c["cause"]):
            suggestions = suggested_hypotheses(linked_types)
            hint = (f" Name the hypothesis instead, e.g. \"{suggestions[0]} may be contributing\", and keep "
                    f"the finding in evidence." if suggestions else "")
            issues.append(f"{label} restates a metric or finding instead of naming a business hypothesis "
                          f"(e.g. quality, fulfillment, pricing, demand, competition, availability, costs): "
                          f"\"{c['cause'][:80]}\".{hint}")
        for pattern, types, needed in CAUSE_TOPICS:
            if pattern.search(c["cause"]) and linked_types and not linked_types & types:
                issues.append(f"{label} (\"{c['cause'][:60]}\") must cite {needed}.")
    return issues


def validate_explanation(explanation, report: dict) -> list[str]:
    """Return a list of problems; an empty list means the explanation can be used."""
    issues = _check_shape(explanation)
    if issues:
        return issues

    insight_ids = {i["id"] for i in report.get("key_insights", [])}
    evidence_ids = {e["id"] for e in report.get("evidence", [])}
    rec_ids = {r["id"] for r in report.get("recommendations", [])}

    def unknown(ids, known, where):
        for x in ids:
            if x not in known:
                issues.append(f"{where} cites unknown ID '{x}'.")

    for n, f in enumerate(explanation["key_findings"]):
        if not f["insight_ids"]:
            issues.append(f"key_findings[{n}] does not cite any insight.")
        unknown(f["insight_ids"], insight_ids, f"key_findings[{n}]")
        unknown(f.get("evidence_ids", []), evidence_ids, f"key_findings[{n}]")
    for n, r in enumerate(explanation["reasoning"]):
        unknown(r["insight_ids"], insight_ids, f"reasoning[{n}]")
        if r["possible_explanations"] and not r["needs_further_investigation"]:
            issues.append(f"reasoning[{n}] offers explanations without flagging further investigation.")
        issues += check_causes(r["possible_explanations"], report, f"reasoning[{n}].possible_explanations")
    for n, r in enumerate(explanation["recommendations"]):
        if not r["insight_ids"]:
            issues.append(f"recommendations[{n}] does not cite any detected finding.")
        unknown(r["insight_ids"], insight_ids, f"recommendations[{n}]")
        unknown([r["recommendation_id"]], rec_ids, f"recommendations[{n}]")

    if report.get("key_insights") and not explanation["key_findings"]:
        issues.append("The report has findings but the explanation lists none.")
    covered = {r["recommendation_id"] for r in explanation["recommendations"]}
    for missing in sorted(rec_ids - covered):
        issues.append(f"Recommendation {missing} from the report is not covered.")
    issues += coverage_issues([c for r in explanation["reasoning"] for c in r["possible_explanations"]
                               if isinstance(c, dict)], report)

    allowed, allowed_pp = allowed_numbers(report), allowed_pp_numbers(report)
    facts, hypotheses = explanation_texts(explanation)
    for text in facts + hypotheses:
        for number in unsupported_numbers(text, allowed, allowed_pp):
            issues.append(f"Number not found in the report: '{number}' in \"{text[:80]}\".")
        if CERTAINTY.search(text):
            issues.append(f"Claims certainty: \"{text[:80]}\".")
    for text in facts:
        for sentence in _sentences(text):
            if states_cause(sentence):
                issues.append(f"States a cause as fact: \"{sentence[:80]}\".")
        if (m := UNAVAILABLE_METRICS.search(text)) and not HEDGE.search(text):
            issues.append(f"Mentions a metric that is not in the report ('{m.group(0)}').")
        issues += misattributed_numbers(text, report)
    return issues


INVESTIGATE = re.compile(r"\b(investigat\w*|identify|determine|find out|understand|look into)\b", re.I)


def states_cause(sentence: str) -> bool:
    """True if the sentence asserts a cause as fact.

    "Investigate the root cause" / "find the reason for" is a recommendation, not a claim.
    """
    matches = [m.group(0).lower() for m in CAUSAL.finditer(sentence)]
    if not matches or HEDGE.search(sentence):
        return False
    if INVESTIGATE.search(sentence):
        matches = [m for m in matches if not m.startswith(("root cause", "the reason"))]
    return bool(matches)


def _retry_message(issues: list[str]) -> str:
    """Feedback for the model's one retry: each distinct problem once."""
    return ("Your response broke these rules:\n- " + "\n- ".join(list(dict.fromkeys(issues))[:10])
            + "\nReturn the corrected JSON only.")


def _parse_json(content: str):
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
    return json.loads(content)


# ---------------------------------------------------------------------------
# Deterministic fallback
# ---------------------------------------------------------------------------

# Which insight types support which recommendation rule, for per-action rationale.
RULE_INSIGHT_TYPES = {
    "decline_with_rising_returns": {"return_rate_increase", "revenue_decline", "units_decline",
                                    "concentrated_decline", "region_specific_decline",
                                    "product_specific_decline"},
    "decline_with_stable_returns": {"revenue_decline", "units_decline", "concentrated_decline",
                                    "region_specific_decline", "product_specific_decline"},
    "rising_returns_without_decline": {"return_rate_increase"},
    "profit_decline_with_stable_revenue": {"profit_decline"},
    "margin_erosion": {"profit_decline"},
}


def fallback_explanation(report: dict) -> dict:
    """An explanation assembled directly from the report's own text and IDs."""
    insights = report.get("key_insights", [])
    if not insights:
        summary = report.get("executive_summary") or "The report contains no findings to explain."
        return {"summary": summary, "key_findings": [], "reasoning": [], "recommendations": []}

    by_id = {i["id"]: i for i in insights}
    reasoning, recommendations = [], []
    for rec in report.get("recommendations", []):
        own = [by_id[i] for i in rec["based_on_insights"]
               if i in by_id and by_id[i]["dimension_value"] == rec["segment"]]
        reasoning.append({
            "observed_facts": list(rec["rationale"]),
            "possible_explanations": linked_causes(rec["rules_triggered"], own),
            "needs_further_investigation": True,
            "insight_ids": list(rec["based_on_insights"]),
            "note": CORRELATION_NOTE,
        })
        for rule, action in zip(rec["rules_triggered"], rec["actions"]):
            support = [i for i in own if i["type"] in RULE_INSIGHT_TYPES.get(rule, set())] or own
            rationale = [i["summary"] for i in support]
            if rule == "margin_erosion":
                rationale += [line for line in rec["rationale"] if line.startswith("Profit margin")]
            recommendations.append({
                "action": action,
                "rationale": " ".join(rationale),
                "recommendation_id": rec["id"],
                "insight_ids": [i["id"] for i in support],
            })
    return {
        "summary": report.get("executive_summary", ""),
        "key_findings": [{"statement": i["summary"], "insight_ids": [i["id"]]} for i in insights],
        "reasoning": reasoning,
        "recommendations": recommendations,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _evidence_for(insight_ids: list[str], report: dict) -> list[str]:
    by_id = {i["id"]: i for i in report.get("key_insights", [])}
    return list(dict.fromkeys(
        e["id"] for i in insight_ids if i in by_id for e in by_id[i]["evidence"]
    ))


def _resolve_evidence(ids: list[str], report: dict) -> list[dict]:
    """The actual evidence records behind the IDs, taken from the report."""
    evidence = {e["id"]: e for e in report.get("evidence", [])}
    keep = ("id", "description", "metric", "filters", "period_previous", "period_current",
            "previous_value", "current_value", "change", "change_percent")
    return [{k: evidence[i][k] for k in keep if k in evidence[i]} for i in ids if i in evidence]


def _finalize(explanation: dict, report: dict, source: str, model, issues, reason=None) -> dict:
    """Attach evidence (deterministically, from cited insights) and provenance metadata."""
    result = {k: explanation[k] for k in ("summary", "key_findings", "reasoning", "recommendations")}
    for finding in result["key_findings"]:
        finding["evidence_ids"] = _evidence_for(finding["insight_ids"], report)
    cited = [i for section in ("key_findings", "reasoning", "recommendations")
             for item in result[section] for i in item["insight_ids"]]
    result["evidence_references"] = _resolve_evidence(_evidence_for(cited, report), report)
    result["metadata"] = {
        "source": source,
        "model": model,
        "fallback_reason": reason,
        "validation_issues": issues,
    }
    return result


def explain_decision_report(
    report: dict,
    model: str | None = None,
    base_url: str | None = None,
    use_llm: bool = True,
    max_attempts: int = MAX_ATTEMPTS,
) -> dict:
    """Explain a decision report from engine.generate_decision_report().

    Returns summary, key_findings, reasoning, recommendations and
    evidence_references, plus metadata saying whether the LLM or the
    deterministic fallback produced the text and why.
    """
    if not isinstance(report, dict):
        report = {}
    model = model or os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)

    if not report.get("key_insights"):
        return _finalize(fallback_explanation(report), report, "fallback", None, [],
                         "No findings to explain.")
    if not use_llm:
        return _finalize(fallback_explanation(report), report, "fallback", None, [],
                         "LLM disabled.")

    messages = build_messages(report)
    issues: list[str] = []
    for _ in range(max_attempts):
        try:
            content = call_ollama(messages, model=model, base_url=base_url)
        except OllamaUnavailable as exc:
            return _finalize(fallback_explanation(report), report, "fallback", model, issues,
                             f"Ollama unavailable ({exc}).")
        try:
            candidate = _parse_json(content)
        except (json.JSONDecodeError, TypeError):
            issues = ["Response is not valid JSON."]
        else:
            issues = validate_explanation(candidate, report)
            if not issues:
                return _finalize(candidate, report, "llm", model, [])
        # Give the model one chance to correct itself.
        messages = messages + [
            {"role": "assistant", "content": content},
            {"role": "user", "content": _retry_message(issues)},
        ]
    return _finalize(fallback_explanation(report), report, "fallback", model, issues,
                     "LLM response failed validation.")


# ---------------------------------------------------------------------------
# Questions about the report
# ---------------------------------------------------------------------------

MAX_QUESTION_CHARS = 300

QA_SYSTEM_PROMPT = """You are a business analyst answering a manager's question about a decision report.

""" + GROUNDING_RULES + """

Answer ONLY from the report. If the report does not contain the information needed,
say so plainly instead of guessing. The report only covers the two past periods: never
forecast or predict future values; if asked, say the report cannot forecast and give the
observed trend instead. If asked what to do, quote the report's recommended action and
say which findings it is based on. Only attach a number to the segment it belongs to
(e.g. a South / Laptop Pro figure is not a South region figure).
Keep the answer to 2-4 sentences of observed facts.
Put any hypotheses in possible_explanations, never in the answer, each linked to the
findings that support it (rule 9).

Respond with JSON only, in exactly this shape:
{
  "answer": "2-4 sentences of observed facts from the report",
  "possible_explanations": [
    {"cause": "hypothesis phrased with may/might/could",
     "evidence": "the observed finding that makes it plausible, quoted from the report",
     "linked_insight_ids": ["I1"]}
  ],
  "insight_ids": ["I1"]
}"""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "possible_explanations": {"type": "array", "items": CAUSE_SCHEMA},
        "insight_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "possible_explanations", "insight_ids"],
}

FORECAST = re.compile(r"\b(will|forecast|predict|projection|project(?:ed)?|expect(?:ed)?|next "
                      r"(?:week|month|quarter|year|period)|future)\b", re.I)

# Question words -> the insight metric they are about (used by the fallback).
QUESTION_TOPICS = {
    r"revenue|sales": "revenue",
    r"profit|margin|earn": "profit",
    r"unit|volume|sold|quantit": "units",
    r"return|refund": "return_rate",
}
# Generic decline words mean revenue only when no metric is named ("why did it drop?").
GENERIC_DECLINE = re.compile(r"declin|drop|fall|fell|down|worse", re.I)


def validate_answer(answer, report: dict) -> list[str]:
    """The same grounding checks as validate_explanation, for a question answer."""
    if not isinstance(answer, dict):
        return ["Response is not a JSON object."]
    issues = []
    if not isinstance(answer.get("answer"), str) or not answer["answer"].strip():
        issues.append("'answer' must be a non-empty string.")
    if not isinstance(answer.get("insight_ids"), list) or not all(
            isinstance(v, str) for v in answer["insight_ids"]):
        issues.append("'insight_ids' must be a list of strings.")
    issues += _cause_shape_issues(answer.get("possible_explanations"), "possible_explanations")
    if issues:
        return issues

    known = {i["id"] for i in report.get("key_insights", [])}
    issues += [f"Cites unknown insight '{x}'." for x in answer["insight_ids"] if x not in known]
    issues += check_causes(answer["possible_explanations"], report, "possible_explanations")
    causes = answer["possible_explanations"]
    allowed, allowed_pp = allowed_numbers(report), allowed_pp_numbers(report)
    for text in [answer["answer"]] + [c["cause"] for c in causes] + [c["evidence"] for c in causes]:
        issues += [f"Number not found in the report: '{n}'." for n in unsupported_numbers(text, allowed, allowed_pp)]
        if CERTAINTY.search(text):
            issues.append(f"Claims certainty: \"{text[:80]}\".")
    for sentence in _sentences(answer["answer"]):
        if states_cause(sentence):
            issues.append(f"States a cause as fact: \"{sentence[:80]}\".")
    if (m := UNAVAILABLE_METRICS.search(answer["answer"])) and not HEDGE.search(answer["answer"]):
        issues.append(f"Mentions a metric that is not in the report ('{m.group(0)}').")
    issues += misattributed_numbers(answer["answer"], report)
    return issues


def fallback_answer(report: dict, question: str) -> dict:
    """Answer by picking the detected insights that match the question's words."""
    insights = report.get("key_insights", [])
    if not insights:
        return {"answer": report.get("executive_summary")
                or "The report contains no findings, so there is nothing to answer from.",
                "possible_explanations": [], "insight_ids": []}

    q = question.lower()
    metrics = {metric for pattern, metric in QUESTION_TOPICS.items() if re.search(pattern, q)}
    if not metrics and GENERIC_DECLINE.search(q):
        metrics = {"revenue"}

    def score(insight):
        name = insight["dimension_value"].lower()
        if name in q:
            segment_hit = 2          # the exact segment the question names
        else:
            segment_hit = any(part.lower() in q for part in insight["dimension_value"].split(" / "))
        return (insight["metric"] in metrics) * 2 + segment_hit

    ranked = sorted(insights, key=lambda i: -score(i))  # stable: keeps severity order on ties
    on_topic = any(score(i) > 0 for i in insights)
    chosen, seen = [], set()
    for insight in (ranked if on_topic else insights):
        key = (insight["dimension_value"], insight["metric"])
        if (on_topic and score(insight) == 0) or key in seen:
            continue
        seen.add(key)
        chosen.append(insight)
        if len(chosen) == 3:
            break
    chosen_ids = {i["id"] for i in chosen}

    possible, actions = [], []
    by_id = {i["id"]: i for i in insights}
    for rec in report.get("recommendations", []):
        if chosen_ids & set(rec["based_on_insights"]):
            support = [by_id[i] for i in rec["based_on_insights"] if i in by_id]
            possible += linked_causes(rec["rules_triggered"], support)
            actions.append(rec["actions"][0])
    text = " ".join(i["summary"] for i in chosen)
    if FORECAST.search(q):
        meta = report.get("metadata", {})
        periods = " and ".join(meta.get(k, {}).get("label", "") for k in ("period_previous", "period_current"))
        text = (f"The report cannot forecast future results; it only compares {periods}. "
                f"The observed trend: " + text)
    elif not on_topic:
        text = "The report does not directly address this question. Its main findings: " + text
    if actions:
        text += " Recommended action: " + actions[0]
    return {"answer": text, "possible_explanations": possible,
            "insight_ids": [i["id"] for i in chosen]}


def answer_question(
    report: dict,
    question: str,
    model: str | None = None,
    base_url: str | None = None,
    use_llm: bool = True,
    max_attempts: int = MAX_ATTEMPTS,
) -> dict:
    """Answer one question using only the decision report (never the CSV).

    Returns answer, possible_explanations, insight_ids, evidence_references and
    metadata (source "llm" or "fallback"). Validation and fallback work the
    same way as in explain_decision_report.
    """
    if not isinstance(report, dict):
        report = {}
    question = (question or "").strip()[:MAX_QUESTION_CHARS]
    model = model or os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)

    def finish(result, source, issues, reason=None):
        result = {k: result[k] for k in ("answer", "possible_explanations", "insight_ids")}
        result["evidence_references"] = _resolve_evidence(
            _evidence_for(result["insight_ids"], report), report)
        result["metadata"] = {"source": source, "model": model if source == "llm" else None,
                              "fallback_reason": reason, "validation_issues": issues}
        return result

    if not question:
        return finish({"answer": "Please enter a question about the business data.",
                       "possible_explanations": [], "insight_ids": []},
                      "fallback", [], "No question asked.")
    if not report.get("key_insights"):
        return finish(fallback_answer(report, question), "fallback", [], "No findings to answer from.")
    if not use_llm:
        return finish(fallback_answer(report, question), "fallback", [], "LLM disabled.")

    context = json.dumps(build_llm_context(report), indent=1)
    messages = [
        {"role": "system", "content": QA_SYSTEM_PROMPT},
        {"role": "user", "content": f"REPORT:\n{context}\n\nQUESTION: {question}"},
    ]
    issues: list[str] = []
    for _ in range(max_attempts):
        try:
            content = call_ollama(messages, model=model, base_url=base_url, schema=ANSWER_SCHEMA)
        except OllamaUnavailable as exc:
            return finish(fallback_answer(report, question), "fallback", issues,
                          f"Ollama unavailable ({exc}).")
        try:
            candidate = _parse_json(content)
        except (json.JSONDecodeError, TypeError):
            issues = ["Response is not valid JSON."]
        else:
            issues = validate_answer(candidate, report)
            if not issues:
                return finish(candidate, "llm", [])
        messages = messages + [
            {"role": "assistant", "content": content},
            {"role": "user", "content": _retry_message(issues)},
        ]
    return finish(fallback_answer(report, question), "fallback", issues,
                  "LLM response failed validation.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain a decision report with a local LLM.")
    parser.add_argument("report", nargs="?", default=str(Path(__file__).parent / "data" / "decision_report.json"))
    parser.add_argument("--model", help=f"Ollama model (default: $OLLAMA_MODEL or {DEFAULT_MODEL})")
    parser.add_argument("--fallback-only", action="store_true", help="skip the LLM")
    parser.add_argument("--out", help="write the JSON explanation here instead of printing it")
    parser.add_argument("--question", action="append",
                        help="answer this question instead of explaining (repeatable)")
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    if args.question:
        result = [{"question": q, **answer_question(report, q, model=args.model,
                                                    use_llm=not args.fallback_only)}
                  for q in args.question]
    else:
        result = explain_decision_report(report, model=args.model, use_llm=not args.fallback_only)
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        sources = [r["metadata"]["source"] for r in (result if isinstance(result, list) else [result])]
        print(f"Written to {args.out} (source: {', '.join(sources)})")
    else:
        print(text)


if __name__ == "__main__":
    main()
