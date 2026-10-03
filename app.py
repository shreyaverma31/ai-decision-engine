"""Streamlit UI for the AI Decision Engine.

    streamlit run app.py

The page tells one story about the single most important problem:
overview -> main problem -> why it was flagged (3 steps) -> traceability
-> AI explanation -> recommended action -> ask a question.

The engine still detects every finding, and the full report keeps all of them.
story.py picks the top issue. All numbers come from analyzer.py / engine.py.
The LLM (llm.py) only sees the structured report, never the CSV.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analyzer import REQUIRED_COLUMNS, DataValidationError, validate_data
from engine import generate_decision_report
from llm import DEFAULT_MODEL, answer_question, explain_decision_report
from story import build_story, fmt_money, fmt_pct, fmt_rate, fmt_value, json_key

DEMO_PATH = Path(__file__).parent / "data" / "sales.csv"
FOOTER = ("All numerical insights are calculated from the uploaded business data. "
          "Possible causes are hypotheses and require investigation.")
log = logging.getLogger("ai_decision_engine")


def _hash_sources(paths) -> str:
    """Short hash of the given source files' contents."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()[:12]


# Passed to every cached function below. st.cache_data keys on a function's own source and its
# arguments, not on the code it calls, so without this a running server could keep serving
# results built by an older story.py/engine.py/... after a deploy.
CODE_VERSION = _hash_sources(Path(__file__).parent / name
                             for name in ("story.py", "engine.py", "analyzer.py", "llm.py"))


class DataError(Exception):
    """A problem with the user's data, with a message that is safe to show."""


# ---------------------------------------------------------------------------
# Data & analysis (thin wrappers around analyzer / engine / story / llm)
# ---------------------------------------------------------------------------

def load_csv(data: bytes) -> pd.DataFrame:
    """Parse and validate CSV bytes, raising DataError with a friendly message."""
    if not data or not data.strip():
        raise DataError("The file is empty.")
    try:
        raw = pd.read_csv(io.BytesIO(data))
    except pd.errors.EmptyDataError:
        raise DataError("The file is empty.")
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError):
        raise DataError("This file could not be read as a CSV. Please upload a comma-separated text file.")
    try:
        return validate_data(raw)
    except DataValidationError as exc:
        raise DataError("The data has problems:\n\n" + "\n".join(f"- {i}" for i in exc.issues))


@st.cache_data(show_spinner=False)
def analyze(data: bytes, code_version: str = CODE_VERSION) -> tuple[pd.DataFrame, dict, dict]:
    df = load_csv(data)
    try:
        report = generate_decision_report(df)
    except ValueError as exc:  # e.g. fewer than two months of data
        raise DataError(f"The data can't be analyzed: {exc}")
    return df, report, build_story(df, report)


@st.cache_data(show_spinner=False)
def explain(report_json: str, use_llm: bool, code_version: str = CODE_VERSION) -> dict:
    return explain_decision_report(json.loads(report_json), use_llm=use_llm)


@st.cache_data(show_spinner=False)
def ask(report_json: str, question: str, use_llm: bool, code_version: str = CODE_VERSION) -> dict:
    return answer_question(json.loads(report_json), question, use_llm=use_llm)


def md(text: str) -> str:
    """Escape '$' so Streamlit doesn't render "$3.19M to $2.39M" as LaTeX math."""
    return text.replace("$", r"\$")


def arrow_pct(value) -> str:
    if value is None:
        return "n/a"
    arrow = "↑" if value > 0 else "↓" if value < 0 else "→"
    return f"{arrow} {abs(value):.1f}%"


def unique_causes(causes) -> list[dict]:
    seen, out = set(), []
    for c in causes:
        if c["cause"] not in seen:
            seen.add(c["cause"])
            out.append(c)
    return out


def causes_markdown(causes: list[dict]) -> str:
    """Each possible cause with the evidence (and finding IDs) it is linked to."""
    return "\n".join(
        f"- {md(c['cause'])}\n  - *Linked evidence ({', '.join(c['linked_insight_ids'])}):* {md(c['evidence'])}"
        for c in causes)


def fmt_filters(filters: dict) -> str:
    return ", ".join(f"{k} = {' / '.join(v)}" for k, v in filters.items()) or "all data"


# ---------------------------------------------------------------------------
# Chart: the segment vs. its peers (the one visual that tells the story)
# ---------------------------------------------------------------------------

def peer_chart(problem: dict) -> go.Figure:
    try:
        dark = st.context.theme.type == "dark"
    except Exception:
        dark = False
    highlight = "#d95926" if dark else "#eb6834"   # reference palette slot 2 (orange)
    neutral = "#6b6a65" if dark else "#a3a29c"     # de-emphasized peers
    rows = [(p["label"], p["change_percent"] or 0.0, neutral) for p in problem["peers"]]
    rows.append((problem["segment"], problem["segment_change_percent"] or 0.0, highlight))
    labels, values, colors = zip(*rows)
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color=colors,
        text=[fmt_pct(v) for v in values], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:+.1f}% revenue<extra></extra>",
    ))
    grid = "rgba(128,128,128,0.18)"
    fig.update_layout(
        title=dict(text="Revenue change, same periods", x=0, font=dict(size=14)),
        height=230, margin=dict(l=8, r=48, t=40, b=8), barcornerradius=4, bargap=0.35,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
    )
    fig.update_xaxes(ticksuffix="%", gridcolor=grid, zeroline=True, zerolinecolor=grid)
    fig.update_yaxes(showgrid=False)
    return fig


# ---------------------------------------------------------------------------
# Page sections
# ---------------------------------------------------------------------------

def status_message(result: dict) -> None:
    meta = result["metadata"]
    if meta["source"] == "llm":
        st.success(f"AI explanation generated from verified evidence (model: {meta['model']})", icon="✅")
    else:
        st.info("Using evidence-based fallback", icon="ℹ️")
        reason = meta.get("fallback_reason") or ""
        if reason.startswith("Ollama unavailable"):
            st.caption("The local AI model (Ollama) is not reachable, so the explanation is assembled "
                       "directly from the evidence.")
        elif reason == "LLM response failed validation.":
            st.caption("The AI response did not pass the evidence checks (for example, it used a number "
                       "not in the data), so it was replaced with the evidence-based explanation.")
            issues = list(dict.fromkeys(meta.get("validation_issues") or []))
            if issues:
                with st.expander("🛡️ What the evidence checks caught"):
                    st.markdown("\n".join(f"- {md(i)}" for i in issues[:8]))
        elif reason == "LLM disabled.":
            st.caption("Local AI is turned off in the sidebar.")


def section_overview(story: dict) -> None:
    st.header("Business Overview")
    o = story["overview"]
    prev, cur = story["period_previous"]["label"], story["period_current"]["label"]
    cols = st.columns(5)
    cols[0].metric("Revenue", fmt_money(o["revenue"]), border=True)
    cols[1].metric("Profit", fmt_money(o["profit"]), border=True)
    cols[2].metric("Units", f"{o['units']:,}", border=True)
    cols[3].metric("Return Rate", fmt_rate(o["return_rate"]), border=True)
    cols[4].metric("Overall Revenue Growth", fmt_pct(o["revenue_growth_percent"]),
                   help=f"{prev} → {cur}", border=True)
    st.markdown(f"**{md(o['sentence'])}**")


# The three kinds of statement on the page, always labelled so certainty is never implied.
FINDING_BADGE = ":blue-badge[📊 Data-backed finding]"
POSSIBLE_BADGE = ":orange-badge[💭 Possible explanation — not proven]"
INVESTIGATE_BADGE = ":green-badge[🔍 Recommended investigation]"


def section_reading_guide() -> None:
    """The principle the whole page follows: what changed ≠ why it changed."""
    cols = st.columns(3)
    guide = [
        (FINDING_BADGE, "**What the data shows**", "Measured changes: what changed, where, and by how much."),
        (POSSIBLE_BADGE, "**What the system suspects**", "Hypotheses consistent with the evidence. Not proven."),
        (INVESTIGATE_BADGE, "**What should be investigated**", "The next step to confirm or rule out each hypothesis."),
    ]
    for col, (badge, title, text) in zip(cols, guide):
        with col.container(border=True):
            st.markdown(f"{badge}\n\n{title}\n\n{text}")
    st.caption("The data shows what changed and where. It does not, by itself, prove why it happened.")


def section_problem(p: dict, story: dict) -> None:
    st.header("🚨 Main Business Problem")
    badge = {"high": ":red-badge[High severity]", "medium": ":orange-badge[Medium severity]",
             "low": ":gray-badge[Low severity]"}[p["severity"]]
    st.subheader(p["segment"])
    st.markdown(f"{badge} {FINDING_BADGE}")
    prev, cur = story["period_previous"]["label"], story["period_current"]["label"]
    st.caption(f"Detected from measurable changes in the uploaded data for {p['segment']}, "
               f"{prev} compared with {cur}.")
    m = p["metrics"]
    rr = m["return_rate"]
    rr_arrow = "↑" if rr["change"] > 0 else "↓" if rr["change"] < 0 else "→"
    cols = st.columns(4)
    cols[0].metric("Revenue", arrow_pct(m["revenue"]["change_percent"]), border=True)
    cols[1].metric("Profit", arrow_pct(m["profit"]["change_percent"]), border=True)
    cols[2].metric("Units", arrow_pct(m["units"]["change_percent"]), border=True)
    cols[3].metric("Return Rate", f"{rr_arrow} {fmt_rate(rr['previous'])} → {fmt_rate(rr['current'])}",
                   border=True)
    st.markdown(md(p["explanation"]))


def section_why(p: dict, story: dict) -> None:
    st.header("Why did the system flag it?")
    st.markdown(f"{FINDING_BADGE} Every step below is measured from the data; none of it is a guess.")
    prev, cur = story["period_previous"], story["period_current"]

    st.subheader("Step 1 — What changed?")
    st.caption(f"{prev['label']} ({prev['start']} to {prev['end']}) → "
               f"{cur['label']} ({cur['start']} to {cur['end']})")
    names = {"revenue": "Revenue", "profit": "Profit", "units": "Units", "return_rate": "Return rate"}
    st.markdown("\n".join(
        f"- **{names[k]}:** {md(fmt_value(k, v['previous']))} → {md(fmt_value(k, v['current']))}"
        for k, v in p["metrics"].items()))

    st.subheader("Step 2 — How significant is the change?")
    st.markdown("\n".join(
        f"- {'✅' if c['significant'] else '➖'} **{c['label']}: {md(c['value'])}** — "
        f"{'significant' if c['significant'] else 'below the significance threshold'} "
        f"({c['criterion']})" + (f" · {c['severity']} severity" if c["severity"] else "")
        for c in p["significance"]))
    st.caption("These are the engine's fixed detection thresholds. For return rate, z measures how unlikely "
               "the change is to be random noise; 3 or more means very unlikely.")

    st.subheader("Step 3 — Is the problem concentrated in this segment?")
    if p["peers"]:
        left, right = st.columns([2, 3])
        with left:
            lines = [f"- {x['label']}: **{fmt_pct(x['change_percent'])}**" for x in p["peers"]]
            lines.append(f"- {p['segment']}: **{fmt_pct(p['segment_change_percent'])}**")
            st.markdown("Revenue change over the same periods:\n" + "\n".join(lines))
            if p["worse_than_peers"]:
                st.error(f"**{p['emphasis']}**", icon="📉")
            else:
                st.info(p["emphasis"])
        right.plotly_chart(peer_chart(p), config={"displayModeBar": False})
    else:
        st.markdown("There is no peer group to compare against in this data.")

    st.subheader("Step 4 — What evidence supports the finding?")
    st.markdown("The engine detected these findings for this problem. Each one points to evidence you "
                "can check in **🔎 Show how every number was calculated** below.")
    st.markdown("\n".join(
        f"- **{f['label']}** · {f['segment']} · {f['severity']} severity · evidence "
        f"{', '.join(f['evidence_ids'])}" for f in p["findings"]))


def section_possible_causes(p: dict) -> None:
    st.header("Possible causes — NOT proven by the data")
    st.markdown(f"{POSSIBLE_BADGE} The data shows **what** changed and **where**. It cannot prove **why**. "
                f"Each possibility below is linked to the evidence that suggested it, and needs investigation.")
    if not p["possible_causes"]:
        st.info("No possible causes are suggested: none of the supporting signals moved in a way that "
                "would point to one.")
        return
    for c in p["possible_causes"]:
        with st.container(border=True):
            st.markdown(f"💭 **{md(c['statement'])}**\n\n"
                        f"*Evidence that led to this hypothesis:* {md(c['evidence_text'])}")


METRIC_NAMES = {"revenue": "Revenue", "profit": "Profit", "units": "Units", "returns": "Returns",
                "return_rate": "Return rate", "profit_margin": "Profit margin", "cost": "Cost"}


def fmt_segment(filters: dict) -> str:
    """Readable segment name from evidence filters (a multi-value filter means 'the others')."""
    single = " / ".join(v[0] for v in filters.values() if len(v) == 1)
    multi = [d for d, v in filters.items() if len(v) > 1]
    if not multi:
        return single or "All business"
    dim = multi[0]
    if not single:
        return f"All other {dim}s"
    return f"Other products in {single}" if dim == "product" else f"{single} in other {dim}s"


def fmt_change(metric: str, change: float) -> str:
    """Absolute change: money with sign, units as a count, rates in percentage points."""
    if metric in ("return_rate", "profit_margin"):
        return f"{change * 100:+.1f} pp"
    if metric in ("revenue", "profit", "cost"):
        return ("+" if change >= 0 else "-") + fmt_money(abs(change))
    return f"{int(change):+,}"


def section_traceability(p: dict, story: dict, full_report: dict) -> None:
    with st.expander("🔎 Show how every number was calculated", expanded=False):
        prev, cur = story["period_previous"], story["period_current"]
        st.markdown(
            "Every number on this page is recomputed from rows of the uploaded data. Each row below is one "
            f"piece of evidence. **Previous** is {prev['label']} ({prev['start']} to {prev['end']}); "
            f"**Current** is {cur['label']} ({cur['start']} to {cur['end']}). *Source filter* shows exactly "
            "which rows were used.")
        # Numbers shown on the page first; the same fact recorded under several IDs is one row.
        rows: dict[tuple, dict] = {}
        for e in p["evidence"] + story["focused_report"]["evidence"]:
            key = json_key(e)
            if key in rows:
                if e["id"] not in rows[key]["Evidence"].split(", "):
                    rows[key]["Evidence"] += f", {e['id']}"
                continue
            metric = METRIC_NAMES.get(e["metric"], e["metric"])
            if e.get("role") == "driver":
                metric += " (largest contributor to the wider change)"
            rows[key] = {
                "Evidence": e["id"],
                "Metric": metric,
                "Segment": fmt_segment(e["filters"]),
                f"Previous ({prev['label']})": fmt_value(e["metric"], e["previous_value"]),
                f"Current ({cur['label']})": fmt_value(e["metric"], e["current_value"]),
                "Absolute change": fmt_change(e["metric"], e["change"]),
                "% change": fmt_pct(e["change_percent"]),
                "Source filter": fmt_filters(e["filters"]),
                "Rows used": f"{e['rows_previous']:,} / {e['rows_current']:,}",
            }
        def id_order(evidence_id: str) -> tuple:
            return (evidence_id[0], int(evidence_id[1:]) if evidence_id[1:].isdigit() else 0)

        for row in rows.values():
            row["Evidence"] = ", ".join(sorted(row["Evidence"].split(", "), key=id_order))
        st.dataframe(pd.DataFrame(rows.values()), hide_index=True)
        n = full_report["metadata"]["insight_count"]
        st.caption(f"This page focuses on the top issue. The engine detected {n} findings in total; "
                   f"all of them, with their evidence, are in the full report.")
        st.download_button("Download full report (JSON)", json.dumps(full_report, indent=2),
                           file_name="decision_report.json", mime="application/json")


def section_explanation(story: dict, use_llm: bool) -> None:
    st.header("🤖 AI Explanation")
    spinner = ("Generating the AI explanation with the local model (this can take a minute)…"
               if use_llm else "Preparing the explanation…")
    with st.spinner(spinner):
        result = explain(json.dumps(story["focused_report"]), use_llm, CODE_VERSION)
    status_message(result)
    st.caption("The AI only explains the evidence above. It was not asked to find the cause itself, and its "
               "output is checked against the evidence before it is shown.")

    st.markdown(md(result["summary"]))
    findings = result["key_findings"][:3]
    if findings:
        st.markdown(f"**Key findings** {FINDING_BADGE}\n"
                    + "\n".join(f"- {md(f['statement'])}" for f in findings))
    possible = unique_causes(c for r in result["reasoning"] for c in r["possible_explanations"])[:3]
    if possible:
        st.warning("**Possible causes — not facts:** these may be contributing and require investigation.\n"
                   + causes_markdown(possible), icon="⚠️")


def section_action(p: dict | None) -> None:
    st.header("Recommended Action")
    action = p["action"] if p else None
    if not action or not action["primary"]:
        st.success("No action needed: no significant problems were detected in this data.")
        return
    st.markdown(INVESTIGATE_BADGE)
    st.success(f"**{md(action['primary'])}**", icon="🎯")
    st.markdown(f"**Why?** {md(action['why'])}")
    st.markdown(f"**Rule:** `{p['rule']}`" + (f" — {md(action['rule_description'])}"
                                              if action.get("rule_description") else ""))
    if action["secondary"]:
        st.markdown(f"**Secondary action:** {md(action['secondary'])}")
        if action["secondary_why"]:
            st.caption(md(action["secondary_why"]))
    st.caption("Recommended by a fixed rule from the data-backed findings above. It is an investigation to "
               "run, not a conclusion about the cause.")


def section_question(report: dict, placeholder: str, use_llm: bool) -> None:
    st.header("Ask the Business Data")
    with st.form("question"):
        question = st.text_input("Your question", placeholder=placeholder, max_chars=300)
        submitted = st.form_submit_button("Ask")
    if not submitted:
        return
    if not question.strip():
        st.warning("Please type a question first.")
        return
    with st.spinner("Answering from the verified evidence…"):
        result = ask(json.dumps(report), question.strip(), use_llm, CODE_VERSION)
    status_message(result)
    st.markdown(md(result["answer"]))
    if result["possible_explanations"]:
        st.warning("**Possibilities, not facts:**\n"
                   + causes_markdown(unique_causes(result["possible_explanations"])[:3]), icon="⚠️")
    if result["evidence_references"]:
        st.caption("Evidence used: " + "; ".join(
            md(f"{e['id']} {e['description']}") for e in result["evidence_references"]))


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def sidebar() -> tuple[bytes | None, bool]:
    """Returns the selected CSV bytes (or None) and whether to use the LLM."""
    with st.sidebar:
        st.header("Data")
        uploaded = st.file_uploader("Upload a business CSV", type=["csv"],
                                    on_change=lambda: st.session_state.update(source="upload"))
        if st.button("Use Demo Dataset", width="stretch"):
            st.session_state.source = "demo"
            st.session_state.pop("analyzed", None)

        if st.session_state.get("source") == "demo" and DEMO_PATH.exists():
            data, name = DEMO_PATH.read_bytes(), "Demo dataset (sales.csv)"
        elif uploaded is not None:
            data, name = uploaded.getvalue(), uploaded.name
        else:
            data, name = None, None
        st.caption(f"Selected: **{md(name)}**" if name else "No data selected yet.")

        if st.button("Analyze Business Data", type="primary", width="stretch", disabled=data is None):
            st.session_state.analyzed = data
        st.caption("Required columns: " + ", ".join(f"`{c}`" for c in REQUIRED_COLUMNS))

        st.divider()
        use_llm = st.toggle("Use local AI (Ollama)", value=True,
                            help="Off: explanations are assembled from the evidence without an LLM (instant).")
        st.caption(f"Model: `{os.environ.get('OLLAMA_MODEL', DEFAULT_MODEL)}`")
    return st.session_state.get("analyzed"), use_llm


def main() -> None:
    st.set_page_config(page_title="AI Decision Engine", page_icon="📊", layout="wide")
    st.title("AI Decision Engine")
    st.markdown("*Turn business data into traceable insights and actionable decisions.*")

    data, use_llm = sidebar()
    if data is None:
        st.info("Upload a CSV or click **Use Demo Dataset** in the sidebar, then click "
                "**Analyze Business Data**.")
        return

    try:
        df, report, story = analyze(data, CODE_VERSION)
    except DataError as exc:
        st.error(md(str(exc)), icon="🚫")
        st.caption("Expected columns: " + ", ".join(REQUIRED_COLUMNS))
        return

    section_overview(story)
    problem = story["problem"]
    if problem is None:
        st.header("🚨 Main Business Problem")
        st.success("No significant problems detected: no segment shows a significant decline or "
                   "return-rate increase between the two periods.")
    else:
        section_reading_guide()
        section_problem(problem, story)
        section_why(problem, story)
        section_traceability(problem, story, report)
        section_possible_causes(problem)
    section_explanation(story, use_llm)
    section_action(problem)
    # Answers use the same verified evidence as the page (the focused top-issue report).
    section_question(story["focused_report"], story["question_placeholder"], use_llm)
    st.divider()
    st.caption(FOOTER)


def run() -> None:
    """Never show a Python traceback to the user."""
    try:
        main()
    except Exception:  # noqa: BLE001 - last-resort guard for the demo UI
        log.exception("Unexpected error in app")
        st.error("Something went wrong while processing this data. Please check the file and try again.",
                 icon="🚫")


if __name__ == "__main__":
    run()
