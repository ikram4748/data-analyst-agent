"""Streamlit UI: ask a question, watch the agent work step by step.

    streamlit run app.py
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import pandas as pd
import streamlit as st

from analyst_agent import charts
from analyst_agent.config import ROOT, get_settings
from analyst_agent.db import DuckDBDatabase, bird_databases, open_bird
from analyst_agent.graph import VARIANTS, build_agent
from analyst_agent.llm import Gemini, LLMUnavailable
from analyst_agent.schema_linker import SchemaLinker

st.set_page_config(page_title="Data Analyst Agent", page_icon="📊", layout="wide")

# A public deployment shares one free-tier key, so each visitor gets a small budget.
SESSION_LIMIT = int(os.getenv("SESSION_QUESTION_LIMIT", "0")) or None

NODE_LABEL = {
    "link_schema": "Linking the question to the schema",
    "write_sql": "Writing SQL",
    "execute": "Checking and running the query",
    "fix_sql": "Fixing the query from the database's feedback",
    "analyze": "Explaining the result",
}


@st.cache_resource
def get_llm():
    return Gemini()


@st.cache_resource
def get_bird(name: str):
    db = open_bird(name)
    return db, SchemaLinker(db)


@st.cache_data
def bird_examples() -> dict[str, list[dict]]:
    path = get_settings().bird_dir / "mini_dev_sqlite.json"
    if not path.exists():
        return {}
    out: dict[str, list[dict]] = {}
    for q in json.loads(path.read_text(encoding="utf-8")):
        out.setdefault(q["db_id"], []).append(q)
    return out


def sidebar():
    st.sidebar.header("Data")
    sources = list(bird_databases())
    choice = st.sidebar.radio("Source", ["BIRD benchmark database", "Upload CSV files"],
                              index=0 if sources else 1)
    db = linker = None
    if choice == "BIRD benchmark database":
        if not sources:
            st.sidebar.warning("No BIRD databases found. Run `python scripts/setup_data.py` first.")
        else:
            name = st.sidebar.selectbox("Database", sources, index=sources.index("financial") if "financial" in sources else 0)
            db, linker = get_bird(name)
    else:
        files = st.sidebar.file_uploader("CSV files (one table each)", type="csv", accept_multiple_files=True)
        if files:
            blobs = {f.name: f.getvalue() for f in files}
            key = hashlib.sha256(b"".join(sorted(blobs.values()))).hexdigest()
            if st.session_state.get("upload_key") != key:
                with st.spinner("Loading CSV files into DuckDB..."):
                    up = DuckDBDatabase.from_csv_files("uploaded", blobs)
                    st.session_state.update(upload_key=key, upload_db=up, upload_linker=SchemaLinker(up))
            db, linker = st.session_state["upload_db"], st.session_state["upload_linker"]

    st.sidebar.header("Agent")
    variant = st.sidebar.selectbox(
        "Variant", list(VARIANTS), index=list(VARIANTS).index("agent"),
        help="baseline = bare schema, one shot · linked = + schema linking · linked+fix = + self-correction · "
             "agent = + literal-value check. Compare them on the same question.")
    if db is not None:
        with st.sidebar.expander(f"Schema: {len(db.tables)} tables"):
            for t in db.tables.values():
                rows = f"{t.row_count:,} rows" if t.row_count is not None else ""
                st.markdown(f"**{t.name}** · {rows}  \n" + ", ".join(f"`{c.name}`" for c in t.columns))
    return db, linker, variant


def example_picker(db) -> None:
    examples = bird_examples().get(db.name, []) if db is not None else []
    if not examples:
        return
    with st.expander("Example questions from the benchmark"):
        for q in examples[:8]:
            if st.button(q["question"], key=f"ex{q['question_id']}", width="stretch"):
                st.session_state["question"] = q["question"]
                st.session_state["evidence"] = q.get("evidence", "")
                st.rerun()


def show_attempts(state: dict) -> None:
    attempts = state.get("attempts", [])
    for i, a in enumerate(attempts, 1):
        status = "error" if a["error"] else ("needs a fix" if a["note"] else "ok")
        icon = {"error": "❌", "needs a fix": "⚠️", "ok": "✅"}[status]
        st.markdown(f"**Attempt {i}** {icon} · {a['rows']} rows · {a['seconds'] * 1000:.0f} ms")
        st.code(a["sql"] or "(empty)", language="sql")
        if a["error"] or a["note"]:
            st.caption(a["error"] or a["note"])


def main() -> None:
    st.title("📊 Data Analyst Agent")
    st.caption("Ask a question in plain English. The agent finds the right tables, writes SQL, runs it read-only, "
               "fixes its own mistakes, and explains the answer.")
    db, linker, variant = sidebar()

    tab_ask, tab_results = st.tabs(["Ask", "Benchmark results"])
    with tab_results:
        found = [p for p in (ROOT / "results" / "RESULTS_test.md", ROOT / "results" / "RESULTS_dev.md") if p.exists()]
        st.markdown("\n\n---\n\n".join(p.read_text() for p in found) if found
                    else "Run `python -m evaluation.run_eval` to produce results.")

    with tab_ask:
        if db is None:
            st.info("Pick a BIRD database or upload CSV files in the sidebar.")
            return
        example_picker(db)
        question = st.text_input("Question", key="question", placeholder="e.g. How many clients opened accounts in 1995?")
        evidence = st.text_input("Definitions / hints (optional)", key="evidence",
                                 help="Business definitions the agent should follow, e.g. 'weekly issuance means frequency = POPLATEK TYDNE'.")
        if not st.button("Ask", type="primary", disabled=not question.strip()):
            return
        used = st.session_state.get("asked", 0)
        if SESSION_LIMIT and used >= SESSION_LIMIT:
            st.warning(f"This public demo allows {SESSION_LIMIT} questions per visitor (shared free-tier key). "
                       "Clone the repo to run it with your own key.")
            return
        st.session_state["asked"] = used + 1

        try:
            agent = build_agent(db, get_llm(), variant, linker)
        except LLMUnavailable as e:
            st.error(str(e))
            return

        state: dict = {}
        t0 = time.monotonic()
        with st.status("Working...", expanded=True) as status:
            try:
                for mode, chunk in agent.stream({"question": question, "evidence": evidence},
                                                {"recursion_limit": 30}, stream_mode=["updates", "values"]):
                    if mode == "values":
                        state = chunk
                        continue
                    for node, update in chunk.items():
                        line = NODE_LABEL.get(node, node)
                        if node == "link_schema" and update.get("linked_columns"):
                            top = ", ".join(f"{t}.{c}" for t, c, _ in update["linked_columns"][:5])
                            line += f": most relevant columns {top}"
                        if node == "execute":
                            a = update["attempts"][-1]
                            line += f": {a['rows']} rows" if not a["error"] else f": {a['error'][:120]}"
                            if a["note"] and not a["error"]:
                                line += f" ⚠️ {a['note'][:160]}"
                        st.write("• " + line)
                status.update(label=f"Done in {time.monotonic() - t0:.1f}s", state="complete", expanded=False)
            except LLMUnavailable as e:
                status.update(label="Gemini quota exhausted", state="error")
                st.error(str(e))
                return

        result = state.get("result")
        st.subheader("Answer")
        st.write(state.get("answer") or "(no answer)")
        if result is not None and result.ok:
            df = pd.DataFrame(result.rows, columns=result.columns)
            fig = charts.figure(state.get("chart"), df)
            if fig is not None:
                st.plotly_chart(fig, width="stretch")
            st.dataframe(df.head(get_settings().max_rows_shown), width="stretch")
            if result.truncated:
                st.caption("Result truncated for display.")
        c1, c2, c3 = st.columns(3)
        c1.metric("LLM calls", state.get("llm_calls", 0))
        c2.metric("Query attempts", len(state.get("attempts", [])))
        c3.metric("Time", f"{time.monotonic() - t0:.1f}s")
        with st.expander("How the agent got there", expanded=len(state.get("attempts", [])) > 1):
            show_attempts(state)
            if state.get("linked_columns"):
                st.markdown("**Schema linking**: top columns by similarity to the question")
                st.dataframe(pd.DataFrame(state["linked_columns"][:10], columns=["table", "column", "score"]),
                             hide_index=True)
            st.caption("Models: " + ", ".join(state.get("models", [])))


main()
