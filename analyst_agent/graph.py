"""The agent as a LangGraph state machine.

    link_schema -> write_sql -> execute --ok / refused (a write)-------------> analyze -> END
                                  ^   \\--error / empty / zero / bad literal--> fix_sql
                                  \\_______________________________________/   (at most max_fixes times)

Why a graph and not one long prompt: the fix loop needs real feedback from the database (the error
text, the empty result), and that only exists after the query has run. Each node does one job, so
each can be measured on its own (see evaluation/), and the state records every attempt for the UI.
"""
from __future__ import annotations

import json
import operator
import time
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph

from . import checks, guard, prompts
from .db import Database, QueryResult
from .llm import parse_json
from .schema_linker import SchemaLinker


@dataclass(frozen=True)
class Variant:
    """Which parts of the agent are switched on. The evaluation compares these."""
    name: str
    rich_schema: bool      # schema linking + column descriptions / example values
    max_fixes: int         # self-correction attempts after a failed or suspicious execution
    literal_check: bool    # verify every text literal in WHERE exists in its column (no LLM call)
    analyze: bool = True   # write the plain-English answer and chart (off in the SQL benchmark)


VARIANTS = {
    "baseline": Variant("baseline", rich_schema=False, max_fixes=0, literal_check=False),
    "linked": Variant("linked", rich_schema=True, max_fixes=0, literal_check=False),
    "linked+fix": Variant("linked+fix", rich_schema=True, max_fixes=3, literal_check=False),
    "agent": Variant("agent", rich_schema=True, max_fixes=3, literal_check=True),
}


class AgentState(TypedDict, total=False):
    question: str
    evidence: str
    schema: str
    linked_columns: list
    sql: str
    attempts: Annotated[list, operator.add]   # every query tried: sql, error, rows, note
    result: Any                               # QueryResult of the latest attempt
    problem: str                              # why the latest attempt needs fixing ("" = fine)
    refused: bool                             # the request would change data: answer with a refusal, do not "fix"
    fixes: int
    answer: str
    chart: dict
    llm_calls: Annotated[int, operator.add]
    llm_seconds: Annotated[float, operator.add]   # time inside API calls (excludes rate-limit waiting)
    models: Annotated[list, operator.add]
    seconds: float


def _problem(r: QueryResult) -> str:
    """What looks wrong with a result. Empty or all-NULL results are usually a wrong literal or join."""
    if not r.ok:
        return f"the database returned an error: {r.error}"
    if not r.rows:
        return "the query ran but returned 0 rows."
    if all(v is None for row in r.rows for v in row):
        return "the query ran but every returned value is NULL."
    if len(r.rows) == 1 and len(r.rows[0]) == 1 and r.rows[0][0] == 0:
        return "the query returned a single 0. If a filter is wrong this looks like a valid answer; check the filters."
    return ""


def _preview(r: QueryResult, n: int = 30) -> str:
    head = " | ".join(r.columns)
    body = "\n".join(" | ".join("NULL" if v is None else str(v)[:60] for v in row) for row in r.rows[:n])
    return f"{head}\n{body}"


def build_agent(db: Database, llm, variant: Variant | str = "agent", linker: SchemaLinker | None = None,
                max_rows: int = 10_000):
    v = VARIANTS[variant] if isinstance(variant, str) else variant
    if v.rich_schema and linker is None:
        linker = SchemaLinker(db)

    def ask(system: str, prompt: str) -> tuple[dict, str, float]:
        r = llm.generate(prompt, system, json_mode=True)
        try:
            return parse_json(r.text), r.model, r.seconds
        except (json.JSONDecodeError, ValueError):
            return {"sql": r.text}, r.model, r.seconds

    def link_schema(s: AgentState) -> AgentState:
        if not v.rich_schema:
            return {"schema": db.ddl(), "linked_columns": [], "fixes": 0, "llm_calls": 0, "llm_seconds": 0.0}
        link = linker.link(s["question"], s.get("evidence", ""))
        return {"schema": db.ddl(rich=link.column_set), "linked_columns": link.columns, "fixes": 0, "llm_calls": 0,
                "llm_seconds": 0.0}

    def write_sql(s: AgentState) -> AgentState:
        out, model, secs = ask(prompts.WRITER_SYSTEM.format(dialect=db.dialect),
                               prompts.WRITER_PROMPT.format(schema=s["schema"], question=s["question"],
                                                            evidence_block=prompts.evidence_block(s.get("evidence", ""))))
        return {"sql": str(out.get("sql", "")), "llm_calls": 1, "models": [model], "llm_seconds": secs}

    def execute(s: AgentState) -> AgentState:
        g = guard.check(s["sql"], db.dialect)
        if not g.ok and g.policy:
            # A write is a request to refuse, not a bug to repair into some unrelated SELECT.
            r = QueryResult([], [], 0.0, error=f"refused by the SQL guard: {g.reason}")
            attempt = {"sql": g.sql, "error": r.error, "rows": 0, "truncated": False, "seconds": 0.0, "note": ""}
            return {"sql": g.sql, "result": r, "problem": "", "refused": True, "attempts": [attempt]}
        if not g.ok:
            r = QueryResult([], [], 0.0, error=f"rejected by the SQL guard: {g.reason}")
        else:
            r = db.execute(g.sql, max_rows=max_rows)
        problem = _problem(r)
        if r.ok and v.literal_check:
            problem = "; ".join(filter(None, [problem, *checks.literal_problems(db, g.sql)]))
        attempt = {"sql": g.sql, "error": r.error, "rows": len(r.rows), "truncated": r.truncated,
                   "seconds": round(r.seconds, 3), "note": problem}
        return {"sql": g.sql, "result": r, "problem": problem, "attempts": [attempt]}

    def fix_sql(s: AgentState) -> AgentState:
        probe = checks.filter_probe(db, s["sql"])
        out, model, secs = ask(prompts.FIXER_SYSTEM.format(dialect=db.dialect),
                               prompts.FIXER_PROMPT.format(schema=s["schema"], question=s["question"], sql=s["sql"],
                                                           problem=s["problem"],
                                                           probe=f"\nStored values of the filtered columns:\n{probe}" if probe else "",
                                                           evidence_block=prompts.evidence_block(s.get("evidence", ""))))
        new = guard.strip_fences(str(out.get("sql", "")))
        unchanged = not new or new.strip() == s["sql"].strip()
        # The fixer may say "the original is right" (an empty answer can be true): then stop looping.
        return {"sql": s["sql"] if unchanged else new, "fixes": s["fixes"] + 1, "llm_calls": 1,
                "models": [model], "problem": "" if unchanged else s["problem"], "llm_seconds": secs}

    def analyze(s: AgentState) -> AgentState:
        r: QueryResult = s["result"]
        if s.get("refused"):
            return {"answer": "This agent only reads data. That request would change the database "
                              f"({s['attempts'][-1]['error'].split(': ', 1)[-1]}), so it was not run.",
                    "chart": {"type": "none"}}
        if not r.ok:
            return {"answer": f"I could not produce a working query. Last error: {r.error}", "chart": {"type": "none"}}
        out, model, secs = ask(prompts.ANALYST_SYSTEM,
                               prompts.ANALYST_PROMPT.format(question=s["question"], sql=s["sql"], n_rows=len(r.rows),
                                                             truncated=", truncated" if r.truncated else "",
                                                             preview=_preview(r)))
        return {"answer": str(out.get("answer", "")), "chart": out.get("chart") or {"type": "none"},
                "llm_calls": 1, "models": [model], "llm_seconds": secs}

    def after_execute(s: AgentState) -> str:
        if s["problem"] and s["fixes"] < v.max_fixes:
            return "fix_sql"
        return "analyze" if v.analyze else END

    def after_fix(s: AgentState) -> str:
        if s["problem"]:                 # a new query to try
            return "execute"
        return "analyze" if v.analyze else END     # fixer kept the original query

    g = StateGraph(AgentState)
    g.add_node("link_schema", link_schema)
    g.add_node("write_sql", write_sql)
    g.add_node("execute", execute)
    g.add_node("fix_sql", fix_sql)
    g.add_node("analyze", analyze)
    g.set_entry_point("link_schema")
    g.add_edge("link_schema", "write_sql")
    g.add_edge("write_sql", "execute")
    g.add_conditional_edges("execute", after_execute, ["fix_sql", "analyze", END])
    g.add_conditional_edges("fix_sql", after_fix, ["execute", "analyze", END])
    g.add_edge("analyze", END)
    return g.compile()


def run(agent, question: str, evidence: str = "") -> AgentState:
    t0 = time.monotonic()
    state = agent.invoke({"question": question, "evidence": evidence}, {"recursion_limit": 30})
    state["seconds"] = time.monotonic() - t0
    return state
