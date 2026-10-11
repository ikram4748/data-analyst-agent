"""Offline tests: no API key, no BIRD download, no network. The LLM is a scripted fake."""
from __future__ import annotations

import json
import sqlite3

import pandas as pd
import pytest

from analyst_agent import charts, checks, guard
from analyst_agent.db import DuckDBDatabase, SQLiteDatabase
from analyst_agent.graph import VARIANTS, build_agent, run
from analyst_agent.llm import LLMResult, parse_json
from evaluation.run_eval import paired, stratified_subset


@pytest.fixture
def shop(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    from analyst_agent import config
    config.get_settings.cache_clear()
    path = tmp_path / "shop.sqlite"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE region (region_id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE customer (customer_id INTEGER PRIMARY KEY, region_id INTEGER REFERENCES region(region_id),
                               currency TEXT);
        INSERT INTO region VALUES (1, 'east Bohemia'), (2, 'Prague');
        INSERT INTO customer VALUES (1, 1, 'EUR'), (2, 1, 'CZK'), (3, 2, 'CZK'), (4, 1, 'CZK');
    """)
    con.commit()
    con.close()
    yield SQLiteDatabase(path)
    config.get_settings.cache_clear()


class FakeLLM:
    """Returns scripted replies in order and records the prompts it was given."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def generate(self, prompt, system="", **_):
        self.prompts.append(prompt)
        return LLMResult(json.dumps(self.replies.pop(0)), "fake", False, 0.0)


# -- guard ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("sql", [
    "SELECT 1", "select * from t;", "WITH a AS (SELECT 1) SELECT * FROM a", "SELECT 1 UNION SELECT 2",
    "```sql\nSELECT name FROM t\n```", "SELECT CAST(SUM(IIF(a = 1, 1, 0)) AS REAL) FROM t",
])
def test_guard_allows_reads(sql):
    assert guard.check(sql).ok


@pytest.mark.parametrize("sql", [
    "DELETE FROM t", "UPDATE t SET a = 1", "DROP TABLE t", "INSERT INTO t VALUES (1)", "SELECT 1; DROP TABLE t",
    "PRAGMA table_info(t)", "ATTACH DATABASE 'x.db' AS x", "CREATE TABLE x AS SELECT 1",
    "SELECT * FROM read_csv('/etc/passwd')", "", "ALTER TABLE t ADD COLUMN b",
    'SELECT 1; SELECT "`T-CHO"` FROM t',
])
def test_guard_blocks_everything_else(sql):
    assert not guard.check(sql).ok


def test_guard_survives_tokenizer_errors():
    # regression: the model once wrote T2."`T-CHO"` and sqlglot raised TokenError, not ParseError
    r = guard.check('SELECT * FROM a AS T1 JOIN b AS T2 ON T1.ID = T2.ID WHERE T2."`T-CHO"` >= 25')
    assert isinstance(r.ok, bool)


def test_connection_is_read_only_even_if_guard_is_bypassed(shop):
    r = shop.execute("DELETE FROM customer")
    assert not r.ok and "readonly" in r.error.replace(" ", "").lower()
    assert shop.execute("SELECT COUNT(*) FROM customer").rows == [(4,)]


def test_duckdb_cannot_read_files(tmp_path):
    secret = tmp_path / "secret.csv"
    secret.write_text("a\n1\n")
    db = DuckDBDatabase("t", {"data": pd.DataFrame({"x": [1, 2]})})
    assert db.execute("SELECT SUM(x) FROM data").rows == [(3,)]
    assert not db.execute(f"SELECT * FROM read_csv_auto('{secret}')").ok
    assert not db.execute("SET enable_external_access = true").ok


# -- schema + checks ---------------------------------------------------------------------------------

def test_schema_has_keys_and_examples(shop):
    ddl = shop.ddl(rich=True)
    assert 'FOREIGN KEY ("region_id") REFERENCES "region"("region_id")' in ddl
    assert "'east Bohemia'" in ddl


def test_literal_check_finds_case_mismatch(shop):
    sql = "SELECT COUNT(*) FROM customer c JOIN region r ON c.region_id = r.region_id WHERE r.name = 'East Bohemia'"
    problems = checks.literal_problems(shop, sql)
    assert len(problems) == 1 and "'east Bohemia'" in problems[0]
    assert checks.literal_problems(shop, sql.replace("East", "east")) == []


# -- agent loop ---------------------------------------------------------------------------------------

class NoLinker:
    """Stands in for the embedding linker so tests need no model download."""

    def link(self, question, evidence="", k=20):
        from analyst_agent.schema_linker import LinkResult
        return LinkResult([("region", "name", 0.9)], [("region", 0.9)])


def test_agent_fixes_a_wrong_literal(shop):
    bad = "SELECT COUNT(*) FROM customer c JOIN region r ON c.region_id = r.region_id WHERE r.name = 'East Bohemia'"
    good = bad.replace("East", "east")
    llm = FakeLLM([{"sql": bad}, {"diagnosis": "case", "sql": good}, {"answer": "3 customers", "chart": {"type": "none"}}])
    s = run(build_agent(shop, llm, "agent", NoLinker()), "How many customers in East Bohemia?")
    assert s["sql"] == good and s["result"].rows == [(3,)]
    assert len(s["attempts"]) == 2 and s["fixes"] == 1 and s["llm_calls"] == 3
    assert "east Bohemia" in llm.prompts[1]           # the fixer was shown the stored value


def test_agent_recovers_from_an_sql_error(shop):
    llm = FakeLLM([{"sql": "SELECT COUNT(*) FROM customers"}, {"sql": "SELECT COUNT(*) FROM customer"},
                   {"answer": "4", "chart": {"type": "none"}}])
    s = run(build_agent(shop, llm, "linked+fix", NoLinker()), "How many customers?")
    assert s["result"].rows == [(4,)] and "no such table" in s["attempts"][0]["error"]


def test_baseline_never_retries(shop):
    llm = FakeLLM([{"sql": "SELECT COUNT(*) FROM customers"}, {"answer": "?", "chart": {"type": "none"}}])
    s = run(build_agent(shop, llm, "baseline"), "How many customers?")
    assert len(s["attempts"]) == 1 and s["fixes"] == 0


def test_fix_loop_is_bounded(shop):
    llm = FakeLLM([{"sql": "SELECT nope0 FROM customer"}] + [{"sql": f"SELECT nope{i} FROM customer"} for i in range(1, 4)])
    import dataclasses
    v = dataclasses.replace(VARIANTS["linked+fix"], analyze=False)
    s = run(build_agent(shop, llm, v, NoLinker()), "?")
    assert s["fixes"] == 3 and len(s["attempts"]) == 4


def test_write_requests_are_refused_not_fixed(shop):
    llm = FakeLLM([{"sql": "DELETE FROM customer"}])
    s = run(build_agent(shop, llm, "agent", NoLinker()), "Delete all customers")
    assert s["refused"] and len(s["attempts"]) == 1 and s["llm_calls"] == 1     # no fixer call, no analyst call
    assert "only reads data" in s["answer"]
    assert shop.execute("SELECT COUNT(*) FROM customer").rows == [(4,)]


def test_malformed_sql_still_goes_to_the_fixer(shop):
    llm = FakeLLM([{"sql": "SELEC COUNT(*) FROM customer"}, {"sql": "SELECT COUNT(*) FROM customer"},
                   {"answer": "4", "chart": {"type": "none"}}])
    s = run(build_agent(shop, llm, "agent", NoLinker()), "How many customers?")
    assert "rejected by the SQL guard" in s["attempts"][0]["error"] and s["result"].rows == [(4,)]


# -- small pieces --------------------------------------------------------------------------------------

def test_parse_json_tolerates_fences():
    assert parse_json('```json\n{"sql": "SELECT 1"}\n```') == {"sql": "SELECT 1"}
    assert parse_json('Sure! {"sql": "SELECT 1"}') == {"sql": "SELECT 1"}


def test_chart_spec_is_validated():
    df = pd.DataFrame({"region": ["a", "b"], "n": [1, 2]})
    assert charts.figure({"type": "bar", "x": "region", "y": "n"}, df) is not None
    assert charts.figure({"type": "bar", "x": "missing", "y": "n"}, df) is not None    # falls back to a guess
    assert charts.figure({"type": "exec('rm -rf /')"}, df) is None
    assert charts.figure({"type": "bar"}, df.head(1)) is None


def test_subset_keeps_difficulty_mix():
    qs = [{"question_id": i, "db_id": f"db{i % 3}", "difficulty": ["simple", "moderate", "challenging"][i % 3]}
          for i in range(300)]
    sub = stratified_subset(qs, 30)
    assert len(sub) == 30 and sub == stratified_subset(qs, 30)
    assert {q["difficulty"] for q in sub} == {"simple", "moderate", "challenging"}


def test_paired_bootstrap():
    runs = {"a": [{"question_id": i, "correct": i < 5} for i in range(10)],
            "b": [{"question_id": i, "correct": i < 8} for i in range(10)]}
    p = paired(runs, "a", "b", set(range(10)), iters=2000)
    assert p["delta"] == pytest.approx(0.3) and p["won"] == 3 and p["lost"] == 0 and p["lo"] > 0


def test_filter_probe_shows_stored_values_of_filtered_columns(shop):
    probe = checks.filter_probe(shop, "SELECT COUNT(*) FROM customer c JOIN region r ON c.region_id = r.region_id "
                                      "WHERE r.name LIKE 'East%' AND c.currency = 'EUR'")
    assert "region.name: 'east Bohemia', 'Prague'" in probe and "customer.currency:" in probe


def test_single_zero_count_triggers_a_fix(shop):
    zero = "SELECT COUNT(*) FROM region WHERE name = 'East Bohemia'"   # literal check is off in linked+fix
    llm = FakeLLM([{"sql": zero}, {"sql": "SELECT COUNT(*) FROM region WHERE name = 'east Bohemia'"},
                   {"answer": "1", "chart": {"type": "none"}}])
    s = run(build_agent(shop, llm, "linked+fix", NoLinker()), "How many regions start with East?")
    assert s["result"].rows == [(1,)] and "single 0" in s["attempts"][0]["note"]
    assert "region.name: 'east Bohemia'" in llm.prompts[1]          # the fixer saw the stored values
