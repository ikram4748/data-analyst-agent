"""SQL guard: the first of two safety layers (the second is the read-only connection in db.py).

A generated query is accepted only if it parses as exactly one read-only statement: SELECT / WITH /
UNION / VALUES. Anything that writes, changes the schema, attaches files or alters settings is rejected
with a reason the model can read and fix.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import sqlglot
from sqlglot import exp

FORBIDDEN = (exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Create, exp.Drop, exp.Alter,
             exp.Command, exp.Pragma, exp.Set, exp.Transaction, exp.Commit, exp.Rollback,
             exp.Attach, exp.Detach, exp.Copy, exp.Use)
# Table functions that read files or the network (DuckDB): blocked here, and DuckDB's external
# access is also switched off, so this is belt and braces.
FILE_FUNCTIONS = re.compile(r"\b(read_(csv|json|parquet|text|blob)\w*|glob|load_extension|readfile|writefile)\s*\(", re.I)


@dataclass
class GuardResult:
    ok: bool
    sql: str
    reason: str = ""
    policy: bool = False      # rejected because it would change something (refuse), not because it is malformed (fix)


def strip_fences(sql: str) -> str:
    sql = re.sub(r"^\s*```(?:sql)?|```\s*$", "", sql.strip(), flags=re.I).strip()
    return sql.rstrip(";").strip()


def check(sql: str, dialect: str = "sqlite") -> GuardResult:
    sql = strip_fences(sql)
    if not sql:
        return GuardResult(False, sql, "empty query")
    if FILE_FUNCTIONS.search(sql):
        return GuardResult(False, sql, "file / extension functions are not allowed", policy=True)
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except sqlglot.errors.SqlglotError as e:      # ParseError and TokenError (e.g. an unclosed quote)
        # A parse failure is not proof of danger (sqlglot does not know every dialect quirk), so we let
        # the database decide, but only for text that starts like a read query.
        if re.match(r"^\s*(select|with)\b", sql, re.I) and ";" not in sql:
            return GuardResult(True, sql)
        return GuardResult(False, sql, f"could not parse SQL: {str(e).splitlines()[0][:200]}")
    if len(statements) != 1:
        return GuardResult(False, sql, f"expected exactly one statement, got {len(statements)}",
                           policy=any(not isinstance(x, (exp.Query, exp.Values)) for x in statements))
    root = statements[0]
    if not isinstance(root, (exp.Query, exp.Values)):
        return GuardResult(False, sql, f"only SELECT queries are allowed (got {root.key.upper()})", policy=True)
    for node in root.walk():
        if isinstance(node, FORBIDDEN):
            return GuardResult(False, sql, f"{node.key.upper()} is not allowed inside a read-only query", policy=True)
    return GuardResult(True, sql)
