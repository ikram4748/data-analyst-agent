"""Cheap result checks that need no LLM call.

The most common silent failure in text-to-SQL is a text literal that does not match the stored value:
`A3 = 'East Bohemia'` when the column holds 'east Bohemia'. The query runs, returns COUNT = 0, and
nothing looks wrong. This module finds every `column = 'text'` / `column IN ('text', ...)` comparison,
checks with a one-row lookup that the value exists, and if not suggests the closest stored values.
"""
from __future__ import annotations

import sqlglot
from sqlglot import exp

from .db import Database, _q


def _aliases(tree: exp.Expression, db: Database) -> dict[str, str]:
    """alias or table name (lower case) -> real table name, for every table the query reads."""
    names = {t.name.lower(): t.name for t in db.tables.values()}
    out: dict[str, str] = {}
    for t in tree.find_all(exp.Table):
        real = names.get(t.name.lower())
        if real:
            out[t.name.lower()] = real
            if t.alias:
                out[t.alias.lower()] = real
    return out


def _table_for(col: exp.Column, aliases: dict[str, str], db: Database) -> str | None:
    if col.table:
        return aliases.get(col.table.lower())
    owners = [t.name for t in db.tables.values() if any(c.name.lower() == col.name.lower() for c in t.columns)
              and t.name.lower() in {a for a in aliases.values()}]
    return owners[0] if len(owners) == 1 else None


def _comparisons(tree: exp.Expression):
    for node in tree.walk():
        if isinstance(node, exp.EQ):
            left, right = node.left, node.right
            if isinstance(right, exp.Column) and isinstance(left, exp.Literal):
                left, right = right, left
            if isinstance(left, exp.Column) and isinstance(right, exp.Literal) and right.is_string:
                yield left, [right.this]
        elif isinstance(node, exp.In) and isinstance(node.this, exp.Column):
            vals = [e.this for e in node.expressions if isinstance(e, exp.Literal) and e.is_string]
            if vals:
                yield node.this, vals


def literal_problems(db: Database, sql: str, limit: int = 5) -> list[str]:
    try:
        tree = sqlglot.parse_one(sql, read=db.dialect)
    except Exception:  # noqa: BLE001 - unparseable SQL is reported by the database instead
        return []
    aliases = _aliases(tree, db)
    problems = []
    for col, values in _comparisons(tree):
        table = _table_for(col, aliases, db)
        if not table:
            continue
        cname = next((c.name for c in db.tables[table].columns if c.name.lower() == col.name.lower()), None)
        if not cname:
            continue
        for v in values[:limit]:
            hit = db.execute(f"SELECT 1 FROM {_q(table)} WHERE {_q(cname)} = ? LIMIT 1".replace("?", _lit(v)),
                             max_rows=1, timeout_s=5)
            if not hit.ok or hit.rows:
                continue
            similar = db.execute(
                f"SELECT DISTINCT {_q(cname)} FROM {_q(table)} WHERE LOWER(CAST({_q(cname)} AS TEXT)) "
                f"LIKE {_lit('%' + v.lower()[:20] + '%')} LIMIT 5", max_rows=5, timeout_s=5)
            hint = ", ".join(repr(r[0]) for r in similar.rows) if similar.ok and similar.rows else "none found"
            problems.append(f"the value {v!r} does not exist in {table}.{cname} (similar stored values: {hint})")
    return problems


def _lit(v: str) -> str:
    return "'" + v.replace("'", "''") + "'"


def filter_probe(db: Database, sql: str, per_column: int = 4, max_columns: int = 6) -> str:
    """Real stored values of every column the query filters on (WHERE / HAVING), any type.

    Given to the fixer when a result looks wrong. It catches what the literal check cannot see: dates written
    in the wrong format (`date LIKE '93%'` against '1993-02-26'), numeric codes, ranges that do not overlap.
    """
    try:
        tree = sqlglot.parse_one(sql, read=db.dialect)
    except Exception:  # noqa: BLE001
        return ""
    aliases = _aliases(tree, db)
    seen, lines = set(), []
    for clause in [*tree.find_all(exp.Where), *tree.find_all(exp.Having)]:
        for col in clause.find_all(exp.Column):
            table = _table_for(col, aliases, db)
            if not table:
                continue
            cname = next((c.name for c in db.tables[table].columns if c.name.lower() == col.name.lower()), None)
            if not cname or (table, cname) in seen:
                continue
            seen.add((table, cname))
            r = db.execute(f"SELECT DISTINCT {_q(cname)} FROM (SELECT {_q(cname)} FROM {_q(table)} "
                           f"WHERE {_q(cname)} IS NOT NULL LIMIT 5000) LIMIT {per_column}", max_rows=per_column, timeout_s=5)
            if r.ok and r.rows:
                lines.append(f"- {table}.{cname}: " + ", ".join(repr(v[0]) for v in r.rows))
            if len(lines) >= max_columns:
                return "\n".join(lines)
    return "\n".join(lines)
