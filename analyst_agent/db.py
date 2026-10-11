"""Database access: schema introspection for prompts, and safe query execution.

Two engines behind one interface:
  * SQLite  - the BIRD benchmark databases (gold SQL is written in SQLite's dialect, so scoring must use it)
  * DuckDB  - CSV files a user uploads in the app (fast columnar analytics, no server)

Execution is always read-only. The SQL guard (guard.py) rejects writes before they get here; the
read-only connection is a second, independent layer in case the guard is ever bypassed.
"""
from __future__ import annotations

import csv
import io
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import get_settings


@dataclass
class Column:
    name: str
    type: str
    description: str = ""
    values: str = ""                       # BIRD "value_description": meaning of codes, units, formats
    examples: list[str] = field(default_factory=list)


@dataclass
class Table:
    name: str
    columns: list[Column]
    primary_key: list[str] = field(default_factory=list)
    foreign_keys: list[tuple[str, str, str]] = field(default_factory=list)   # (column, ref_table, ref_column)
    row_count: int | None = None


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple]
    seconds: float
    truncated: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


class Database:
    dialect = "sqlite"

    def __init__(self, name: str):
        self.name = name
        self._tables: dict[str, Table] | None = None

    # -- implemented by subclasses ---------------------------------------------------------------
    def _connect(self):
        raise NotImplementedError

    def _introspect(self) -> dict[str, Table]:
        raise NotImplementedError

    def execute(self, sql: str, max_rows: int | None = None, timeout_s: float | None = None) -> QueryResult:
        raise NotImplementedError

    # -- shared -------------------------------------------------------------------------------------
    @property
    def tables(self) -> dict[str, Table]:
        if self._tables is None:
            self._tables = self._introspect()
        return self._tables

    def ddl(self, tables: list[str] | None = None, *, rich: bool | set[tuple[str, str]] = False) -> str:
        """Schema text for a prompt.

        rich=True       every column gets its description, value meanings and example values
        rich={(t, c)}   only these columns get notes; the rest are listed bare (keeps prompts focused)
        """
        out = []
        for t in self.tables.values():
            if tables is not None and t.name not in tables:
                continue
            lines = []
            for c in t.columns:
                line = f"  {_q(c.name)} {c.type}"
                if rich is True or (rich and (t.name, c.name) in rich):
                    notes = [n for n in (c.description, c.values) if n]
                    if c.examples:
                        notes.append("e.g. " + ", ".join(repr(x) for x in c.examples))
                    if notes:
                        line += "  -- " + " | ".join(notes)
                lines.append(line)
            if t.primary_key:
                lines.append(f"  PRIMARY KEY ({', '.join(_q(c) for c in t.primary_key)})")
            for col, rt, rc in t.foreign_keys:
                ref = f"{_q(rt)}({_q(rc)})" if rc else _q(rt)     # no column = the referenced table's primary key
                lines.append(f"  FOREIGN KEY ({_q(col)}) REFERENCES {ref}")
            rows = f"  -- {t.row_count:,} rows" if (rich and t.row_count is not None) else ""
            out.append(f"CREATE TABLE {_q(t.name)} ({rows}\n" + ",\n".join(lines) + "\n);")
        return "\n\n".join(out)


class SQLiteDatabase(Database):
    dialect = "sqlite"

    def __init__(self, path: Path, description_dir: Path | None = None):
        super().__init__(path.stem)
        self.path = path
        self.description_dir = description_dir

    def _connect(self):
        # mode=ro: the file is opened read-only by SQLite itself; query_only blocks writes even to temp tables
        con = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
        con.execute("PRAGMA query_only = ON")
        con.text_factory = lambda b: b.decode("utf-8", errors="replace")
        return con

    def execute(self, sql: str, max_rows: int | None = None, timeout_s: float | None = None) -> QueryResult:
        timeout_s = timeout_s or get_settings().query_timeout_s
        con = self._connect()
        start = time.monotonic()
        # SQLite has no statement timeout; the progress handler aborts the query when time is up.
        con.set_progress_handler(lambda: 1 if time.monotonic() - start > timeout_s else 0, 10_000)
        try:
            cur = con.execute(sql)
            cols = [d[0] for d in cur.description or []]
            if max_rows is None:
                rows, truncated = cur.fetchall(), False
            else:
                rows = cur.fetchmany(max_rows + 1)
                truncated = len(rows) > max_rows
                rows = rows[:max_rows]
            return QueryResult(cols, [tuple(r) for r in rows], time.monotonic() - start, truncated)
        except sqlite3.OperationalError as e:
            msg = f"query exceeded {timeout_s:.0f}s timeout" if "interrupted" in str(e) else str(e)
            return QueryResult([], [], time.monotonic() - start, error=msg)
        except sqlite3.Error as e:
            return QueryResult([], [], time.monotonic() - start, error=str(e))
        finally:
            con.close()

    def _introspect(self) -> dict[str, Table]:
        con = self._connect()
        try:
            names = [r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
            tables = {}
            for n in names:
                info = con.execute(f"PRAGMA table_info({_q(n)})").fetchall()
                cols = [Column(r[1], r[2] or "TEXT") for r in info]
                pk = [r[1] for r in sorted(info, key=lambda r: r[5]) if r[5]]
                fks = [(r[3], r[2], r[4]) for r in con.execute(f"PRAGMA foreign_key_list({_q(n)})")]
                count = con.execute(f"SELECT COUNT(*) FROM {_q(n)}").fetchone()[0]
                tables[n] = Table(n, cols, pk, fks, count)
        finally:
            con.close()
        self._add_descriptions(tables)
        self._add_examples(tables)
        return tables

    def _add_descriptions(self, tables: dict[str, Table]) -> None:
        """BIRD ships one CSV per table describing each column; merge it into the schema."""
        if not self.description_dir or not self.description_dir.exists():
            return
        by_lower = {t.lower(): t for t in tables}
        for f in self.description_dir.glob("*.csv"):
            t = tables.get(by_lower.get(f.stem.lower(), ""))
            if not t:
                continue
            raw = f.read_bytes()
            try:
                text = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = raw.decode("cp1252", errors="replace")
            cols = {c.name.lower(): c for c in t.columns}
            for row in csv.DictReader(io.StringIO(text)):
                row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
                c = cols.get(row.get("original_column_name", "").lower())
                if not c:
                    continue
                desc = row.get("column_description") or row.get("column_name", "")
                if desc.lower() != c.name.lower():
                    c.description = " ".join(desc.split())
                c.values = " ".join(row.get("value_description", "").split())[:300]

    def _add_examples(self, tables: dict[str, Table]) -> None:
        """A few real values per text and date column, so the model writes 'EUR' instead of guessing 'euro',
        and sees that a date is stored as '1995-03-24' even when the documentation says "YYMMDD".
        Cached to disk because scanning big tables is slow the first time."""
        cache = get_settings().cache_dir / f"examples_v2_{self.name}.json"
        if cache.exists():
            saved = json.loads(cache.read_text())
            for t in tables.values():
                for c in t.columns:
                    c.examples = saved.get(t.name, {}).get(c.name, [])
            return
        con = self._connect()
        saved: dict[str, dict[str, list[str]]] = {}
        try:
            for t in tables.values():
                for c in t.columns:
                    if not any(k in c.type.upper() for k in ("CHAR", "TEXT", "CLOB", "DATE", "TIME")) and c.type:
                        continue
                    try:
                        vals = [r[0] for r in con.execute(
                            f"SELECT DISTINCT {_q(c.name)} FROM (SELECT {_q(c.name)} FROM {_q(t.name)} "
                            f"WHERE {_q(c.name)} IS NOT NULL AND {_q(c.name)} != '' LIMIT 2000) LIMIT 3")]
                    except sqlite3.Error:
                        continue
                    c.examples = [str(v)[:40] for v in vals if isinstance(v, (str, int, float))]
                    saved.setdefault(t.name, {})[c.name] = c.examples
        finally:
            con.close()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(saved, ensure_ascii=False))


class DuckDBDatabase(Database):
    """In-memory DuckDB over uploaded CSV files. External file/network access is switched off after
    loading, so a generated query cannot read anything except the uploaded tables."""
    dialect = "duckdb"

    def __init__(self, name: str, tables: dict):
        """tables: table name -> pandas DataFrame, or raw CSV bytes (types detected by DuckDB's CSV sniffer,
        which recognises dates and numbers that pandas would leave as text)."""
        super().__init__(name)
        import duckdb
        self.con = duckdb.connect(":memory:")
        for table, src in tables.items():
            rel = self.con.read_csv(io.BytesIO(src)) if isinstance(src, bytes) else self.con.from_df(src)
            rel.create(table)
        self.con.execute("SET enable_external_access = false")
        self.con.execute("SET lock_configuration = true")

    @classmethod
    def from_csv_files(cls, name: str, files: dict[str, bytes]) -> "DuckDBDatabase":
        import re
        tables = {}
        for fname, data in files.items():
            tables[re.sub(r"\W+", "_", Path(fname).stem).strip("_").lower() or "data"] = data
        return cls(name, tables)

    def execute(self, sql: str, max_rows: int | None = None, timeout_s: float | None = None) -> QueryResult:
        start = time.monotonic()
        try:
            cur = self.con.cursor()
            cur.execute(sql)
            cols = [d[0] for d in cur.description or []]
            rows = cur.fetchall() if max_rows is None else cur.fetchmany(max_rows + 1)
            truncated = max_rows is not None and len(rows) > max_rows
            return QueryResult(cols, [tuple(r) for r in rows[:max_rows]], time.monotonic() - start, truncated)
        except Exception as e:  # noqa: BLE001 - duckdb raises many error types; all become feedback
            return QueryResult([], [], time.monotonic() - start, error=str(e).split("\n")[0])

    def _introspect(self) -> dict[str, Table]:
        tables = {}
        for (n,) in self.con.execute("SELECT table_name FROM information_schema.tables ORDER BY 1").fetchall():
            cols = [Column(c, t) for c, t in self.con.execute(
                "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = ? "
                "ORDER BY ordinal_position", [n]).fetchall()]
            count = self.con.execute(f"SELECT COUNT(*) FROM {_q(n)}").fetchone()[0]
            for c in cols:
                if c.type.upper() in ("VARCHAR", "DATE", "TIMESTAMP"):
                    c.examples = [str(v)[:40] for (v,) in self.con.execute(
                        f"SELECT DISTINCT {_q(c.name)} FROM {_q(n)} WHERE {_q(c.name)} IS NOT NULL LIMIT 3").fetchall()]
            tables[n] = Table(n, cols, row_count=count)
        return tables


def bird_databases() -> dict[str, Path]:
    """BIRD database name -> folder (each folder holds <name>.sqlite and database_description/)."""
    root = get_settings().bird_dir / "dev_databases"
    return {p.name: p for p in sorted(root.iterdir()) if (p / f"{p.name}.sqlite").exists()} if root.exists() else {}


def open_bird(name: str) -> SQLiteDatabase:
    folder = bird_databases()[name]
    return SQLiteDatabase(folder / f"{name}.sqlite", folder / "database_description")
