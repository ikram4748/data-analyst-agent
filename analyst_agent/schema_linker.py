"""Schema linking: which columns does this question need?

Each column becomes a short document ("table.column: description; values; examples") embedded once
with a small local model (bge-small, ONNX via fastembed: ~130 MB, CPU, no PyTorch, no API quota).
At question time we embed the question + evidence and rank columns by cosine similarity.

Design choice: we never *drop* tables. Every BIRD schema fits in the prompt, and pruning tables
risks deleting the one the query needs (a recall loss nothing downstream can recover from).
Instead the top-k columns get rich notes and every other column is listed bare.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .config import get_settings
from .db import Database


@dataclass
class LinkResult:
    columns: list[tuple[str, str, float]]       # (table, column, score), best first
    tables: list[tuple[str, float]]             # (table, best column score), best first

    @property
    def column_set(self) -> set[tuple[str, str]]:
        return {(t, c) for t, c, _ in self.columns}


@lru_cache(maxsize=1)
def _model():
    from fastembed import TextEmbedding
    s = get_settings()
    return TextEmbedding(model_name=s.embed_model, cache_dir=str(s.cache_dir / "models"))


def _column_doc(table: str, col) -> str:
    parts = [f"{table}.{col.name}"]
    if col.description:
        parts.append(col.description)
    if col.values:
        parts.append(col.values[:200])
    if col.examples:
        parts.append("values: " + ", ".join(col.examples))
    return ": ".join(parts[:2]) + ("; " + "; ".join(parts[2:]) if len(parts) > 2 else "")


class SchemaLinker:
    def __init__(self, db: Database):
        self.db = db
        self.keys = [(t.name, c.name) for t in db.tables.values() for c in t.columns]
        docs = [_column_doc(t.name, c) for t in db.tables.values() for c in t.columns]
        self.matrix = self._embed_docs(docs)

    def _embed_docs(self, docs: list[str]) -> np.ndarray:
        s = get_settings()
        digest = hashlib.sha256(("\n".join(docs) + s.embed_model).encode()).hexdigest()[:16]
        path = s.cache_dir / f"colvec_{self.db.name}_{digest}.npy"
        if path.exists():
            return np.load(path)
        m = np.array(list(_model().passage_embed(docs)), dtype=np.float32)
        m /= np.linalg.norm(m, axis=1, keepdims=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, m)
        return m

    def link(self, question: str, evidence: str = "", k: int = 20) -> LinkResult:
        q = np.array(next(iter(_model().query_embed([f"{question} {evidence}".strip()]))), dtype=np.float32)
        q /= np.linalg.norm(q)
        scores = self.matrix @ q
        order = np.argsort(-scores)
        cols = [(*self.keys[i], float(scores[i])) for i in order[:k]]
        best: dict[str, float] = {}
        for i in order:
            t = self.keys[i][0]
            best.setdefault(t, float(scores[i]))
        return LinkResult(cols, sorted(best.items(), key=lambda x: -x[1]))
