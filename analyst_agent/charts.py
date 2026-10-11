"""Turn the analyst's chart *spec* into a Plotly figure.

The model never writes plotting code. It returns a small JSON spec ({type, x, y, title}); this module
validates it against the real result columns and builds the figure. A spec cannot execute anything,
so a malicious or confused model output can at worst produce a wrong chart, never run code.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px

CHARTS = {"bar": px.bar, "line": px.line, "scatter": px.scatter, "pie": px.pie}


def figure(spec: dict | None, df: pd.DataFrame):
    if not spec or df.empty or len(df) < 2:
        return None
    kind = str(spec.get("type", "none")).lower()
    if kind not in CHARTS:
        return None
    x, y = spec.get("x") or "", spec.get("y") or ""
    if x not in df.columns or y not in df.columns:
        x, y = _guess(df)
        if not x:
            return None
    if not pd.api.types.is_numeric_dtype(df[y]):
        converted = pd.to_numeric(df[y], errors="coerce")
        if converted.isna().all():
            return None
        df = df.assign(**{y: converted})
    df = df.head(50)
    title = str(spec.get("title", ""))[:120]
    if kind == "pie":
        return px.pie(df, names=x, values=y, title=title)
    return CHARTS[kind](df, x=x, y=y, title=title)


def _guess(df: pd.DataFrame) -> tuple[str, str]:
    """First non-numeric column as x, first numeric column as y."""
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    other = [c for c in df.columns if c not in numeric]
    if numeric and other:
        return other[0], numeric[0]
    if len(numeric) >= 2:
        return numeric[0], numeric[1]
    return "", ""
