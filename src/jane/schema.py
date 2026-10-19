"""Table schema summarization → compact text for LLM prompts, plus DuckDB registration helpers."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .logging_setup import get_logger

log = get_logger("jane.schema")


@dataclass
class SchemaInfo:
    table: str
    row_count: int
    columns: list[dict] = field(default_factory=list)
    sample_rows: list[dict] = field(default_factory=list)
    duplicate_rows: int = 0
    active: bool = False

    def to_prompt_text(self) -> str:
        lines = [f"TABLE `{self.table}` — {self.row_count} rows"]
        if self.active:
            lines[-1] += "  ← ACTIVE TABLE (the one you were asked about)"
        for c in self.columns:
            extra = ""
            if c["nulls"]:
                extra += f" (nulls: {c['nulls']})"
            if c["sample"]:
                sample = ", ".join(json.dumps(s, default=str) for s in c["sample"][:3])
                extra += f" e.g. {sample}"
            st = c.get("stats")
            if st:
                if "med" in st:
                    extra += f" med {st['med']}, range {st['min']}–{st['max']}"
                else:
                    extra += (f" spans {st['first']} → {st['last']}, "
                              f"{st['distinct']} distinct values")
            lines.append(f"  - {c['name']}: {c['dtype']}{extra}")
        if self.duplicate_rows:
            lines.append(f"  NOTE: {self.duplicate_rows} rows are EXACT duplicates of "
                         "another row — dedupe before counting/averaging unless "
                         "duplicates are meaningful (e.g. repeated purchases).")
        reserved = [c["name"] for c in self.columns
                    if str(c["name"]).lower() in _SQL_RESERVED]
        if reserved:
            lines.append(f"  NOTE: column(s) {reserved} are SQL RESERVED WORDS — you "
                         "MUST double-quote them in SQL, e.g. SUM(CASE WHEN "
                         "\"left\" = 1 THEN 1 ELSE 0 END), or the parser will fail "
                         "every time. Same for any column that looks like a "
                         "function name.")
        return "\n".join(lines)


_SQL_RESERVED = {
    "left", "right", "order", "group", "limit", "offset", "select", "from",
    "where", "join", "on", "as", "by", "case", "when", "then", "else", "end",
    "and", "or", "not", "in", "is", "like", "between", "union", "all",
    "having", "values", "default", "check", "primary", "foreign", "table",
    "column", "index", "unique", "desc", "asc", "current_date", "year", "month", "day",
}


def _jsonable(v: Any) -> Any:
    if v is None or isinstance(v, (str, bool)):
        return v
    if isinstance(v, (int, float)):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v
    if hasattr(v, "isoformat"):  # timestamps
        return v.isoformat()
    return str(v)


def describe_table(
    name: str,
    df: pd.DataFrame,
    sample_rows: int = 3,
    *,
    active: bool = False,
) -> SchemaInfo:
    from .preprocess import _is_date_like

    cols: list[dict] = []

    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        nulls = int(series.isna().sum())
        samples: list[Any] = []
        for v in series.dropna().unique()[:3]:
            samples.append(_jsonable(v))
        entry: dict[str, Any] = {"name": str(col), "dtype": dtype,
                                 "nulls": nulls, "sample": samples}
        # compact stats so the model never has to guess distributions
        if pd.api.types.is_numeric_dtype(series) and nulls < len(series):
            entry["stats"] = {
                "med": round(float(series.median()), 3),
                "min": round(float(series.min()), 3),
                "max": round(float(series.max()), 3),
            }
        elif _is_date_like(series):
            parsed = pd.to_datetime(series.dropna().head(500), errors="coerce", format="mixed")
            if parsed.notna().any():
                entry["stats"] = {
                    "first": str(parsed.min())[:10],
                    "last": str(parsed.max())[:10],
                    "distinct": int(series.nunique()),
                }
        cols.append(entry)
    info = SchemaInfo(
        table=name,
        row_count=int(len(df)),
        columns=cols,
        duplicate_rows=int(df.duplicated().sum()),
        active=bool(active),
    )
    # sample rows (records) — keeps prompt small
    if sample_rows > 0 and len(df) > 0:
        head = df.head(sample_rows)
        info.sample_rows = [
            {str(k): _jsonable(v) for k, v in row.items()}
            for row in head.to_dict(orient="records")
        ]
    return info


def build_schema_block(
    tables: dict[str, pd.DataFrame],
    sample_rows: int = 3,
    active_table: str | None = None,
) -> str:
    """Full schema block to inject into the system prompt."""
    parts = ["=== DATA SCHEMA (the ONLY tables you may query) ==="]
    active = active_table if isinstance(active_table, str) else None
    for tname, df in tables.items():
        info = describe_table(
            tname, df, sample_rows=sample_rows,
            active=(tname == active),
        )
        parts.append(info.to_prompt_text())
        if info.sample_rows:
            parts.append("  sample rows: " + json.dumps(info.sample_rows, default=str))
    if active and active not in tables:
        parts.append(
            f"  ACTIVE TABLE REFERENCE: `{active}` (ask is about this table -- "
            "confirm its spelling against the schema above before querying)."
        )
    parts.append("=== END SCHEMA ===")
    return "\n".join(parts)


def register_in_duckdb(con, tables: dict[str, pd.DataFrame]) -> list[str]:
    """Register every DataFrame as a queryable DuckDB view. Returns view names.

    Registration order carries no meaning: every table is queryable by name,
    and nothing here ranks one view above another.
    """
    import duckdb  # local import: only needed on the execute path

    views = []
    for tname, df in tables.items():
        safe = _safe_name(tname)
        try:
            con.register(safe, df)
            views.append(safe)
        except Exception as e:  # duckdb registration can fail on exotic dtypes
            log.warning("could not register %s: %s — coercing to strings", tname, e)
            con.register(safe, df.astype(str))
            views.append(safe)
    return views


def _safe_name(name: str) -> str:
    out = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name.strip())
    if not out or not (out[0].isalpha() or out[0] == "_"):
        out = "t_" + out
    return out
