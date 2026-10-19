"""Standard preprocessing for Jane — the things an analyst does before asking.

Every operation is deterministic pandas: no model involvement, no cloud.
`run_ops` always returns a NEW DataFrame plus a human-readable log; the
original table is never mutated, so the Prep tab stays a no-risk sandbox.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class PrepResult:
    df: pd.DataFrame
    log: list[str] = field(default_factory=list)
    applied: list[str] = field(default_factory=list)   # op names that ran
    skipped: list[str] = field(default_factory=list)   # op names that no-oped


class PreprocessError(ValueError):
    """Raised on invalid ops, never on empty results — no-ops are reported."""


OPS: dict[str, Callable[..., tuple[pd.DataFrame, str]]] = {}
ALIASES: dict[str, str] = {
    "impute": "fill_missing",
    "fillna": "fill_missing",
    "dedupe": "dedupe",
    "drop_duplicates": "dedupe",
    "standardize": "scale",
    "normalize": "scale",
    "onehot": "encode",
    "one_hot": "encode",
    "winsorize": "clamp_outliers",
}


def op(name: str) -> Callable:
    def deco(fn: Callable[..., tuple[pd.DataFrame, str]]) -> Callable[..., tuple[pd.DataFrame, str]]:
        OPS[name] = fn
        return fn
    return deco


def _is_date_like(s: pd.Series) -> bool:
    """True when ≥90% of a non-numeric column's sample parses as dates."""
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
        return False
    sample = s.dropna().astype(str).head(200)
    if not len(sample):
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() > 0.9


def _resolve(name: str) -> str:
    n = name.strip().lower().replace("-", "_")
    n = ALIASES.get(n, n)
    if n not in OPS:
        raise PreprocessError(f"unknown op '{name}'. Available: {', '.join(sorted(OPS))}")
    return n


def _check_columns(df: pd.DataFrame, columns: list[str] | None, opname: str) -> None:
    """Explicitly requested columns must exist — fail loudly, not silently."""
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise PreprocessError(f"{opname}: column(s) not in table: {', '.join(missing)}")


def run_ops(df: pd.DataFrame, ops: list[tuple[str, dict]]) -> PrepResult:
    """Apply a sequence of (name, kwargs) ops; returns the new frame + log."""
    out = df.copy()
    res = PrepResult(df=out)
    for name, kwargs in ops:
        fn = OPS[_resolve(name)]
        try:
            out, msg = fn(out, **(kwargs or {}))
        except PreprocessError:
            raise
        except Exception as e:  # surface bad column choices readably
            raise PreprocessError(f"{name}: {e}") from e
        res.df = out
        res.applied.append(name)
        if msg:
            res.log.append(msg)
    return res


def available_ops() -> list[str]:
    return sorted(OPS)


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------

@op("fill_missing")
def fill_missing(df: pd.DataFrame, strategy: str = "auto",
                 columns: list[str] | None = None) -> tuple[pd.DataFrame, str]:
    """Fill NaNs: numeric→median, text→'Unknown', datetime→forward-fill.

    strategy: auto | zero | drop  (drop removes rows with any NaN instead).
    """
    if strategy == "drop":
        before = len(df)
        out = df.dropna().reset_index(drop=True)
        dropped = before - len(out)
        return out, f"fill_missing(drop): removed {dropped} row(s) with any missing value"

    out = df.copy()
    _check_columns(out, columns, "fill_missing")
    wanted = set(columns) if columns else None
    filled: list[str] = []
    for col in out.columns:
        if wanted is not None and col not in wanted:
            continue
        s = out[col]
        n = int(s.isna().sum())
        if n == 0:
            continue
        if pd.api.types.is_numeric_dtype(s):
            out[col] = s.fillna(s.median())
            filled.append(f"{col} ← median ({n})")
        elif pd.api.types.is_datetime64_any_dtype(s):
            out[col] = s.ffill()
            filled.append(f"{col} ← forward-fill ({n})")
        else:
            out[col] = s.fillna("Unknown")
            filled.append(f"{col} ← 'Unknown' ({n})")
    if not filled:
        return out, "fill_missing: nothing to fill — no missing values found"
    return out, "fill_missing: " + "; ".join(filled)


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------

@op("dedupe")
def dedupe(df: pd.DataFrame, subset: list[str] | None = None) -> tuple[pd.DataFrame, str]:
    """Drop exact duplicate rows (optionally judged on a column subset)."""
    before = len(df)
    out = df.drop_duplicates(subset=subset or None).reset_index(drop=True)
    removed = before - len(out)
    msg = f"dedupe: removed {removed} duplicate row(s)"
    if subset:
        msg += f" (matched on {', '.join(subset)})"
    return out, msg


# ---------------------------------------------------------------------------
# Text hygiene
# ---------------------------------------------------------------------------

@op("clean_text")
def clean_text(df: pd.DataFrame, columns: list[str] | None = None) -> tuple[pd.DataFrame, str]:
    """Trim whitespace, collapse runs of spaces, drop trailing whitespace headers."""
    out = df.copy()
    renames: dict[str, str] = {}
    touched = 0
    for col in out.columns:
        new_name = re.sub(r"\s+", " ", str(col)).strip()
        if new_name != str(col):
            renames[col] = new_name
    if renames:
        out = out.rename(columns=renames)
    wanted = set(columns) if columns else None
    targets = [c for c in out.columns
               if (wanted is None or c in wanted)
               and (out[c].dtype == object or out[c].dtype == "string")]
    for col in targets:
        stripped = out[col].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
        changed = int((stripped.fillna("§") != out[col].astype("string").fillna("§")).sum())
        if changed:
            out[col] = stripped
            touched += changed
    msg = f"clean_text: normalized whitespace in {touched} cell(s)"
    if renames:
        msg += f"; renamed {len(renames)} column(s)"
    return out, msg


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

@op("to_datetime")
def to_datetime(df: pd.DataFrame, columns: list[str] | None = None) -> tuple[pd.DataFrame, str]:
    """Coerce object/string columns that look like dates into datetime64."""
    out = df.copy()
    converted: list[str] = []
    targets = columns
    if targets is None:
        targets = []
        for col in out.columns:
            s = out[col]
            if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_numeric_dtype(s):
                continue
            sample = s.dropna().astype(str).head(200)
            if len(sample) and pd.to_datetime(sample, errors="coerce", format="mixed").notna().mean() > 0.9:
                targets.append(col)
    for col in targets:
        if col not in out.columns:
            continue
        coerced = pd.to_datetime(out[col], errors="coerce", format="mixed")
        ok = int(coerced.notna().sum())
        if ok:
            out[col] = coerced
            converted.append(f"{col} ({ok}/{len(out)} parsed)")
    if not converted:
        return out, "to_datetime: no date-like columns found"
    return out, "to_datetime: converted " + ", ".join(converted)


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def _zscore(s: pd.Series) -> pd.Series:
    sd = s.std()
    return (s - s.mean()) / sd if sd and not np.isnan(sd) else s * 0.0


@op("scale")
def scale(df: pd.DataFrame, method: str = "zscore",
          columns: list[str] | None = None) -> tuple[pd.DataFrame, str]:
    """Standardize numeric columns. method: zscore | minmax | robust."""
    out = df.copy()
    _check_columns(out, columns, "scale")
    numeric = out.select_dtypes(include=[np.number]).columns
    targets = [c for c in (columns or numeric) if c in numeric]
    if not targets:
        return out, "scale: no numeric columns to scale"
    for col in targets:
        s = out[col].astype(float)
        if method == "minmax":
            lo, hi = s.min(), s.max()
            out[col] = (s - lo) / (hi - lo) if hi > lo else s * 0.0
        elif method == "robust":
            iqr = s.quantile(0.75) - s.quantile(0.25)
            med = s.median()
            out[col] = (s - med) / iqr if iqr else s * 0.0
        else:
            out[col] = _zscore(s)
    return out, f"scale({method}): standardized {len(targets)} column(s): {', '.join(targets)}"


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

@op("encode")
def encode(df: pd.DataFrame, columns: list[str] | None = None,
           max_categories: int = 12) -> tuple[pd.DataFrame, str]:
    """One-hot encode low-cardinality text columns (≤ max_categories levels)."""
    out = df.copy()
    _check_columns(out, columns, "encode")
    made = 0
    wanted = set(columns) if columns else None
    targets = [
        c for c in out.columns
        if (wanted is None or c in wanted)
        and not _is_date_like(out[c])
        and out[c].nunique(dropna=True) <= max_categories
    ]
    skipped = [c for c in (columns or []) if c in out.columns and _is_date_like(out[c])]
    for col in targets:
        n = out[col].nunique(dropna=True)
        if n < 2:
            continue
        dummies = pd.get_dummies(out[col], prefix=str(col), dtype=int)
        out = pd.concat([out.drop(columns=[col]), dummies], axis=1)
        made += dummies.shape[1]
    if not made:
        return out, "encode: no eligible text columns (need 2–12 distinct values, not date-like)"
    msg = f"encode: one-hot encoded {len(targets)} column(s) → {made} new columns"
    if skipped:
        msg += f"; skipped date-like: {', '.join(skipped)}"
    return out, msg


# ---------------------------------------------------------------------------
# Outliers
# ---------------------------------------------------------------------------

@op("clamp_outliers")
def clamp_outliers(df: pd.DataFrame, columns: list[str] | None = None,
                   k: float = 1.5) -> tuple[pd.DataFrame, str]:
    """Winsorize numeric columns at the IQR fence (q1 − k·iqr, q3 + k·iqr)."""
    out = df.copy()
    _check_columns(out, columns, "clamp_outliers")
    numeric = out.select_dtypes(include=[np.number]).columns
    targets = [c for c in (columns or numeric) if c in numeric]
    clamped = 0
    for col in targets:
        s = out[col].dropna()
        if len(s) < 4:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if not iqr:
            # Constant column: any different value is definitionally outside.
            lo = hi = q1
        else:
            lo, hi = q1 - k * iqr, q3 + k * iqr
        before = out[col].copy()
        out[col] = out[col].clip(lower=lo, upper=hi)
        clamped += int(((before < lo) | (before > hi)).sum())
    if not clamped:
        return out, "clamp_outliers: no values outside the IQR fence"
    return out, f"clamp_outliers: clamped {clamped} value(s) in {len(targets)} numeric column(s)"


# ---------------------------------------------------------------------------
# Convenience presets
# ---------------------------------------------------------------------------

def clean_default(df: pd.DataFrame) -> PrepResult:
    """The 'just make it tidy' preset: text → dates → missing → dupes."""
    return run_ops(df, [
        ("clean_text", {}),
        ("to_datetime", {}),
        ("fill_missing", {}),
        ("dedupe", {}),
    ])
