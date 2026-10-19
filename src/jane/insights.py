"""Dataset insights — profile reports and pattern hunting, no LLM required.

Profile: an analyst's first briefing on any table (shape, dtypes, nulls,
cardinality, numeric distributions, outliers, top correlations).
Insights: a deterministic sweep for spikes, drops, outliers and strong
correlations, each with enough context that the agent can narrate it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Profile report
# ---------------------------------------------------------------------------

def profile_table(name: str, df: pd.DataFrame, max_corr: int = 6) -> dict:
    """Full analyst briefing on one table. Pure pandas — deterministic."""
    report: dict[str, Any] = {
        "name": name,
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "columns": [],
        "correlations": [],
    }
    num_cols = list(df.select_dtypes(include=[np.number]).columns)
    for col in df.columns:
        s = df[col]
        entry: dict[str, Any] = {
            "name": str(col),
            "dtype": str(s.dtype),
            "nulls": int(s.isna().sum()),
            "null_pct": round(100 * s.isna().mean(), 1) if len(s) else 0.0,
            "unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s) and s.notna().sum():
            clean = s.dropna()
            q1, q3 = clean.quantile([0.25, 0.75])
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            entry.update({
                "min": _safe_round(clean.min()),
                "median": _safe_round(clean.median()),
                "mean": _safe_round(clean.mean()),
                "max": _safe_round(clean.max()),
                "std": _safe_round(clean.std()),
                "zeros": int((clean == 0).sum()),
                "negatives": int((clean < 0).sum()),
                "outliers_iqr": int(((clean < lo) | (clean > hi)).sum()),
            })
        else:
            vc = s.value_counts(dropna=True)
            entry["top_values"] = [
                {"value": _shorten(str(v)), "count": int(c)} for v, c in vc.head(3).items()]
        report["columns"].append(entry)

    # strongest correlations among numeric columns
    if len(num_cols) >= 2:
        corr = df[num_cols].corr(numeric_only=True)
        pairs = []
        for i, a in enumerate(num_cols):
            for b in num_cols[i + 1:]:
                c = corr.loc[a, b]
                if pd.notna(c) and abs(c) > 0.35:
                    pairs.append((a, b, float(c)))
        pairs.sort(key=lambda p: -abs(p[2]))
        report["correlations"] = [
            {"a": a, "b": b, "r": round(r, 3)} for a, b, r in pairs[:max_corr]]
    return report


def profile_summary_text(reports: list[dict]) -> str:
    """Compact human-readable briefing (used by LLM narration + CLI)."""
    lines = []
    for r in reports:
        lines.append(f"TABLE {r['name']} — {r['rows']:,} rows × {r['cols']} cols "
                     f"({r['duplicate_rows']} duplicate rows, {r['memory_mb']} MB)")
        for c in r["columns"]:
            base = (f"  {c['name']} ({c['dtype']}): {c['nulls']} nulls "
                    f"({c['null_pct']}%), {c['unique']} unique")
            if "median" in c:
                base += (f", median {c['median']}, mean {c['mean']}, "
                         f"range {c['min']}–{c['max']}, {c['outliers_iqr']} IQR outliers")
            elif c.get("top_values"):
                tops = ", ".join(f"'{t['value']}'×{t['count']}" for t in c["top_values"])
                base += f", top: {tops}"
            lines.append(base)
        for cr in r["correlations"]:
            strength = "strong" if abs(cr["r"]) > 0.7 else "moderate"
            lines.append(f"  correlation: {cr['a']} ↔ {cr['b']} r={cr['r']} ({strength})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Auto-insights (deterministic pattern hunt)
# ---------------------------------------------------------------------------

@dataclass
class Insight:
    kind: str          # "outlier" | "spike" | "drop" | "correlation" | "dominance" | "missing"
    title: str
    detail: str
    table: str
    score: float = 0.0  # rough interestingness, for ranking


def insight_question(ins: Insight) -> str:
    """The question that investigates this insight.

    Shared by the suggestion pills and the dashboard's insight cards, so a card
    and its pill always ask the agent for the same thing.
    """
    if ins.kind in ("spike", "drop"):
        # title is "<metric> dropped 86% in <when>"
        tense = "spiked" if ins.kind == "spike" else "dropped"
        word = "spike" if ins.kind == "spike" else "drop"
        metric, _, rest = ins.title.partition(f" {tense} ")
        pct, _, where = rest.partition(" in ")
        if metric and pct and where:
            return f"What drove the {pct} {word} in {yname(metric)} in {where}?"
        return f"What drove the {ins.title}?"
    if ins.kind == "correlation":
        # title is "<a> and <b> move together (r=0.73)"
        head = ins.title.split(" (")[0].split(" move ")[0]
        pair = head.split(" and ")
        if len(pair) == 2:
            return (f"How strong is the link between {yname(pair[0])} and "
                    f"{yname(pair[1])}? Plot it.")
        return f"Is the relationship in {head} real? Plot it."
    if ins.kind == "missing":
        col, _, rest = ins.title.partition(" is ")
        pct = rest.split(" ")[0] if rest else ""
        if col and pct:
            return f"{pct} of {yname(col)} is missing — does the answer hold?"
        return f"Does the missing data in {col} change the answer?"
    if ins.kind == "dominance":
        val, _, rest = ins.title.partition(" is ")
        if val and rest:
            return f"{val} is {rest} — does that skew the numbers?"
        return f"Does the skew in {ins.title} change the numbers?"
    if ins.kind == "outlier":
        col = ins.title.split(" in ")[-1]
        return f"Which rows are extreme on {yname(col)}, and are they real?"
    return f"Tell me more about {ins.title}"


def find_insights(tables: dict[str, pd.DataFrame], max_per_table: int = 6) -> list[Insight]:
    """Sweep every table for notable patterns. Deterministic; ranked."""
    insights: list[Insight] = []
    for tname, df in tables.items():
        insights.extend(_table_insights(tname, df, max_per_table))
    insights.sort(key=lambda i: -i.score)
    return insights[:max_per_table * max(1, len(tables))]


def _table_insights(tname: str, df: pd.DataFrame, cap: int) -> list[Insight]:
    out: list[Insight] = []
    if len(df) < 8:
        return out
    # identifiers never make interesting insights (order_id vs revenue = noise)
    id_cols = {c for c in df.columns if _is_id_like(c, df[c])}
    keys = {c for c in df.columns
            if c not in id_cols and _entity_key_like(df[c])}
    # entity keys are groupable categories, never measures: exclude them
    # from the metric pool (they may still be grouped on elsewhere).
    nonmeasure = id_cols | keys

    # heavy missing columns
    for col in df.columns:
        pct = df[col].isna().mean()
        if pct >= 0.4:
            out.append(Insight(
                "missing", f"{col} is {pct:.0%} empty",
                f"Column '{col}' in {tname} is missing {pct:.0%} of its values "
                f"({int(df[col].isna().sum())} rows). Treat it as sparse.",
                tname, score=pct * 3))

    # temporal spikes & drops on the first datetime + numeric pair
    date_col = next((c for c in df.columns
                     if c not in id_cols
                     and (pd.api.types.is_datetime64_any_dtype(df[c])
                          or (_is_text(df[c]) and _mostly_dates(df[c])))), None)
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in nonmeasure]
    if date_col and num_cols:
        metric = num_cols[0]
        t = df[[date_col, metric]].dropna().copy()
        if len(t) >= 8:
            t = t.sort_values(date_col)
            dates = pd.to_datetime(t[date_col], errors="coerce")
            span_days = (dates.max() - dates.min()).days
            freq = "D" if span_days <= 70 else ("W" if span_days <= 400 else "M")
            t["bucket"] = dates.dt.to_period(freq)
            series = t.groupby("bucket")[metric].sum()
            if len(series) >= 3:
                pct_change = series.pct_change().dropna()
                if len(pct_change):
                    biggest = pct_change.abs().idxmax()
                    change = pct_change[biggest]
                    loc = pct_change.index.get_loc(biggest)
                    prev = series[pct_change.index[loc - 1]] if loc > 0 else series.iloc[0]
                    word = "spiked" if change > 0 else "dropped"
                    unit = {"D": "day", "W": "week", "M": "month"}[freq]
                    # a period reads like "2025-05-05/2025-05-11"; name it instead
                    where = (f"the {unit} of {biggest.start_time.date()}" if freq == "W"
                             else f"{biggest}")
                    out.append(Insight(
                        "spike" if change > 0 else "drop",
                        f"{metric} {word} {abs(change):.0%} in {where}",
                        f"In {tname}, per-{unit} {metric} {word} {abs(change):.0%} in {biggest} "
                        f"(from {_short_num(prev)} to {_short_num(series[biggest])}). "
                        f"Worth investigating what happened in that {unit}.",
                        tname, score=min(1.0, abs(change))))

    # numeric outliers via IQR
    for col in num_cols[:4]:
        clean = df[col].dropna()
        if len(clean) < 8:
            continue
        q1, q3 = clean.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr <= 0:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out_mask = (clean < lo) | (clean > hi)
        n_out = int(out_mask.sum())
        if n_out and n_out <= 0.05 * len(clean):
            extreme = clean[out_mask].max()
            out.append(Insight(
                "outlier", f"{n_out} extreme values in {col}",
                f"'{col}' in {tname} has {n_out} IQR outliers "
                f"(largest {_short_num(extreme)} vs median {_short_num(clean.median())}). "
                f"Ask: 'show the rows with the largest {col}' to see them.",
                tname, score=min(1.0, n_out / max(1, len(clean)) * 8)))

    # dominance: one category dwarfs the rest
    cat_cols = [c for c in df.columns
                if c not in id_cols
                and not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) <= 20]
    for col in cat_cols[:3]:
        vc = df[col].value_counts(dropna=True)
        if len(vc) >= 2 and vc.iloc[0] >= 0.7 * len(df):
            out.append(Insight(
                "dominance", f"'{vc.index[0]}' is {vc.iloc[0]/len(df):.0%} of {col}",
                f"In {tname}, {col} is dominated by '{vc.index[0]}' "
                f"({vc.iloc[0]:,} of {len(df):,} rows). Keep that skew in mind "
                f"when comparing groups.",
                tname, score=vc.iloc[0] / len(df)))

    # strong correlations
    if len(num_cols) >= 2:
        corr = df[num_cols].corr(numeric_only=True)
        seen = set()
        for a in num_cols:
            for b in num_cols:
                if a >= b:
                    continue
                r = corr.loc[a, b]
                if pd.notna(r) and abs(r) > 0.7 and (a, b) not in seen:
                    seen.add((a, b))
                    direction = "positively" if r > 0 else "negatively"
                    out.append(Insight(
                        "correlation",
                        f"{a} and {b} move {'together' if r > 0 else 'opposite'} "
                        f"(r={r:.2f})",
                        f"In {tname}, {a} and {b} are {direction} correlated "
                        f"(r={r:.2f}). Ask: 'plot {yname(b)} against {yname(a)}' "
                        f"to see it.", tname, score=abs(r)))

    out.sort(key=lambda i: -i.score)
    return out[:cap]


def _is_text(s: pd.Series) -> bool:
    """Text column under both pandas 2 (object) and pandas 3 (str dtype)."""
    return s.dtype == object or pd.api.types.is_string_dtype(s)


def _mostly_dates(s: pd.Series) -> bool:
    sample = s.dropna().astype(str).head(50)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() > 0.8


def _short_num(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    a = abs(v)
    if a >= 1e6:
        return f"{v/1e6:.1f}M"
    if a >= 1e3:
        return f"{v/1e3:.1f}k"
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _shorten(s: str, n: int = 22) -> str:
    return s if len(s) <= n else s[:n - 1] + "…"


def _safe_round(v) -> float:
    try:
        return round(float(v), 4)
    except (TypeError, ValueError):
        return 0.0


def yname(col: str) -> str:
    return str(col).replace("_", " ")


# ---------------------------------------------------------------------------
# Example-question suggestions — data-aware, ID-proof
# ---------------------------------------------------------------------------

_ID_TOKENS = ("id", "uuid", "guid", "code", "sku", "ssn", "phone",
              "postal", "zip", "index", "no", "num_", "_no")


def _is_id_like(col: str, s: pd.Series) -> bool:
    """Identifiers make meaningless averages/breakdowns — 'avg Product ID'.
    Name heuristics plus value fingerprints. Continuous FLOATS are metrics
    (prices, amounts) even when every value is unique — only integer-like
    mostly-unique sequences are treated as IDs on fingerprints alone."""
    name = str(col).lower().replace(" ", "_").replace("-", "_")
    if any(t in name for t in _ID_TOKENS) and "rate" not in name:
        return True
    clean = s.dropna()
    if len(clean) < 8:
        return False
    uniq_ratio = clean.nunique() / len(clean)
    if pd.api.types.is_integer_dtype(clean) and uniq_ratio > 0.9:
        vals = clean.sort_values()
        diffs = vals.diff().dropna()
        if (diffs == 1).mean() > 0.9:     # 1,2,3,... sequence
            return True
        if uniq_ratio > 0.98 and len(clean) >= 50:
            return True                   # 50+ mostly-unique integers: an ID
    return False


def _entity_key_like(s: pd.Series) -> bool:
    """Dense integer entity keys (Store 1-45, ward 1-30) and float-coded
    categoricals (restecg 0.0/1.0/2.0): groupable, not a measure. Differs
    from _is_id_like: fully dense (each value repeats many times), so
    uniqueness heuristics never fire — but summing 45 store numbers is still
    nonsense."""
    if not (pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s)):
        return False
    clean = s.dropna()
    n = len(clean)
    if n < 40:
        return False
    k = clean.nunique()
    if not (2 <= k <= 50 and n / k >= 50 and not set(clean.unique()) <= {0, 1}):
        return False
    # floats must be integral-valued (0.0/1.0/2.0) to be a coded category;
    # fractional values (0.5, 2.3) mean a real measure
    return bool((clean % 1 == 0).all())


def _bin_friendly(s: pd.Series) -> bool:
    """Continuous numeric with real spread — binning tells a story."""
    clean = pd.to_numeric(s, errors="coerce").dropna()
    if len(clean) < 30 or clean.nunique() < 12:
        return False
    q1, q3 = clean.quantile([0.25, 0.75])
    return (q3 - q1) > 1e-9


def suggest_questions(tables: dict[str, pd.DataFrame],
                      active: str | None = None) -> list[str]:
    """4-6 example questions that fit the DATA, never 'average of Product_ID'.

    Draws on the deterministic insight sweep + column fingerprints. Every pill
    is shaped like the question an analyst would open with — subject, measure,
    and the comparison or the so-what — never a bare chart command. They are
    kept close to 60 characters because the chat renders them as pills.
    """
    if not tables:
        return []
    tname = active if active in tables else next(iter(tables))
    df = tables[tname]
    cols = list(df.columns)
    id_cols = {c for c in cols if _is_id_like(c, df[c])}
    # dense entity keys (Store 1-45) group like categories, never measure
    keys = {c for c in cols if c not in id_cols and _entity_key_like(df[c])}
    num_cols = [c for c in df.select_dtypes("number").columns
                if c not in id_cols and c not in keys
                and df[c].nunique(dropna=True) > 1]
    cat_cols = [c for c in df.columns
                if c not in id_cols and c not in keys
                and not pd.api.types.is_numeric_dtype(df[c])
                and 1 < df[c].nunique(dropna=True) <= 25] + sorted(keys)
    flags = [c for c in num_cols if set(df[c].dropna().unique()) <= {0, 1}]
    cont = [c for c in num_cols if _bin_friendly(df[c])]
    date_col = next((c for c in df.columns if c not in id_cols
                     and (pd.api.types.is_datetime64_any_dtype(df[c])
                          or (_is_text(df[c]) and _mostly_dates(df[c])))), None)

    # which category actually changes the metric? pick the highest-spread pair
    def spread(cnum: str, ccat: str) -> float:
        g = df.groupby(ccat)[cnum].mean()
        if len(g) < 2 or g.isna().all():
            return 0.0
        rng = float(g.max() - g.min())
        return rng / (abs(float(g.mean())) + 1e-9)

    questions: list[str] = []
    seen: set[str] = set()

    def add(q: str):
        if q and q not in seen and len(questions) < 6:
            seen.add(q)
            questions.append(q)

    # 1. deterministic insights first — these are the real story
    for ins in find_insights({tname: df}):
        # only the kinds with a specific question make good pills; 'outlier'
        # cards (dashboard only) fall through to the generic fallback below.
        if ins.kind in ("spike", "drop", "correlation", "missing", "dominance"):
            add(insight_question(ins))
        if len(questions) >= 3:
            break

    # 2. comparison with the most informative categorical
    if num_cols and cat_cols:
        # true measures only: a 0/1 flag's rate question is covered above —
        # as a breakdown metric ("break down promoted by dept") it reads nonsense
        meas = [c for c in num_cols if c not in flags]
        if meas:
            best = max(((n, c, spread(n, c)) for n in meas[:6]
                        for c in cat_cols[:6]), key=lambda t: t[2])
            if best[2] > 0.15:
                # "break down … by" is the vocabulary the answer rules key on:
                # it asks for the whole ranked breakdown, not a single winner
                add(f"Break down {yname(best[0])} by {yname(best[1])} — who leads, "
                    f"and by how much?")

    # 3. rate question for 0/1 flags
    if flags:
        add(f"What percentage have {yname(flags[0])}, and where is it highest?")

    # 4. banding: do the big rows behave differently from the small ones
    if cont and (flags or cat_cols):
        grp = flags[0] if flags else cat_cols[0]
        add(f"Split {yname(cont[0])} into bands — does {yname(grp)} differ across "
            f"them?")

    # 5. trend over time — always worth a slot when a date exists
    if date_col and num_cols:
        add(f"How has {yname(num_cols[0])} moved over {yname(date_col)}, and lately?")

    # 6. distribution question
    if cont:
        add(f"What's the distribution of {yname(cont[0])} — typical and extreme?")

    # 7. fallbacks that are still data-aware (never IDs)
    if cat_cols:
        add(f"Which {yname(cat_cols[0])} appears most, and does it skew {yname(num_cols[0])}?"
            if num_cols else f"Which {yname(cat_cols[0])} appears most often?")

    return questions[:6]
