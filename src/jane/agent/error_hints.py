"""Error-kind -> actionable retry hint. Hint strings are byte-identical to the
pre-taxonomy rules; kind classification lives in jane.errors."""
import re

from ..errors import classify_error

_ERROR_HINTS: dict[str, str] = {
    # pandas 3.x reset_index collision — the most common sandbox crash
    "python_reset_collision":
     "pandas name collision: you named a result column the same as the index column being reset. "
     "FIX: use groupby('<key>', as_index=False) for the aggregation, and do NOT call "
     ".to_frame('<key>') or .rename('<key>') where <key> is the group-by column name — "
     "give the aggregated column a different name (e.g. 'total_revenue') or skip reset_index entirely.",
    "python_keyerror":
     "a column name in your code does not exist. Copy names EXACTLY from the schema (case-sensitive). "
     "If names may contain stray whitespace, normalize first: df.columns = [str(c).strip() for c in df.columns].",
    "schema_missing_column":
     "SQL references a column or function that is not in the schema. Use ONLY the exact column names "
     "listed in the schema, quoted with double quotes if they contain spaces or special characters.",
    "sql_groupby":
     "you mixed plain columns with aggregates. Every selected column must be grouped or aggregated. "
     "Simplest fix: select ONLY the grouped column and the aggregate, e.g. "
     "SELECT month, churned_subs FROM t ORDER BY churned_subs DESC LIMIT 3 — no CTE, no window "
     "functions, no percentile needed for find-the-outlier questions.",
    "sql_parse":
     "SQL syntax or unknown function. DuckDB aggregates are: SUM, COUNT, AVG, MIN, MAX, MEDIAN, "
     "COUNT(DISTINCT x). There is NO PERCENTILE_CONT — use MEDIAN or QUANTILE_CONT(x, p) if truly "
     "needed. For find-the-outlier questions skip aggregates entirely: "
     "SELECT <time_col>, <metric_col> FROM <table> ORDER BY <metric_col> DESC LIMIT 5.",
    "python_scalar_df":
     "you are building a DataFrame from scalars or nested lists. Wrap each scalar in a list: "
     "pd.DataFrame({'metric': [value]}), and aggregate with named aggregation, not .agg(list).",
    "sql_multi_statement":
     "you emitted two statements. SQL path allows exactly ONE statement — combine both groups in a "
     "single query with conditional aggregation: SELECT MEDIAN(CASE WHEN churned = 'Yes' THEN "
     "monthly_charges END) AS med_churned, MEDIAN(CASE WHEN churned = 'No' THEN monthly_charges "
     "END) AS med_stayed FROM churn; — one row, one column per group. (Or switch language to "
     "python and do df.groupby(...).median().)",
    "sql_union_alias":
     "the UNION result repeats the same value in both columns — classic bug of selecting the SAME "
     "expression under two different aliases. Compute per-group values with CASE inside one query "
     "instead: SELECT MEDIAN(CASE WHEN churned = 'Yes' THEN monthly_charges END) AS med_churned, "
     "MEDIAN(CASE WHEN churned = 'No' THEN monthly_charges END) AS med_stayed FROM churn — one row, "
     "one column per group, each with its own value.",
    "python_series_scalar":
     "you extracted a Series where a scalar is needed. Use .item(), .iloc[0], or wrap the value in a list "
     "when building result_df.",
}

_GENERIC_HINT = ("Remember: set result_df to a DataFrame; wrap scalar values in lists when building "
                 "one-row summaries; use only pd, np, the table variables, tables, df, con.")

_RESERVED_HINT_TMPL = (
    "'{col}' is a SQL RESERVED WORD — you cannot use it as a bare column "
    "name. Double-quote it everywhere: SUM(CASE WHEN \"{col}\" = 1 THEN 1 "
    "ELSE 0 END) ... GROUP BY \"{col}\". Same for any column named like a "
    "function (left, right, order, ...).")


def _error_hint(err: str) -> str:
    """Turn a raw traceback/engine error into an actionable, idiom-level retry hint."""
    kind = classify_error(err)
    if kind == "schema_reserved_word":
        col = re.search(r'syntax error at or near "(\w+)"', err or "", re.I).group(1)
        return _RESERVED_HINT_TMPL.format(col=col)
    hint = _ERROR_HINTS.get(kind)
    if hint is not None:
        return "Likely cause: " + hint
    return _GENERIC_HINT
