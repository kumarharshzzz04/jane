"""Error taxonomy: classification per kind + byte-equality vs pre-taxonomy hints.

The byte-equality half is the important one: the hint text the model receives
on retry must be EXACTLY what the old regex-table produced (spec §2).
"""
import re

from jane.errors import ERROR_KINDS, classify_error
from jane.agent.error_hints import _GENERIC_HINT, _error_hint


# ---- classification --------------------------------------------------------

def test_reset_collision_kind():
    assert classify_error("ValueError: cannot insert product, already exists") \
        == "python_reset_collision"


def test_keyerror_kind():
    assert classify_error("KeyError: 'unit_price'") == "python_keyerror"


def test_missing_column_kind():
    assert classify_error('Binder Error: Referenced column "employee_rating" not found') \
        == "schema_missing_column"


def test_groupby_kind():
    assert classify_error("column \"region\" must appear in the GROUP BY clause") \
        == "sql_groupby"


def test_parse_kind():
    assert classify_error("Parser Error: syntax error at or near \"FROM\"") \
        in ("sql_parse", "schema_reserved_word")   # reserved-word may win


def test_scalar_df_kind():
    assert classify_error("Length of values (3) does not match length of index (1)") \
        == "python_scalar_df"


def test_multi_statement_kind():
    assert classify_error("Error: multiple SQL statements are not allowed") \
        == "sql_multi_statement"


def test_union_kind():
    assert classify_error("UNION ALL result types differ") == "sql_union_alias"


def test_series_scalar_kind():
    assert classify_error("can only convert an array of size 1 to a scalar") \
        == "python_series_scalar"


def test_reserved_word_beats_parse():
    assert classify_error('Parser Error: syntax error at or near "left"') \
        == "schema_reserved_word"
    # a non-reserved word in the same shape stays sql_parse
    assert classify_error('Parser Error: syntax error at or near "zigzag"') \
        == "sql_parse"


def test_generic_kind():
    assert classify_error("Ollama timed out") == "generic"
    assert classify_error("") == "generic"


def test_every_kind_classifiable():
    """Each ERROR_KIND except reserved-word/generic has a matcher that fires."""
    kinds_seen = {classify_error(p) for _, p in [
        (None, "cannot insert x, already exists"),
        (None, "KeyError: z"),
        (None, "Binder Error: no such column"),
        (None, "must appear in the GROUP BY"),
        (None, "Parser Error: syntax error at or near \"qq\""),
        (None, "Length of values (1)"),
        (None, "multiple SQL statements are not allowed"),
        (None, "UNION ALL"),
        (None, "size 1 to a scalar"),
    ]}
    assert kinds_seen <= set(ERROR_KINDS)
    assert len(kinds_seen) >= 8


# ---- byte-equality with the pre-taxonomy behavior --------------------------

def _legacy_hint(err: str) -> str:
    """The exact pre-taxonomy logic, copied from git history (agent.py @ 07d244b)."""
    _ERROR_HINTS = [
        (r"cannot insert \w+, already exists",
         "pandas name collision: you named a result column the same as the index column being reset. "
         "FIX: use groupby('<key>', as_index=False) for the aggregation, and do NOT call "
         ".to_frame('<key>') or .rename('<key>') where <key> is the group-by column name — "
         "give the aggregated column a different name (e.g. 'total_revenue') or skip reset_index entirely."),
        (r"KeyError",
         "a column name in your code does not exist. Copy names EXACTLY from the schema (case-sensitive). "
         "If names may contain stray whitespace, normalize first: df.columns = [str(c).strip() for c in df.columns]."),
        (r"(Binder Error|UndefinedColumn|does not exist|column .* not found)",
         "SQL references a column or function that is not in the schema. Use ONLY the exact column names "
         "listed in the schema, quoted with double quotes if they contain spaces or special characters."),
        (r"must appear in the GROUP BY|must be part of an aggregate|non-aggregated",
         "you mixed plain columns with aggregates. Every selected column must be grouped or aggregated. "
         "Simplest fix: select ONLY the grouped column and the aggregate, e.g. "
         "SELECT month, churned_subs FROM t ORDER BY churned_subs DESC LIMIT 3 — no CTE, no window "
         "functions, no percentile needed for find-the-outlier questions."),
        (r"(Parser Error|Catalog Error|syntax error)",
         "SQL syntax or unknown function. DuckDB aggregates are: SUM, COUNT, AVG, MIN, MAX, MEDIAN, "
         "COUNT(DISTINCT x). There is NO PERCENTILE_CONT — use MEDIAN or QUANTILE_CONT(x, p) if truly "
         "needed. For find-the-outlier questions skip aggregates entirely: "
         "SELECT <time_col>, <metric_col> FROM <table> ORDER BY <metric_col> DESC LIMIT 5."),
        (r"(unhashable type|Length of values|must be 1-dimensional)",
         "you are building a DataFrame from scalars or nested lists. Wrap each scalar in a list: "
         "pd.DataFrame({'metric': [value]}), and aggregate with named aggregation, not .agg(list)."),
        (r"multiple SQL statements are not allowed",
         "you emitted two statements. SQL path allows exactly ONE statement — combine both groups in a "
         "single query with conditional aggregation: SELECT MEDIAN(CASE WHEN churned = 'Yes' THEN "
         "monthly_charges END) AS med_churned, MEDIAN(CASE WHEN churned = 'No' THEN monthly_charges "
         "END) AS med_stayed FROM churn; — one row, one column per group. (Or switch language to "
         "python and do df.groupby(...).median().)"),
        (r"UNION ALL|UNION",
         "the UNION result repeats the same value in both columns — classic bug of selecting the SAME "
         "expression under two different aliases. Compute per-group values with CASE inside one query "
         "instead: SELECT MEDIAN(CASE WHEN churned = 'Yes' THEN monthly_charges END) AS med_churned, "
         "MEDIAN(CASE WHEN churned = 'No' THEN monthly_charges END) AS med_stayed FROM churn — one row, "
         "one column per group, each with its own value."),
        (r"(size 1 to a scalar|only length-1|cannot convert)",
         "you extracted a Series where a scalar is needed. Use .item(), .iloc[0], or wrap the value in a list "
         "when building result_df."),
    ]
    _GENERIC = ("Remember: set result_df to a DataFrame; wrap scalar values in lists when building "
                "one-row summaries; use only pd, np, the table variables, tables, df, con.")
    m = re.search(r'syntax error at or near "(\w+)"', err or "", re.I)
    if m and m.group(1).upper() in (
            "LEFT", "RIGHT", "ORDER", "GROUP", "LIMIT", "OFFSET", "SELECT",
            "FROM", "WHERE", "JOIN", "ON", "AS", "BY", "CASE", "WHEN", "THEN",
            "ELSE", "END", "AND", "OR", "NOT", "IN", "IS", "LIKE", "BETWEEN",
            "UNION", "ALL", "HAVING", "VALUES", "DEFAULT", "CHECK", "PRIMARY",
            "FOREIGN", "TABLE", "COLUMN", "INDEX", "UNIQUE", "DESC", "ASC"):
        col = m.group(1)
        return (f"'{col}' is a SQL RESERVED WORD — you cannot use it as a bare column "
                f"name. Double-quote it everywhere: SUM(CASE WHEN \"{col}\" = 1 THEN 1 "
                f"ELSE 0 END) ... GROUP BY \"{col}\". Same for any column named like a "
                "function (left, right, order, ...).")
    for pattern, hint in _ERROR_HINTS:
        if re.search(pattern, err, re.IGNORECASE):
            return "Likely cause: " + hint
    return _GENERIC


_CASES = [
    "ValueError: cannot insert product, already exists",
    "KeyError: 'unit_price'",
    'Binder Error: Referenced column "employee_rating" not found in FROM clause!',
    'column "region" must appear in the GROUP BY clause or be used in an aggregate',
    "Parser Error: syntax error at or near \"zzz\"",
    "Parser Error: syntax error at or near \"left\"",
    "Length of values (3) does not match length of index (1)",
    "unhashable type: 'list'",
    "Error: multiple SQL statements are not allowed",
    "set operations are only supported between... UNION ALL",
    "can only convert an array of size 1 to a Python scalar",
    "only length-1 arrays can be converted",
    "totally unmapped error text",
    "",
]


def test_hints_byte_identical_to_legacy():
    for err in _CASES:
        assert _error_hint(err) == _legacy_hint(err), err
    assert _GENERIC_HINT == _legacy_hint("")


def test_reserved_word_hint_is_parameterized():
    h = _error_hint('Parser Error: syntax error at or near "order"')
    assert "'order' is a SQL RESERVED WORD" in h
    assert 'GROUP BY "order"' in h
