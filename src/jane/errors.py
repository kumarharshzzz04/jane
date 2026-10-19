"""Error taxonomy — classify raw failure text into kinds, key hints off kinds.

Kinds are 1:1 with the hint rules that existed before this module (plus the
reserved-word rule), so classification changes no retry behavior: the hint
strings are byte-identical to the pre-taxonomy output. The spec
(docs/superpowers/specs/2026-09-30-agent-split-hardening-design.md §2) pins
this: coarser kinds would collapse the reserved-word hint into the generic
schema bucket and silently revert a real fix.
"""
from __future__ import annotations

import re

ERROR_KINDS = (
    "python_reset_collision",   # cannot insert X, already exists
    "python_keyerror",          # KeyError: a python-side column name
    "schema_missing_column",    # SQL Binder Error / column not found
    "sql_groupby",              # mixed plain columns with aggregates
    "sql_parse",                # Parser/Catalog/syntax errors
    "python_scalar_df",         # DataFrame built from scalars / nested lists
    "sql_multi_statement",      # more than one SQL statement
    "sql_union_alias",          # UNION alias bug
    "python_series_scalar",     # Series where a scalar is needed
    "schema_reserved_word",     # SQL reserved word used as a bare column
    "generic",                  # unmapped
)

# matchers run in this order; first hit wins
_KIND_MATCHERS: tuple[tuple[str, str], ...] = (
    ("python_reset_collision", r"cannot insert \w+, already exists"),
    ("python_keyerror", r"KeyError"),
    ("sql_multi_statement", r"multiple SQL statements are not allowed"),
    ("sql_union_alias", r"UNION ALL|UNION"),
    ("schema_missing_column",
     r"(Binder Error|UndefinedColumn|does not exist|column .* not found)"),
    ("sql_groupby",
     r"must appear in the GROUP BY|must be part of an aggregate|non-aggregated"),
    ("sql_parse", r"(Parser Error|Catalog Error|syntax error)"),
    ("python_scalar_df", r"(unhashable type|Length of values|must be 1-dimensional)"),
    ("python_series_scalar", r"(size 1 to a scalar|only length-1|cannot convert)"),
)

_RESERVED_WORDS = frozenset((
    "LEFT", "RIGHT", "ORDER", "GROUP", "LIMIT", "OFFSET", "SELECT",
    "FROM", "WHERE", "JOIN", "ON", "AS", "BY", "CASE", "WHEN", "THEN",
    "ELSE", "END", "AND", "OR", "NOT", "IN", "IS", "LIKE", "BETWEEN",
    "UNION", "ALL", "HAVING", "VALUES", "DEFAULT", "CHECK", "PRIMARY",
    "FOREIGN", "TABLE", "COLUMN", "INDEX", "UNIQUE", "DESC", "ASC",
))


def classify_error(err: str) -> str:
    """Map raw traceback/engine error text to an ERROR_KINDS member.

    The reserved-word rule runs first: it must win over the generic
    sql_parse matcher for 'syntax error at or near "left"'.
    """
    err = err or ""
    m = re.search(r'syntax error at or near "(\w+)"', err, re.I)
    if m and m.group(1).upper() in _RESERVED_WORDS:
        return "schema_reserved_word"
    for kind, pattern in _KIND_MATCHERS:
        if re.search(pattern, err, re.IGNORECASE):
            return kind
    return "generic"
