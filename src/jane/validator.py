"""SQL whitelist validation — only single read-only SELECT/WITH statements over registered views."""
from __future__ import annotations

import re

# Statements we allow (first keyword)
_ALLOWED_START = re.compile(r"^\s*(with|select)\b", re.IGNORECASE)

# Tokens that mean "not a read-only query" anywhere in the statement (outside quotes).
#
# The list is deliberately narrow: only names that are still reachable *after*
# the start check — DML verbs that a WITH clause can prefix, and the in-expression
# file/network readers. Everything else (DROP/ATTACH/COPY/LOAD/CALL/SET/INSTALL/
# PRAGMA/VACUUM/…) is blocked by position, since the statement must start with
# SELECT/WITH and may not contain a second statement, so those words have nowhere
# to be a statement. Token-blocking them bought nothing and broke real data:
# `SELECT load FROM kiosks`, `SELECT call FROM logs`, `SELECT copy FROM t` and
# `SELECT set FROM t` were all refused as "forbidden token", so a table with such
# a column could not be queried at all (measured) — the round trip then spent
# retries rewriting a query that was already correct.
#
# A column named insert/update/delete/merge is still refused unquoted; quoting it
# works (`SELECT "update" FROM t`) because quoted identifiers are stripped before
# this scan and DuckDB resolves them as plain names.
#
# What a name list cannot do: validate_sql strips string literals before scanning,
# so an in-expression executor such as DuckDB's `query('…')` hides its payload from
# any scan here. The guarantee for that is the connection itself —
# `enable_external_access=false` (engine.SQLEngine.__init__) plus the engine only
# ever holding in-memory frames, never the user's own files.
_FORBIDDEN = [
    # measured: `WITH x AS (…) INSERT/UPDATE/DELETE/MERGE …` all execute, so a
    # WITH prefix gets these past the start check — they must stay token-blocked
    r"\binsert\b", r"\bupdate\b", r"\bdelete\b", r"\bmerge\b",
    r"\bpg_read\b", r"\bread_[a-z_0-9]*", r"\bwrite_[a-z_0-9]*",
    r"\bhttpfs\b",
    # DuckDB table functions that reach the filesystem without a read_/write_
    # prefix. They are named here rather than discovered: `glob()` listed a real
    # 47-entry directory and `sniff_csv()` returned a local file's contents
    # through read-only SQL. The connection is also opened with
    # enable_external_access=false, so this is the outer of two barriers.
    r"\bglob\b", r"\bsniff_csv\b", r"\bparquet_scan\b", r"\bparquet_metadata\b",
    r"\bcsv_scan\b", r"\biceberg_scan\b", r"\bdelta_scan\b", r"\bfile_scan\b",
    r"\bpostgres_scan\b", r"\bmysql_scan\b", r"\bsqlite_scan\b",
]
_FORBIDDEN_RE = re.compile("|".join(_FORBIDDEN), re.IGNORECASE)

# multiple statements separated by ; are not allowed (except trailing)
_SEMI = ";"


class SQLValidationError(ValueError):
    pass


def _strip_strings_and_comments(sql: str) -> str:
    """Remove quoted literals and comments so token scan doesn't false-positive."""
    out = []
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch in ("'", '"'):
            quote = ch
            i += 1
            while i < n:
                if sql[i] == quote:
                    if i + 1 < n and sql[i + 1] == quote:  # escaped quote
                        i += 2
                        continue
                    break
                i += 1
            i += 1
            out.append("''")
        elif ch == "-" and sql[i:i + 2] == "--":
            while i < n and sql[i] != "\n":
                i += 1
        elif ch == "/" and sql[i:i + 2] == "/*":
            i = sql.find("*/", i + 2)
            i = n if i == -1 else i + 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def validate_sql(sql: str) -> str:
    """Validate a single read-only SELECT/WITH statement. Returns cleaned SQL.

    Raises SQLValidationError otherwise.
    """
    if not sql or not sql.strip():
        raise SQLValidationError("empty SQL")
    sql = sql.strip().rstrip(_SEMI).strip()

    # reject multiple statements
    body = _strip_strings_and_comments(sql)
    if _SEMI in body:
        raise SQLValidationError("multiple SQL statements are not allowed")

    if not _ALLOWED_START.match(body):
        raise SQLValidationError("only SELECT/WITH (read-only) queries are allowed")

    m = _FORBIDDEN_RE.search(body)
    if m:
        raise SQLValidationError(f"forbidden token in SQL: {m.group(0).upper()}")

    # hard size cap (runaway queries / prompt abuse)
    if len(sql) > 10_000:
        raise SQLValidationError("SQL too long (max 10k chars)")

    return sql


# --------------------------------------------------------------------------
# Code block parsing (assistant JSON responses)
# --------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:sql|python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code_blocks(text: str) -> list[tuple[str, str]]:
    """Return [(lang, code)] from fenced blocks; bare text yields ('text', text)."""
    blocks = []
    for m in _FENCE_RE.finditer(text):
        lang = (m.group(0).split("\n")[0][3:] or "").strip().lower()
        blocks.append((lang or "code", m.group(1).rstrip()))
    if not blocks:
        stripped = text.strip()
        if stripped:
            blocks.append(("text", stripped))
    return blocks


def extract_sql(text: str) -> str | None:
    """Pull the first SQL-looking block out of an assistant message."""
    for lang, code in extract_code_blocks(text):
        if lang == "sql":
            return code
        if lang in ("text", "code") and re.match(r"^\s*(with|select)\b", code, re.IGNORECASE):
            return code
    return None
