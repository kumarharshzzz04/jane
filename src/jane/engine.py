"""DuckDB-backed SQL engine — the primary analysis path (NOT retrieval)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from .logging_setup import get_logger
from .schema import register_in_duckdb, _jsonable
from .validator import SQLValidationError, validate_sql

log = get_logger("jane.engine")

MAX_QUERY_ROWS = 10_000   # hard cap on rows a single query may return


@dataclass
class QueryResult:
    ok: bool
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    error: Optional[str] = None
    sql: Optional[str] = None


class SQLEngine:
    """Owns an in-memory DuckDB connection with registered DataFrames."""

    def __init__(self, tables: dict[str, pd.DataFrame]):
        import duckdb  # noqa: local import keeps import cost off the UI path

        # enable_external_access=false is the load-bearing guard, not a nicety:
        # DuckDB's external access is ON by default, which left read-only SQL
        # able to reach the local filesystem through table functions the
        # validator's token list does not name — `glob('C:/Windows/*')` listed a
        # real directory and `sniff_csv('…/win.ini')` returned that file's
        # contents (both verified). Closing it at the connection kills the whole
        # class in one place (file readers, INSTALL, ATTACH, COPY) while leaving
        # registered-frame analysis — register/groupby/CTE/JOIN/DESCRIBE —
        # unchanged (verified). The validator's read_* tokens stay as a second,
        # independent barrier.
        self._con = duckdb.connect(":memory:", config={"enable_external_access": "false"})
        self.views = register_in_duckdb(self._con, tables)
        self._tables = tables

    def close(self) -> None:
        try:
            self._con.close()
        except Exception:
            pass

    def execute(self, sql: str, max_rows: int = MAX_QUERY_ROWS) -> QueryResult:
        """Validate + run a read-only query, return up to max_rows rows."""
        try:
            sql = validate_sql(sql)
        except SQLValidationError as e:
            return QueryResult(ok=False, error=f"SQL rejected: {e}", sql=sql)

        try:
            rel = self._con.sql(sql)
            out_df = rel.df()
        except Exception as e:
            return QueryResult(ok=False, error=f"{type(e).__name__}: {e}", sql=sql)

        truncated = len(out_df) > max_rows
        out_df = out_df.head(max_rows)
        cols = [str(c) for c in out_df.columns]
        rows = [
            [_jsonable(v) for v in row]
            for row in out_df.itertuples(index=False, name=None)
        ]
        return QueryResult(
            ok=True,
            columns=cols,
            rows=rows,
            row_count=int(len(out_df)),
            truncated=truncated,
            sql=sql,
        )

    # -- result-shape contract checks (the "output validator") ---------------

    @staticmethod
    def check_result_shape(res: QueryResult) -> list[str]:
        """Cheap sanity contracts on a successful result."""
        issues = []
        if res.ok and res.row_count == 0:
            issues.append("query returned 0 rows — is the filter too strict or column names wrong?")
        null_cells = 0
        for row in res.rows:
            null_cells += sum(1 for v in row if v is None)
        if res.rows and null_cells / max(1, len(res.rows) * max(1, len(res.columns))) > 0.5:
            issues.append("more than half the result cells are NULL — check joins/misspellings")
        return issues
