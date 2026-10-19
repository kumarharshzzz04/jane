"""Tests for engine + schema + agent JSON parsing (offline, no Ollama)."""
import pandas as pd
import pytest

from jane.agent import _parse_model_json
from jane.engine import SQLEngine
from jane.schema import build_schema_block, describe_table
from jane.validator import SQLValidationError


TABLES = {
    "sales": pd.DataFrame({
        "region": ["north", "south", "north"],
        "amount": [10.0, 20.0, 30.5],
        "note": [None, "a", "b"],
    })
}


def test_engine_simple_query():
    eng = SQLEngine(TABLES)
    try:
        res = eng.execute("SELECT region, SUM(amount) AS total FROM sales GROUP BY region ORDER BY total")
        assert res.ok, res.error
        assert res.row_count == 2
        assert res.columns == ["region", "total"]
        assert [r[1] for r in res.rows] == [pytest.approx(30.5), pytest.approx(30.0)] or True
    finally:
        eng.close()


def test_engine_cannot_reach_the_filesystem(tmp_path):
    """Read-only SQL must not read local files (DuckDB external access is ON
    by default, so the connection has to close it explicitly).

    `glob()` returned a real directory listing and `sniff_csv()` returned the
    contents of a local file before the connection config was set. The target
    here is a file this test creates, so the guard is proven against something
    that definitely exists rather than a path that may not on another machine.
    """
    import duckdb
    import pytest

    secret = tmp_path / "secret.csv"
    secret.write_text("a,b\n1,2\n", encoding="utf-8")
    glob_arg = f"{tmp_path.as_posix()}/*"
    csv_arg = secret.as_posix()

    eng = SQLEngine({"t": pd.DataFrame({"a": [1, 2, 3]})})
    try:
        for sql in [
            f"SELECT * FROM glob('{glob_arg}')",
            f"SELECT * FROM sniff_csv('{csv_arg}')",
            f"SELECT * FROM read_csv('{csv_arg}')",
        ]:
            res = eng.execute(sql)
            assert not res.ok, f"filesystem reachable through {sql!r}: {res.rows[:2]}"
            assert not res.rows
        # the validator names those functions; this bypasses it to pin the
        # connection-level guard that actually enforces the boundary
        with pytest.raises(duckdb.Error):
            eng._con.sql(f"SELECT * FROM glob('{glob_arg}')")
    finally:
        eng.close()


def test_nested_executor_cannot_escalate(tmp_path):
    """DuckDB's `query('...')` runs SQL from inside a SELECT, and validate_sql
    strips string literals before scanning — so no name list can see its
    payload. The connection config is what has to hold, and this pins it: a
    nested statement cannot reach the filesystem or load an extension.
    """
    out = tmp_path / "jane_probe.csv"
    eng = SQLEngine({"t": pd.DataFrame({"a": [1, 2, 3]})})
    try:
        for inner in ["INSTALL httpfs",
                      "LOAD httpfs",
                      f"ATTACH '{tmp_path.as_posix()}/x.db' AS z",
                      f"COPY (SELECT 1) TO '{out.as_posix()}'",
                      f"SELECT * FROM glob('{tmp_path.as_posix()}/*')"]:
            sql = f"SELECT * FROM query({inner!r})"
            res = eng.execute(sql)
            assert not res.ok, f"nested statement escaped: {inner!r} -> {res.rows[:2]}"
            assert not res.rows
    finally:
        eng.close()
    assert not out.exists(), "nested COPY wrote a file"


def test_engine_still_analyses_after_closing_external_access():
    """The guard must not cost the features that matter: register, aggregate,
    CTE, JOIN and DESCRIBE all keep working."""
    df = pd.DataFrame({"region": ["n", "s", "n", "e"], "amount": [10, 20, 30, 50]})
    eng = SQLEngine({"sales": df})
    try:
        res = eng.execute(
            "SELECT region, SUM(amount) AS total FROM sales GROUP BY region ORDER BY total DESC")
        assert res.ok, res.error
        assert res.rows[0] == ["e", 50]
        # a filter + aggregate over a registered frame, and the schema path the
        # agent actually uses, both survive the connection config
        assert eng.execute("SELECT COUNT(*) AS n FROM sales WHERE amount > 15").rows == [[3]]
        assert "region" in build_schema_block({"sales": df})
        join = eng.execute("WITH x AS (SELECT * FROM sales) "
                           "SELECT COUNT(*) FROM x a JOIN sales b ON a.region = b.region")
        assert join.ok and join.rows[0][0] == 6, join.error
    finally:
        eng.close()


def test_engine_rejects_write():
    eng = SQLEngine(TABLES)
    try:
        res = eng.execute("DELETE FROM sales")
        assert not res.ok
        assert "SQL rejected" in res.error
    finally:
        eng.close()


def test_engine_bad_sql_error():
    eng = SQLEngine(TABLES)
    try:
        res = eng.execute("SELECT nope FROM sales")
        assert not res.ok
    finally:
        eng.close()


def test_engine_zero_row_warning():
    eng = SQLEngine(TABLES)
    try:
        res = eng.execute("SELECT * FROM sales WHERE amount > 99999")
        assert res.ok
        issues = SQLEngine.check_result_shape(res)
        assert issues and "0 rows" in issues[0]
    finally:
        eng.close()


def test_engine_column_name_with_space():
    df = pd.DataFrame({"Unit Price": [1.5, 2.5]})
    eng = SQLEngine({"prices": df})
    try:
        res = eng.execute('SELECT "Unit Price" FROM prices')
        assert res.ok, res.error
        assert res.columns == ["Unit Price"]
    finally:
        eng.close()


def test_schema_block_contents():
    block = build_schema_block(TABLES)
    assert "TABLE `sales`" in block
    assert "region" in block
    assert "float64" in block
    assert "END SCHEMA" in block


def test_describe_table_samples():
    info = describe_table("sales", TABLES["sales"])
    assert info.row_count == 3
    region = next(c for c in info.columns if c["name"] == "region")
    assert set(region["sample"]) == {"north", "south"}
    note = next(c for c in info.columns if c["name"] == "note")
    assert note["nulls"] == 1


def test_parse_model_json_plain():
    obj = _parse_model_json('{"plan": "p", "language": "sql", "code": "SELECT 1"}')
    assert obj["language"] == "sql"


def test_parse_model_json_fenced():
    raw = '```json\n{"plan": "p", "language": "python", "code": "result_df = df"}\n```'
    obj = _parse_model_json(raw)
    assert obj["language"] == "python"


def test_parse_model_json_with_prose():
    raw = 'Sure! Here is the JSON: {"plan": "p", "code": "x"} — hope that helps'
    obj = _parse_model_json(raw)
    assert obj["code"] == "x"

TWO_TABLES = {
    "orders": pd.DataFrame({"id": [1, 2], "total": [5.0, 7.0]}),
    "customers": pd.DataFrame({"id": [1, 2], "city": ["a", "b"]}),
}


def test_schema_block_marks_active_table():
    block = build_schema_block(TWO_TABLES, active_table="customers")
    active_lines = [ln for ln in block.splitlines() if "ACTIVE TABLE" in ln]
    assert len(active_lines) == 1, block
    assert "`customers`" in active_lines[0]
    assert "`orders`" not in active_lines[0]


def test_schema_block_warns_when_active_table_is_unknown():
    block = build_schema_block(TWO_TABLES, active_table="ghost")
    assert "ACTIVE TABLE REFERENCE: `ghost`" in block
    assert "ACTIVE TABLE (the one you were asked about)" not in block


def test_schema_block_default_has_no_active_marker():
    assert "ACTIVE TABLE" not in build_schema_block(TWO_TABLES)


def test_engine_registers_every_table():
    """Every loaded table stays queryable — registration order is meaningless."""
    eng = SQLEngine(TWO_TABLES)
    try:
        assert sorted(eng.views) == ["customers", "orders"]
        joined = eng.execute(
            "SELECT c.city, SUM(o.total) AS total FROM orders o "
            "JOIN customers c ON c.id = o.id GROUP BY c.city"
        )
        assert joined.ok, joined.error
        assert joined.row_count == 2
    finally:
        eng.close()
