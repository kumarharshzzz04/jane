"""Tests for SQL whitelist validator."""
import pytest

from jane.validator import SQLValidationError, extract_sql, validate_sql


def test_simple_select_ok():
    sql = "SELECT region, SUM(amount) AS total FROM sales GROUP BY region ORDER BY total DESC"
    assert validate_sql(sql).lower().startswith("select")


def test_cte_ok():
    sql = "WITH t AS (SELECT 1 AS x) SELECT x FROM t"
    validate_sql(sql)


def test_reject_insert():
    with pytest.raises(SQLValidationError):
        validate_sql("INSERT INTO t VALUES (1)")


def test_reject_drop():
    with pytest.raises(SQLValidationError):
        validate_sql("DROP TABLE sales")


def test_reject_attach():
    with pytest.raises(SQLValidationError):
        validate_sql("ATTACH '/etc/passwd' AS x")


def test_reject_pragma_and_read_file():
    with pytest.raises(SQLValidationError):
        validate_sql("SELECT * FROM read_csv_auto('data.csv')")
    with pytest.raises(SQLValidationError):
        validate_sql("PRAGMA database_list")


def test_reject_filesystem_table_functions():
    """Filenames *and* the functions that read them, not just `read_*`.

    `glob('C:/Windows/*')` listed a real directory through read-only SQL and
    `sniff_csv()` returned a local file's contents; neither name was blocked.
    """
    for sql in [
        "SELECT * FROM glob('/etc/*')",
        "SELECT * FROM sniff_csv('/etc/passwd')",
        "SELECT * FROM parquet_scan('data.parquet')",
        "SELECT * FROM parquet_metadata('data.parquet')",
        "SELECT * FROM sqlite_scan('x.db', 't')",
    ]:
        with pytest.raises(SQLValidationError):
            validate_sql(sql)


def test_normal_queries_survive_the_filesystem_tokens():
    """The new tokens must not eat ordinary analysis SQL."""
    for sql in [
        "SELECT region, SUM(amount) AS total FROM sales GROUP BY region",
        "WITH t AS (SELECT * FROM sales) SELECT COUNT(*) FROM t",
        "SELECT a.region FROM sales a JOIN sales b ON a.region = b.region",
        "SELECT * FROM sales WHERE region NOT LIKE 'glob%'",
    ]:
        validate_sql(sql)


def test_columns_named_after_statement_keywords_are_queryable():
    """A column is not a statement. `SELECT load FROM kiosks` used to be
    refused as "forbidden token: LOAD", so any table with a `load`/`call`/
    `update`/`set`/`copy` column was unanalysable — the round trip then burned
    retries trying to rewrite a query that was already correct.
    """
    for col in ["load", "call", "set", "copy", "import", "export", "drop",
                "alter", "create", "attach", "detach", "begin", "commit",
                "rollback", "reset", "vacuum", "checkpoint", "execute",
                "prepare", "transaction", "pragma", "truncate", "show"]:
        validate_sql(f"SELECT {col} FROM logs")
        validate_sql(f"SELECT SUM({col}) AS total FROM logs GROUP BY region")


def test_dml_column_names_need_quoting():
    """insert/update/delete/merge stay blocked by name — measured: DuckDB runs
    `WITH x AS (…) INSERT/UPDATE/DELETE/MERGE`, so a WITH prefix does get them
    past the start check. Quoting is the escape hatch, and it works because
    quoted identifiers are stripped before the scan.
    """
    for col in ["insert", "update", "delete", "merge"]:
        with pytest.raises(SQLValidationError):
            validate_sql(f"SELECT {col} FROM logs")
        validate_sql(f'SELECT "{col}" FROM logs')


def test_statement_keywords_still_rejected_as_statements():
    """The other half of the contract: dropping those tokens from the list must
    not make a single write statement reachable. Position is the guard now, so
    every one of these is refused for what it *is*, not for what it names.
    """
    for sql in [
        "INSERT INTO t VALUES (1)",
        "UPDATE t SET x = 1",
        "DELETE FROM t",
        "DROP TABLE sales",
        "ALTER TABLE sales ADD COLUMN z INT",
        "CREATE TABLE t AS SELECT * FROM sales",
        "ATTACH '/etc/passwd' AS x",
        "DETACH x",
        "COPY sales TO '/tmp/x.csv'",
        "EXPORT DATABASE '/tmp/d'",
        "IMPORT DATABASE '/tmp/d'",
        "INSTALL httpfs",
        "LOAD httpfs",
        "CALL pragma_show('memory_limit')",
        "PRAGMA database_list",
        "SET enable_external_access = true",
        "RESET enable_external_access",
        "VACUUM",
        "CHECKPOINT",
        "EXECUTE stmt",
        "PREPARE stmt AS SELECT 1",
        "BEGIN TRANSACTION",
        "COMMIT",
        "ROLLBACK",
        "/* c */ DROP TABLE sales",
        "  \n\t delete from t",
        # the one shape where position alone is not enough
        "WITH x AS (SELECT 1) INSERT INTO t SELECT * FROM x",
        "WITH x AS (SELECT 1) UPDATE t SET a = 1",
        "WITH x AS (SELECT 1) DELETE FROM t",
        "WITH x AS (SELECT 1) MERGE INTO t USING x ON TRUE WHEN MATCHED THEN UPDATE SET a = 1",
    ]:
        with pytest.raises(SQLValidationError):
            validate_sql(sql)


def test_no_smuggling_a_second_statement():
    """Semicolon variants must not get a second statement past the start check."""
    for sql in [
        "SELECT 1; DROP TABLE sales",
        "SELECT 1;  \n DROP TABLE sales",
        "select 1;delete from t",
        "SELECT 1 ; \n DROP TABLE sales",
    ]:
        with pytest.raises(SQLValidationError):
            validate_sql(sql)


def test_reject_multiple_statements():
    with pytest.raises(SQLValidationError):
        validate_sql("SELECT 1; DROP TABLE sales")


def test_reject_empty():
    with pytest.raises(SQLValidationError):
        validate_sql("   ")


def test_semicolon_inside_string_is_fine():
    sql = "SELECT * FROM t WHERE note = 'a;b'"
    validate_sql(sql)


def test_trailing_semicolon_stripped():
    assert validate_sql("SELECT 1;").startswith("SELECT")


def test_extract_sql_from_fences():
    text = "Here you go:\n```sql\nSELECT 1\n```"
    assert extract_sql(text) == "SELECT 1"


def test_extract_sql_bare():
    assert extract_sql("SELECT 42") == "SELECT 42"
