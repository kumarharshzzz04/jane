"""Tests for the sandbox: static analysis + real subprocess execution."""
import pandas as pd
import pytest

from jane.sandbox import analyze_code_safety, run_code


# -- static analysis ---------------------------------------------------------

def test_clean_code_passes():
    code = "import pandas as pd\nresult_df = df.head(10)\n"
    assert analyze_code_safety(code) == []


def test_block_os_import():
    errs = analyze_code_safety("import os\nos.system('rm -rf /')")
    assert errs


def test_block_subprocess():
    assert analyze_code_safety("import subprocess")


def test_block_open_call():
    assert analyze_code_safety("open('x.txt', 'w')")


def test_block_eval_exec():
    assert analyze_code_safety("eval('1+1')")
    assert analyze_code_safety("exec('x=1')")


def test_block_dunder():
    assert analyze_code_safety("df.__class__")


def test_block_requests():
    assert analyze_code_safety("import requests")


def test_syntax_error_reported():
    errs = analyze_code_safety("def oops(:\n    pass")
    assert any("SyntaxError" in e for e in errs)


# -- execution ----------------------------------------------------------------

TABLES = {
    "sales": pd.DataFrame({
        "region": ["north", "south", "north", "east"],
        "amount": [10, 20, 30, 50],
    })
}


def test_run_simple_groupby():
    code = "result_df = df.groupby('region', as_index=False)['amount'].sum()"
    res = run_code(code, TABLES, active_table="sales", timeout=60)
    assert res.ok, res.error
    assert res.tables and res.tables[0]["name"] == "result_df"
    t = res.tables[0]
    totals = {r["region"]: r["amount"] for r in t["rows"]}
    assert totals["north"] == 40


def test_run_with_print_and_stdout():
    code = "print('hello from sandbox')\nresult_df = df.head(2)"
    res = run_code(code, TABLES, active_table="sales", timeout=60)
    assert res.ok
    assert "hello from sandbox" in res.stdout


def test_runtime_error_captured():
    code = "result_df = df['missing_col'] / 0"
    res = run_code(code, TABLES, active_table="sales", timeout=60)
    assert not res.ok
    assert res.error and "Traceback" in res.error


def test_safety_rejected_before_execution():
    code = "import os\nresult_df = df"
    res = run_code(code, TABLES, active_table="sales", timeout=60)
    assert not res.ok
    assert res.safety_errors


def test_duckdb_available_in_child():
    code = "result_df = con.sql('SELECT region, SUM(amount) s FROM sales GROUP BY region ORDER BY s DESC').df()"
    res = run_code(code, TABLES, active_table="sales", timeout=60)
    assert res.ok, res.error
    assert res.tables[0]["rows"][0]["region"] == "east"


def test_chart_passthrough():
    code = (
        "charts = [{'mark': 'bar', 'encoding': {'x': {'field': 'region', 'type': 'nominal'}, "
        "'y': {'field': 'amount', 'type': 'quantitative'}}}]\n"
        "result_df = df"
    )
    res = run_code(code, TABLES, active_table="sales", timeout=60)
    assert res.ok
    assert res.charts and res.charts[0]["mark"] == "bar"


def test_timeout_kills_runaway_code():
    code = "while True:\n    pass"
    res = run_code(code, TABLES, active_table="sales", timeout=2)
    assert not res.ok
    assert res.timed_out


# -- table-name exposure & result contracts -----------------------------------

def test_table_variable_exposed_by_name():
    # models naturally write `sales.groupby(...)` using the table name
    code = "result_df = sales.groupby('region', as_index=False)['amount'].sum()"
    res = run_code(code, TABLES, active_table="sales", timeout=60)
    assert res.ok, res.error
    totals = {r["region"]: r["amount"] for r in res.tables[0]["rows"]}
    assert totals["north"] == 40


def test_table_name_sanitized():
    tables = {"My File!": pd.DataFrame({"x": [1, 2]})}
    code = "result_df = My_File_.x.sum() * 0 + len(My_File_)\nresult_df = pd.DataFrame({'n': [int(result_df)]})"
    res = run_code(code, tables, active_table="My File!", timeout=60)
    assert res.ok, res.error
    assert res.tables[0]["rows"][0]["n"] == 2


def test_table_name_never_shadow_builtin_namespaces():
    # a table literally named 'pd' must not clobber the pandas module:
    # it becomes pd_table while pd.DataFrame still works
    tables = {"pd": pd.DataFrame({"x": [1]})}
    code = "result_df = pd.DataFrame({'n': [len(pd_table)], 'x_sum': [int(pd_table['x'].sum())]})"
    res = run_code(code, tables, active_table="pd", timeout=60)
    assert res.ok, res.error
    row = res.tables[0]["rows"][0]
    assert row["n"] == 1 and row["x_sum"] == 1


def test_scalar_frame_hint_scenario_works():
    # one-row summaries must wrap scalars in lists (documented in agent prompt)
    code = ("result_df = pd.DataFrame({'rate': [df['amount'].mean()]})")
    res = run_code(code, TABLES, active_table="sales", timeout=60)
    assert res.ok, res.error
    assert res.tables[0]["rows"][0]["rate"] == 27.5
