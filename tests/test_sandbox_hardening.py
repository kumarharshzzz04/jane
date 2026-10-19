"""Adversarial sandbox tests — escape attempts and resource limits (spec §3).

Every escape attempt must come back ok=False with a safety error. Resource
cases must terminate cleanly (timeout/error), never hang the suite and never
crash the test process. Windows has no memory rlimit — wall-clock timeout is
the only hard limit, so resource tests assert clean termination, not OOM kills.
"""
import pandas as pd
import pytest

from jane.sandbox import analyze_code_safety, run_code

TABLES = {
    "sales": pd.DataFrame({
        "region": ["north", "south", "north", "east"],
        "amount": [10, 20, 30, 50],
    })
}

SHORT = 20  # generous-but-bounded timeout for adversarial runs


# ---- static-analysis escapes: direct ---------------------------------------

@pytest.mark.parametrize("code", [
    "import os",
    "import subprocess",
    "import importlib",
    "from importlib import import_module",
    "import sys as _s",
    "import pathlib",
    "import shutil",
    "open('/etc/passwd')",
    "eval('1+1')",
    "exec('x=1')",
    "compile('1', '<s>', 'eval')",
    "__import__('os').system('echo hi')",
    "globals()",
    "locals()",
    "setattr(df, 'x', 1)",
    "getattr(df, '__class__')",
    "vars()",
    "dir(df)",
    "delattr(df, 'shape')",
    "help()",
    "input()",
    "exit()",
    "quit()",
    "breakpoint()",
    "df.__class__",
    "().__class__",
    "().__class__.__bases__[0].__subclasses__()",
    "df.__dict__",
    "().__init__",
    "x = 1; x.__class__.__name__",
])
def test_escape_attempts_blocked_statically(code):
    assert analyze_code_safety(code), f"not blocked: {code!r}"


# ---- indirect / obfuscated escapes -----------------------------------------

def test_builtins_shadow_blocked():
    # reaching builtins through a dunder name
    assert analyze_code_safety("__builtins__['eval']('1+1')")
    assert analyze_code_safety("__builtins__.eval('1+1')")


def test_chr_composition_blocked():
    # building '__import__' from chr() pieces — the dunder-Name rule and
    # BLOCKED_NAMES both fire before any value flows
    code = "f = getattr(__builtins__, chr(95)*2 + 'import' + chr(95)*2); f('os')"
    assert analyze_code_safety(code)


def test_subclass_crawl_blocked():
    code = ("for c in ().__class__.__bases__[0].__subclasses__():\n"
            "    if 'os' in str(c):\n        c.Popen('echo hi', shell=True)")
    assert analyze_code_safety(code)


def test_type_sneak_blocked():
    assert analyze_code_safety("type('X', (), {'__init__': lambda s: None})")


def test_module_attr_sneak_blocked():
    # allowed module, forbidden attribute root
    assert analyze_code_safety("import pandas as pd\npd.io")
    assert analyze_code_safety("import numpy as np\nnp.__loader__")
    assert analyze_code_safety("import json\njson.__spec__")


def test_string_built_import_blocked():
    code = "import builtins\nbuiltins.__import__('os')"
    assert analyze_code_safety(code)


# ---- execution-level: static passes but runtime must stay contained --------

def test_runtime_escape_neutralized():
    """If something slips the static net, the restricted runtime must still
    deny it — defense in depth. (Currently the static net catches all above;
    this pins the runtime denial for a lambda-shaped getattr that the AST
    whitelist approach rejects at import time.)"""
    res = run_code("result_df = df.head(1)\nx = __builtins__\n",
                   TABLES, active_table="sales", timeout=SHORT)
    assert not res.ok and res.safety_errors


def test_infinite_loop_times_out_cleanly():
    res = run_code("while True:\n    pass\n",
                   TABLES, active_table="sales", timeout=4)
    assert not res.ok
    assert res.timed_out or (res.error and ("tim" in (res.error or "").lower()
                                            or "kill" in (res.error or "").lower()))


def test_giant_allocation_terminates():
    res = run_code("x = [0] * (10**10)\nresult_df = df.head(1)\n",
                   TABLES, active_table="sales", timeout=SHORT)
    assert not res.ok   # MemoryError (posix rlimit), TimeoutError, or C-level


def test_huge_dataframe_terminates_bounded():
    """A billion-row DataFrame either completes (output capped by the sandbox)
    or dies (memory/timeout) — either way the call RETURNS, bounded, and Jane
    stays usable. (The sandbox caps result rows at 200, so an ok result here
    is legitimate resource handling, not a leak.)"""
    res = run_code(
        "import pandas as pd\n"
        "result_df = pd.DataFrame({'a': range(10**9)})\n",
        TABLES, active_table="sales", timeout=30)
    assert res is not None
    if res.ok:
        assert len(res.tables[0]["rows"]) <= 200
    else:
        assert res.timed_out or res.error


def test_deep_recursion_reported():
    res = run_code(
        "def f(n):\n    return f(n + 1)\nresult_df = f(0)\n",
        TABLES, active_table="sales", timeout=SHORT)
    assert not res.ok
    assert not res.timed_out   # RecursionError surfaces as a normal failure


def test_flooded_stdout_capped_not_hung():
    res = run_code(
        "for i in range(10**7):\n    print('x' * 100)\nresult_df = df.head(1)\n",
        TABLES, active_table="sales", timeout=SHORT)
    # either it finished with capped stdout or timed out — never hangs/crashes
    assert res is not None
    assert len(res.stdout or "") < 5_000_000


def test_legitimate_work_still_runs():
    res = run_code(
        "result_df = df.groupby('region', as_index=False)['amount'].sum()",
        TABLES, active_table="sales", timeout=SHORT)
    assert res.ok, res.error


def test_duckdb_inside_the_child_cannot_read_files(tmp_path):
    """The child's own DuckDB connection is a second, independent way out.

    The AST whitelist allows `con` (analysts use it), and it cannot inspect the
    SQL *string* passed to it, so the connection itself has to refuse external
    access. Before the config was set, this returned a real directory listing.
    The directory listed here is one the test creates, so the check does not
    depend on any machine's layout.
    """
    (tmp_path / "secret.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    target = f"{tmp_path.as_posix()}/*"
    res = run_code(
        f'result_df = con.sql("SELECT * FROM glob(\'{target}\')").df()',
        TABLES, active_table="sales", timeout=SHORT)
    assert not res.ok, f"sandbox reached the filesystem: {res.tables[:1]}"
    assert "Permission" in (res.error or "") or "disabled" in (res.error or ""), res.error


def test_duckdb_inside_the_child_still_analyses():
    """Closing external access must leave registered-frame SQL working."""
    res = run_code(
        "result_df = con.sql(\"SELECT region, SUM(amount) AS total FROM sales "
        "GROUP BY region ORDER BY total DESC\").df()",
        TABLES, active_table="sales", timeout=SHORT)
    assert res.ok, res.error
    rows = res.tables[0]["rows"]
    assert rows[0] == {"region": "east", "total": 50}, rows
