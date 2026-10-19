"""Sandboxed execution of LLM-generated analysis code.

Layers:
  1. AST static analysis — whitelisted imports/names/attrs, blocked dunders.
  2. Fresh subprocess: wall-clock timeout, (POSIX) memory rlimit, stdout capture.

The child receives tables as JSON on stdin and prints one JSON result blob
after a unique marker. Data flow is stdin/stdout only — no files, no network.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Optional

from .logging_setup import get_logger

log = get_logger("jane.sandbox")

MAX_INPUT_ROWS = 300_000  # hard cap on rows shipped to the child

# ---------------------------------------------------------------------------
# 1. Static analysis
# ---------------------------------------------------------------------------

ALLOWED_IMPORTS = {
    "pandas", "numpy", "duckdb", "math", "statistics", "datetime",
    "json", "re", "collections", "itertools", "functools", "matplotlib",
}
BLOCKED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals",
    "breakpoint", "help", "input", "exit", "quit",
    "setattr", "getattr", "delattr", "vars", "dir",
    "execfile", "reload", "memoryview", "bytearray",
}
BLOCKED_ROOTS = {
    "os", "sys", "subprocess", "socket", "requests", "urllib", "http",
    "shutil", "pathlib", "glob", "tempfile", "io", "importlib", "ctypes",
    "multiprocessing", "threading", "signal", "resource", "pickle",
    "shelve", "asyncio", "webbrowser", "platform",
}
BLOCKED_ATTRS = {
    "system", "popen", "spawn", "spawnl", "spawnv", "fork", "execve", "execv",
    "execvp", "unlink", "mkdir", "rmdir",
    "chmod", "socket", "connect", "bind", "listen", "accept", "environ",
    "argv", "_exit", "kill", "terminate", "setrlimit", "getenv", "putenv",
    "frame", "f_globals", "f_locals", "f_builtins",
    # sneaky reachability: sub-modules/attributes that grant fresh roots
    "io", "loader", "spec", "spec_loader", "__loader__", "__spec__",
}
BLOCKED_CALLS = frozenset({"type"})   # type() builds classes -> MRO escapes
# NOTE: 'rename', 'remove', 'truncate' were removed from BLOCKED_ATTRS — they
# collide with legitimate pandas (DataFrame.rename/truncate, list.remove).
# Filesystem access is still impossible: every dangerous module (os, sys,
# pathlib, shutil…) is denied at the ROOT level in BLOCKED_ROOTS, and open()
# is in BLOCKED_NAMES, so those method names have no reachable attacker here.


class CodeSafetyError(ValueError):
    pass


class _Visitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.errors: list[str] = []

    def _deny(self, node: ast.AST, reason: str) -> None:
        self.errors.append(f"line {getattr(node, 'lineno', '?')}: {reason}")

    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            root = a.name.split(".")[0]
            if root not in ALLOWED_IMPORTS:
                self._deny(node, f"import '{a.name}' is not allowed")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        root = (node.module or "").split(".")[0]
        if root not in ALLOWED_IMPORTS:
            self._deny(node, f"from-import of '{node.module}' is not allowed")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        f = node.func
        if isinstance(f, ast.Name) and f.id in BLOCKED_NAMES:
            self._deny(node, f"call to '{f.id}()' is not allowed")
        elif isinstance(f, ast.Name) and f.id in BLOCKED_CALLS:
            self._deny(node, f"call to '{f.id}()' is not allowed")
        elif isinstance(f, ast.Attribute) and f.attr in BLOCKED_NAMES:
            self._deny(node, f"call to '.{f.attr}()' is not allowed")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in BLOCKED_NAMES or (node.id.startswith("__") and node.id.endswith("__")):
            self._deny(node, f"use of '{node.id}' is not allowed")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in BLOCKED_ATTRS:
            self._deny(node, f"attribute '.{node.attr}' is not allowed")
        if node.attr.startswith("__") and node.attr.endswith("__"):
            self._deny(node, f"dunder attribute '.{node.attr}' is not allowed")
        if isinstance(node.value, ast.Name) and node.value.id in BLOCKED_ROOTS:
            self._deny(node, f"use of module '{node.value.id}' is not allowed")
        self.generic_visit(node)


def analyze_code_safety(code: str) -> list[str]:
    """Return a list of violations (empty == safe enough to run)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError: {e}"]
    v = _Visitor()
    v.visit(tree)
    return v.errors


# ---------------------------------------------------------------------------
# 2. Child process
# ---------------------------------------------------------------------------

_CHILD_TEMPLATE = '''\
import sys, os, json, math, re, datetime

if os.name == "posix":  # memory cap (POSIX only)
    try:
        import resource
        limit = {memory_mb} * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except Exception:
        pass

def _default(o):
    if hasattr(o, "isoformat"):
        return o.isoformat()
    if isinstance(o, (bytes, bytearray)):
        return o.decode("utf-8", "replace")
    return str(o)

def _clean(v):
    if v is None or isinstance(v, (str, bool, int)):
        return v
    if isinstance(v, float):
        return None if (v != v or v in (float("inf"), float("-inf"))) else v
    return _default(v)

payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))

import pandas as pd
import numpy as np

tables = {}
for tname, tbl in payload["tables"].items():
    tables[tname] = pd.DataFrame(tbl["data"], columns=tbl["columns"])
if payload.get("tables_truncated"):
    print("[warning] tables were truncated to {max_input_rows} rows before analysis", file=sys.stderr)

con = None
if payload.get("duckdb"):
    import duckdb
    # same guard as the SQL engine: the child's own DuckDB must not reach the
    # filesystem either (glob/read_csv/sniff_csv were reachable from here too)
    con = duckdb.connect(":memory:", config={"enable_external_access": "false"})
    for tname, df in tables.items():
        con.register(tname, df)

df = tables.get(payload.get("active_table"))
if df is None:
    df = next(iter(tables.values())) if tables else pd.DataFrame()

g = {
    "pd": pd, "np": np, "tables": tables, "df": df, "con": con,
    "math": math, "json": json, "re": re, "datetime": datetime,
}
# expose each table as its own variable (sanitized name) — models naturally
# write `sales.groupby(...)` using the table name from the schema
_reserved = {"pd", "np", "tables", "df", "con", "math", "json", "re", "datetime",
             "result_df", "result_df2", "result_df3", "charts"}
for tname, tdf in tables.items():
    gname = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in str(tname).strip())
    if not gname or not (gname[0].isalpha() or gname[0] == "_"):
        gname = "t_" + gname
    if gname in _reserved:
        gname = gname + "_table"
    g[gname] = tdf
_ALLOWED_ROOTS = {"pandas", "numpy", "duckdb", "math", "statistics", "datetime", "json", "re", "collections", "itertools", "functools", "matplotlib"}

def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in _ALLOWED_ROOTS:
        raise ImportError(f"import of '{name}' is blocked in the sandbox")
    return __import__(name, globals, locals, fromlist, level)

g["__builtins__"] = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict, "divmod": divmod,
    "enumerate": enumerate, "filter": filter, "float": float, "format": format,
    "frozenset": frozenset, "hash": hash, "int": int, "isinstance": isinstance,
    "issubclass": issubclass, "iter": iter, "len": len, "list": list, "map": map,
    "max": max, "min": min, "next": next, "None": None, "object": object,
    "pow": pow, "print": print, "range": range, "repr": repr, "reversed": reversed,
    "round": round, "set": set, "slice": slice, "sorted": sorted, "str": str,
    "sum": sum, "tuple": tuple, "zip": zip,
    "__import__": _restricted_import, "__build_class__": __build_class__,
    "ArithmeticError": ArithmeticError, "Exception": Exception, "IndexError": IndexError,
    "KeyError": KeyError, "TypeError": TypeError, "ValueError": ValueError,
    "ZeroDivisionError": ZeroDivisionError, "StopIteration": StopIteration,
}

MAX_OUT_ROWS = {max_rows}
_CODE = {code_repr}

def _records(dframe):
    if not isinstance(dframe, pd.DataFrame) or dframe.empty:
        return [], []
    cols = [str(c) for c in dframe.columns]
    recs = []
    for row in dframe.head(MAX_OUT_ROWS).to_dict(orient="records"):
        recs.append({str(k): _clean(v) for k, v in row.items()})
    return cols, recs

result = {"stdout": "", "tables": [], "charts": [], "figures": []}
import io, contextlib, traceback
_buf = io.StringIO()
try:
    with contextlib.redirect_stdout(_buf):
        exec(_CODE, g)
    result["stdout"] = _buf.getvalue()[-20000:]
    for name in ("result_df", "result_df2", "result_df3"):
        obj = g.get(name)
        if isinstance(obj, pd.DataFrame):
            cols, recs = _records(obj)
            result["tables"].append({"name": name, "columns": cols, "rows": recs})
    charts = g.get("charts")
    if isinstance(charts, list):
        for c in charts[:3]:
            if isinstance(c, dict):
                result["charts"].append(c)
    # matplotlib figures produced in-sandbox: serialize to PNG data URIs
    figs = []
    for fname in ("fig", "fig1", "fig2", "fig3", "figs"):
        fobj = g.get(fname)
        if fname == "figs" and isinstance(fobj, (list, tuple)):
            figs.extend(f for f in fobj if f.__class__.__name__ == "Figure")
        elif fobj is not None and fobj.__class__.__name__ == "Figure":
            figs.append(fobj)
    if figs:
        import base64 as _b64mod
        result["figures"] = []
        for f in figs[:3]:
            try:
                f.tight_layout()
                bio = io.BytesIO()
                f.savefig(bio, format="png", bbox_inches="tight")
                result["figures"].append(
                    "data:image/png;base64," + _b64mod.b64encode(bio.getvalue()).decode())
            except Exception:
                pass
except Exception:
    result["error"] = traceback.format_exc()
    result["stdout"] = _buf.getvalue()[-20000:]

sys.stdout.write("<<<JANE>>>" + json.dumps(result, default=_default))
'''


@dataclass
class SandboxResult:
    ok: bool
    stdout: str = ""
    tables: list = field(default_factory=list)   # [{name, columns, rows}]
    charts: list = field(default_factory=list)   # [Vega-Lite spec dicts]
    figures: list = field(default_factory=list)  # [png data-URI strings]
    error: Optional[str] = None
    safety_errors: list[str] = field(default_factory=list)
    timed_out: bool = False


def _table_payload(tables: dict[str, Any]) -> tuple[dict, bool]:
    """Serialize tables columnar, capped at MAX_INPUT_ROWS total.

    Dtype-specific fast paths keep this O(bytes) via C-speed tolist() for
    numeric/bool columns; object columns get a Python-level scalar pass.
    """
    out: dict[str, dict] = {}
    budget = MAX_INPUT_ROWS
    truncated = False
    from .schema import _jsonable

    for name, df in tables.items():
        cols = [str(c) for c in df.columns]
        if budget <= 0:
            truncated = True
            out[name] = {"columns": cols, "data": {c: [] for c in cols}}
            continue
        take = df.head(budget)
        if len(take) < len(df):
            truncated = True
        budget -= len(take)
        data: dict[str, list] = {}
        for col_name, series in take.items():
            key = str(col_name)
            k = series.dtype.kind
            if k in "if":
                vals = series.tolist()  # C-speed; NaN/inf -> float('nan'/'inf')
            elif k == "b":
                vals = series.tolist()
            elif k in "Mm":  # datetime / timedelta
                vals = [None if v is None or v != v else v.isoformat()
                        for v in series.tolist()]
            else:
                vals = [None if v is None or v != v else _jsonable(v)
                        for v in series]
            data[key] = vals
        out[name] = {"columns": cols, "data": data}
    return out, truncated


def run_code(
    code: str,
    tables: dict[str, Any],
    active_table: str | None = None,
    timeout: int = 30,
    max_memory_mb: int = 4096,
    max_output_rows: int = 200,
    use_duckdb: bool = True,
) -> SandboxResult:
    violations = analyze_code_safety(code)
    if violations:
        return SandboxResult(
            ok=False,
            safety_errors=violations,
            error="code failed static safety analysis:\n" + "\n".join(violations),
        )

    payload_rows, truncated = _table_payload(tables)
    payload = {
        "tables": payload_rows,
        "active_table": active_table,
        "duckdb": use_duckdb,
        "tables_truncated": truncated,
    }

    script = (
        _CHILD_TEMPLATE
        .replace("{memory_mb}", str(max_memory_mb))
        .replace("{max_input_rows}", str(MAX_INPUT_ROWS))
        .replace("{max_rows}", str(max_output_rows))
        .replace("{code_repr}", repr(code))
    )

    with tempfile.TemporaryDirectory() as td:
        child = os.path.join(td, "child.py")
        with open(child, "w", encoding="utf-8") as f:
            f.write(script)
        try:
            proc = subprocess.run(
                [sys.executable, child],
                input=json.dumps(payload).encode("utf-8"),
                capture_output=True,
                timeout=timeout,
                cwd=td,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(ok=False, timed_out=True,
                                 error=f"execution exceeded {timeout}s — code was killed")
        except OSError as e:
            return SandboxResult(ok=False, error=f"could not spawn sandbox process: {e}")

    out = proc.stdout.decode("utf-8", errors="replace")
    err = proc.stderr.decode("utf-8", errors="replace")
    marker = "<<<JANE>>>"
    if marker not in out:
        msg = err.strip() or "child process died before producing a result"
        if proc.returncode != 0:
            msg += f" (exit code {proc.returncode})"
        return SandboxResult(ok=False, error=msg[-4000:])

    try:
        data = json.loads(out.split(marker, 1)[1])
    except json.JSONDecodeError:
        return SandboxResult(ok=False, error="sandbox produced unparseable result JSON")

    if err.strip() and not data.get("error"):
        data["stdout"] = (data.get("stdout", "") + "\n" + err.strip())[-20000:]

    return SandboxResult(
        ok="error" not in data,
        stdout=data.get("stdout", ""),
        tables=data.get("tables", []),
        charts=data.get("charts", []),
        figures=data.get("figures", []),
        error=data.get("error"),
    )
