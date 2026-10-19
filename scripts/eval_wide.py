"""Grade Jane on wide-terrain data: hospital, 2-table e-commerce join,
wide sensor telemetry, personal banking. 10 questions, planted truths.

Run:  python scripts/eval_wide.py
"""
from __future__ import annotations

import re
import sys
import time

sys.path.insert(0, ".")

from jane.agent import AgentContext, AnalystAgent  # noqa: E402
from jane.config import settings  # noqa: E402
from jane.ingest import load_file  # noqa: E402
from jane.ollama_client import OllamaClient  # noqa: E402

WIDE = "data/eval_wide"


def _first_num(text: str):
    m = re.search(r"-?\d[\d,]*\.?\d*", (text or "").replace(",", ""))
    return float(m.group()) if m else None


def check_num(want, tol=0.02):
    def c(answer, res):
        got = _first_num(answer)
        if got is None:
            return False, f"no number in answer (want {want})"
        ok = abs(got - want) <= max(abs(want) * tol, 0.11)
        return ok, f"want ~{want}, got {got}"
    return c


def check_str(want):
    def c(answer, res):
        ok = want.lower() in (answer or "").lower()
        return ok, f"want '{want}' in answer, got: {(answer or '')[:120]}"
    return c


QUESTIONS = [
    ("hospital", ["hospital"], "Which department has the highest 30-day readmission rate?",
     check_str("oncology")),
    ("hospital", ["hospital"], "What is the median length of stay across all patients, in days?",
     check_num(3.0, tol=0.0)),
    ("hospital", ["hospital"],
     "How many unique patients are in this data? The table may contain duplicate rows.",
     check_num(900, tol=0.0)),
    ("ecom", ["orders", "customers"],
     "Join orders with customers: which customer segment generates the highest total revenue (unit_price * quantity)?",
     check_str("standard")),
    ("ecom", ["orders", "customers"],
     "Join orders with customers: what is the total revenue of that top segment?",
     check_num(328234.92, tol=0.01)),
    ("ecom", ["orders"], "Which product category has the highest return rate?",
     check_str("home")),
    ("sensors", ["sensors"], "Which machine runs the hottest on average temperature?",
     check_str("press-b")),
    ("sensors", ["sensors"], "What percentage of all sensor readings have status fault?",
     check_num(3.58, tol=0.05)),
    ("sensors", ["sensors"], "Which numeric sensor channel has the highest average value across all machines?",
     check_str("rpm")),
    ("bank", ["bank"], "Among accounts with balance below 500, what percentage have overdraft_flag equal to 1?",
     check_num(26.5, tol=0.05)),
    ("bank", ["bank"], "Which account type holds the highest average balance?",
     check_str("credit")),
]


def main() -> int:
    client = OllamaClient(settings.ollama_host, settings.request_timeout)
    agent = AnalystAgent(client)
    total = passed = 0
    cache = {}
    for dom, tables, q, checker in QUESTIONS:
        key = tuple(tables)
        if key not in cache:
            ctx_tables = {}
            active = tables[0]
            for t in tables:
                loaded = load_file(f"{WIDE}/{t}.csv", progress=lambda s: None)
                for s in loaded.sheets:
                    ctx_tables[s.name] = s.df
            cache[key] = (ctx_tables, active)
        tables_map, active = cache[key]
        ctx = AgentContext(tables=tables_map, active_table=active)
        t0 = time.perf_counter()
        try:
            res = agent.analyze(q, ctx, progress=lambda *a, **k: None)
        except Exception as e:
            print(f"  [EXC] {q}\n        {type(e).__name__}: {e}")
            total += 1
            continue
        dt = time.perf_counter() - t0
        ok, why = checker(res.answer or res.error or "", res)
        total += 1
        passed += ok
        print(f"  [{'PASS' if ok else 'FAIL'}] ({dt:5.1f}s, tries={res.attempts_used}) {q}")
        if not ok:
            print(f"         {why}")
            print(f"         got: {(res.answer or res.error or '')[:200]}")
    print(f"\n=== WIDE TERRAIN: {passed}/{total} ({passed / total:.0%}) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
