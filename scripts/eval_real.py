"""Grade Jane on REAL public datasets (the Kaggle/UCI classics):
- IBM HR Analytics attrition (1470 x 35, BOM, constant columns)
- UCI Bank Marketing (41188 x 20, semicolon-separated)
- Titanic (891 x 12, missing Ages/Cabin)

Ground truths computed with pandas in scripts/gen_real_truths.py.
Run:  python scripts/eval_real.py
"""
from __future__ import annotations

import re
import time

from jane.agent import AgentContext, AnalystAgent
from jane.config import settings
from jane.ingest import load_file
from jane.ollama_client import OllamaClient


def _first_num(text: str):
    m = re.search(r"-?\d[\d,]*\.?\d*", (text or "").replace(",", ""))
    return float(m.group()) if m else None


def check_num(want, tol=0.02):
    def c(answer, res):
        got = _first_num(answer)
        if got is None:
            return False, f"no number (want {want})"
        return abs(got - want) <= max(abs(want) * tol, 0.11), f"want ~{want}, got {got}"
    return c


def check_str(*wants):
    def c(answer, res):
        a = (answer or "").lower()
        return any(w.lower() in a for w in wants), \
            f"want one of {wants}, got: {(answer or '')[:120]}"
    return c


QUESTIONS = [
    ("hr_attrition", "hr_attrition.csv", {},
     "What is the overall attrition rate as a percentage?", check_num(16.12)),
    ("hr_attrition", "hr_attrition.csv", {},
     "Which job role has the highest attrition rate?", check_str("Sales Representative")),
    ("hr_attrition", "hr_attrition.csv", {},
     "How many employees work in the Research & Development department?", check_num(961)),
    ("hr_attrition", "hr_attrition.csv", {},
     "Compare attrition rates for employees who work overtime versus those who don't.",
     check_str("30.5", "30.53", "10.4", "10.44")),
    ("hr_attrition", "hr_attrition.csv", {},
     "What is the average monthly income of employees whose attrition is Yes?",
     check_num(4787.09)),
    ("bank", "bank-additional/bank-additional-full.csv", {"sep": ";"},
     "What percentage of clients subscribed to a term deposit (column y = yes)?",
     check_num(11.27)),
    ("bank", "bank-additional/bank-additional-full.csv", {"sep": ";"},
     "What is the median age of the clients?", check_num(38.0)),
    ("bank", "bank-additional/bank-additional-full.csv", {"sep": ";"},
     "Which job category has the highest subscription rate?", check_str("student")),
    ("bank", "bank-additional/bank-additional-full.csv", {"sep": ";"},
     "Is the subscription rate higher when contacted by cellular or by telephone?",
     check_str("cellular")),
    ("titanic", "titanic.csv", {},
     "What percentage of passengers survived?", check_num(38.38)),
    ("titanic", "titanic.csv", {},
     "Which passenger class had the highest survival rate?", check_str("1", "first")),
    ("titanic", "titanic.csv", {},
     "What was the average ticket fare?", check_num(32.2)),
]


def main() -> int:
    client = OllamaClient(settings.ollama_host, settings.request_timeout)
    agent = AnalystAgent(client)
    total = passed = 0
    for name, fname, kw, q, checker in QUESTIONS:
        try:
            loaded = load_file(f"data/real/{fname}", progress=lambda s: None, **kw)
        except TypeError:
            loaded = load_file(f"data/real/{fname}", progress=lambda s: None)
        tables = {s.name: s.df for s in loaded.sheets}
        ctx = AgentContext(tables=tables, active_table=loaded.sheets[0].name)
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
    print(f"\n=== REAL-DATA GAUNTLET: {passed}/{total} ({passed / total:.0%}) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
