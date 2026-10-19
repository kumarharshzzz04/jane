"""Grade Jane on the user's own `go data/` Kaggle drops:
- HR churn (14999 x 10, reserved-word 'left', misleading 'sales' column)
- Walmart Sales (6435 x 8, store/week grain)
- Heart patients (303 x 14, health domain, 0-4 target scale)

Truths computed with pandas (session 17). Run: python scripts/eval_godata.py
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
    ("go data/archive/HR.csv",
     "What percentage of employees left the company?",
     check_num(23.81)),
    ("go data/archive/HR.csv",
     "Employees who left were working more or fewer average monthly hours than those who stayed?",
     check_str("more")),
    ("go data/archive/HR.csv",
     "Which department has the highest employee churn? The department values are in the column named sales.",
     check_str("hr")),
    ("go data/archive/HR.csv",
     "Employees with how many projects have the highest churn rate: 2, 3, 4, 6 or 7?",
     check_str("7")),
    ("go data/archive (2)/Walmart_Sales.csv",
     "Which store has the LOWEST total weekly sales?",
     check_str("33")),
    ("go data/archive (2)/Walmart_Sales.csv",
     "Which store had the single biggest weekly sale, and in what date?",
     check_str("14", "24-12-2010")),
    ("go data/archive (2)/Walmart_Sales.csv",
     "What is the average fuel price across all rows?",
     check_num(3.359, tol=0.01)),
    ("go data/health_extract/health d3/data/heart.csv",
     "What is the average cholesterol level of the patients?",
     check_num(246.7, tol=0.01)),
    ("go data/health_extract/health d3/data/heart.csv",
     "What is the most common chest pain type (cp) among the patients?",
     check_str("4", "four", "asymptomatic")),
    ("go data/health_extract/health d3/data/heart.csv",
     "What is the average age of the patients?",
     check_num(54.4, tol=0.01)),
]


def main() -> int:
    client = OllamaClient(settings.ollama_host, settings.request_timeout)
    agent = AnalystAgent(client)
    total = passed = 0
    for path, q, checker in QUESTIONS:
        try:
            loaded = load_file(path, progress=lambda s: None)
        except Exception as e:
            print(f"  [LOAD FAIL] {path}: {e}")
            total += 1
            continue
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
    print(f"\n=== GO-DATA GAUNTLET: {passed}/{total} ({passed / total:.0%}) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
