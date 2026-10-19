"""Diagnose the 5 remaining eval failures with full code + table output.

Usage: .venv/Scripts/python scripts/debug_eval_failures.py [case ...]
Cases: movies_count, spike, spike_value, missing_month, emp_count
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for s in (sys.stdout, sys.stderr):
    if hasattr(s, "reconfigure"):
        s.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

from jane.agent import AnalystAgent, AgentContext
from jane.config import settings
from jane.ingest import load_file
from jane.ollama_client import OllamaClient

EVAL = ROOT / "data" / "eval"

CASES = {
    "movies_count": ("movies.csv", "How many movies have a rating above 8?"),
    "spike": ("subscriptions.csv",
              "Was there any month with an abnormal churn spike? When and how many churned?"),
    "spike_value": ("subscriptions.csv",
                    "In the churn spike month, how many subscriptions churned?"),
    "missing_month": ("subscriptions.csv",
                      "Is any month missing from this time series? Check the month column for gaps."),
    "emp_count": ("employees.csv", "How many employees work in engineering?"),
    "churn_rate": ("churn.csv", "What is the overall churn rate? Answer with the percentage."),
    "worst_contract": ("churn.csv", "Which contract type churns the most, and what is its churn rate?"),
    "charges_median": ("churn.csv", "Do customers who churned pay higher monthly charges than those who stayed? Compare the medians for both groups."),
}


def show(result):
    print(f"ok={result.ok} tries={result.attempts_used}")
    print(f"answer: {result.answer}")
    print(f"code: {result.code}")
    for t in result.tables:
        print(f"table {t['name']}: cols={t['columns']} rows={len(t['rows'])}")
        for r in t["rows"][:8]:
            print(f"  {r}")
    if result.error:
        print(f"error: {result.error[:200]}")


def main():
    wanted = [a for a in sys.argv[1:] if not a.startswith("-")] or list(CASES)
    client = OllamaClient(settings.ollama_host, settings.request_timeout)
    agent = AnalystAgent(client)
    for key in wanted:
        fname, q = CASES[key]
        loaded = load_file(EVAL / fname, progress=lambda s: None)
        tables = {s.name: s.df for s in loaded.sheets}
        active = loaded.sheets[0].name
        print("=" * 70)
        print(f"[{key}] {fname}: {q}")
        ctx = AgentContext(tables=tables, active_table=active)
        result = agent.analyze(q, ctx, progress=lambda step, detail="": None)
        show(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
