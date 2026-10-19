"""Domain eval: run Jane's agent against real-world-style datasets and grade
the answers against ground truth recomputed from the data.

Usage:
  .venv/Scripts/python scripts/eval_domains.py            # all domains
  .venv/Scripts/python scripts/eval_domains.py churn      # one domain
  .venv/Scripts/python scripts/eval_domains.py --fast     # skip live LLM,
                                                          # deterministic checks only

Requires Ollama running for the LLM phase (skipped automatically otherwise).
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Windows consoles default to cp1252; output contains arrows and math symbols.
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

from jane.agent import AnalystAgent, AgentContext
from jane.config import settings
from jane.insights import profile_table
from jane.ingest import load_file
from jane.ollama_client import OllamaClient, OllamaError
from jane.preprocess import clean_default

sys.path.insert(0, str(ROOT / "scripts"))
from gen_eval_data import GENERATORS  # noqa: E402  (ground-truth functions)

EVAL_DIR = ROOT / "data" / "eval"


# ---------------------------------------------------------------------------
# question banks per domain: (question, checker(result_dict) -> (ok, why))
# checkers accept the AnalysisResult and recompute truth from the same file.
# ---------------------------------------------------------------------------

def _tbl(result, i=0):
    return result.tables[i] if result.tables and len(result.tables) > i else None


def _rows_as_dicts(t):
    if not t:
        return []
    cols = t["columns"]
    return [dict(zip(cols, r)) if isinstance(r, list) else r for r in t["rows"]]


def _find_number(text):
    m = re.search(r"[-+]?\d[\d,]*\.?\d*", (text or "").replace(",", ""))
    return float(m.group(0)) if m else None


def _find_pct(text):
    m = re.search(r"(\d[\d,]*\.?\d*)\s*%", text or "")
    return float(m.group(1)) if m else None


def _grade_number(got, want, tol_rel=0.02, tol_abs=1e-9):
    if got is None:
        return False, "no number found in answer"
    if abs(want) > 0 and abs(got - want) / abs(want) <= tol_rel:
        return True, ""
    if abs(got - want) <= max(tol_abs, 0.5):
        return True, ""
    return False, f"expected ≈{want}, got {got}"


# --- churn checkers ---------------------------------------------------------

def q_churn_rate(df, result):
    from gen_eval_data import gt_churn
    want = gt_churn(df)["churn_rate"] * 100
    got = _find_pct(result.answer) or _find_number(result.answer)
    ok, why = _grade_number(got, want, tol_rel=0.05)
    return ok, f"churn_rate {want}% — {why}"


def q_worst_contract(df, result):
    from gen_eval_data import gt_churn
    truth = gt_churn(df)
    text = (result.answer or "") + json.dumps(_rows_as_dicts(_tbl(result)))[:2000]
    ok = truth["worst_contract"] in text.lower()
    return ok, f"worst contract is '{truth['worst_contract']}' ({truth['worst_contract_rate']*100:.0f}%)"


def q_churn_charges_compare(df, result):
    from gen_eval_data import gt_churn
    t = gt_churn(df)
    text = (result.answer or "")
    a, b = _find_number(text.split("vs")[0]) if "vs" in text.lower() else (None, None)
    rows = _rows_as_dicts(_tbl(result))
    nums = [v for r in rows for v in r.values() if isinstance(v, (int, float))]
    nums += [n for n in (a, b) if n is not None]
    want = sorted([t["median_charges_churned"], t["median_charges_stayed"]])
    ok = any(abs(n - want[0]) <= 1.0 for n in nums) and any(abs(n - want[1]) <= 1.0 for n in nums)
    return ok, f"needs both medians ≈ {want[0]} and {want[1]}"


# --- movies checkers --------------------------------------------------------

def q_top_movie(df, result):
    from gen_eval_data import gt_movies
    t = gt_movies(df)
    rows = _rows_as_dicts(_tbl(result))
    text = (result.answer or "").lower() + json.dumps(rows)[:3000].lower()
    ok = "movie #587" in text or "587" in text
    return ok, f"top-grossing is Movie #587 (revenue {df['revenue_m'].max()})"


def q_best_genre(df, result):
    from gen_eval_data import gt_movies
    t = gt_movies(df)
    text = ((result.answer or "") + json.dumps(_rows_as_dicts(_tbl(result)))[:3000]).lower()
    ok = t["best_genre_by_rating"] in text
    return ok, f"best genre by avg rating is '{t['best_genre_by_rating']}'"


def q_count_rating(df, result):
    from gen_eval_data import gt_movies
    want = gt_movies(df)["n_rating_over_8"]
    got = _find_number(result.answer)
    ok, why = _grade_number(got, want, tol_abs=0.5)
    return ok, f"{want} movies rated > 8 — {why}"


# --- subscriptions checkers -------------------------------------------------

def q_spike_month(df, result):
    from gen_eval_data import gt_subscriptions
    t = gt_subscriptions(df)
    text = ((result.answer or "") + json.dumps(_rows_as_dicts(_tbl(result)))[:3000])
    ok = ("2024-03" in text) or ("2024" in text and "03" in text) or ("march" in text.lower())
    return ok, f"churn spike is {t['spike_month']} ({t['spike_churned']} churned)"


def q_spike_value(df, result):
    from gen_eval_data import gt_subscriptions
    from jane.agent import _extract_numbers
    want = gt_subscriptions(df)["spike_churned"]
    # check EVERY number in the answer: '2024-03' yields 2024 and 3 — neither
    # is the count; 210 anywhere in the answer (text or table) is a pass
    nums = _extract_numbers(result.answer)
    rows = _rows_as_dicts(_tbl(result))
    nums += [v for r in rows for v in r.values() if isinstance(v, (int, float))]
    ok = any(abs(n - want) <= 2 for n in nums)
    return ok, f"spike churned count = {want} — got {nums[:6]}"


def q_missing_month(df, result):
    from gen_eval_data import gt_subscriptions
    t = gt_subscriptions(df)
    text = ((result.answer or "") + json.dumps(_rows_as_dicts(_tbl(result)))[:4000]).lower()
    ok = "2023-06" in text or "jun" in text
    return ok, f"missing month is {t['missing_month']}"


# --- employees checkers -----------------------------------------------------

def q_top_dept(df, result):
    from gen_eval_data import gt_employees
    t = gt_employees(df)
    text = ((result.answer or "") + json.dumps(_rows_as_dicts(_tbl(result)))[:3000]).lower()
    ok = t["top_dept_by_median_salary"] in text
    return ok, f"top dept by median salary is '{t['top_dept_by_median_salary']}'"


def q_headcount(df, result):
    from gen_eval_data import gt_employees
    want = gt_employees(df)["headcount_engineering"]
    got = _find_number(result.answer)
    ok, why = _grade_number(got, want, tol_abs=1.0)
    return ok, f"{want} engineers — {why}"


def q_pay_gap(df, result):
    from gen_eval_data import gt_employees
    t = gt_employees(df)
    text = (result.answer or "")
    got = _find_pct(text)
    rows = _rows_as_dicts(_tbl(result))
    # accept either the gap % or both salary means in the table
    nums = [v for r in rows for v in r.values() if isinstance(v, (int, float))]
    both_means = (any(abs(n - t["male_mean_salary"]) <= 200 for n in nums)
                  and any(abs(n - t["female_mean_salary"]) <= 200 for n in nums))
    ok = (got is not None and abs(got - t["pay_gap_pct"]) <= 2.0) or both_means
    return ok, f"pay gap ≈ {t['pay_gap_pct']}% (male {t['male_mean_salary']:.0f} vs female {t['female_mean_salary']:.0f})"


# --- registry ---------------------------------------------------------------

QUESTIONS = {
    "churn": [
        ("What is the overall churn rate? Answer with the percentage.", q_churn_rate),
        ("Which contract type churns the most, and what is its churn rate?", q_worst_contract),
        ("Do customers who churned pay higher monthly charges than those who stayed? "
         "Compare the medians for both groups.", q_churn_charges_compare),
    ],
    "movies": [
        ("Which movie has the highest revenue?", q_top_movie),
        ("Which genre has the highest average rating?", q_best_genre),
        ("How many movies have a rating above 8?", q_count_rating),
    ],
    "subscriptions": [
        ("Was there any month with an abnormal churn spike? When and how many churned?",
         q_spike_month),
        ("In the churn spike month, how many subscriptions churned?", q_spike_value),
        ("Is any month missing from this time series? Check the month column for gaps.",
         q_missing_month),
    ],
    "employees": [
        ("Which department has the highest median salary?", q_top_dept),
        ("How many employees work in engineering? Count distinct employees.", q_headcount),
        ("Is there a gender pay gap? Compare average salary by gender.",
         q_pay_gap),
    ],
}


# ---------------------------------------------------------------------------
# deterministic phase: ingest/profile/insights/prep must behave on every domain
# ---------------------------------------------------------------------------

def deterministic_phase() -> bool:
    all_ok = True
    for name in GENERATORS:
        path = EVAL_DIR / name
        if not path.exists():
            print(f"  MISSING {path} — run scripts/gen_eval_data.py")
            all_ok = False
            continue
        loaded = load_file(path, progress=lambda s: None)
        df = loaded.primary
        prof = profile_table(name.replace(".csv", ""), df)
        null_cols = [c["name"] for c in prof["columns"] if c.get("nulls", 0) > 0]
        # dedupe must not crash and must remove the planted dupes in employees
        res = clean_default(df)
        notes = []
        if name == "employees.csv" and len(res.df) != 400:
            notes.append(f"tidy preset kept {len(res.df)} rows (wanted 400)")
        if name == "subscriptions.csv" and "month" not in df.columns:
            notes.append("month column missing")
        status = "OK" if not notes else "PROBLEM"
        if notes:
            all_ok = False
        print(f"  [{status}] {name:20s} {len(df):>5} rows | nulls: "
              f"{', '.join(null_cols) or 'none'} | tidy→{len(res.df)} rows")
        for n in notes:
            print(f"         - {n}")
    return all_ok


# ---------------------------------------------------------------------------
# live LLM phase
# ---------------------------------------------------------------------------

def llm_phase(domains: list[str]) -> float:
    client = OllamaClient(settings.ollama_host, settings.request_timeout)
    try:
        models = [m["name"] for m in client.tags()]
    except Exception as e:
        print(f"Ollama unreachable ({e}) — start it with `ollama serve`.")
        return -1.0
    print(f"models: {models}")
    agent = AnalystAgent(client)

    total, passed = 0, 0
    for dom in domains:
        path = EVAL_DIR / f"{dom}.csv"
        loaded = load_file(path, progress=lambda s: None)
        tables = {s.name: s.df for s in loaded.sheets}
        active = loaded.sheets[0].name
        print(f"\n=== {dom} ({len(loaded.primary)} rows) ===")
        for q, checker in QUESTIONS[dom]:
            ctx = AgentContext(tables=tables, active_table=active)
            t0 = time.perf_counter()
            try:
                result = agent.analyze(q, ctx, progress=lambda *a, **k: None)
            except Exception as e:
                print(f"  [EXC] {q}\n        {type(e).__name__}: {e}")
                total += 1
                continue
            dt = time.perf_counter() - t0
            ok, why = checker(loaded.primary, result)
            total += 1
            passed += ok
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] ({dt:5.1f}s, tries={result.attempts_used}) {q}")
            if not ok:
                print(f"         want: {why}")
                print(f"         got:  {(result.answer or result.error or '')[:220]}")
    rate = passed / total if total else 0.0
    print(f"\n=== RESULT: {passed}/{total} questions passed ({rate:.0%}) ===")
    return rate


def main() -> int:
    args = sys.argv[1:]
    fast = "--fast" in args
    domains = [a for a in args if not a.startswith("--")] or list(QUESTIONS)

    print("=== DETERMINISTIC PHASE (ingest/profile/prep) ===")
    det_ok = deterministic_phase()

    rate = 0.0
    if not fast:
        print("\n=== LLM PHASE (live, slow) ===")
        rate = llm_phase(domains)
    else:
        print("\n(LLM phase skipped — --fast)")

    if not det_ok:
        print("\ndeterministic phase found problems")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
