"""Benchmark Jane's data phases (no LLM) + optional LLM round-trips.

Usage:
  .venv/Scripts/python scripts/benchmark.py [datadir]            # data phases only
  .venv/Scripts/python scripts/benchmark.py --llm [datadir]      # + live agent calls

Measures: ingest, schema build, DuckDB registration, SQL execution,
sandbox round-trip, and (with --llm) full agent runs with real Ollama calls.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from jane.agent import AnalystAgent, AgentContext
from jane.config import settings
from jane.engine import SQLEngine
from jane.ingest import load_file
from jane.ollama_client import OllamaClient
from jane.sandbox import run_code
from jane.schema import build_schema_block, register_in_duckdb

USE_LLM = "--llm" in sys.argv
argv = [a for a in sys.argv[1:] if not a.startswith("--")]
data_dir = Path(argv[0]) if argv else Path("data/bench")


class Timer:
    def __init__(self) -> None:
        self.marks: list[tuple[str, float]] = []

    def step(self, label: str, extra: str = ""):
        t0 = self.marks[-1][1] if self.marks else time.perf_counter()
        dt = time.perf_counter() - t0
        self.marks.append((label, time.perf_counter()))
        print(f"    {label:32s} {dt:7.2f}s  {extra}")
        return dt


def bench_ingest(path: Path) -> tuple[dict[str, pd.DataFrame], list[float]]:
    t = Timer()
    t.marks.append(("start", time.perf_counter()))
    times: list[float] = []
    t0 = time.perf_counter()
    loaded = load_file(path, progress=lambda s: None)
    times.append(time.perf_counter() - t0)
    t.step(f"ingest {path.name} ({path.stat().st_size/1e6:.1f} MB)",
           f"{len(loaded.sheets)} sheet(s), {len(loaded.primary):,} rows")
    tables = {s.name: s.df for s in loaded.sheets}

    t0 = time.perf_counter()
    block = build_schema_block(tables)
    times.append(time.perf_counter() - t0)
    t.step("schema block", f"{len(block):,} chars")

    t0 = time.perf_counter()
    eng = SQLEngine(tables)
    times.append(time.perf_counter() - t0)
    t.step("duckdb registration")

    tname = list(tables)[0]
    cols = list(tables[tname].columns)
    # pick a numeric column to aggregate and a low-cardinality one to group by
    num_col = next((c for c in cols
                    if pd.api.types.is_numeric_dtype(tables[tname][c])), cols[0])
    cat_col = next((c for c in cols
                    if not pd.api.types.is_numeric_dtype(tables[tname][c])
                    and tables[tname][c].nunique() <= 50), cols[0])
    sql = (f"SELECT {cat_col}, SUM({num_col}) AS total, COUNT(*) AS n "
           f"FROM {tname} GROUP BY {cat_col} ORDER BY total DESC")
    t0 = time.perf_counter()
    res = eng.execute(sql)
    times.append(time.perf_counter() - t0)
    t.step("SQL: group by category",
           f"{res.row_count} rows, ok={res.ok}" if res.ok else f"ERROR {res.error}")
    t0 = time.perf_counter()
    res2 = eng.execute(f"SELECT {cat_col}, AVG({num_col}) a FROM {tname} GROUP BY {cat_col}")
    times.append(time.perf_counter() - t0)
    t.step("SQL: avg rollup", f"{res2.row_count} rows")

    t0 = time.perf_counter()
    res2 = eng.execute(f"SELECT {cat_col}, COUNT(DISTINCT {num_col}) d, MEDIAN({num_col}) m FROM {tname} GROUP BY {cat_col}")
    times.append(time.perf_counter() - t0)
    t.step("SQL: distinct+median", f"{res2.row_count} rows")

    # sandbox: ship tables to child, groupby in pandas
    t0 = time.perf_counter()
    name = list(tables)[0]
    sres = run_code(
        f"result_df = df.groupby('{cat_col}', as_index=False)['{num_col}'].sum()",
        tables, active_table=name, timeout=120,
    )
    times.append(time.perf_counter() - t0)
    t.step("sandbox: pandas groupby",
           f"ok={sres.ok}, {len(sres.tables[0]['rows']) if sres.ok and sres.tables else 0} rows"
           + ("" if sres.ok else f" — {sres.error[:120]}"))

    eng.close()
    return tables, times


def bench_llm(tables: dict[str, pd.DataFrame], questions: list[str]) -> None:
    client = OllamaClient(settings.ollama_host, settings.request_timeout)
    try:
        models = [m["name"] for m in client.tags()]
    except Exception as e:
        print(f"  Ollama unreachable ({e}) — skipping LLM phase")
        return
    agent = AnalystAgent(client)
    active = list(tables)[0]
    print(f"  models: {models}")
    for q in questions:
        ctx = AgentContext(tables=tables, active_table=active)
        t0 = time.perf_counter()
        try:
            result = agent.analyze(q, ctx, progress=lambda step, detail="": None)
        except Exception as e:
            print(f"  Q: {q}\n    EXCEPTION: {e}")
            continue
        dt = time.perf_counter() - t0
        status = "OK " if result.ok else "FAIL"
        print(f"  Q: {q}")
        print(f"    {status} {dt:6.1f}s attempts={result.attempts_used} model={result.model_used}")
        if result.ok:
            first = result.tables[0] if result.tables else None
            if first:
                print(f"    answer rows: {len(first['rows'])} | first: {first['rows'][0] if first['rows'] else '[]'}")
        else:
            print(f"    error: {(result.error or '')[:200]}")


def main() -> int:
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        print(f"no CSVs in {data_dir} — run scripts/gen_datasets.py first")
        return 1

    print(f"=== DATA PHASES ({'with' if USE_LLM else 'no'} LLM) — {len(files)} datasets ===")
    all_times: dict[str, list[float]] = {}
    for path in files:
        print(f"\n--- {path.name} ---")
        tables, times = bench_ingest(path)
        all_times[path.name] = times

    print("\n=== SUMMARY (seconds) ===")
    labels = ["ingest", "schema", "duckdb reg", "sql grp", "sql rollup",
              "sql median", "sandbox"]
    header = "dataset".ljust(20) + "".join(l.ljust(12) for l in labels)
    print(header)
    for name, times in all_times.items():
        print(name.ljust(20) + "".join(f"{t:<12.2f}" for t in times))

    if USE_LLM:
        mid = next((f for f in files if "mid" in f.name), None)
        big = next((f for f in files if "big" in f.name), None)
        questions = [
            "Which region has the highest total revenue?",
            "What is the average discount by product, as a table sorted descending?",
            "Show monthly revenue trend and a bar chart by region.",
        ]
        if mid:
            print(f"\n=== LLM PHASE — {mid.name} ===")
            t0 = time.perf_counter()
            tables_mid, _ = bench_ingest(mid)
            all_times[mid.name + " (2nd)"] = _
            print(f"  (re-ingest {time.perf_counter() - t0:.1f}s)")
            bench_llm(tables_mid, questions[:2])
        if big:
            print(f"\n=== LLM PHASE — {big.name} ===")
            tables_big, _ = bench_ingest(big)
            all_times[big.name + " (2nd)"] = _
            bench_llm(tables_big, questions[:1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
