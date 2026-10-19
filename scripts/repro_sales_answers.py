"""Reproduce (and later grade) Jane's answers on the user's own sales-ish CSVs.

smartphone_data.csv: camera, age, ram, cpu_score, slot_sd, sims, price
housing.csv:         California housing prices (median_house_value, ocean_proximity, ...)

Run: .venv/Scripts/python.exe scripts/repro_sales_answers.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

import pandas as pd

from jane.agent import AgentContext, AnalystAgent
from jane.config import settings
from jane.ingest import load_file
from jane.ollama_client import OllamaClient

SMARTPHONE = Path.home() / "Downloads" / "smartphone_data.csv"
HOUSING = Path.home() / "Downloads" / "housing.csv.csv"

QUESTIONS: list[tuple[Path, str, str]] = [
    # (path, question, truth label + value)
    (SMARTPHONE, "What is the average price of the phones?", "mean(price)"),
    (SMARTPHONE, "Which RAM size has the highest average price?",
     "argmax(mean price by ram)"),
    (SMARTPHONE, "Do phones with 2 SIM slots cost more than phones with 1?",
     "mean(price) by sims"),
    (SMARTPHONE, "Is there a link between RAM and price? Plot it.", "corr(ram, price)"),
    (SMARTPHONE, "Break down the average price by camera count.",
     "mean(price) by camera"),
    (HOUSING, "What is the average house value?", "mean(median_house_value)"),
    (HOUSING, "Which ocean proximity has the highest average house value?",
     "argmax(mean value by ocean_proximity)"),
    (HOUSING, "Break down the average house value by ocean proximity.",
     "mean(value) by ocean_proximity"),
    (HOUSING, "How strong is the link between median income and median house value? Plot it.",
     "corr(median_income, median_house_value)"),
    (HOUSING, "Are houses near the bay more expensive than inland ones?",
     "mean(value) NEAR BAY vs INLAND"),
]


def truths() -> dict[str, str]:
    out: dict[str, str] = {}
    sp = pd.read_csv(SMARTPHONE)
    out["mean(price)"] = f"{sp['price'].mean():.2f}"
    g = sp.groupby("ram")["price"].mean()
    out["argmax(mean price by ram)"] = f"ram={g.idxmax()} -> {g.max():.2f}"
    g2 = sp.groupby("sims")["price"].mean()
    out["mean(price) by sims"] = " | ".join(f"sims={k}: {v:.2f}" for k, v in g2.items())
    out["corr(ram, price)"] = f"{sp['ram'].corr(sp['price']):.4f}"
    g3 = sp.groupby("camera")["price"].mean()
    out["mean(price) by camera"] = " | ".join(f"cam={k}: {v:.2f}" for k, v in g3.items())

    hs = pd.read_csv(HOUSING)
    out["mean(median_house_value)"] = f"{hs['median_house_value'].mean():.2f}"
    g4 = hs.groupby("ocean_proximity")["median_house_value"].mean().sort_values(ascending=False)
    out["argmax(mean value by ocean_proximity)"] = (
        " | ".join(f"{k}: {v:.0f}" for k, v in g4.items()))
    out["mean(value) by ocean_proximity"] = out["argmax(mean value by ocean_proximity)"]
    out["corr(median_income, median_house_value)"] = \
        f"{hs['median_income'].corr(hs['median_house_value']):.4f}"
    b = hs[hs["ocean_proximity"] == "NEAR BAY"]["median_house_value"].mean()
    i = hs[hs["ocean_proximity"] == "INLAND"]["median_house_value"].mean()
    out["mean(value) NEAR BAY vs INLAND"] = f"NEAR BAY: {b:.0f} vs INLAND: {i:.0f}"
    return out


def main() -> int:
    truth = truths()
    print("=== TRUTHS ===")
    for k, v in truth.items():
        print(f"  {k}: {v}")
    print()

    client = OllamaClient(settings.ollama_host, settings.request_timeout)
    agent = AnalystAgent(client)

    cache: dict[Path, tuple[str, AgentContext]] = {}
    for path, question, label in QUESTIONS:
        if path not in cache:
            loaded = load_file(str(path), progress=lambda s: None)
            tables = {s.name: s.df for s in loaded.sheets}
            cache[path] = (loaded.sheets[0].name,
                           AgentContext(tables=tables, active_table=loaded.sheets[0].name))
        active, ctx = cache[path]
        t0 = time.perf_counter()
        try:
            res = agent.analyze(question, ctx, progress=lambda *a, **k: None)
            dt = time.perf_counter() - t0
            answer = res.answer or res.error or "(empty)"
            guards = getattr(res, "guards", None)
            gtxt = ""
            if guards:
                gtxt = " ".join(f"{g.name}:{g.status}" for g in guards)
            print(f"--- [{path.name}] {question}")
            print(f"    truth:  {truth[label]}")
            print(f"    answer: {answer}")
            print(f"    meta:   ok={res.ok} tries={res.attempts_used} {dt:.1f}s guards=[{gtxt}]")
            if res.code:
                print(f"    sql:    {' '.join(res.code.split())}")
            if res.tables and res.columns:
                try:
                    print(f"    rows:   {res.columns[0]} {res.tables[0][:6]}")
                except Exception:
                    pass
            print()
        except Exception as e:  # noqa: BLE001
            print(f"--- [{path.name}] {question}\n    [EXC] {type(e).__name__}: {e}\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
