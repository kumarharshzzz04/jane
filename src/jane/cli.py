"""Terminal entry point: ask a question about a file without launching the UI.

Also: `jane data.csv --prep tidy --out clean.csv` runs deterministic
preprocessing (see jane.preprocess) without touching the model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .agent import AnalystAgent, AgentContext
from .config import settings
from .ingest import load_file
from .ollama_client import OllamaClient, OllamaError
from .preprocess import OPS, PreprocessError, _resolve, clean_default, run_ops


def _run_prep(df: pd.DataFrame, spec: str, out: str | None, src: Path) -> int:
    """--prep: run ops on the first sheet, write a cleaned CSV, print the log."""
    try:
        if spec.strip().lower() == "tidy":
            res = clean_default(df)
        else:
            ops = [(_resolve(name), {}) for name in spec.split(",")]
            res = run_ops(df, ops)
    except PreprocessError as e:
        print(f"error: {e}", file=sys.stderr)
        print(f"available ops: {', '.join(sorted(OPS))}", file=sys.stderr)
        return 2

    out_path = Path(out) if out else src.with_name(src.stem + "_cleaned.csv")
    res.df.to_csv(out_path, index=False)
    print(f"[prep] {src.name}: {len(df)} rows -> {out_path.name}: {len(res.df)} rows")
    for line in res.log:
        print(f"[prep]   {line}")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to cp1252; model answers and prep logs use UTF-8.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        prog="jane",
        description="Chat with a spreadsheet using a local LLM (Ollama + DuckDB).",
    )
    parser.add_argument("file", help="CSV/Excel/JSON/Parquet file to analyze")
    parser.add_argument("question", nargs="*", help="question to ask about the data")
    parser.add_argument("--sheet", default=None, help="pick one sheet of an Excel file")
    parser.add_argument("--model", default=None, help="override the Ollama model")
    parser.add_argument("--prep", default=None,
                        help="preprocess instead of asking: 'tidy' or comma-separated "
                             "ops (clean_text,to_datetime,fill_missing,dedupe,scale,encode,clamp_outliers)")
    parser.add_argument("--out", default=None,
                        help="output CSV path for --prep (default: <file>_cleaned.csv)")
    args = parser.parse_args(argv)

    path = Path(args.file)
    if not path.exists():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 2

    try:
        loaded = load_file(path, sheet=args.sheet,
                           progress=lambda s: print(f"[ingest] {s}"))
    except Exception as e:
        print(f"error: could not load {path.name}: {e}", file=sys.stderr)
        return 2

    if args.prep:
        return _run_prep(loaded.sheets[0].df, args.prep, args.out, path)

    if not args.question:
        parser.error("a question is required unless --prep is given")

    tables = {s.name: s.df for s in loaded.sheets}
    client = OllamaClient(settings.ollama_host, settings.request_timeout)
    if args.model:
        client.model_override = args.model

    agent = AnalystAgent(client)
    ctx = AgentContext(tables=tables, active_table=loaded.sheets[0].name)
    question = " ".join(args.question)

    def progress(step: str, detail: str = "") -> None:
        print(f"[agent] {step}" + (f": {detail}" if detail else ""))

    try:
        result = agent.analyze(question, ctx, progress=progress)
    except OllamaError as e:
        print(f"error: {e}", file=sys.stderr)
        print("hint: start Ollama with `ollama serve` and pull a model, e.g. "
              "`ollama pull qwen2.5-coder:7b`", file=sys.stderr)
        return 1

    print()
    print(result.answer or result.error or "(no answer)")
    for t in result.tables:
        print(f"\n--- {t['name']} ({len(t['rows'])} rows) ---")
        print(json.dumps(t["rows"][:20], indent=2, default=str))
    if result.code:
        print(f"\n--- generated {('python' if 'import' in result.code else 'SQL')} ---")
        print(result.code)
    for chart in result.charts:
        print("\n--- vega-lite chart ---")
        print(json.dumps(chart, indent=2, default=str)[:2000])
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
