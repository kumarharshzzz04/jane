"""Live end-to-end smoke test: CSV -> schema -> model -> SQL -> result.

Requires Ollama running. Uses whatever model is available.
Run:  .venv/Scripts/python scripts/smoke_test.py "your question"
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jane.agent import AnalystAgent, AgentContext
from jane.config import settings
from jane.ingest import load_file
from jane.ollama_client import OllamaClient

CSV = """region,product,units,unit_price
north,widget,10,2.50
north,gadget,5,10.00
south,widget,8,2.50
south,gadget,12,10.00
east,widget,15,2.50
east,gadget,3,10.00
"""


def main() -> int:
    question = sys.argv[1] if len(sys.argv) > 1 else "Which region has the highest total revenue (units * unit_price)?"

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(CSV.encode("utf-8"))
        tmp_path = tmp.name

    loaded = load_file(tmp_path)
    tables = {s.name: s.df for s in loaded.sheets}

    client = OllamaClient(settings.ollama_host, settings.request_timeout)
    models = [m["name"] for m in client.tags()]
    print(f"[smoke] Ollama models available: {models}")

    agent = AnalystAgent(client)
    ctx = AgentContext(tables=tables, active_table=loaded.sheets[0].name)
    steps = []

    def progress(step, detail=""):
        steps.append(f"{step}: {detail}" if detail else step)
        print(f"  [step] {step} {detail}")

    print(f"[smoke] question: {question}")
    result = agent.analyze(question, ctx, progress=progress)

    print("\n=== RESULT ===")
    print("ok:", result.ok)
    print("model:", result.model_used, "| attempts:", result.attempts_used)
    print("answer:", result.answer)
    print("code:")
    print(result.code)
    if result.error:
        print("error/warnings:", result.error)
    for t in result.tables:
        print(f"table {t['name']}: {len(t['rows'])} rows")
        for row in t["rows"][:10]:
            print("   ", row)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
