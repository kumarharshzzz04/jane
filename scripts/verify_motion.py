"""Verify the split moved every original code line (behavior-neutral check)."""
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
orig = subprocess.run(
    ["git", "show", "HEAD:src/jane/agent.py"],
    capture_output=True, encoding="utf-8", errors="replace",
).stdout
orig_lines = orig.splitlines()
start = next(i for i, l in enumerate(orig_lines, 1) if l.startswith("SYSTEM_PROMPT = "))
orig_code = {l.strip() for l in orig_lines[start - 1:] if l.strip()}

moved = set()
for f in Path("src/jane/agent").glob("*.py"):
    moved |= {l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.strip()}

missing = [l for l in sorted(orig_code) if l not in moved
           and not l.startswith(("from .config", "from .engine", "from .logging_setup",
                                 "from .ollama_client", "from .sandbox", "from .schema",
                                 "from .types", "from .validator", "from .rag",
                                 "from .history"))]
print("orig code lines:", len(orig_code), "| missing:", len(missing))
for m in missing[:10]:
    print("  MISS:", m[:110])
sys.exit(1 if missing else 0)
