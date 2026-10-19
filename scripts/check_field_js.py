"""Extract the injected field JS from app.py and node --check it."""
import ast
import re
import subprocess
import sys
from pathlib import Path

src = Path("src/jane/app.py").read_text(encoding="utf-8")
ast.parse(src)
print("py parse OK,", len(src.splitlines()), "lines")

arr = re.search(r"s\.textContent = \[(.*?)\]\.join", src, re.S).group(1)
parts = re.findall(r'"((?:[^"\\]|\\.)*)"', arr)
js = "".join(p.replace('\\"', '"').replace("\\\\", "\\") for p in parts)
out = Path("data/_field_check.js")
out.write_text(js, encoding="utf-8")
r = subprocess.run(["node", "--check", str(out)], capture_output=True, text=True)
out.unlink(missing_ok=True)
if r.returncode != 0:
    print(r.stderr)
    sys.exit(1)
print("field JS OK,", len(js), "chars")
