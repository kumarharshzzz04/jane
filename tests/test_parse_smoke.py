"""Smoke: every module must parse. Catches syntax breaks that pytest never
sees because no unit test imports app.py (live regression: a collapsed
newline in app.py killed every fresh server start)."""
import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "jane"


def test_app_py_parses():
    app = SRC / "app.py"
    ast.parse(app.read_text(encoding="utf-8"), filename=str(app))


def test_all_package_modules_parse():
    broken = []
    for py in sorted(SRC.glob("*.py")):
        try:
            ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except SyntaxError as e:
            broken.append(f"{py.name}: {e}")
    assert not broken, "syntax errors:\n" + "\n".join(broken)
