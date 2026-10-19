"""One-shot: split src/jane/agent.py into the agent/ package (pure code motion).

Every slice is asserted against its expected anchor before writing, and each
new module gets only a docstring + the imports it needs (orchestrator keeps the
full original import block to stay behavior-neutral). Deleted after success.
"""
import ast
import pathlib
import shutil
import sys

root = pathlib.Path(__file__).resolve().parents[1]
src = root / "src" / "jane" / "agent.py"
text = src.read_text(encoding="utf-8")
lines = text.splitlines(keepends=True)


def find(marker: str, start: int = 0) -> int:
    for i in range(start, len(lines)):
        if marker in lines[i]:
            return i
    raise SystemExit(f"anchor not found: {marker!r}")


def cut(a: int, b: int) -> str:
    """Lines [a, b) 0-indexed."""
    return "".join(lines[a:b])


# ---- anchors (line numbers are 1-based in the read output) ----------------
i_system = find('SYSTEM_PROMPT = """You are Jane')          # 27
i_ctx = find("@dataclass")                                   # 50 (class AgentContext)
i_bind = find("def _bind_chart_data")                        # 57
i_parse = find("def _parse_model_json")                      # 65
i_hints = find("_ERROR_HINTS: list")                         # 109-ish
i_generic = find("_GENERIC_HINT = (")                        # 123
i_extract = find("def _extract_numbers")                     # 127
i_rate = find("def _rate_question")                          # 213
i_factual = find("def _factual_answer")                      # 219
i_cond = find("def _conditional_pct_sanity")                 # 274
i_sup = find('_SUPERLATIVE_MAX = (')                         # 324
i_verify_req = find("def _verification_request")             # 451
i_contra = find("def _contradicts_result")                   # 465
i_error_hint = find("def _error_hint")                       # 576
i_cls = find("class AnalystAgent")                           # 597

# sanity: ordering as read
assert i_system < i_ctx < i_bind < i_parse < i_hints < i_generic \
       < i_extract < i_rate < i_factual < i_cond < i_sup \
       < i_verify_req < i_contra < i_error_hint < i_cls, "anchor order"

L = lambda n: n - 1  # 1-based -> 0-based

# ---- slices (0-based, end-exclusive) --------------------------------------
H = cut(0, L(i_system))                         # module docstring + imports
# orchestrator docstring is prepended INSIDE the head, after the original
# docstring but before `from __future__` — __future__ must stay file-first
# among real imports, so instead: keep head byte-exact and place the new
# docstring by rewriting the first docstring line is skipped — orchestrator
# gets no extra docstring; the original module docstring serves.
s_prompt = cut(L(i_system), L(i_ctx))           # SYSTEM_PROMPT
s_ctx = cut(L(i_ctx), L(i_bind))                # AgentContext
s_bind = cut(L(i_bind), L(i_parse))             # _bind_chart_data
s_parse = cut(L(i_parse), L(i_hints))           # _parse_model_json
s_err = cut(L(i_hints), L(i_extract))           # _ERROR_HINTS + _GENERIC_HINT + _error_hint? no:
# _error_hint is at 576, AFTER verification helpers. _ERROR_HINTS.._GENERIC_HINT ends before _extract_numbers.
s_verif = cut(L(i_extract), L(i_error_hint))    # all verification helpers + _SUPERLATIVE + _verification_request
s_hint_fn = cut(L(i_error_hint), L(i_cls))      # _error_hint function
s_cls = cut(L(i_cls), len(lines))               # AnalystAgent to EOF

# _error_hint references _ERROR_HINTS/_GENERIC_HINT (in s_err) — both go to error_hints.py.
# verification helpers reference _rate_question etc. — all inside s_verif.
assert "_ERROR_HINTS" in s_err and "_GENERIC_HINT" in s_err
assert "def _error_hint" in s_hint_fn
assert "def _extract_numbers" in s_verif and "def _profit_sanity" in s_verif
assert "def _winner_mismatch" in s_verif and "def _direction_contradiction" in s_verif
assert "class AnalystAgent" in s_cls and "SYSTEM_PROMPT" in s_prompt
assert "from __future__" in H.splitlines()[3] or "from __future__" in H

pkg = root / "src" / "jane" / "agent"
if pkg.exists():
    shutil.rmtree(pkg)
pkg.mkdir()

H = cut(0, L(i_system))  # original docstring + full import block, byte-exact

(pkg / "orchestrator.py").write_text(
    H + s_ctx + s_cls, encoding="utf-8")

(pkg / "prompting.py").write_text(
    '"""System prompt + verification request text (prompt text lives here)."""\n'
    + s_prompt, encoding="utf-8")

(pkg / "parsing.py").write_text(
    '"""Model reply parsing — defensive JSON extraction."""\n'
    'import json\nimport re\n\n\n'
    + s_parse, encoding="utf-8")

(pkg / "verification.py").write_text(
    '"""Deterministic answer verification guards (no model calls here)."""\n'
    'import re\n\n\n'
    + s_verif, encoding="utf-8")

(pkg / "error_hints.py").write_text(
    '"""Raw-error -> actionable retry hints, keyed by classified kind."""\n'
    'import re\n\n\n'
    + s_err + "\n\n" + s_hint_fn, encoding="utf-8")

(pkg / "charting.py").write_text(
    '"""Bind executed rows into model-proposed Vega-Lite specs."""\n\n\n'
    + s_bind, encoding="utf-8")

(pkg / "__init__.py").write_text(
    '"""Jane analysis agent package — re-exports the historical module surface."""\n'
    'from .orchestrator import AgentContext, AnalystAgent  # noqa: F401\n'
    'from .prompting import SYSTEM_PROMPT  # noqa: F401\n'
    'from .parsing import _parse_model_json  # noqa: F401\n'
    'from .charting import _bind_chart_data  # noqa: F401\n'
    'from .verification import (  # noqa: F401\n'
    '    _answer_matches_data, _conditional_pct_sanity, _contradicts_result,\n'
    '    _direction_contradiction, _extract_numbers, _factual_answer,\n'
    '    _percentize_answer, _profit_sanity, _rate_question,\n'
    '    _verification_request, _winner_mismatch)\n'
    'from .error_hints import _error_hint  # noqa: F401\n', encoding="utf-8")

# orchestrator must now import the moved names from siblings FIRST (anchored
# on the original single-dot text), then gain a dot on project imports (it
# lives one level deeper now)
orch = pkg / "orchestrator.py"
otext = orch.read_text(encoding="utf-8")
otext = otext.replace(
    "from .validator import extract_code_blocks, extract_sql\n",
    "from .validator import extract_code_blocks, extract_sql\n"
    "from .charting import _bind_chart_data\n"
    "from .error_hints import _error_hint\n"
    "from .parsing import _parse_model_json\n"
    "from .prompting import SYSTEM_PROMPT\n"
    "from .verification import (_answer_matches_data, _conditional_pct_sanity,\n"
    "                           _contradicts_result, _direction_contradiction,\n"
    "                           _factual_answer, _percentize_answer,\n"
    "                           _profit_sanity, _rate_question,\n"
    "                           _verification_request, _winner_mismatch)\n")
assert "from .charting import" in otext, "sibling import injection failed"
for mod in ("config", "engine", "logging_setup", "ollama_client",
            "sandbox", "schema", "types", "validator"):
    otext = otext.replace(f"from .{mod} import", f"from ..{mod} import")
otext = otext.replace("from .rag import", "from ..rag import")
otext = otext.replace("from .history import", "from ..history import")
# the just-injected sibling imports stay single-dot: they are intra-package
for sib in ("charting", "error_hints", "parsing", "prompting", "verification"):
    otext = otext.replace(f"from ..{sib} import", f"from .{sib} import")
orch.write_text(otext, encoding="utf-8")

src.unlink()  # agent.py replaced by the package

# ---- verify: every module parses, package imports, names present ----------
for p in sorted(pkg.glob("*.py")):
    ast.parse(p.read_text(encoding="utf-8"))
sys.path.insert(0, str(root / "src"))
import jane.agent as A  # noqa: E402
for name in ("AnalystAgent", "AgentContext", "SYSTEM_PROMPT", "_parse_model_json",
             "_bind_chart_data", "_answer_matches_data", "_extract_numbers",
             "_error_hint", "_profit_sanity", "_verification_request"):
    assert hasattr(A, name), name
print("split OK:", sorted(p.name for p in pkg.glob("*.py")))
