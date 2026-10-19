# Phase 1+3 — Agent Package Split + Test Hardening (Design Spec)

Date: 2026-09-30
Status: awaiting user review
Scope decision: Phase 1 (structure) and Phase 3 (adversarial/hardening tests) built together, because the new tests protect the refactor. Phases 2 and 4 come later as separate approved efforts.

## Context

`src/jane/agent.py` is 1,046 lines holding orchestration, prompting, JSON parsing,
verification guards, error hints, and chart binding in one module. Public surface
in active use:

- `AnalystAgent`, `AgentContext` — app.py, benchmark.py, all 4 eval scripts, smoke_test.py
- `SYSTEM_PROMPT` — internal to agent, but prompt text must not drift
- Private helpers imported by tests/evals: `_answer_matches_data`, `_parse_model_json`,
  `_bind_chart_data`, `_extract_numbers`

This phase restructures without behavior change, and adds the test hardening that
keeps the refactor honest.

## Design

### 1. Package split (behavior-neutral)

`src/jane/agent.py` becomes `src/jane/agent/` with `agent.py`'s current content
distributed one-responsibility-per-module:

```
src/jane/agent/
├── __init__.py      # re-exports: AnalystAgent, AgentContext, SYSTEM_PROMPT,
│                    #   _parse_model_json, _bind_chart_data, _answer_matches_data,
│                    #   _extract_numbers  → every existing import keeps working
├── orchestrator.py  # AnalystAgent, AgentContext, the attempt loop, retry plumbing
├── prompting.py     # SYSTEM_PROMPT + _verification_request + schema-block assembly
├── parsing.py       # _parse_model_json and model-JSON repair logic
├── verification.py  # _extract_numbers, _result_numbers, _derivable,
│                    #   _answer_matches_data, _contradicts_result, _rate_question,
│                    #   _factual_answer, _percentize_answer, _winner_mismatch,
│                    #   _direction_contradiction, _conditional_pct_sanity,
│                    #   _profit_sanity, _SUPERLATIVE_*
├── error_hints.py   # _error_hint, _GENERIC_HINT
└── charting.py      # _bind_chart_data
```

Rules:
- Pure code motion first, committed separately before anything else touches it.
- `orchestrator.py` imports from siblings; siblings never import orchestrator
  (acyclic). The one shared dependency (`AgentContext` dataclass) moves to
  `orchestrator.py` and is imported where needed.
- No signature changes, no logic changes, no prompt text changes in this step.
  Diff-driven verification: `git diff` between old module and new package must
  show moved, not rewritten, code.

### 2. Error taxonomy (small, feeds the existing hint system)

New `src/jane/errors.py`:

```python
class JaneError(Exception):
    kind: str          # one of ERROR_KINDS below
    detail: str = ""

# kinds are 1:1 with the CURRENT _error_hint rules — coarser kinds would
# collapse e.g. the reserved-word hint into the generic schema hint and
# silently change retry behavior (the regression session 16 fixed)
ERROR_KINDS = (
    "schema_missing_column",   # BinderException: column not found
    "schema_reserved_word",    # left/ORDER/... used as a column name
    "sql_parse",               # Parser/Binder syntax-level failures
    "python_reset_collision",  # cannot insert X, already exists
    "python_scalar",           # scalar used where DataFrame required
    "python_safety",           # static safety analysis rejection
    "python_exec",             # runtime failure in sandboxed python
    "result_empty",            # empty result set
    "timeout",                 # wall-clock limit hit
    "generic",                 # unmapped — current generic hint
)
```

- `classify_error(text: str) -> str` maps raw exception text to a kind; the
  regexes from today's `_error_hint` become its matchers.
- `_error_hint(kind, err)` keys off the kind; the returned hint strings are
  byte-identical to today's.
- The retry loop classifies once and passes the kind through.
- Tests: every kind has a classify test; the hint for each kind is asserted
  equal to the current hint text (snapshot per kind).

### 3. Adversarial sandbox tests (no product code changes expected)

Extend `tests/test_sandbox.py` with escape-attempt and resource cases. All must
come back `ok=False` with a safety error, or time out cleanly — never hang the
suite (each gets a short explicit timeout) and never touch the real filesystem.

- Direct + indirect escape variants:
  - `__import__("os").system("...")` and the string-built
    `getattr(builtins, "__import__")` style — including via `chr()` composition
    and `getattr` chains (`().__class__.__bases__[0].__subclasses__()`)
  - `eval`, `exec`, `compile`, `open`, `os.system`, `subprocess`,
    `globals`, `locals`, `setattr`, `vars`
  - `builtins` reachability: `__builtins__` lookup, `help()`, `input()`,
    `breakpoint()`, `exit()`
  - `importlib` directly and via `from x import y` aliases
  - attribute-chain sneak paths: `pandas.io`, `numpy.__loader__`, `json.decoder`
    used as a module root
- Resource limits (Windows has no memory rlimit — wall-clock timeout is the only
  hard limit, so these assert clean termination + `timed_out`/error, not OOM kills):
  - `while True: pass` → timeout, suite continues
  - giant list allocation (e.g. `[0] * 10**10`) → error or timeout, never crash
  - huge DataFrame construction from range → timeout at the cap
  - deep recursion → RecursionError surfaced as a normal failure
  - very large printed output → stdout capped, no hang

Acceptance: sandbox suite runs green in < ~90s total on this machine.

### 4. Analytical failure-mode tests (deterministic, no LLM)

New `tests/test_failure_modes.py` — feeds hand-built DataFrames through
`_answer_matches_data`, `_winner_mismatch`, `_direction_contradiction`,
`_conditional_pct_sanity`, `_profit_sanity`, `classify_error`, and the sandbox:

- wrong denominator (percentage computed over all rows vs filtered subset)
- `COUNT(*)` vs conditional count on a 0/1 flag
- average presented where the question asked for median (and vice versa)
- percentage vs percentage-point confusion (0.385 vs 38.5 vs +3.5pp)
- incorrect date grouping (month boundaries, year rollover)
- duplicate rows inflating a SUM (and dedupe changing the answer)
- NULL handling: NULLs dropped vs counted as zero
- zero denominators (rate over an empty group)
- negative values (profit sanity must not reject negative margins outright)
- ranking ties (two rows share the max — winner check must not guess)
- top-N off-by-one (4 rows returned for top-3)
- wrong sort direction (ASC where DESC was asked)
- join double-counting (fact table joined to a duplicated dimension)
- empty groups and missing categories

Each case: construct the lie, assert the specific guard rejects it (or the
documented behavior where the guard deliberately passes — e.g. negative margins).

### 5. Concurrency guard test

Two rapid `analyze()` calls share no mutable state (the agent holds tables per
call). Test: two agents, interleaved table dicts, no cross-contamination of
results. This pins a property the refactor must not break.

## Non-goals

- No structured answer contracts / metric semantics (Phase 2, later).
- No Planner→Validator→Executor pipeline rewrite.
- No prompt text changes beyond moving them to `prompting.py` (versioning is a
  Phase 4 item).
- No new dependencies; property testing stays hand-rolled.

## Test & verification plan

1. After the pure code-motion commit: `pytest -q` 147/147 + all 4 eval suites
   re-run (deterministic phases; LLM phases optional if Ollama is up).
2. After the taxonomy commit: full suite + eval suites again.
3. After test-hardening commits: new tests green, old 147 still green.
4. The app boots on :8501 and one real chat answer verifies end-to-end.

## Done when

- `src/jane/agent.py` no longer exists as a module; the package exposes the same
  names; zero import changes needed anywhere in the repo.
- Every error kind is classified and hint-selected by test.
- Escape and resource tests all green with documented behavior.
- Failure-mode tests green; no guard behavior change.
- Total suite time stays reasonable (< ~3 min).
