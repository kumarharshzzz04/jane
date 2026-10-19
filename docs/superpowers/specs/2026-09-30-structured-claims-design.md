# Phase 2 — Structured Answer Claims (Design Spec)

Date: 2026-09-30
Status: awaiting user review
Parent: 2026-09-30-agent-split-hardening-design.md (Phase 2 was a non-goal there; this is its own approved effort)

## Problem

Today the model writes its explanation BEFORE the code runs, and the truth guard
parses numbers out of prose to check them against the result. Parsing prose is
heuristic: percent forms (38.5 vs 0.385), comma grouping, derivations
("12.5% more"), and small-integer exemptions all exist because prose is lossy.
TODO #3's fix: the model ALSO returns machine-readable claims; Jane verifies
each independently; prose stays but is no longer the only source of truth.

## Principle (from the TODO, unchanged)

> The LLM proposes the analysis. Jane decides whether it is valid.
> The verifier is deterministic wherever possible.

## Design

### 1. Claims ride the existing single JSON reply (no extra model call)

A second LLM round-trip per answer doubles latency on a local 7B model. Instead,
rule 5's JSON shape gains one OPTIONAL field:

```json
{
  "plan": "...", "language": "sql", "code": "...", "explanation": "...",
  "claims": [
    {"metric": "churn rate", "group": "month-to-month", "value": 38.5, "unit": "percent"},
    {"metric": "profit", "group": "Furniture", "value": 12345,
     "comparison": {"group": "Technology", "value": 11200, "operation": "pct_change"}}
  ]
}
```

- `unit`: `"number" | "percent" | "fraction"` — percent (38.5) and fraction
  (0.385) normalize to the same magnitude before checking (formalizes the
  existing ×100 blessing as data, TODO #5).
- `comparison.operation`: `"difference"` (a−b), `"pct_change"` ((a−b)/b×100),
  `"ratio"` (a/b). The comparison's numbers are verified the same way.
- Omitting `claims` is ALWAYS legal. The prompt says so in one line; models
  that skip it fall back to today's prose path unchanged (the fallback contract).

### 2. Verification: `verification.py::verify_claims(claims, result)`

Deterministic, no model calls:

- For each claim: `value` must match some result cell within tolerance
  (same tolerance scale the truth guard uses: max(0.051, |t|×0.002)),
  after unit normalization. Metric/group are descriptive labels — the NUMBERS
  are what get verified (a free-text metric taxonomy is deliberately NOT built;
  noted as Phase 2b if a need emerges).
- With `comparison`: both values must exist, and the stated operation must hold
  arithmetically between them within tolerance.
- Property tests (hand-rolled, no new deps): fraction↔percent round-trip,
  0 ≤ percent ≤ 100 on normalized claims, tolerance symmetry for large values.
- Returns (ok, failures) where each failure names the claim — this is what
  makes the retry hint targeted.

### 3. Orchestrator wiring: verify immediately after execution succeeds

Placement: right after `res.ok` / `sres.ok` is established and BEFORE the
truth-guard block. A failed claim triggers the CODE-level retry before any
verification round-trip is spent on prose that was built on wrong numbers.

- claims present + all verified → proceed exactly as today (truth guard still
  runs on the prose; claims verify the numbers, prose guards the narrative).
- claims present + any failed → set `last_error` with the named failing claim
  ("Your claim [churn rate / month-to-month] says 38.5 but the result contains
  54.2 — recompute and re-answer") and `continue` the attempt loop — exactly
  the winner-check pattern, consuming a normal attempt from max_attempts, no
  separate retry budget.
- claims absent / malformed → byte-identical to today's behavior. Any JSON
  weirdness in claims degrades to no-claims, never to an error.

### 4. What is explicitly NOT built (YAGNI)

- No second model call for claims (latency).
- No metric-type registry (COUNT/SUM/MEDIAN/... semantics) — the number
  arithmetic is what gets verified; metric taxonomy deferred.
- No prose regeneration from claims — the existing _factual_answer /
  _percentize answers already serve as model-free fallbacks.
- No chart changes.

## Files touched

- `src/jane/agent/prompting.py` — rule 5 JSON shape + ~4 lines of claims spec.
- `src/jane/agent/parsing.py` — tolerant claims extraction (malformed → None).
- `src/jane/agent/verification.py` — verify_claims + unit normalization.
- `src/jane/agent/orchestrator.py` — wire verification after execution; the
  targeted-retry branch mirrors the winner-check pattern.
- `tests/test_claims.py` — new: unit tests for verification, stub-client
  end-to-end for all three paths (claims-pass / claims-fail-retry /
  claims-absent → unchanged), percentage-semantics property tests.

## Fallback contract (the risk control)

For a 7B model the risk is prompt-bloat degrading compliance. Controlled by:
- claims are optional and documented as such in the prompt;
- ANY parse oddity → no-claims path;
- if live evals show regression, the prompt section is one deletable block —
  the feature turns off by removing the prompt lines (claims become absent
  everywhere and behavior is exactly pre-Phase-2).

## Done when

- All three paths proven by stub-client tests (pass / targeted-retry / absent).
- Percentage semantics covered: fraction vs percent vs pct_change vs
  pct_points confusion cases rejected or normalized correctly.
- Full suite green (233 + new), all 4 eval suites import + deterministic phases
  unchanged, app boots on :8501.
- A live Ollama run on retail_sales shows claims in the reply and the receipt
  unchanged for the user.
