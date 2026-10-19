# Jane — Engineering Improvements TODO

A prioritized roadmap based on the code/architecture review of Jane.

---

## 🔴 P0 — High Priority

### 1. Split `agent.py`

**Current problem:** `src/jane/agent.py` has grown very large and mixes many responsibilities.

It currently handles things such as:
- Agent orchestration
- Prompt construction
- Model response parsing
- Retry/correction logic
- Answer verification
- Numerical claim checking
- Error interpretation
- Chart generation
- Result sanity checks

**Goal:** Keep the agent orchestrator small and move specialized logic into separate modules.

Suggested structure:

```text
src/jane/
├── agent/
│   ├── orchestrator.py
│   ├── prompting.py
│   ├── parsing.py
│   ├── verification.py
│   ├── corrections.py
│   ├── error_hints.py
│   └── charting.py
```

**Done when:**
- `agent.py` is reduced to orchestration.
- Each module has one clear responsibility.
- Existing tests still pass.
- No behavior changes are introduced during the refactor.

---

### 2. Reduce domain-specific correctness heuristics

**Current problem:** Some correctness checks contain logic tied to specific concepts/datasets such as churn, default, returned, readmission, etc.

This works for known datasets but makes Jane less general.

**Goal:** Move toward schema/question-derived semantic validation.

Instead of hardcoding:

```text
"churn" → special handling
"default" → special handling
"returned" → special handling
```

derive the expected operation from the question and available schema.

Example:

```text
User question
    ↓
Semantic intent
    ↓
Metric + grouping + filters
    ↓
Expected denominator / aggregation
    ↓
Generated query
    ↓
Independent verification
```

**Done when:**
- Adding a new dataset does not require adding a new hardcoded domain rule.
- Common analytical operations work across unrelated datasets.
- Domain-specific rules are isolated and documented where they are genuinely necessary.

---

### 3. Introduce structured analytical answer contracts

**Current problem:** Jane currently validates numerical claims after the model produces natural-language prose.

That works, but parsing numbers from prose is inherently heuristic.

**Goal:** Make the model return structured claims before generating the final explanation.

Example:

```json
{
  "claim": {
    "metric": "profit",
    "group": "Furniture",
    "value": 12345,
    "comparison": {
      "group": "Technology",
      "value": 11200,
      "operation": "relative_difference"
    }
  }
}
```

Jane can independently calculate/verify the claim and then generate prose from verified values.

**Done when:**
- Important numerical claims have machine-readable representations.
- Jane can independently verify the values.
- Final prose is generated from verified data rather than being the source of truth.
- Unsupported claims are rejected before reaching the user.

---

## 🟠 P1 — Reliability & Generalization

### 4. Formalize metric semantics

Create an internal representation for common analytical operations:

```text
COUNT
COUNT_DISTINCT
SUM
AVG
MEDIAN
MIN
MAX
PERCENTAGE
RATE
GROWTH
RANK
TOP_N
COMPARISON
```

For each metric, define:
- Required columns
- Aggregation
- Filter conditions
- Denominator
- Comparison operation
- Expected output type

Example:

```json
{
  "metric": "conversion_rate",
  "numerator": "converted_users",
  "denominator": "eligible_users",
  "operation": "percentage"
}
```

This gives the verifier something deterministic to reason about.

---

### 5. Make percentage verification more formal

Percentage calculations are one of the easiest places for an LLM to make subtle mistakes.

Explicitly model:

```text
percentage = numerator / denominator × 100

percentage_change =
(new - old) / old × 100

percentage_point_change =
new_percentage - old_percentage
```

Jane should distinguish between:

- Percentage
- Percentage change
- Percentage-point difference
- Ratio
- Rate

**Done when:** Tests cover all of these cases.

---

### 6. Improve independent verification

Whenever possible, verify model-generated analytical results independently.

For example:

```text
LLM generates SQL
       ↓
SQL executes
       ↓
Result obtained
       ↓
Independent verifier calculates expected metric
       ↓
Compare
       ↓
Accept / reject
```

The verifier should not simply ask the same LLM whether the answer is correct.

---

### 7. Expand adversarial analytical tests

Add tests specifically designed to catch:

- Wrong denominator
- Wrong aggregation
- `COUNT()` vs conditional count
- Average vs median
- Percentage vs percentage-point difference
- Incorrect date grouping
- Duplicate rows
- NULL handling
- Zero denominators
- Negative values
- Missing categories
- Empty groups
- Ties in rankings
- Top-N mistakes
- Incorrect sorting
- Incorrect joins
- Double-counting after joins
- Incorrect profit/revenue calculations

The goal is to test **failure modes**, not just happy paths.

---

## 🟠 P1 — Security / Sandbox Hardening

### 8. Keep Python execution aggressively restricted

Maintain the current AST validation and sandbox approach, but continuously expand the test suite.

Test attempts involving:

```python
__import__(...)
eval(...)
exec(...)
open(...)
os.system(...)
subprocess(...)
globals()
locals()
getattr(...)
setattr(...)
object.__subclasses__()
```

Also test indirect variants rather than only obvious strings.

**Done when:** Every blocked capability has a regression test.

---

### 9. Add resource-limit tests

Test:

- Infinite loops
- Extremely large allocations
- Huge DataFrames
- Excessive recursion
- Very expensive operations
- Extremely large generated outputs
- Timeout behavior

Verify that the process terminates cleanly and Jane remains usable.

---

## 🟡 P2 — Architecture / Maintainability

### 10. Introduce a clearer pipeline object

Instead of passing many loosely related values around, consider a structured analysis context:

```python
AnalysisContext(
    question=...,
    schema=...,
    query=...,
    result=...,
    metric=...,
    verification=...,
)
```

This makes the execution pipeline easier to understand and test.

---

### 11. Separate planning from execution

Aim for:

```text
Planner
   ↓
Analysis Plan
   ↓
Validator
   ↓
Executor
   ↓
Verifier
   ↓
Presenter
```

The planner should decide **what should happen**.

The executor should decide **how to execute it**.

The verifier should decide **whether the result is trustworthy**.

The presenter should decide **how to communicate it**.

---

### 12. Make retries state-aware

Instead of:

```text
attempt 1 failed
→ try again
→ try again
→ stop
```

make each retry understand the previous failure:

```text
Attempt 1
    ↓
Failure type:
  wrong column
  wrong aggregation
  invalid SQL
  incorrect result
    ↓
Targeted correction
    ↓
Attempt 2
```

This avoids wasting attempts on the same mistake.

---

### 13. Standardize error types

Create explicit error categories:

```text
SchemaError
PlanningError
SQLValidationError
SQLExecutionError
PythonValidationError
PythonExecutionError
ResultValidationError
AnswerVerificationError
ChartValidationError
TimeoutError
ResourceLimitError
```

This makes retry behavior much easier to control.

---

## 🟡 P2 — Visualization

### 14. Validate Vega-Lite specs before rendering

Before displaying a generated chart:

```text
LLM spec
   ↓
Schema validation
   ↓
Column existence check
   ↓
Data-type compatibility
   ↓
Row/data-size limit
   ↓
Render
```

Reject charts referencing nonexistent columns or incompatible encodings.

---

### 15. Add chart intent validation

If the user asks:

> "Show monthly revenue"

Jane should verify that the chart actually represents:

```text
date/month → revenue
```

rather than blindly trusting the model's chart specification.

---

## 🟢 P3 — UX / Product Improvements

### 16. Explain why an analysis failed

Instead of generic:

```text
Analysis failed.
```

show:

```text
Jane couldn't complete this analysis because:

The generated query referenced `customer_age`,
but your dataset contains `age`.

You can:
• Retry
• Edit the query
• Open manual analysis
```

Keep technical details available without overwhelming normal users.

---

### 17. Make generated SQL/Python easier to inspect

Keep the transparency feature, but add:

- Copy button
- Run button
- Clear error highlighting
- Line numbers
- Execution time
- Row count
- Export result

This turns the generated code from a debugging artifact into a useful analytical interface.

---

### 18. Show provenance for every answer

A useful result panel could show:

```text
Answer
↓
Source data
↓
Query executed
↓
Rows returned
↓
Metric calculated
↓
Verification status
```

Example:

```text
Revenue: ₹1.24M

✓ Calculated from 9,994 rows
✓ SQL executed successfully
✓ Numerical claim verified
```

---

## 🔵 P3 — Performance

### 19. Measure execution performance

Track:

```text
LLM planning time
SQL execution time
Python execution time
Verification time
Chart generation time
Total response time
```

This will help identify where Jane is actually slow.

---

### 20. Add dataset-size benchmarks

Benchmark Jane against datasets such as:

```text
10K rows
100K rows
1M rows
10M rows
50M+ rows
```

Track:

- Import time
- Schema generation time
- Query time
- Memory usage
- Chart rendering time
- End-to-end latency

---

## 🔵 P3 — Testing Infrastructure

### 21. Create a benchmark question suite

Build a fixed dataset + question benchmark.

Example:

```text
Question
Expected operation
Expected result
Allowed tolerance
Failure category
```

Then run it whenever Jane changes.

This becomes Jane's regression suite.

---

### 22. Add model comparison tests

Run the same benchmark across supported models.

Track:

```text
SQL success rate
Python success rate
Correct-answer rate
Verification rejection rate
Average retries
Latency
Token usage
```

This gives you actual evidence for which models work best with Jane.

---

### 23. Add property-based tests where useful

For numerical verification, test mathematical properties rather than only fixed examples.

Examples:

```text
0 ≤ percentage ≤ 100
AVG(x) lies between MIN(x) and MAX(x)
COUNT(DISTINCT x) ≤ COUNT(x)
```

This can uncover edge cases that manually written tests miss.

---

# 🧹 Code Quality Cleanup

### 24. Add type hints everywhere practical

Prioritize public functions and pipeline boundaries.

Prefer:

```python
def execute_query(
    query: str,
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    ...
```

over ambiguous interfaces.

---

### 25. Reduce duplicated validation logic

If several places independently check:

```text
column exists
result isn't empty
value is numeric
```

centralize those checks.

One validation utility should ideally be reused everywhere.

---

### 26. Keep prompts versioned

Treat important prompts like code.

For example:

```text
prompts/
├── planner_v1.txt
├── planner_v2.txt
├── answer_v1.txt
└── chart_v1.txt
```

This makes model behavior changes reproducible.

---

# 🚀 Bigger Future Ideas

These are not necessary immediately, but they could push Jane into a much more serious system.

### 27. Build an intermediate analysis plan

Instead of going directly from question → SQL:

```text
"What percentage of customers churned last year?"
```

produce:

```json
{
  "metric": "percentage",
  "numerator": "churned_customers",
  "denominator": "customers",
  "filter": "year = 2025"
}
```

Then generate SQL from the plan.

This gives you a deterministic intermediate layer.

---

### 28. Independent calculation engine

For common metrics, Jane could calculate the answer independently of the generated SQL.

For example:

```text
SQL says:
churned = 217
customers = 1024

Verifier:
217 / 1024 × 100
= 21.1914%

LLM says:
21.2%

✓ verified
```

This would significantly strengthen trust.

---

### 29. Dataset semantic layer

Build a lightweight semantic representation:

```text
Column
├── name
├── dtype
├── nullable
├── cardinality
├── likely semantic meaning
├── example values
└── statistics
```

Then Jane can reason about datasets without relying entirely on raw column names.

---

### 30. Analysis provenance / reproducibility

Allow users to save:

```text
Dataset
+
Question
+
Model
+
Prompt version
+
Generated query
+
Result
+
Verification
+
Chart
```

Then an analysis can be reproduced later.

---

# Suggested priority order

Don't try to implement everything at once.

### Phase 1 — Reliability

```text
1. Split agent.py
2. Structured answer contracts
3. Formal metric semantics
4. Independent numerical verification
5. Adversarial test suite
```

### Phase 2 — Robustness

```text
6. Better retry/error categories
7. Sandbox/resource-limit tests
8. Vega-Lite validation
9. Dataset-size benchmarks
10. Model benchmark suite
```

### Phase 3 — Architecture

```text
11. AnalysisContext
12. Planner → Validator → Executor → Verifier → Presenter
13. Dataset semantic layer
14. Provenance/reproducibility
```

### Phase 4 — Polish

```text
15. Better error UX
16. Better code inspection
17. Performance instrumentation
18. Prompt versioning
```

---

# ⭐ North-star architecture

The eventual Jane pipeline should look roughly like this:

```text
                         USER QUESTION
                              │
                              ▼
                    ┌──────────────────┐
                    │  Semantic Planner │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Analysis Plan   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │     Validator    │
                    └────────┬─────────┘
                             │
                             ▼
                 ┌─────────────────────────┐
                 │ SQL / Python Generator  │
                 └────────────┬────────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │ Sandbox/Executor│
                     └────────┬────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │ Result Validator│
                     └────────┬────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │ Independent     │
                     │ Verifier        │
                     └────────┬────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
                  FAIL                 PASS
                    │                   │
                    ▼                   ▼
              Targeted retry       Presentation
                                        │
                              ┌─────────┴─────────┐
                              ▼                   ▼
                           Answer              Chart
```

## Final principle

> **The LLM should propose the analysis. Jane should decide whether the analysis is valid.**

That's the architectural direction I'd preserve above everything else.

The model should be replaceable.

The dataset should be arbitrary.

The verifier should be deterministic wherever possible.

And the user should always be able to inspect what happened.

That would take Jane from a cool local AI project into a genuinely robust analytical system.
