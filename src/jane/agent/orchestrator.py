"""Analysis agent — schema injection + code generation + bounded self-correction.

Primary path:  NL -> SQL (whitelisted, executed by DuckDB).
Python path:   NL -> sandboxed pandas code (for things SQL can't do).
Fallback:      after max_attempts, hand back code + traceback for the manual pane.
"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from ..config import settings
from ..engine import SQLEngine, QueryResult
from ..logging_setup import get_logger
from ..ollama_client import OllamaClient, OllamaError
from ..sandbox import SandboxResult, analyze_code_safety, run_code
from ..schema import build_schema_block
from ..types import AnalysisCancelled, AnalysisResult, ChatMessage
from ..validator import extract_code_blocks, extract_sql
from .charting import _bind_chart_data
from .error_hints import _error_hint
from .parsing import _parse_claims, _parse_model_json
from .prompting import SYSTEM_PROMPT
from .verification import (_answer_matches_data, _answerable_refusal,
                           _association_missing_coefficient, _breakdown_coverage,
                           _conditional_pct_sanity, _contradicts_result,
                           _direction_contradiction, _factual_answer,
                           _flatten_json_answer, _has_prompt_placeholders,
                           _looks_like_json_answer, _missing_label_column,
                           _pct_attribution, _pct_retry_hint, _percent_claim,
                           _percentize_answer,
                           _profit_sanity, _rate_question,
                           _superlative_value_mismatch, _tidy_answer,
                           _verification_request, _winner_mismatch,
                           guard_report, verify_claims)

log = get_logger("jane.agent")


def _raise_if_cancelled(cancel: "threading.Event | None") -> None:
    """Stop a run the user has cancelled. Checked between attempts and right
    after each model call, so the worst case is one model call's latency, not
    the whole retry budget."""
    if cancel is not None and cancel.is_set():
        raise AnalysisCancelled("stopped by the user")


def _as_prose(value) -> str:
    """Explanation/plan as text. A 7B sometimes nests the answer as a JSON
    object ({"verdict": …}); the guards regex the prose, so a dict here used to
    crash the whole ask ('expected string or bytes-like object, got dict', then
    "'dict' object has no attribute 'strip'"). Serializing keeps its numbers
    traceable and the run alive."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)

@dataclass
class AgentContext:
    tables: dict[str, pd.DataFrame]
    active_table: Optional[str] = None
    history: list[ChatMessage] = field(default_factory=list)
    extra_context: str = ""  # retrieved notes/profiles (RAG), never numbers


class AnalystAgent:
    def __init__(self, client: OllamaClient | None = None):
        self.client = client or OllamaClient(settings.ollama_host, settings.request_timeout)

    # -- model selection ------------------------------------------------------

    def _pick_model(self) -> str:
        override = getattr(self.client, "model_override", None)
        if override and self.client.has_model(override):
            return override
        if self.client.has_model(settings.primary_model):
            return settings.primary_model
        if self.client.has_model(settings.fallback_model):
            return settings.fallback_model
        try:
            models = self.client.tags()
        except OllamaError:
            raise
        if models:
            return models[0]["name"]
        raise OllamaError("no models available in Ollama — run `ollama pull qwen2.5-coder:7b`")

    # -- main entry ------------------------------------------------------------

    @staticmethod
    def _notes() -> list[str]:
        """Load the analyst's notes for RAG (best-effort)."""
        try:
            from ..rag import load_notes
            return load_notes()
        except Exception:
            return []

    def analyze(self, question: str, ctx: AgentContext,
                progress=lambda step, detail="": None,
                cancel: threading.Event | None = None) -> AnalysisResult:
        """Answer one question. `cancel` is checked between attempts and after
        each model call, so a stopped run raises AnalysisCancelled (nothing is
        shipped or stored) instead of holding the bridge's lock to the end."""
        model = self._pick_model()
        progress("model", model)
        schema_block = build_schema_block(ctx.tables, active_table=ctx.active_table)
        # local RAG: user notes + column profiles, relevance-filtered
        try:
            from ..rag import retrieve_context
            from ..history import HistoryStore
            ctx.extra_context = retrieve_context(
                question, ctx.tables, self._notes())
            if ctx.extra_context:
                progress("context", "notes/profiles retrieved")
        except Exception as e:
            log.warning("rag skipped: %s", e)
        engine = SQLEngine(ctx.tables)
        try:
            return self._loop(question, ctx, model, schema_block, engine, progress,
                              cancel=cancel)
        finally:
            engine.close()

    def _loop(self, question: str, ctx: AgentContext, model: str,
              schema_block: str, engine: SQLEngine, progress,
              cancel: threading.Event | None = None) -> AnalysisResult:
        max_attempts = settings.max_attempts
        last_error = ""
        last_code = ""
        attempt = 0

        empty_count = 0
        repairs = 0
        claims_retries = 0
        while attempt < max_attempts:
            _raise_if_cancelled(cancel)
            attempt += 1
            progress("attempt", f"{attempt}/{max_attempts}")
            messages = self._build_messages(question, ctx, schema_block, attempt,
                                            last_error, last_code)
            try:
                raw = self.client.chat(messages, model)
            except OllamaError as e:
                return AnalysisResult(ok=False, error=f"model error: {e}",
                                      attempts_used=attempt, model_used=model)
            _raise_if_cancelled(cancel)   # do not start another attempt on a stopped run

            try:
                parsed = _parse_model_json(raw)
            except json.JSONDecodeError:
                last_error = f"Your reply was not valid JSON: {raw[:400]!r}. Respond with ONLY the JSON object."
                progress("error", "model output was not valid JSON")
                continue

            language = (parsed.get("language") or "sql").lower()
            code = (parsed.get("code") or "").strip()
            # comparison-question flag must exist BEFORE statement splitting:
            # the multi-SELECT refusal reads it on ANY question whose first
            # reply contains two statements (was an UnboundLocalError crash).
            q = question.lower()
            is_compare_q = any(w in q for w in ("compare", " vs ", "versus",
                                                "both groups"))
            if language == "sql" and ";" in code:
                # the model loves two-statement fixes; keep only the first
                # statement if it is complete, else drop trailing semicolon.
                # EXCEPTION: a first attempt of "one side per statement" means
                # the question wants BOTH sides — refuse immediately with the
                # CASE-pattern hint instead of silently answering half of it.
                stmts = [s.strip() for s in code.split(";") if s.strip()]
                if len(stmts) > 1 and is_compare_q:
                    last_error = ("You sent multiple SQL statements (one per group) — that is why the "
                                  "validator rejected you. ONE query with conditional aggregation does "
                                  "both groups: SELECT MEDIAN(CASE WHEN churned = 'Yes' THEN "
                                  "monthly_charges END) AS med_churned, MEDIAN(CASE WHEN churned = 'No' "
                                  "THEN monthly_charges END) AS med_stayed FROM churn — copy that "
                                  "pattern with your own table/column names. Two SELECTs separated by "
                                  "a comma or semicolon are INVALID; the CASE WHEN form is the only "
                                  "way. Your next reply MUST use that exact shape.")
                    progress("error", "multiple statements on a comparison question")
                    continue
                if len(stmts) > 1:
                    progress("note", "model sent multiple statements — using the first")
                code = stmts[0]
            last_code = code
            plan = _as_prose(parsed.get("plan", ""))
            explanation = _as_prose(parsed.get("explanation", ""))
            chart = parsed.get("chart") if isinstance(parsed.get("chart"), dict) else None
            if not code:
                last_error = "Your JSON had no 'code' field."
                continue

            if language == "sql":
                progress("executing", "SQL via DuckDB")
                res: QueryResult = engine.execute(code)
                # a comparison question deserves computed comparison columns:
                # nudge the model when it returns a bare projection.
                # A bare auxiliary ("is", "are", "do") matches ordinary
                # questions ("how many rows ARE in data") — a comparison
                # needs a DIRECTION word too ("does X affect Y", "is A higher").
                aux = any(w in q for w in ("does", "do ", "is ", "are ",
                                           "was ", "were "))
                direction = any(w in q for w in ("more", "less", "better",
                                                 "worse", "higher", "lower",
                                                 "bigger", "smaller", "affect",
                                                 "impact", "difference",
                                                 "outperform"))
                comparative = (any(w in q for w in ("vs", "versus", "compare",
                                                    "compared", "between",
                                                    "than"))
                               or (aux and direction))
                # a rate question's overall answer IS a single value — don't
                # let the comparison guard fight it. A COMPARISON question
                # ("compare the medians for both groups") needs ≥2 result
                # columns or rows, not a single value.
                if res.ok and comparative and _rate_question(question) and not is_compare_q:
                    comparative = False
                # An ASSOCIATION question ("is there a relationship / link /
                # correlation between A and B") is answered by one coefficient,
                # a scatter, or a grouped table — never by BOTH sides side by
                # side. "between" alone must not drag it into the comparison
                # guards: CORR(a, b) is the whole answer, and demanding a
                # breakdown burned all 5 attempts on a question that was already
                # right.
                if res.ok and any(w in q for w in ("correlat", "relationship",
                                                   "link", "associated",
                                                   "association")):
                    comparative = False
                if res.ok and comparative and is_compare_q and res.row_count == 1 \
                        and len(res.columns) == 1:
                    last_error = ("You compared a value with itself: one row, one column. Compute BOTH "
                                  "sides in one query with conditional aggregation: SELECT "
                                  "MEDIAN(CASE WHEN <outcome> = 'Yes' THEN <col> END) AS med_yes, "
                                  "MEDIAN(CASE WHEN <outcome> = 'No' THEN <col> END) AS med_no FROM "
                                  "<table>; — one row, one column per group.")
                    progress("error", "comparison needs both sides")
                    continue
                # counting questions expect exactly one number back
                counting_q = any(w in q for w in ("how many", "how much",
                                                  "count", "total number",
                                                  "total rows"))
                if res.ok and comparative and not is_compare_q and not counting_q \
                        and res.row_count == 1 \
                        and len(res.columns) == 1:
                    last_error = ("Your result is a single bare value — it does not answer the comparison. "
                                  "Re-run computing BOTH sides side-by-side (e.g. group by the condition "
                                  "and aggregate the metric), so the user can see which wins.")
                    progress("error", "comparison needs both sides")
                    continue
                # UNION-alias bug detector: a comparative question whose result
                # has 2+ numeric columns that are IDENTICAL in every row means
                # the model selected the same expression twice under different
                # aliases (UNION ALL pattern) — the 'comparison' compares a
                # value with itself. Demand the CASE-per-group shape instead.
                if res.ok and comparative and len(res.columns) >= 2 and res.row_count >= 1:
                    def _cell(r, i):
                        return r[i] if isinstance(r, (list, tuple)) else None
                    num_cols = [i for i in range(len(res.columns))
                                if all(isinstance(_cell(r, i), (int, float))
                                       and not isinstance(_cell(r, i), bool)
                                       for r in res.rows[:20])]
                    if len(num_cols) >= 2 and all(
                            _cell(r, num_cols[0]) == _cell(r, j)
                            for r in res.rows[:20] for j in num_cols[1:]):
                        last_error = ("Your comparison columns hold IDENTICAL values in every row — "
                                      "you selected the same expression twice (typical UNION ALL with "
                                      "re-aliased columns). Compute per-group values in ONE query with "
                                      "CASE inside the aggregate: SELECT "
                                      "MEDIAN(CASE WHEN <outcome> = 'Yes' THEN <col> END) AS med_yes, "
                                      "MEDIAN(CASE WHEN <outcome> = 'No' THEN <col> END) AS med_no "
                                      "FROM <table>; — one row, one column per group, different values.")
                        progress("error", "comparison columns identical (UNION bug)")
                        continue
                # arithmetic-sanity guard for rate/percentage questions: a rate
                # grouped by the outcome itself is always 100.0 / 0.0 and means
                # the wrong query shape — demand the overall rate instead.
                # Only fires when EVERY rate-ish cell is degenerate AND the
                # result varies by the outcome column itself (per-group rates
                # like 54% / 12% / 9% are legitimate and must pass).
                # superlative-rate questions ('which group churns MOST?') may
                # legitimately return exactly 1.0 (e.g. 100% of 7-project
                # employees left) — do not call that degenerate.
                superlative_rate_q = any(w in q for w in (
                    "highest", "most", "lowest", "least", "maximum",
                    "minimum", "top"))
                if (res.ok and _rate_question(question)
                        and res.row_count >= 1 and not is_compare_q
                        and not superlative_rate_q):
                    rate_cols = [i for i, c in enumerate(res.columns)
                                 if any(isinstance(r[i] if isinstance(r, (list, tuple)) else None, (int, float))
                                        and (r[i] in (0.0, 100.0, 1.0))
                                        for r in res.rows[:20])]
                    if rate_cols and all(
                            (r[i] in (0.0, 100.0, 1.0))
                            for r in res.rows[:20] for i in rate_cols):
                        last_error = ("Your rate column only contains 0/1/100 — you almost certainly "
                                      "grouped by the outcome itself, so each group's 'rate' is trivial. "
                                      "For an OVERALL rate do NOT group by the outcome column: compute "
                                      "the single ratio over the whole table, e.g. SELECT "
                                      "SUM(CASE WHEN <outcome> = '<positive>' THEN 1 ELSE 0 END) * 1.0 "
                                      "/ COUNT(*) * 100 AS <rate> FROM <table>; — one row, one rate. "
                                      "For per-group rates group by a DIFFERENT dimension (contract, "
                                      "region, …) and each group's rate will differ. If your numerator is "
                                      "COUNT(CASE WHEN <cond> THEN 1 ELSE 0 END): that is the bug — COUNT "
                                      "counts every non-null value, and the ELSE 0 rows are non-null, so "
                                      "the ratio is always 1.0. Replace COUNT( with SUM( ... THEN 1 ELSE "
                                      "0 END) so the zeros count as zero. NEVER select both the rate and "
                                      "COUNT(*) under names like churned_count/churn_count for a rate "
                                      "question — show rate per group plus the group size.")
                        progress("error", "degenerate rate detected")
                        continue
                if not res.ok:
                    last_error = f"SQL error:\n{res.error}\n\n{_error_hint(res.error)}"
                    progress("error", "SQL failed")
                    continue
                # empty-result guard: a filtered query matching nothing usually
                # means a date-format/WHERE mismatch, not a true empty answer.
                # Retry 1: teach inspection. Retry 2: paste REAL sample rows —
                # models that ignore advice still read data shown to them.
                if res.ok and res.row_count == 0 and attempt < max_attempts:
                    empty_count += 1
                    if empty_count == 1:
                        last_error = ("Your query returned NO ROWS. That usually means a WHERE/JOIN/HAVING "
                                      "filter matched nothing (date-format mismatch, wrong column value, "
                                      "over-filtering). First inspect the data plainly — SELECT <time_col>, "
                                      "<metric_col> FROM <table> ORDER BY <metric_col> DESC LIMIT 10 — then "
                                      "answer from what you actually see.")
                        progress("error", "empty result — retrying with inspection hint")
                        continue
                    glimpse = []
                    active_df = ctx.tables.get(ctx.active_table)
                    if active_df is not None:
                        glimpse = json.loads(json.dumps(
                            active_df.head(8).to_dict("records"), default=str))
                    last_error = ("Your query returned NO ROWS twice — your filters/CTEs are wrong, "
                                  "stop using them. Here are real rows from "
                                  f"{ctx.active_table}:\n{json.dumps(glimpse, ensure_ascii=False)}\n"
                                  "The answer is visible in this data. Write ONE plain query — no CTEs, "
                                  "no PERCENTILE_CONT, no HAVING — e.g. SELECT <time_col>, <metric_col> "
                                  "FROM <table> ORDER BY <metric_col> DESC LIMIT 5, and report what it "
                                  "shows. Only say 'no results' if the answer is genuinely absent.")
                    progress("error", "empty result twice — retrying with data glimpse")
                    continue
                # grouped result lost its group column: one numeric column
                # over >=2 rows says nothing about which value is which group.
                if (res.ok and attempt < max_attempts
                        and _missing_label_column(res.columns, res.rows, code)):
                    last_error = (
                        "Your result has one value per group but never says WHICH "
                        "group each value belongs to — the group column is missing "
                        "from the SELECT. Return the group and its aggregate in one "
                        "plain query, e.g. SELECT <group_col>, AVG(<metric>) AS "
                        "<metric> FROM <table> GROUP BY <group_col> ORDER BY <metric> "
                        "DESC — no UNION, no two separate SELECTs.")
                    progress("error", "grouped result lost its group column — retrying")
                    continue
                # association guard: "is there a link between A and B" was
                # answered by a raw listing of the two columns — a listing
                # cannot tell the STRENGTH of a link, and the 7B then invents
                # one (live: ram/price scatter produced r = 0.73 copied from
                # the prompt on data whose true coefficient is -0.024).
                if (res.ok and attempt < max_attempts
                        and _association_missing_coefficient(question, res.columns,
                                                             res.rows)):
                    last_error = (
                        "Your result lists the two columns but never computes the "
                        "strength of the link, so it cannot answer 'how strong'. Return "
                        "the coefficient WITH the points in ONE query: SELECT <a>, <b>, "
                        "CORR(<a>, <b>) OVER () AS r FROM <table> LIMIT 200 — chart a vs b, "
                        "then state r's value, sign and strength. (For a single number: "
                        "SELECT CORR(<a>, <b>) AS r FROM <table>.)")
                    progress("error", "association question returned points without a coefficient — retrying")
                    continue
                # coverage guard: a "break down / who leads" question must
                # return EVERY group — models love LIMIT 1/2, and a trimmed
                # result reads as if the hidden regions do not exist. On the
                # final attempt the table ships with a warning instead of a
                # give-up over an otherwise good result.
                coverage_missing = (_breakdown_coverage(res.columns, res.rows,
                                                       ctx, question, code=code)
                                    if res.ok else None)
                if coverage_missing is not None and attempt < max_attempts:
                    cov_col, cov_n, cov_all, cov_missing = coverage_missing
                    last_error = (
                        f"Your result covers {cov_n} of {cov_all} {cov_col} values — it is "
                        f"missing {', '.join(str(m) for m in cov_missing[:8])}. A "
                        "'break down / who leads' question must return EVERY group, "
                        "ranked. Remove any LIMIT or filter and run ONE plain query: "
                        f"SELECT {cov_col}, <aggregate> FROM <table> GROUP BY {cov_col} "
                        "ORDER BY <aggregate> DESC — every group must be in the result.")
                    progress("error", "breakdown is missing groups — retrying")
                    continue
                # profit sanity: recompute (price-cost)*qty from the table and
                # reject results that contradict it (negative millions from a
                # revenue-looking column). Fires only on profit/margin questions
                # when price/cost/quantity-like columns all exist.
                bad_profit = (_profit_sanity(res.columns, res.rows, ctx, question,
                                             code=code)
                              if res.ok else None)
                if bad_profit is not None:
                    true_total, (p, co, qt), bad_val = bad_profit
                    last_error = (f"Your Profit values are IMPOSSIBLE: you reported {bad_val!r}. Recomputed from "
                                  f"the table itself, total profit is {true_total:,.2f} and per-row profit is "
                                  f"({p} - {co}) * {qt} — every row is positive. Do NOT use Sales_Amount or any "
                                  "other revenue-looking column; it is NOT price*quantity in this data. Rewrite: "
                                  f"SELECT <group_col>, SUM(({p} - {co}) * {qt}) AS Profit FROM <table> "
                                  "GROUP BY <group_col> ORDER BY Profit DESC — then report the top row.")
                    progress("error", "profit contradicts (price-cost)*qty — retrying")
                    continue
                # conditional-percentage sanity: 'Among <subset>, what % have
                # <condition>' — recompute both denominators in pandas. A result
                # near subset_count/all_count when subset_count << all_count means
                # the model divided by the WHOLE table (0.9% instead of 26.5%).
                if res.ok and res.row_count == 1 and len(res.columns) == 1:
                    _pct = _conditional_pct_sanity(res.columns[0], res.rows[0], ctx,
                                                   question, code=code)
                    if _pct is not None:
                        want_pct, n_sub, n_all = _pct
                        last_error = (f"Wrong denominator: you reported {_pct and res.rows[0]} of a total that includes ALL "
                                      f"{n_all} rows. The question restricts to a SUBSET of {n_sub} rows — filter "
                                      f"FIRST (WHERE ...), then compute the percentage INSIDE that subset. Correct "
                                      f"answer is about {want_pct:.1f}. Rewrite as: SELECT SUM(CASE WHEN <condition> "
                                      f"THEN 1 ELSE 0 END)*1.0/COUNT(*)*100 AS pct FROM <table> WHERE <subset-condition>.")
                        progress("error", "percentage denominator error — retrying")
                        continue
                shape_issues = SQLEngine.check_result_shape(res)
                sql_result = AnalysisResult(
                    ok=True, answer=explanation or plan, code=code,
                    stdout=plan, tables=[{"name": "query_result", "columns": res.columns, "rows": res.rows}],
                    attempts_used=attempt, model_used=model,
                )
                result = sql_result
                # --- structured claims (Phase 2): machine-readable numbers
                # verified deterministically. Absent/malformed claims -> None
                # -> legacy path, byte-identical (fallback contract). A failed
                # claim retries at the code level BEFORE prose verification.
                claims = _parse_claims(raw)
                if claims:
                    ok_claims, claim_fails = verify_claims(claims, sql_result)
                    if not ok_claims:
                        claims_retries += 1
                        if claims_retries <= 2:
                            # Show the ACTUAL rows: a model that ignores 'fix
                            # your claims' still reads the data it is handed
                            # (the trick that rescued the spike path). Without
                            # them a 7B repeats the same invented claim until
                            # the budget burns out.
                            shown = json.dumps({"columns": res.columns,
                                                "rows": res.rows[:8]}, default=str)
                            last_error = ("Your structured claims contradict the executed result:\n"
                                          + "\n".join(claim_fails)
                                          + f"\nThe executed result was: {shown[:900]}"
                                          + "\nRecompute and re-answer; copy every claim value from "
                                            "those rows and fix the claims to match the result.")
                            if any(_percent_claim(c) for c in claims
                                   if isinstance(c, dict)):
                                hint = _pct_retry_hint(res.columns, res.rows)
                                if hint:
                                    last_error += "\n" + hint
                            progress("error", "structured claim failed verification — retrying")
                            continue
                        # two retries could not fix the claims, but the
                        # executed result itself is sound: stop burning the
                        # budget — fall through to the prose guards + repair
                        # (which can still ship the answer) instead of a
                        # give-up with no table (live: the perfect 4-row
                        # region breakdown was thrown away exactly this way).
                        progress("note", "claims still inconsistent — verifying the prose instead")
                # --- truth guard: the explanation was written BEFORE the code
                # ran; make sure its numbers actually exist in the result ---
                # Three ways prose can be wrong even when the numbers match:
                # the prompt's own <placeholder> slots shipped verbatim, a
                # refusal/meta-comment where the grouped result IS the answer,
                # or numbers that are not in the result at all.
                repair_reason = None
                if _has_prompt_placeholders(explanation):
                    repair_reason = (
                        "Your explanation still contains the placeholder slots from the "
                        "instructions (<...>). Replace EVERY one with the actual label or "
                        "value from the executed result below.")
                elif _looks_like_json_answer(explanation):
                    repair_reason = (
                        "Your explanation is a JSON object. Rewrite it as ONE analyst's "
                        "sentence naming the finding and its value from the executed "
                        "result below — plain prose, no JSON, no markdown table.")
                elif _answerable_refusal(explanation, res.columns, res.rows, question):
                    repair_reason = (
                        "Your explanation refuses or talks about verification instead of "
                        "answering, but the executed result below IS the grouped answer the "
                        "question asked for. State the finding from those rows.")
                if res.ok and (repair_reason is not None
                               or not _answer_matches_data(explanation, sql_result, question)
                               or _contradicts_result(explanation, sql_result)):
                    tables_json = json.dumps(
                        {"columns": res.columns, "rows": res.rows[:25]}, default=str)
                    progress("verify", "answer needs a rewrite against the result — re-asking")
                    repairs += 1
                    try:
                        fix_raw = self.client.chat(
                            [ChatMessage("system", SYSTEM_PROMPT + "\n" + schema_block),
                             ChatMessage("user", question),
                             ChatMessage("assistant", raw),
                             ChatMessage("user", _verification_request(
                                 explanation, tables_json, repair_reason))],
                            model, temperature=0.0)
                        fixed = _parse_model_json(fix_raw)
                        fixed_expl = _as_prose(fixed.get("explanation")).strip()
                        if (fixed_expl and _answer_matches_data(fixed_expl, sql_result, question)
                                and not _has_prompt_placeholders(fixed_expl)
                                and not _looks_like_json_answer(fixed_expl)
                                and not _contradicts_result(fixed_expl, sql_result)):
                            result.answer = fixed_expl
                        else:
                            # still off — flatten a nested rewrite, else the
                            # model-free factual line
                            result.answer = (_flatten_json_answer(fixed_expl)
                                             or _factual_answer(res.columns, res.rows, question))
                    except (OllamaError, json.JSONDecodeError, ValueError):
                        result.answer = _factual_answer(res.columns, res.rows, question)
                else:
                    # verified prose IS the answer — the factual line is the
                    # last resort, never a replacement for a passing answer
                    # (live defect: this else discarded every verified
                    # explanation and shipped a truncated 'Executed result — …',
                    # so 7/7 questions on retail_sales read like a data dump).
                    if not (explanation or "").strip():
                        result.answer = _factual_answer(res.columns, res.rows)
                # format fix AFTER the truth guard (which may replace the
                # answer): if the question asks for a percentage and the
                # result is a lone 0-1 fraction, turn the echoed '0.385' into
                # '38.5%' — the guard's x100 blessing accepts that form.
                result.answer = _percentize_answer(result.answer, res.columns,
                                                   res.rows, question)
                result.answer = _tidy_answer(result.answer, question)
                # winner check: the answer must crown the row that actually
                # wins by the metric column (deterministic re-sort)
                bad_winner = _winner_mismatch(result.answer, res.columns,
                                              res.rows, question)
                if bad_winner is not None:
                    true_label, metric_col, sup = bad_winner
                    last_error = (f"Your answer names the wrong winner. Sorting by {metric_col} "
                                  f"{'descending' if sup == 'max' else 'ascending'}, the top row is "
                                  f"{true_label!r} — THAT is the {'highest' if sup == 'max' else 'lowest'}. "
                                  f"Re-answer naming {true_label!r} as the winner, with its "
                                  f"{metric_col} value from the result table.")
                    progress("error", "answer names the wrong winner — retrying")
                    continue
                sup_bad = _superlative_value_mismatch(result.answer, res.columns,
                                                      res.rows)
                if sup_bad is not None:
                    word, stated, true_v, true_lbl, sup = sup_bad
                    last_error = (
                        f"Your answer calls {stated:,.2f} the {word} value, but the result's "
                        f"rows say the {word} is {true_v:,.2f} ({true_lbl}). "
                        f"Re-answer with that row's label and value from the result, or drop "
                        f"the superlative entirely.")
                    progress("error", "superlative value contradicts the result — retrying")
                    continue
                pct_bad = _pct_attribution(result.answer, res.columns,
                                           res.rows, question)
                if pct_bad is not None:
                    subj, other, v_subj, v_other, stated, true_pct = pct_bad
                    last_error = (
                        f"Your percentage is misattributed: you say {subj!r} leads "
                        f"{other!r} by {stated}%, but the result has {subj!r} = "
                        f"{v_subj:,.2f} and {other!r} = {v_other:,.2f}, so "
                        f"({subj!r} - {other!r}) / {other!r} * 100 = {true_pct:.4g}%. "
                        "State the percentage for the pair you name (or the raw "
                        "difference), computed from the result.")
                    progress("error", "percentage does not match the pair it names — retrying")
                    continue
                direction_bad = _direction_contradiction(result.answer, res.columns,
                                                         res.rows, question)
                if direction_bad is not None:
                    subj, word, v1, v2 = direction_bad
                    last_error = (f"Your comparison direction contradicts your own numbers: you say "
                                  f"{subj!r} is {word}, but you cite {v1:,.2f} for it first and {v2:,.2f} "
                                  f"for the other group — {v1:,.2f} is NOT {'greater' if 'higher' in word or 'more' in word or 'greater' in word or 'bigger' in word else 'less'} "
                                  f"than {v2:,.2f}. Re-examine which value belongs to which group and "
                                  "answer with the direction the numbers actually show.")
                    progress("error", "comparison direction contradicts data — retrying")
                    continue
                if res.truncated:
                    result.error = f"result truncated to first rows (query matched more)"
                if chart and res.row_count >= 2 and len(res.columns) >= 2:
                    # a chart of one value is not a chart
                    result.charts.append(_bind_chart_data(chart, res.columns, res.rows))
                if coverage_missing is not None:
                    cov_col, cov_n, cov_all, cov_missing = coverage_missing
                    result.error = ((result.error + "\n" if result.error else "") +
                                    f"Coverage warning: {cov_n} of {cov_all} {cov_col} "
                                    f"values in the result (missing "
                                    f"{', '.join(str(m) for m in cov_missing[:8])}).")
                if shape_issues:
                    result.error = ((result.error + "\n" if result.error else "") +
                                    "Data-quality warnings: " + "; ".join(shape_issues))
                # the receipt: measured on the answer being returned, not assumed
                result.guards = guard_report(result.answer, result, question)
                result.repairs = repairs
                return result

            # python path
            violations = analyze_code_safety(code)
            if violations:
                last_error = ("Your code failed the safety check:\n" + "\n".join(violations) +
                              "\nRewrite using ONLY pandas/numpy/duckdb on the provided tables.")
                progress("error", "code blocked by safety analysis")
                continue

            progress("executing", "Python in sandbox")
            sres: SandboxResult = run_code(
                code, ctx.tables, active_table=ctx.active_table,
                timeout=settings.code_timeout, max_memory_mb=settings.max_memory_mb,
            )
            if not sres.ok:
                err = sres.error or ""
                if sres.safety_errors:
                    hint = ("\nUse only: pd, np, the table variables, tables, df, con, math, json, "
                            "re, datetime. Set result_df to a DataFrame; wrap scalar values in lists.")
                else:
                    hint = "\n" + _error_hint(err)
                last_error = f"Python execution failed:\n{err}{hint}"
                progress("error", "sandbox execution failed")
                continue

            result = AnalysisResult(
                ok=True, answer=explanation or plan, code=code, stdout=sres.stdout,
                tables=sres.tables, charts=sres.charts, figures=sres.figures,
                attempts_used=attempt, model_used=model,
            )
            # association guard (python path): a raw two-column listing cannot
            # tell the strength of a link — same contract as the SQL path.
            _first = result.tables[0] if result.tables else None
            if (_first is not None and attempt < max_attempts
                    and _association_missing_coefficient(
                        question, _first["columns"], _first["rows"])):
                last_error = (
                    "Your result lists the two columns but never computes the "
                    "strength of the link, so it cannot answer 'how strong'. Compute "
                    "the coefficient into the SAME result: result_df = df[['<a>', "
                    "'<b>']].head(200); result_df['r'] = df['<a>'].corr(df['<b>']) — "
                    "then state r's value, sign and strength.")
                progress("error", "association question returned points without a coefficient — retrying")
                continue
            # --- structured claims (python path): same contract as SQL ---
            claims = _parse_claims(raw)
            if claims:
                ok_claims, claim_fails = verify_claims(claims, result)
                if not ok_claims:
                    claims_retries += 1
                    if claims_retries <= 2:
                        last_error = ("Your structured claims contradict the executed result:\n"
                                      + "\n".join(claim_fails)
                                      + "\nRecompute and re-answer; fix the claims to match the result.")
                        progress("error", "structured claim failed verification — retrying")
                        continue
                    # the result itself is sound; let the prose guards decide
                    progress("note", "claims still inconsistent — verifying the prose instead")
            # --- truth guard (python path): same contract as the SQL path ---
            _pc, _pr = ((_first["columns"], _first["rows"]) if _first else ([], []))
            repair_reason = None
            if _has_prompt_placeholders(explanation):
                repair_reason = (
                    "Your explanation still contains the placeholder slots from the "
                    "instructions (<...>). Replace EVERY one with the actual label or "
                    "value from the executed result below.")
            elif _looks_like_json_answer(explanation):
                repair_reason = (
                    "Your explanation is a JSON object. Rewrite it as ONE analyst's "
                    "sentence naming the finding and its value from the executed "
                    "result below — plain prose, no JSON, no markdown table.")
            elif _answerable_refusal(explanation, _pc, _pr, question):
                repair_reason = (
                    "Your explanation refuses or talks about verification instead of "
                    "answering, but the executed result below IS the grouped answer the "
                    "question asked for. State the finding from those rows.")
            if result.ok and (repair_reason is not None
                              or not _answer_matches_data(explanation, result, question)
                              or _contradicts_result(explanation, result)):
                tables_json = json.dumps(
                    {"columns": _pc, "rows": _pr[:25]}, default=str)
                progress("verify", "answer needs a rewrite against the result — re-asking")
                repairs += 1
                try:
                    fix_raw = self.client.chat(
                        [ChatMessage("system", SYSTEM_PROMPT + "\n" + schema_block),
                         ChatMessage("user", question),
                         ChatMessage("assistant", raw),
                         ChatMessage("user", _verification_request(
                             explanation, tables_json, repair_reason))],
                        model, temperature=0.0)
                    fixed = _parse_model_json(fix_raw)
                    fixed_expl = _as_prose(fixed.get("explanation")).strip()
                    if (fixed_expl and _answer_matches_data(fixed_expl, result, question)
                            and not _has_prompt_placeholders(fixed_expl)
                            and not _looks_like_json_answer(fixed_expl)
                            and not _contradicts_result(fixed_expl, result)):
                        result.answer = fixed_expl
                    else:
                        result.answer = (_flatten_json_answer(fixed_expl)
                                         or _factual_answer(_pc, _pr, question))
                except (OllamaError, json.JSONDecodeError, ValueError):
                    pass
            result.answer = _tidy_answer(result.answer, question)
            # winner / superlative-value checks (python path parity with SQL):
            # a correct value table can still be crowned with the wrong extreme
            # in prose.
            bad_winner = _winner_mismatch(result.answer, _pc, _pr, question)
            if bad_winner is not None:
                true_label, metric_col, sup = bad_winner
                last_error = (f"Your answer names the wrong winner. Sorting by {metric_col} "
                              f"{'descending' if sup == 'max' else 'ascending'}, the top row is "
                              f"{true_label!r} — THAT is the {'highest' if sup == 'max' else 'lowest'}. "
                              f"Re-answer naming {true_label!r} as the winner, with its "
                              f"{metric_col} value from the result table.")
                progress("error", "answer names the wrong winner — retrying")
                continue
            sup_bad = _superlative_value_mismatch(result.answer, _pc, _pr)
            if sup_bad is not None:
                word, stated, true_v, true_lbl, sup = sup_bad
                last_error = (
                    f"Your answer calls {stated:,.2f} the {word} value, but the result's "
                    f"rows say the {word} is {true_v:,.2f} ({true_lbl}). "
                    f"Re-answer with that row's label and value from the result, or drop "
                    f"the superlative entirely.")
                progress("error", "superlative value contradicts the result — retrying")
                continue
            result.guards = guard_report(result.answer, result, question)
            result.repairs = repairs
            return result

        return AnalysisResult(
            ok=False,
            error=(
                f"Gave up after {attempt} attempts. Last error:\n{last_error}\n\n"
                f"Last generated code:\n```\n{last_code}\n```\n"
                "Open “Fix it yourself — edit and run” just below to edit and run it here."
            ),
            code=last_code or None,
            attempts_used=attempt,
            model_used=model,
        )

    def _build_messages(self, question: str, ctx: AgentContext, schema_block: str,
                        attempt: int, last_error: str, last_code: str) -> list[ChatMessage]:
        system = SYSTEM_PROMPT + "\n" + schema_block
        if ctx.extra_context:
            system += ("\n\n=== CONTEXT FROM YOUR ANALYST'S NOTES AND DATA PROFILES ===\n"
                       + ctx.extra_context + "\n=== END CONTEXT (background info only; "
                       "every number must still come from executing code) ===")
        msgs: list[ChatMessage] = [ChatMessage("system", system)]
        for m in ctx.history[-settings.max_history_messages:]:
            msgs.append(m)
        user = question
        if attempt > 1:
            user = (
                f"{question}\n\n"
                f"Your previous attempt FAILED.\nPrevious code:\n```\n{last_code}\n```\n"
                f"Error:\n{last_error}\n\nFix it and respond with the same JSON shape only."
            )
        msgs.append(ChatMessage("user", user))
        return msgs
