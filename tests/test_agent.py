"""Tests for agent.py: error-hint classifier, prompt contract, retry loop.

No Ollama required — the client is stubbed with canned JSON replies.
"""
import json

import pandas as pd
import pytest

from jane.agent import (AgentContext, AnalystAgent, _answer_matches_data,
                        _contradicts_result, _direction_contradiction,
                        _error_hint, _extract_numbers, _percentize_answer,
                        _profit_sanity, _winner_mismatch, SYSTEM_PROMPT)
from jane.types import ChatMessage
from jane.config import settings


class _FakeResult:
    """Minimal stand-in for AnalysisResult in truth-guard tests."""
    def __init__(self, rows):
        self.tables = [{"name": "r", "columns": ["a"], "rows": rows}]


# ---------------------------------------------------------------------------
# truth guard: answers may only contain numbers the data supports
# ---------------------------------------------------------------------------

def test_extract_numbers_handles_commas_and_percent():
    assert _extract_numbers("38.5% of 1,200 rows") == [38.5, 1200.0]


def test_guard_accepts_matching_answer():
    res = _FakeResult([{"count": 32}])
    assert _answer_matches_data("There are 32 movies above 8.", res, "above 8?")


def test_guard_rejects_hallucinated_lone_number():
    res = _FakeResult([{"count": 32}])
    assert not _answer_matches_data("There are 3 movies above 8.", res, "above 8?")


def test_guard_exempts_question_echoed_thresholds():
    res = _FakeResult([{"count": 32}])
    # '8' comes from the question, 32 matches the result
    assert _answer_matches_data("32 movies have a rating above 8.", res, "rating above 8?")


def test_guard_accepts_derivable_percentage():
    res = _FakeResult([{"yes": 54.21, "no": 48.15}])
    # 12.5% ≈ (54.21-48.15)/48.15*100 — derived, not fabricated
    assert _answer_matches_data("Discounted orders earn 12.5% more.", res, "compare")


def test_guard_accepts_percentage_narration_of_decimal_rate():
    # question asks for a percentage; result stores the decimal 0.385;
    # answer says 38.5% — same number, percent-formatted
    res = _FakeResult([{"churn_rate": 0.385}])
    assert _answer_matches_data(
        "The overall churn rate is 38.5%.", res,
        "What is the overall churn rate? Answer with the percentage.")


def test_guard_rejects_fabricated_rate_with_shared_total():
    # the regression from the live eval: answer shares '800' with the truth
    # but invents the rate and the numerator
    res = _FakeResult([{"churned": 308, "total": 800, "rate": 38.5}])
    assert not _answer_matches_data(
        "The churn rate is 12.5% — 100 customers out of 800 churned.",
        res, "overall churn rate?")


def test_guard_accepts_prose_small_integers_with_context():
    res = _FakeResult([{"region": "north", "total": 5123.4},
                       {"region": "south", "total": 4211.1}])
    assert _answer_matches_data("North leads with 5123.4; the top 3 are close.",
                                res, "top regions?")


def test_guard_accepts_qualitative_answers():
    res = _FakeResult([{"a": 1}])
    assert _answer_matches_data("The data is heavily skewed.", res, "?")


# ---------------------------------------------------------------------------
# percentize: lone 0-1 fraction + "answer with the percentage" question
# ---------------------------------------------------------------------------

def test_percentize_converts_fractional_echo():
    ans = _percentize_answer(
        "Executed result — churn_rate=0.385.", ["churn_rate"],
        [{"churn_rate": 0.385}],
        "What is the overall churn rate? Answer with the percentage.")
    assert ans == "The churn rate is 38.5%."


def test_percentize_leaves_real_prose_alone():
    assert _percentize_answer(
        "The churn rate is 38.5%, higher than expected.", ["churn_rate"],
        [{"churn_rate": 0.385}],
        "What is the overall churn rate? Answer with the percentage.") \
        == "The churn rate is 38.5%, higher than expected."


def test_percentize_ignores_non_fraction_and_multi_col():
    q = "What is the median? Answer with the percentage."
    assert _percentize_answer("x", ["med"], [{"med": 76.91}], q) == "x"
    assert _percentize_answer("x", ["a", "b"], [{"a": 0.5, "b": 2}], q) == "x"
    assert _percentize_answer("x", ["n"], [{"n": 3}], q) == "x"


# ---------------------------------------------------------------------------
# profit sanity: result profit vs recomputed (price-cost)*qty truth
# ---------------------------------------------------------------------------

def _profit_ctx():
    df = pd.DataFrame({"Product_Category": ["a", "a", "b"],
                       "Unit_Price": [100.0, 100.0, 50.0],
                       "Unit_Cost": [40.0, 40.0, 20.0],
                       "Quantity_Sold": [10, 5, 8],
                       "Sales_Amount": [500.0, 600.0, 300.0]})
    return AgentContext(tables={"sales": df}, active_table="sales")


def test_profit_sanity_flags_negative_when_truth_positive():
    bad = _profit_sanity(["Product_Category", "Profit"],
                         [["Food", -12519004.06]], _profit_ctx(),
                         "which product category has the highest profit for us")
    assert bad is not None
    true_total, (p, co, qt), _ = bad
    assert p == "Unit_Price" and co == "Unit_Cost" and qt == "Quantity_Sold"
    assert abs(true_total - (60*15 + 30*8)) < 0.01   # 900 + 240 = 1140


def test_profit_sanity_passes_correct_positive_profit():
    assert _profit_sanity(["Product_Category", "Profit"],
                          [["a", 900.0], ["b", 240.0]], _profit_ctx(),
                          "highest profit by category") is None


def test_profit_sanity_ignores_non_profit_questions():
    assert _profit_sanity(["Product_Category", "Profit"],
                          [["x", -5.0]], _profit_ctx(),
                          "top category by revenue") is None


# ---------------------------------------------------------------------------
# winner check: superlative answers must name the row that actually wins
# ---------------------------------------------------------------------------

def _profit_rows():
    return [["Clothing", 6922.0, 1712957.80],
            ["Furniture", 6729.0, 1779461.16],
            ["Electronics", 6096.0, 1574320.06],
            ["Food", 5608.0, 1421108.05]]


def test_winner_mismatch_catches_wrong_category():
    # the live regression: correct table, wrong winner named
    bad = _winner_mismatch(
        "The most profitable category is Clothing with $1,712,957.80.",
        ["category", "qty", "profit"], _profit_rows(),
        "which product category has the highest profit for us")
    assert bad is not None and bad[0] == "Furniture" and bad[1] == "profit"


def test_winner_mismatch_passes_correct_winner():
    assert _winner_mismatch(
        "The most profitable category is Furniture with $1,779,461.16.",
        ["category", "qty", "profit"], _profit_rows(),
        "which product category has the highest profit for us") is None


def test_winner_mismatch_ignores_other_question_focus():
    # question asks about quantity -> Clothing IS the max by qty; the profit
    # column is not the metric under discussion
    assert _winner_mismatch(
        "The category with the highest quantity is Clothing (6,922).",
        ["category", "qty", "profit"], _profit_rows(),
        "which category has the highest quantity sold") is None


def test_winner_mismatch_skips_ties():
    rows = [["a", 10.0], ["b", 10.0], ["c", 5.0]]
    assert _winner_mismatch("The highest is b.", ["cat", "v"], rows,
                            "which has the highest value") is None


def test_winner_mismatch_detects_lowest():
    bad = _winner_mismatch("The lowest is Electronics.", ["category", "profit"],
                           _profit_rows(), "which category has the lowest profit")
    assert bad is not None and bad[0] == "Food" and bad[2] == "min"


# ---------------------------------------------------------------------------
# reserved-word + direction guards (live Kaggle-data failures)
# ---------------------------------------------------------------------------

def test_hint_reserved_word_left():
    hint = _error_hint('Parser Exception: syntax error at or near "left"\nLINE 1: ... left = 1')
    assert 'RESERVED' in hint.upper() and '"left"' in hint


def test_direction_contradiction_catches_backwards_conclusion():
    # live Walmart failure: correct numbers, backwards conclusion
    bad = _direction_contradiction(
        "Non-holiday weeks have higher average weekly sales than holiday weeks. "
        "The average weekly sales for non-holiday weeks is 1,041,256, while for "
        "holiday weeks it is 1,122,888.",
        ["holiday_flag", "avg_sales"], [[0, 1041256.0], [1, 1122888.0]],
        "Do holiday weeks have higher average weekly sales than non-holiday weeks?")
    assert bad is not None


def test_direction_contradiction_passes_correct_direction():
    assert _direction_contradiction(
        "Holiday weeks have higher average weekly sales: 1,122,888 vs 1,041,256.",
        ["holiday_flag", "avg_sales"], [[1, 1122888.0], [0, 1041256.0]],
        "Do holiday weeks have higher average weekly sales than non-holiday weeks?") is None


def test_direction_contradiction_ignores_other_shapes():
    # three values -> skip
    assert _direction_contradiction(
        "A is higher: 5 vs 3 and 2.", ["g", "v"],
        [["a", 5.0], ["b", 3.0], ["c", 2.0]], "is A higher than B?") is None


# ---------------------------------------------------------------------------
# contradiction guard: qualitative denials vs a single-row result
# ---------------------------------------------------------------------------

def test_contradiction_tolerates_float_repr_noise():
    """Regression: a lone cell was checked as TEXT, so the raw repr of a float
    (CORR() of two perfectly linked columns = 0.9999999999999998) never
    appeared in the prose and every correct correlation answer was bounced
    into a pointless verify round."""
    res = _FakeResult([{"r": 0.9999999999999998}])
    assert not _contradicts_result("They move together: r = 1.0.", res)
    assert not _contradicts_result("The link is very strong (r = 0.99999).", res)
    assert _contradicts_result("They move together.", res)      # no value stated


def test_contradiction_detected_missing_month_case():
    """The exact live failure: result says 2023-06 is missing, text says none."""
    res = _FakeResult([{"missing_months": "2023-06"}])
    assert _contradicts_result(
        "No, there are no gaps in the month column. The time series is complete.", res)


def test_contradiction_detected_list_cell():
    res = _FakeResult([{"missing": ["2023-06"]}])
    assert _contradicts_result("There are no missing months.", res)


def test_contradiction_passes_when_answer_mentions_value():
    res = _FakeResult([{"missing_months": "2023-06"}])
    assert not _contradicts_result(
        "Yes — one month is missing: 2023-06.", res)


def test_contradiction_ignores_multirow_results():
    res = _FakeResult([{"m": "2022-01"}, {"m": "2022-02"}])
    assert not _contradicts_result("Several months appear.", res)


# ---------------------------------------------------------------------------
# _error_hint: raw errors -> actionable retry hints
# ---------------------------------------------------------------------------

def test_hint_pandas_reset_collision():
    err = ("Traceback (most recent call last):\n"
           "  exec(_CODE, g)\n"
           "ValueError: cannot insert product, already exists")
    hint = _error_hint(err)
    assert "cannot insert" in hint or "collision" in hint
    assert "as_index=False" in hint          # names the working idiom
    assert "reset_index" in hint             # explains the trap


def test_hint_keyerror_points_at_schema_names():
    hint = _error_hint("KeyError: 'revenu'")
    assert "column name" in hint.lower() and "schema" in hint.lower()


def test_hint_sql_binder_error():
    hint = _error_hint('Binder Error: column "revenue" not found')
    assert "schema" in hint.lower()


def test_hint_sql_parser_error():
    hint = _error_hint('Parser Error: syntax error at or near "FORM"')
    assert "SELECT" in hint


def test_hint_generic_fallback_mentions_result_df():
    hint = _error_hint("MysteryFailure: something odd")
    assert "result_df" in hint


# ---------------------------------------------------------------------------
# system prompt teaches the safe groupby idiom up front
# ---------------------------------------------------------------------------

def test_system_prompt_teaches_as_index_idiom():
    assert "as_index=False" in SYSTEM_PROMPT
    assert "already exists" in SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# retry loop end-to-end with a stub client: attempt 1 crashes with the
# collision error, attempt 2 (with the hint injected) succeeds
# ---------------------------------------------------------------------------

class StubClient:
    """Scripted OllamaClient double: pops canned replies per chat() call."""
    model_override = None

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[list[ChatMessage]] = []

    def has_model(self, model: str) -> bool:
        return True

    def chat(self, messages, model, temperature=0.1, **kw):
        self.calls.append(messages)
        return self.replies.pop(0)


def _reply(code: str, language: str = "python", explanation: str = "done") -> str:
    return json.dumps({"plan": "p", "language": language, "code": code,
                       "explanation": explanation})


def _retail_ctx() -> AgentContext:
    df = pd.DataFrame({"product": ["a", "b", "a", "c"],
                       "revenue": [10.0, 20.0, 30.0, 5.0]})
    return AgentContext(tables={"retail_sales": df}, active_table="retail_sales")


_BAD = ("result_df = tables['retail_sales'].groupby('product')['revenue'].sum()"
        ".sort_values(ascending=False).to_frame('product').reset_index()")

_GOOD = ("result_df = (tables['retail_sales'].groupby('product', as_index=False)['revenue']"
         ".sum().sort_values('revenue', ascending=False).rename(columns={'revenue': 'total_revenue'}))")


def test_retry_injects_collision_hint_and_second_attempt_succeeds():
    stub = StubClient([_reply(_BAD), _reply(_GOOD)])
    agent = AnalystAgent(client=stub)
    result = agent.analyze("chart of top 5 products with their revenue", _retail_ctx(),
                           progress=lambda *a, **k: None)
    assert result.ok, result.error
    assert len(stub.calls) == 2
    # the failed attempt's error + the specific hint reached the model
    retry_msg = stub.calls[1][-1].content
    assert "cannot insert product" in retry_msg
    assert "as_index=False" in retry_msg
    top = result.tables[0]
    # sandbox rows are dicts; product 'a' has the highest total (10+30=40)
    assert top["rows"][0] == {"product": "a", "total_revenue": 40.0}


def test_how_many_rows_question_not_treated_as_comparison():
    """Regression: 'how many rows ARE in data' matched the bare 'are ' in the
    old comparative word list, so a correct COUNT(*) was rejected five times
    for 'not answering the comparison'."""
    stub = StubClient([
        _reply("SELECT COUNT(*) AS total_rows FROM retail_sales",
               language="sql", explanation="There are 4 rows in the data."),
    ])
    agent = AnalystAgent(client=stub)
    result = agent.analyze("how many rows are in data", _retail_ctx(),
                           progress=lambda *a, **k: None)
    assert result.ok, result.error
    assert result.tables[0]["rows"][0][0] == 4
    assert len(stub.calls) == 1          # accepted on the first try


def test_multi_statement_first_reply_does_not_crash():
    """Regression: a NON-compare question whose first reply holds two SQL
    statements used to hit `is_compare_q` before assignment
    (UnboundLocalError: 'cannot access local variable')."""
    stub = StubClient([_reply("SELECT 1; SELECT 2", language="sql",
                              explanation="The value is 1.")])
    agent = AnalystAgent(client=stub)
    result = agent.analyze("top products", _retail_ctx(),
                           progress=lambda *a, **k: None)
    assert result.ok
    assert len(result.tables[0]["rows"]) == 1   # first statement only


def test_association_question_accepts_a_single_correlation():
    """Regression: "is there a relationship / link between A and B" contains
    "between", which the loop read as a group comparison — a CORRECT
    SELECT CORR(a, b) was rejected on all five attempts for 'not answering
    the comparison'. An association question is answered by one coefficient."""
    df = pd.DataFrame({"units": [1, 2, 3, 4], "revenue": [10.0, 20.0, 30.0, 40.0]})
    ctx = AgentContext(tables={"retail_sales": df}, active_table="retail_sales")
    stub = StubClient([
        _reply("SELECT CORR(revenue, units) AS r FROM retail_sales", language="sql",
               explanation="They move perfectly together: r = 1.0."),
    ])
    agent = AnalystAgent(client=stub)
    result = agent.analyze("how strong is the link between revenue and units?", ctx,
                           progress=lambda *a, **k: None)
    assert result.ok, result.error
    assert len(stub.calls) == 1          # accepted on the first try
    assert "CORR" in result.code


def test_system_prompt_requires_full_ranked_breakdown():
    """The two shapes a pill asks for most: a ranked breakdown (never a
    LIMIT 1 window difference) and a correlation as a single coefficient."""
    assert "ranked breakdown" in SYSTEM_PROMPT
    assert "CORR(" in SYSTEM_PROMPT


def test_give_up_error_contains_hint_and_code():
    n = settings.max_attempts
    stub = StubClient([_reply(_BAD)] * n)
    agent = AnalystAgent(client=stub)
    result = agent.analyze("top products", _retail_ctx(),
                           progress=lambda *a, **k: None)
    assert not result.ok
    assert "cannot insert product" in result.error
    assert "as_index=False" in result.error       # hint survives into the handoff
    assert "groupby" in result.code               # last code shown for manual pane


_BREAKDOWN_SQL = ("SELECT product, SUM(revenue) AS total_revenue FROM retail_sales "
                  "GROUP BY product ORDER BY total_revenue DESC")


def test_verified_sql_prose_is_not_replaced_by_the_factual_fallback():
    """Regression, seen live on retail_sales: the SQL path's truth-guard
    `else` overwrote a VERIFIED explanation with the model-free
    'Executed result — …' line, so an answer that had passed both guards
    still shipped as a data dump (the fallback-prose collapse)."""
    stub = StubClient([
        _reply(_BREAKDOWN_SQL, language="sql",
               explanation="Product a leads with 40.0 in total revenue."),
    ])
    agent = AnalystAgent(client=stub)
    result = agent.analyze("break down revenue by product - who leads?",
                           _retail_ctx(), progress=lambda *a, **k: None)
    assert result.ok, result.error
    assert len(stub.calls) == 1                  # guard passed; no repair call
    assert result.answer == "Product a leads with 40.0 in total revenue."


def test_sql_answer_without_explanation_still_falls_back_to_factual():
    """The other side of the contract: no prose at all ⇒ the model-free
    factual line is better than an empty answer or a bare plan."""
    stub = StubClient([
        _reply(_BREAKDOWN_SQL, language="sql", explanation=""),
    ])
    agent = AnalystAgent(client=stub)
    result = agent.analyze("break down revenue by product", _retail_ctx(),
                           progress=lambda *a, **k: None)
    assert result.ok, result.error
    # the model-free line now states the finding (leader + gap + group count)
    # instead of dumping cells — 'Executed result — product=a, total_revenue=40.0'
    assert result.answer.startswith("a leads at 40")
    assert "3 groups" in result.answer


def _region_ctx() -> AgentContext:
    df = pd.DataFrame({"region": ["south", "east", "north", "west"],
                       "revenue": [40.0, 30.0, 20.0, 10.0]})
    return AgentContext(tables={"retail_sales": df}, active_table="retail_sales")


_REGION_SQL = ("SELECT region, SUM(revenue) AS total_revenue FROM retail_sales "
               "GROUP BY region ORDER BY total_revenue DESC")


def test_breakdown_coverage_bounces_a_limited_result_then_accepts_all_groups():
    """The live region defect: the model writes LIMIT 1/2 on a break-down and
    the trimmed result reads as if the hidden regions do not exist. The
    coverage guard demands every group before the answer ships."""
    stub = StubClient([
        _reply(_REGION_SQL + " LIMIT 1", language="sql",
               explanation="South leads with 40.0 in total revenue."),
        _reply(_REGION_SQL, language="sql",
               explanation="South leads with 40.0 in total revenue."),
    ])
    result = AnalystAgent(client=stub).analyze(
        "break down revenue by region - who leads, and by how much?",
        _region_ctx(), progress=lambda *a, **k: None)
    assert result.ok, result.error
    assert len(stub.calls) == 2                  # LIMIT 1 bounced, full run accepted
    assert len(result.tables[0]["rows"]) == 4
    assert result.answer == "South leads with 40.0 in total revenue."


def test_breakdown_coverage_exempts_top_n_and_time_filtered_questions():
    """A partial result can be the CORRECT answer — top-N and time-filtered
    questions must not be bounced (a miss is cheaper than breaking a right
    answer; the guard is a net, not a proof)."""
    two = _reply(_REGION_SQL + " LIMIT 2", language="sql",
                 explanation="South leads with 40.0 and east follows with 30.0.")
    result = AnalystAgent(client=StubClient([two])).analyze(
        "break down the top 2 regions by revenue", _region_ctx(),
        progress=lambda *a, **k: None)
    assert result.ok, result.error

    stub2 = StubClient([
        _reply(_REGION_SQL + " LIMIT 2", language="sql",
               explanation="South leads with 40.0 and east follows with 30.0."),
    ])
    result2 = AnalystAgent(client=stub2).analyze(
        "break down revenue by region in 2025", _region_ctx(),
        progress=lambda *a, **k: None)
    assert result2.ok, result2.error
    assert len(stub2.calls) == 1                  # time marker → guard stays out


def _real_region_ctx() -> AgentContext:
    df = pd.DataFrame({"region": ["south", "east", "north", "west"],
                       "revenue": [22663.59, 22333.27, 19256.72, 19105.89]})
    return AgentContext(tables={"retail_sales": df}, active_table="retail_sales")


def test_misattributed_percentage_bounces_then_accepts_the_named_pair():
    """The other half of the live region defect: 16.87% is traceable to the
    east-west pair but attributed to south-west (18.62%) — it must be bounced,
    and the corrected percentage for the named pair accepted."""
    stub = StubClient([
        _reply(_REGION_SQL, language="sql",
               explanation="Verdict: south leads by 16.87% over west."),
        _reply(_REGION_SQL, language="sql",
               explanation="Verdict: south leads by 18.62% over west."),
    ])
    result = AnalystAgent(client=stub).analyze(
        "break down revenue by region - who leads, and by how much?",
        _real_region_ctx(), progress=lambda *a, **k: None)
    assert result.ok, result.error
    assert len(stub.calls) == 2                  # misattributed → bounced → fixed
    # _tidy_answer normalizes the slot separator for display
    assert result.answer == "Verdict — south leads by 18.62% over west."


def test_verdict_placeholders_are_repaired_against_the_result():
    """Live defect: 'VERDICT — <ocean_proximity> has the highest average
    house value at <avg_house_value>' shipped verbatim because placeholders
    carry no numbers to trace. They must trigger a rewrite against the rows."""
    stub = StubClient([
        _reply(_REGION_SQL, language="sql",
               explanation="VERDICT — <region> has the highest total_revenue "
                           "at <total_revenue>."),
        _reply(_REGION_SQL, language="sql",
               explanation="South leads at 40.0 in total revenue."),
    ])
    result = AnalystAgent(client=stub).analyze(
        "Which region has the highest revenue?", _region_ctx(),
        progress=lambda *a, **k: None)
    assert result.ok, result.error
    assert len(stub.calls) == 2                  # placeholder → repaired
    assert "<" not in result.answer
    assert result.answer == "South leads at 40.0 in total revenue."


def test_association_listing_is_bounced_for_a_coefficient():
    """Live defect: a raw ram/price scatter was accepted as the answer to
    'is there a link' and the prose invented r = 0.73 (the prompt's example)
    on data whose true coefficient is -0.024. A listing must come back with
    the coefficient computed in the same result."""
    import pandas as pd  # noqa: PLC0415
    ctx = AgentContext(
        tables={"phones": pd.DataFrame(
            {"ram": [4, 4, 6, 12, 8] * 10,
             "price": [31596, 33530, 62199, 66287, 30418] * 10})},
        active_table="phones")
    stub = StubClient([
        _reply("SELECT ram, price FROM phones", language="sql",
               explanation="RAM and price move together."),
        _reply("SELECT ram, price, CORR(ram, price) OVER () AS r "
               "FROM phones LIMIT 200", language="sql",
               explanation="r = -0.0241 — essentially no linear relationship."),
    ])
    result = AnalystAgent(client=stub).analyze(
        "Is there a link between RAM and price? Plot it.", ctx,
        progress=lambda *a, **k: None)
    assert result.ok, result.error
    assert len(stub.calls) == 2                  # listing bounced, CORR accepted
    assert "r" in [c.lower() for c in result.tables[0]["columns"]]
    assert "-0.0241" in result.answer


def test_union_comparison_without_the_group_column_is_bounced():
    """Live: 'SELECT AVG(price) … WHERE sims = 2 UNION ALL SELECT AVG(price) …'
    returned two unlabelled averages and the answer shipped as
    'Executed result — avg_price=39,806.38 | avg_price=38,602.40'."""
    ctx = AgentContext(
        tables={"phones": pd.DataFrame(
            {"sims": [1, 1, 2, 2], "price": [38000.0, 39000.0, 40000.0, 41000.0]})},
        active_table="phones")
    stub = StubClient([
        _reply("SELECT AVG(price) AS avg_price FROM phones WHERE sims = 2 "
               "UNION ALL SELECT AVG(price) AS avg_price FROM phones "
               "WHERE sims = 1", language="sql",
               explanation="2 SIM slots cost more."),
        _reply("SELECT sims, AVG(price) AS avg_price FROM phones "
               "GROUP BY sims ORDER BY avg_price DESC", language="sql",
               explanation="2 SIM slots cost more: 40,500.00 vs 38,500.00."),
    ])
    result = AnalystAgent(client=stub).analyze(
        "Do phones with 2 SIM slots cost more than phones with 1?", ctx,
        progress=lambda *a, **k: None)
    assert result.ok, result.error
    assert len(stub.calls) == 2                  # unlabelled halves bounced
    assert result.tables[0]["columns"] == ["sims", "avg_price"]


def test_nested_dict_explanation_does_not_crash_the_run():
    """Live through the bridge: the 7B returned "explanation" as a JSON
    object and the placeholder guard's regex crashed the whole ask
    ('expected string or bytes-like object, got \'dict\''). It must be
    serialized and the run must finish."""
    stub = StubClient([
        # attempt 1: nested answer with a number that is NOT in the result
        _reply(_REGION_SQL, language="sql",
               explanation={"verdict": "south leads", "numbers": 999.0}),
        # the repair call is nested too
        _reply(_REGION_SQL, language="sql",
               explanation={"verdict": "south leads", "numbers": 40.0}),
    ])
    result = AnalystAgent(client=stub).analyze(
        "Which region has the highest revenue?", _region_ctx(),
        progress=lambda *a, **k: None)
    assert result.ok, result.error
    assert len(stub.calls) == 2                  # bad nested number → repaired
    assert isinstance(result.answer, str) and result.answer
    assert "south leads" in result.answer
    assert "999" not in result.answer


def test_association_listing_bounced_on_the_python_path_too():
    """The live smartphone run reached the same dead end through pandas
    (`result_df = smartphone_data[['ram','price']]`), so the retry must fire
    on the sandbox path as well."""
    import pandas as pd  # noqa: PLC0415
    ctx = AgentContext(
        tables={"phones": pd.DataFrame(
            {"ram": [4, 4, 6, 12, 8] * 10,
             "price": [31596, 33530, 62199, 66287, 30418] * 10})},
        active_table="phones")
    stub = StubClient([
        _reply("result_df = phones[['ram', 'price']]", language="python",
               explanation="RAM and price move together."),
        _reply("result_df = phones[['ram', 'price']].head(200)\n"
               "result_df['r'] = phones['ram'].corr(phones['price'])",
               language="python",
               explanation="r = -0.0241 — essentially no linear relationship."),
    ])
    result = AnalystAgent(client=stub).analyze(
        "Is there a link between RAM and price? Plot it.", ctx,
        progress=lambda *a, **k: None)
    assert result.ok, result.error
    assert len(stub.calls) == 2
    assert "r" in [str(c).lower() for c in result.tables[0]["columns"]]
    assert "-0.0241" in result.answer
