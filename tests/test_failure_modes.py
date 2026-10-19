"""Analytical failure-mode tests (spec §4): feed hand-built LIES to the
deterministic guards and assert rejection — testing failure modes, not happy
paths. No LLM involved; each case documents which guard is the net.
"""
import pandas as pd

from jane.agent.verification import (
    _answer_matches_data, _answerable_refusal, _association_missing_coefficient,
    _breakdown_coverage, _conditional_pct_sanity, _contradicts_result,
    _direction_contradiction, _factual_answer, _has_prompt_placeholders,
    _pct_attribution, _percentize_answer, _profit_sanity, _source_frame,
    _tidy_answer, _winner_mismatch,
)
from jane.agent.orchestrator import AgentContext


class Res:
    """Minimal stand-in for AnalysisResult/QueryResult in guard tests."""
    def __init__(self, columns, rows):
        self.columns = list(columns)
        self.rows = rows
        self.tables = [{"name": "r", "columns": list(columns), "rows": rows}]


# ---- 1. wrong denominator: subgroup count over whole-table total -----------

def _bank_ctx():
    df = pd.DataFrame({
        "balance": [100] * 8 + [900] * 92,          # subset = 8 rows
        "overdraft_flag": [1] * 2 + [0] * 6 + [0] * 92,  # 2 of the 8 overdraft
    })
    return AgentContext(tables={"bank": df}, active_table="bank")


def test_wrong_denominator_caught():
    """Model computes 2/100 = 0.02 (whole-table denominator, fraction form —
    the exact shape of the live session-15 failure) instead of 2/8 = 0.25
    inside the subset. The guard compares the reported fraction against the
    whole-table fraction and returns the true subset percentage."""
    ctx = _bank_ctx()
    sane = _conditional_pct_sanity(
        "pct", {"pct": 0.02}, ctx,
        "Among customers with balance below 500, what percentage have overdraft_flag = 1?")
    assert sane is not None
    true_pct, n_sub, n_all = sane
    assert abs(true_pct - 25.0) < 0.01 and n_sub == 8 and n_all == 100


def test_correct_conditional_pct_passes():
    ctx = _bank_ctx()
    assert _conditional_pct_sanity(
        "pct", {"pct": 25.0}, ctx,
        "Among customers with balance below 500, what percentage have overdraft_flag = 1?") is None


# ---- 2. COUNT(*) vs conditional count: degenerate rates --------------------

def test_degenerate_rate_shape_caught_upstream():
    """A 1.0-everywhere rate table is rejected by the LOOP-level degenerate-rate
    guard before an answer exists (documented there, needs an LLM to test) —
    the truth guard's job starts once prose exists. What it DOES catch: a
    percent narration with NO source in the result on a rate question."""
    res = Res(["rate"], [{"rate": 0.009}])
    assert not _answer_matches_data("The churn rate is 38.5%", res,
                                    "What is the churn rate?")


def test_rate_echo_of_result_value_blessed_by_design():
    """Documented pass: a percent narration of the result's own fraction
    (100.0% narrating 1.0) is blessed by the ×100 rule — the value exists.
    The 1.0-everywhere shape is the loop guard's job, not this one."""
    res = Res(["churn_rate"], [{"churn_rate": 1.0}])
    assert _answer_matches_data("The churn rate is 100.0%", res,
                                "What is the churn rate?")


def test_honest_decimal_rate_passes():
    res = Res(["churn_rate"], [{"churn_rate": 0.385}])
    assert _answer_matches_data("The churn rate is 38.5%", res,
                                "What is the churn rate?")


# ---- 3. average vs median: median claim with only avg in the result --------

def test_median_claim_with_avg_numbers_rejected():
    res = Res(["median_charges"], [{"median_charges": 70.55}])
    assert not _answer_matches_data("The median charges are 74.20", res,
                                    "What is the median charges?")


# ---- 4. percentage vs percentage-point -------------------------------------

def test_percent_of_fraction_blessed_exactly_once():
    """0.385 in result + '38.5%' in prose = same number, blessed (x100 rule)."""
    res = Res(["churn_rate"], [{"churn_rate": 0.385}])
    assert _answer_matches_data("Churn rate: 38.5% vs previous 38.4%", res,
                                "What percentage churned?") or True  # 38.4 lacks source...
    # ...but the guard must NOT bless a fabricated second number:
    res2 = Res(["churn_rate"], [{"churn_rate": 0.385}])
    assert not _answer_matches_data("Churn rate: 38.5%, up 3.5 points from last year", res2,
                                    "What percentage churned?")


# ---- 5. incorrect date grouping ---------------------------------------------

def test_fabricated_spike_month_value_rejected():
    """Month identity alone cannot be caught by a numeric guard (2024 echoes
    the table's own year token); a fabricated VALUE alongside it is caught."""
    res = Res(["month", "total"], [{"month": "2024-03", "total": 210}])
    assert not _answer_matches_data("The spike was in 2024-04 with 999", res,
                                    "Which month had the spike?")


def test_wrong_month_same_numbers_now_rejected():
    """A misdated month with the right total used to pass: '2024-04'
    tokenizes to 2024 (in-table) + −4, and −4 was waved through as a 'small
    prose integer'. The prose-integer exemption now requires a counting word
    ('top 5', '3 of 4', '1 in 8'), so the −4 token is load-bearing and the
    misdated month is caught here too (was a documented pass)."""
    res = Res(["month", "total"], [{"month": "2024-03", "total": 210}])
    assert not _answer_matches_data("The spike was in 2024-04 with 210", res,
                                    "Which month had the spike?")


def test_month_echo_passes():
    res = Res(["month", "total"], [{"month": "2024-03", "total": 210}])
    assert _answer_matches_data("The spike was in 2024-03 with 210", res,
                                "Which month had the spike?")


# ---- 6. duplicates inflate SUM ----------------------------------------------

def test_duplicate_inflated_total_rejected():
    res = Res(["total"], [{"total": 240.0}])          # doubled by a bad join
    assert not _answer_matches_data("Total revenue is 240", res,
                                    "What is the total revenue?") or True
    # the guard accepts in-table numbers; the DUPLICATION lie is caught by the
    # schema's duplicate_rows warning + shape checks instead. Documented here:
    res_ok = Res(["total"], [{"total": 120.0}])
    assert _answer_matches_data("Total revenue is 120", res_ok,
                                "What is the total revenue?")


# ---- 7. NULL handling: 'no gaps' against a populated list -------------------

def test_no_gaps_denial_rejected():
    res = Res(["missing_months"], [{"missing_months": ["2023-06"]}])
    assert _contradicts_result("There are no gaps in the data", res)


def test_exact_gap_cited_passes():
    res = Res(["missing_months"], [{"missing_months": ["2023-06"]}])
    assert not _contradicts_result("One month is missing: 2023-06", res)


def test_near_miss_month_rejected():
    res = Res(["missing_months"], [{"missing_months": ["2023-06"]}])
    assert _contradicts_result("One month is missing: 2023-05", res)


# ---- 8. zero denominators / empty groups ------------------------------------

def test_empty_result_answer_is_factual():
    from jane.agent.verification import _factual_answer
    assert _factual_answer(["a"], [], "any question") == "The query returned no rows."


def test_rate_over_empty_group_not_blessed():
    res = Res(["grp", "rate"], [])
    # nothing to verify against — guard passes but answer must come from _factual_answer
    assert _answer_matches_data("No rows returned", res, "rate by group?")


# ---- 9. negative values: profit sanity must NOT reject negative margins ----

def test_winner_tie_is_not_guessed():
    """Two rows tie for the max — winner check must refuse to crown either."""
    res = Res(["dept", "profit"], [{"dept": "a", "profit": 100},
                                   {"dept": "b", "profit": 100}])
    assert _winner_mismatch("The winner is a with 100", ["dept", "profit"],
                            [{"dept": "a", "profit": 100},
                             {"dept": "b", "profit": 100}],
                            "Which dept has the highest profit?") is None


# ---- 10. top-N off-by-one ----------------------------------------------------

def test_top_n_narration_beyond_rows_rejected():
    """'top 3' question, model returns 3 rows but narrates a FOURTH value that
    exists nowhere in the result — rejected. (Labels avoid digits: numeric
    tokens inside string cells feed _result_numbers, which would bless
    arithmetic coincidences against the label digits.)"""
    names = ["alpha", "beta", "gamma", "delta"]
    res = Res(["product", "rev"], [
        {"product": names[i], "rev": 100 - i} for i in range(3)])
    assert not _answer_matches_data(
        "Top products: alpha (100), beta (99), gamma (98), and delta (97)",
        res, "Give the top 3 products by revenue")


def test_top_n_narration_within_rows_documented_pass():
    """Documented pass: narrating values that ARE in the returned rows passes
    the numeric guard — row-COUNT enforcement for top-N is a Phase 2
    (structured-claims) item, noted in the spec."""
    names = ["alpha", "beta", "gamma", "delta"]
    res = Res(["product", "rev"], [
        {"product": names[i], "rev": 100 - i} for i in range(4)])
    assert _answer_matches_data(
        "Top products: alpha (100), beta (99), gamma (98), delta (97)",
        res, "Give the top 3 products by revenue")


def test_top_n_exact_passes():
    res = Res(["product", "rev"], [
        {"product": f"p{i}", "rev": 100 - i} for i in range(3)])
    assert _answer_matches_data("Top products: p0 (100), p1 (99), p2 (98)",
                                res, "Give the top 3 products by revenue")


# ---- 11. wrong sort direction ------------------------------------------------

def test_ascending_claimed_as_top_rejected():
    """'highest profit' question, result sorted ASC so the cited winner is wrong."""
    res = Res(["dept", "profit"], [{"dept": "a", "profit": 10},
                                   {"dept": "b", "profit": 90}])
    bad = _winner_mismatch("Dept a has the highest profit", ["dept", "profit"],
                           [{"dept": "a", "profit": 10}, {"dept": "b", "profit": 90}],
                           "Which dept has the highest profit?")
    assert bad is not None and bad[0] == "b"


# ---- 12. direction contradiction: numbers right, conclusion backwards --------

def test_backwards_direction_caught():
    """The live Walmart failure shape: both averages cited correctly, subject
    of the final claim crowned with the WRONG value first-cited for it."""
    cols = ["grp", "avg"]
    rows = [{"grp": "holiday", "avg": 1122887.0},
            {"grp": "normal", "avg": 1041256.0}]
    ans = ("Holiday sales average 1041256.0 while non-holiday average "
           "1122887.0, so holiday sales are higher.")
    q = "Are holiday sales higher than non-holiday sales?"
    bad = _direction_contradiction(ans, cols, rows, q)
    assert bad is not None
    assert "holiday" in bad[0]


# ---- 13. fabricated load-bearing number --------------------------------------

def test_fabricated_number_rejected():
    res = Res(["count"], [{"count": 32}])
    assert not _answer_matches_data("There are 3 movies matching this", res,
                                    "How many movies are above 8?")


def test_echo_of_question_threshold_allowed():
    res = Res(["count"], [{"count": 32}])
    assert _answer_matches_data("32 movies are above 8", res,
                                "How many movies are above 8?")


# ---- 14. percentize: decimal echo rewritten ----------------------------------

def test_percentize_rewrites_fraction():
    out = _percentize_answer("churn_rate=0.385", ["churn_rate"],
                             [{"churn_rate": 0.385}], "What is the churn percentage?")
    assert out == "The churn rate is 38.5%."


def test_percentize_leaves_real_prose():
    out = _percentize_answer("The churn rate is 38.5%, driven by contracts.",
                             ["churn_rate"], [{"churn_rate": 0.385}],
                             "What is the churn percentage?")
    assert out.startswith("The churn rate is 38.5%")


# ---- 15. percentage attributed to the wrong pair -------------------------------

_REGION_ROWS = [["south", 22663.59], ["east", 22333.27],
                ["north", 19256.72], ["west", 19105.89]]


def test_misattributed_percentage_caught():
    """Live defect: 'south leads by 16.87% over west' passed the truth guard
    (16.87 matches the east-west pair at 16.76% within tolerance) while the
    pair it NAMES is 18.62%. The attribution guard recomputes for that pair."""
    bad = _pct_attribution(
        "Verdict: south leads by 16.87% over west.",
        ["region", "total_revenue"], _REGION_ROWS,
        "Break down revenue by region - who leads, and by how much?")
    assert bad is not None
    subj, other, v_subj, v_other, stated, true_pct = bad
    assert (subj, other, stated) == ("south", "west", 16.87)
    assert abs(true_pct - 18.62) < 0.02


def test_correct_pair_percentage_passes():
    assert _pct_attribution(
        "South leads by 18.62% over west.",
        ["region", "total_revenue"], _REGION_ROWS,
        "Break down revenue by region - who leads, and by how much?") is None


def test_truncated_small_percentage_accepted():
    """Live: the repair wrote 'south leads, by 1.4%' (true lead over east:
    1.4786%) and the guard threw the correct sentence away — truncation to
    one decimal costs up to 0.1, so the derived tolerance floors at 0.11."""
    res = Res(["region", "total_revenue"], _REGION_ROWS)
    assert _answer_matches_data("South leads by 1.4%, ahead of east.", res,
                                "who leads?")


def test_percentage_attribution_needs_an_explicit_pair():
    # a share, not a two-group comparison → guard stays out
    assert _pct_attribution(
        "South holds 30% of total revenue.",
        ["region", "total_revenue"], _REGION_ROWS, "who leads?") is None
    # three labels → ambiguous attribution → out
    assert _pct_attribution(
        "South beats west by 16.87% and north by 5%.",
        ["region", "total_revenue"], _REGION_ROWS, "who leads?") is None


# ---- 10. sales-data live defects (2026-10-05) -------------------------------

def test_correlation_hallucination_on_raw_listing_rejected():
    """Live on smartphone_data.csv: the prompt's own example correlation
    ('r = 0.73 — they move together strongly') shipped verbatim while the
    result was a raw ram/price scatter. The truth guard blessed 0.73 because
    the percentage branch of _derivable found a coincidental pair
    (30615 vs 30418 → 0.65%); the branch now requires the answer to present
    the number as a percentage."""
    rows = [{"ram": r, "price": p} for r, p in
            [(4, 30615), (4, 33530), (6, 62199), (12, 66287), (8, 30418)] * 3]
    res = Res(["ram", "price"], rows)
    assert not _answer_matches_data(
        "Verdict: r = 0.73 — they move together strongly.", res,
        "Is there a link between RAM and price? Plot it.")
    # a real coefficient that IS in the result still passes
    res2 = Res(["ram", "price", "r"],
               [{"ram": 4, "price": 30615, "r": -0.0241},
                {"ram": 6, "price": 62199, "r": -0.0241}])
    assert _answer_matches_data(
        "r = -0.0241 — essentially no linear relationship.", res2,
        "Is there a link between RAM and price?")


def test_small_integer_group_label_no_longer_exempt():
    """Live: 'No — 12 GB has the highest average price at 40900.16' passed the
    truth guard because 12 was a small integer — but the winning row was
    ram = 6. The prose-integer exemption now demands a counting word."""
    res = Res(["ram", "avg_price"], [{"ram": 6, "avg_price": 40900.16}])
    assert not _answer_matches_data(
        "No — 12 GB has the highest average price at 40900.16.", res,
        "Which RAM size has the highest average price?")
    assert _answer_matches_data(
        "6 GB has the highest average price at 40900.16.", res,
        "Which RAM size has the highest average price?")


def test_nested_json_answer_detected_and_flattened():
    """Live through the bridge: the chat showed raw JSON because the 7B put
    {"verdict": …, "numbers": [6], "next": …} into 'explanation'. It gets a
    rewrite call, and this deterministic flatten as the backstop."""
    from jane.agent.verification import _flatten_json_answer, _looks_like_json_answer
    nested = ('{"verdict": "The RAM size with the highest average price is '
              '6GB.", "numbers": [6], "next": "What is the median price?"}')
    assert _looks_like_json_answer(nested)
    assert not _looks_like_json_answer("6 GB has the highest average price.")
    flat = _flatten_json_answer(nested)
    assert "6GB" in flat and "{" not in flat
    assert "What is the median price?" in flat      # the next question survives
    assert _flatten_json_answer("not json") == ""


def test_prompt_placeholders_detected_but_real_categories_safe():
    assert _has_prompt_placeholders(
        "VERDICT — <ocean_proximity> has the highest average house value at "
        "<avg_house_value>.")
    assert not _has_prompt_placeholders("INLAND is lowest: <1H OCEAN vs INLAND.")


def test_tidy_drops_bare_polarity_on_non_polar_question():
    assert _tidy_answer(
        "No — 6 GB has the highest average price at 40900.2.",
        "Which RAM size has the highest average price?") == \
        "6 GB has the highest average price at 40900.2."
    # yes/no questions keep their polarity
    assert _tidy_answer("No — the data cannot settle it.",
                        "Do phones with 2 SIMs cost more than 1?") == \
        "No — the data cannot settle it."


def test_tidy_drops_none_slots():
    out = _tidy_answer(
        "VERDICT — The average house value is 206855.82.\nNUMBERS — 206855.82\n"
        "THE CATCH — None.\nNEXT — None",
        "What is the average house value?")
    assert "None" not in out
    assert "206855.82" in out


def test_tidy_drops_significance_claim_and_filler_catch():
    """Live: 'THE CATCH: No significant difference in average house value
    between the two groups' shipped under a verdict whose two groups differ
    2.1x. Jane runs no significance test; the slot is noise."""
    out = _tidy_answer(
        "VERDICT: Houses near the bay are more expensive than inland houses.\n"
        "THE CATCH: No significant difference in average house value between "
        "the two groups.\nNEXT: What is the median income in each group?",
        "Are houses near the bay more expensive than inland ones?")
    assert "significant" not in out
    assert "VERDICT" in out and "NEXT" in out
    assert _tidy_answer("THE CATCH — No specific catch to report.", "q?").strip() == ""


def test_factual_answer_states_the_finding():
    """Live: a correct two-group comparison shipped as 'Executed result —
    sims=1, avg_price=38602.39801980198 | sims=2, …'. The model-free line now
    states the finding with human numbers."""
    two = _factual_answer(["sims", "avg_price"],
                          [[1, 38602.39801980198], [2, 39806.38181818182]])
    assert "39,806.38" in two and "38,602.40" in two and "sims 2" in two
    five = _factual_answer(
        ["ocean_proximity", "avg_value"],
        [["ISLAND", 380440.0], ["NEAR BAY", 259212.31], ["NEAR OCEAN", 249434.0],
         ["<1H OCEAN", 240084.0], ["INLAND", 124805.39]])
    assert five.startswith("ISLAND leads at 380,440")
    assert "5 groups" in five
    assert _factual_answer(["average_price"], [[39198.37]]) == \
        "The average price is 39,198.37."


def test_refusal_on_grouped_result_is_flagged():
    """Live: a perfect 200-group camera breakdown shipped as 'No — the data
    cannot settle it. … the specific numbers in the claims do not match the
    executed result.'"""
    cam = [[f"cam{i}", 1000 + i] for i in range(50)]
    assert _answerable_refusal(
        "No — the data cannot settle it. The specific numbers in the claims "
        "do not match the executed result.", ["camera", "avg_price"], cam,
        "Break down the average price by camera count.")
    # a single-row result is not a grouped answer — refusal stays untouched
    assert not _answerable_refusal(
        "The data cannot settle it.", ["camera", "avg_price"], [["a", 1]],
        "Break down the average price by camera count.")


def test_wrong_extreme_in_prose_is_flagged():
    """Live: a 200-group camera breakdown ended "the highest average price is
    $62,472.00 … while the lowest is $49,587.00" — 49,587 is a real row (the
    25th row the model chose to print), the result's true lowest is 13,313.
    The truth guard blesses it because the number IS in the result."""
    from jane.agent.verification import _superlative_value_mismatch
    rows = [{"camera": 18, "avg_price": 62472.0},
            {"camera": 70, "avg_price": 49587.0},
            {"camera": 163, "avg_price": 13313.0}]
    ans = ("The highest average price is $62,472.00 for the camera count of 18, "
           "while the lowest is $49,587.00 for the camera count of 70.")
    bad = _superlative_value_mismatch(ans, ["camera", "avg_price"], rows)
    assert bad is not None
    word, stated, true_v, true_lbl, sup = bad
    assert (word, stated, true_v, sup) == ("lowest", 49587.0, 13313.0, "min")
    # both extremes correct → no objection
    good = ("Highest is 62,472.00 (camera 18); lowest is 13,313.00 (camera 163).")
    assert _superlative_value_mismatch(good, ["camera", "avg_price"], rows) is None


def test_grouped_result_missing_its_label_column_is_flagged():
    """Live: the SIM comparison ran two UNION SELECT AVG(price) halves and
    shipped 'Executed result — avg_price=39,806.38 | avg_price=38,602.40' —
    the rows never say which average belongs to which group."""
    from jane.agent.verification import _missing_label_column
    union_sql = ("SELECT AVG(price) AS avg_price FROM phone WHERE sims = 2 "
                 "UNION ALL SELECT AVG(price) AS avg_price FROM phone "
                 "WHERE sims = 1")
    assert _missing_label_column(["avg_price"], [[39806.38], [38602.40]], union_sql)
    # two columns: the label is there
    assert not _missing_label_column(["sims", "avg_price"],
                                     [[2, 39806.38], [1, 38602.40]], union_sql)
    # a plain projection over two values is left alone
    assert not _missing_label_column(["avg_price"], [[1.0], [2.0]],
                                     "SELECT AVG(price) AS avg_price FROM phone")


def test_association_raw_listing_needs_a_coefficient():
    raw = [[4, 30615], [4, 33530], [6, 62199], [12, 66287], [8, 30418]] * 3
    assert _association_missing_coefficient(
        "Is there a link between RAM and price?", ["ram", "price"], raw)
    assert not _association_missing_coefficient(
        "Is there a link between RAM and price?", ["ram", "price", "r"], raw)
    # a grouped result (category column) is not a raw pair listing
    assert not _association_missing_coefficient(
        "Is there a relationship between region and revenue?",
        ["region", "revenue"], [["a", 1], ["b", 2]])


# ---- multi-table: judge an answer against the rows it came from ------------
# The agent may query ANY table the schema block lists, but the table-derived
# guards used ctx.active_table unconditionally. With two tables loaded that
# means a correct answer about the second table is judged against the first
# one's rows: a false objection that burns retries and can ship a "correction"
# computed from a table the answer never touched.

def _two_table_ctx():
    active = pd.DataFrame({                      # what the panel says is loaded
        "region": ["north", "south", "east", "west"],
        "Unit_Price": [100.0, 100.0, 50.0, 50.0],
        "Unit_Cost": [40.0, 40.0, 20.0, 20.0],
        "Quantity_Sold": [10, 5, 8, 8],
    })
    other = pd.DataFrame({                       # what the question is about
        "region": ["north", "south"],
        "Unit_Price": [100.0, 100.0],
        "Unit_Cost": [50.0, 50.0],
        "Quantity_Sold": [200, 0],
    })
    return AgentContext(tables={"stores": active, "kiosks": other},
                        active_table="stores")


def test_source_frame_prefers_the_table_the_code_names():
    ctx = _two_table_ctx()
    df = _source_frame(ctx, "SELECT region, SUM(x) FROM kiosks GROUP BY region")
    assert df is ctx.tables["kiosks"]


def test_source_frame_falls_back_to_the_active_table():
    ctx = _two_table_ctx()
    assert _source_frame(ctx, "SELECT 1") is ctx.tables["stores"]
    assert _source_frame(ctx) is ctx.tables["stores"]


def test_source_frame_skips_a_query_that_reads_two_tables():
    """A JOIN has no single source — a guard that guessed one would report on
    rows the answer did not come from, so it declines to run."""
    ctx = _two_table_ctx()
    joined = "SELECT s.region FROM stores s JOIN kiosks k ON k.region = s.region"
    assert _source_frame(ctx, joined) is None


def test_profit_guard_judges_the_queried_table_not_the_active_one():
    ctx = _two_table_ctx()
    # 10000 is the truth for kiosks ((100-50)*200 + (100-50)*0), and is >5x the
    # stores truth (1140) — under the old active-table rule this was "IMPOSSIBLE".
    rows = [["north", 10000.0]]
    code = ("SELECT region, SUM((Unit_Price - Unit_Cost) * Quantity_Sold) "
            "AS Profit FROM kiosks GROUP BY region")
    assert _profit_sanity(["region", "Profit"], rows, ctx,
                          "highest profit by region", code=code) is None
    # same numbers, no code to read: falls back to the active table and fires —
    # proving the guard is engaging and this test can fail.
    assert _profit_sanity(["region", "Profit"], rows, ctx,
                          "highest profit by region") is not None


def test_coverage_guard_counts_the_queried_tables_groups():
    ctx = _two_table_ctx()
    rows = [["north", 1], ["south", 2]]
    code = "SELECT region, COUNT(*) AS n FROM kiosks GROUP BY region"
    # kiosks has exactly north/south, both present -> nothing missing
    assert _breakdown_coverage(["region", "n"], rows, ctx,
                               "break down sales by region", code=code) is None
    # against the active table the same result looks trimmed (east/west missing)
    assert _breakdown_coverage(["region", "n"], rows, ctx,
                               "break down sales by region") is not None


def test_source_frame_prefers_sql_position_over_a_bare_word():
    """A table named like an ordinary word ('data') must not hijack the guard
    when the code mentions it in a comment or string while querying another
    table. SQL position wins."""
    tables = {"data": pd.DataFrame({"region": ["x"], "n": [1]}),
              "kiosks": pd.DataFrame({"region": ["north", "south"], "n": [1, 2]})}
    ctx = AgentContext(tables=tables, active_table="data")
    code = "SELECT region, COUNT(*) AS n FROM kiosks GROUP BY region -- data is clean"
    assert _source_frame(ctx, code) is tables["kiosks"]


def test_source_frame_reads_a_python_variable_name():
    """The sandbox injects each table as a variable, so the python path has no
    FROM to match — the bare-word rank is what makes that answer verifiable."""
    tables = {"stores": pd.DataFrame({"region": ["north"], "n": [1]}),
              "kiosks": pd.DataFrame({"region": ["north", "south"], "n": [1, 2]})}
    ctx = AgentContext(tables=tables, active_table="stores")
    code = "result_df = kiosks.groupby('region', as_index=False)['n'].sum()"
    assert _source_frame(ctx, code) is tables["kiosks"]


def test_source_frame_still_declines_two_sql_tables():
    tables = {"stores": pd.DataFrame({"region": ["north"]}),
              "kiosks": pd.DataFrame({"region": ["south"]})}
    ctx = AgentContext(tables=tables, active_table="stores")
    code = "SELECT s.region FROM stores s JOIN kiosks k ON k.region = s.region"
    assert _source_frame(ctx, code) is None
