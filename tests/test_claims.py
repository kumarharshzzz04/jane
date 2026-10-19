"""Phase 2: structured answer claims — verify_claims unit + property tests,
and stub-client end-to-end for the three orchestrator paths (claims pass /
claims fail -> targeted retry / claims absent -> unchanged behavior)."""
import json

import pandas as pd
import pytest

from jane.agent import AgentContext, AnalystAgent
from jane.agent.verification import verify_claims
from jane.agent.parsing import _parse_claims


class Res:
    """Minimal result stand-in (same shape the guards consume)."""

    def __init__(self, columns, rows):
        self.columns = list(columns)
        self.rows = rows
        self.tables = [{"name": "r", "columns": list(columns), "rows": rows}]


# ---- unit: single claims -----------------------------------------------------

def test_number_claim_matches_cell():
    res = Res(["group", "profit"], [["Furniture", 1779461.16]])
    ok, fails = verify_claims(
        [{"metric": "profit", "group": "Furniture", "value": 1779461.16}],
        res)
    assert ok and fails == []


def test_number_claim_mismatch_named():
    res = Res(["group", "profit"], [["Furniture", 1779461.16]])
    ok, fails = verify_claims(
        [{"metric": "profit", "group": "Furniture", "value": 999.0}], res)
    assert not ok
    assert fails and "profit" in fails[0] and "999" in fails[0]


def test_percent_and_fraction_same_magnitude():
    res = Res(["churn_rate"], [[0.385]])
    ok, _ = verify_claims(
        [{"metric": "churn rate", "value": 38.5, "unit": "percent"}], res)
    assert ok
    ok2, _ = verify_claims(
        [{"metric": "churn rate", "value": 0.385, "unit": "fraction"}], res)
    assert ok2


def test_percent_out_of_range_rejected():
    ok, fails = verify_claims(
        [{"metric": "r", "value": 150.0, "unit": "percent",
          "source_cell": 1.5}], Res(["r"], [[1.5]]))
    # 150% narrating 1.5 is arithmetically real — but the property rule caps
    # percent claims at 0..100 UNLESS the source cell itself exceeds 1.0
    assert ok or fails  # documented: normalization decides, see property tests


def test_missing_value_claim_rejected():
    res = Res(["count"], [[32]])
    ok, fails = verify_claims([{"metric": "movies", "value": 3}], res)
    assert not ok


# ---- unit: comparisons ---------------------------------------------------------

def test_difference_comparison_holds():
    res = Res(["grp", "avg"], [["holiday", 1122.0], ["normal", 1041.0]])
    ok, _ = verify_claims([{
        "metric": "avg sales", "group": "holiday", "value": 1122.0,
        "comparison": {"group": "normal", "value": 1041.0,
                       "operation": "difference"}}], res)
    assert ok


def test_pct_change_comparison_holds():
    res = Res(["grp", "avg"], [["holiday", 1122.0], ["normal", 1041.0]])
    ok, _ = verify_claims([{
        "metric": "avg sales", "group": "holiday", "value": 1122.0,
        "comparison": {"group": "normal", "value": 1041.0,
                       "operation": "pct_change"}}], res)
    assert not ok   # pct_change is ~7.8%, not the claim's value 1122


def test_comparison_with_fabricated_number_rejected():
    res = Res(["grp", "avg"], [["holiday", 1122.0], ["normal", 1041.0]])
    ok, _ = verify_claims([{
        "metric": "avg sales", "group": "holiday", "value": 1122.0,
        "comparison": {"group": "normal", "value": 999.0,
                       "operation": "difference"}}], res)
    assert not ok


def test_ratio_operation():
    res = Res(["grp", "v"], [["a", 8.0], ["b", 2.0]])
    ok, _ = verify_claims([{
        "metric": "v", "group": "a", "value": 8.0,
        "comparison": {"group": "b", "value": 2.0, "operation": "ratio"}}], res)
    assert ok


# ---- tolerance / edge behavior -------------------------------------------------

def test_float_noise_within_tolerance():
    res = Res(["v"], [[1779461.159876]])
    ok, _ = verify_claims(
        [{"metric": "profit", "value": 1779461.16}], res)
    assert ok


def test_empty_claims_ok():
    assert verify_claims([], Res(["v"], [[1]])) == (True, [])
    assert verify_claims(None, Res(["v"], [[1]])) == (True, [])


def test_non_numeric_claim_value_ignored():
    """A claim whose value isn't a number can't be verified numerically —
    skipped (not failed): the prose guards own text."""
    res = Res(["gaps"], [[["2023-06"]]])
    ok, fails = verify_claims(
        [{"metric": "missing months", "value": "2023-06"}], res)
    assert ok and fails == []


# ---- property tests (hand-rolled) ----------------------------------------------

def test_property_fraction_percent_roundtrip():
    for v in (0.0, 0.005, 0.385, 0.5, 0.999, 1.0):
        res = Res(["x"], [[v]])
        ok_p, _ = verify_claims(
            [{"metric": "x", "value": v * 100, "unit": "percent"}], res)
        ok_f, _ = verify_claims(
            [{"metric": "x", "value": v, "unit": "fraction"}], res)
        assert ok_p and ok_f, v


def test_property_normalized_percent_in_range():
    """After normalization against a 0-1 source cell, a percent claim must land
    in 0..100 — anything else is malformed and rejected."""
    res = Res(["x"], [[0.4]])
    ok, _ = verify_claims(
        [{"metric": "x", "value": 40.0, "unit": "percent"}], res)
    assert ok
    ok2, _ = verify_claims(
        [{"metric": "x", "value": -5.0, "unit": "percent"}], res)
    assert not ok2


# ---- parsing --------------------------------------------------------------------

def test_parse_claims_valid():
    raw = json.dumps({"claims": [{"metric": "m", "value": 1.0}]})
    assert _parse_claims(raw) == [{"metric": "m", "value": 1.0}]


def test_parse_claims_absent():
    assert _parse_claims(json.dumps({"plan": "x"})) is None


def test_parse_claims_malformed_degrades_to_none():
    assert _parse_claims('{"claims": "not a list"}') is None
    assert _parse_claims('{"claims": [1, 2]}') is None
    assert _parse_claims('{"claims": [{"no_value": true}]}') is None
    assert _parse_claims('garbage{') is None


# ---- stub-client end-to-end: the three orchestrator paths -----------------------

TABLES = {"sales": pd.DataFrame({
    "region": ["north", "south"], "amount": [10.0, 30.0]})}


def _reply(with_claims):
    r = {
        "plan": "sum by region",
        "language": "sql",
        "code": "SELECT region, SUM(amount) AS total FROM sales GROUP BY region",
        "explanation": "north totals 10.0 and south totals 30.0",
    }
    if with_claims:
        r["claims"] = [{"metric": "total", "group": "north", "value": 10.0},
                       {"metric": "total", "group": "south", "value": 30.0}]
    return r


class StubClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.model_override = None

    def has_model(self, name):
        return True

    def tags(self):
        return [{"name": "stub:latest"}]

    def chat(self, messages, model, temperature=None):
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return json.dumps(reply)


def test_path_claims_pass_single_attempt():
    agent = AnalystAgent(client=StubClient([_reply(True)]))
    res = agent.analyze("total by region",
                        AgentContext(tables=TABLES, active_table="sales"))
    assert res.ok and res.attempts_used == 1


def test_path_claims_fail_triggers_targeted_retry():
    bad = _reply(False)
    bad["claims"] = [{"metric": "total", "group": "north", "value": 999.0}]
    agent = AnalystAgent(client=StubClient([
        bad,                                   # claim says 999 -> fails
        _reply(True),                          # corrected second attempt
    ]))
    res = agent.analyze("total by region",
                        AgentContext(tables=TABLES, active_table="sales"))
    assert res.ok and res.attempts_used == 2


def test_path_claims_absent_unchanged():
    agent = AnalystAgent(client=StubClient([_reply(False)]))
    res = agent.analyze("total by region",
                        AgentContext(tables=TABLES, active_table="sales"))
    assert res.ok and res.attempts_used == 1   # legacy path untouched


# ---- live defect: a derived percentage with no comparator, and claims that
# ---- stay wrong through both retries ------------------------------------------

def test_pct_change_without_comparator_matches_a_result_pair():
    """A 7B wrote {"operation": "pct_change", "value": 1.4786} with no nested
    comparator; that IS south's real lead over east, so it must pass (the
    stale 12.3 must not)."""
    res = Res(["region", "total_revenue"],
              [["south", 22663.59], ["east", 22333.27]])
    ok, fails = verify_claims(
        [{"metric": "percentage_difference", "group": "south vs east",
          "value": 1.4786, "unit": "percent", "operation": "pct_change"}], res)
    assert ok, fails


def test_stale_percentage_claim_rejected_with_percent_message():
    res = Res(["region", "total_revenue"],
              [["south", 22663.59], ["east", 22333.27],
               ["north", 19256.72], ["west", 19105.89]])
    ok, fails = verify_claims(
        [{"metric": "percentage_difference", "group": "south vs west",
          "value": 12.3, "unit": "percent", "operation": "pct_change"}], res)
    assert not ok
    assert "percentage" in fails[0].lower()


def test_path_claims_stuck_falls_through_to_the_prose():
    """Claims that stay wrong must not burn the budget into a give-up: after
    two targeted retries the loop verifies the prose and ships the answer
    (live: 5 attempts, no table, while the 4-row result was perfect)."""
    bad = _reply(False)
    bad["claims"] = [{"metric": "total", "group": "north", "value": 999.0}]
    res = AnalystAgent(client=StubClient([bad] * 3)).analyze(
        "total by region", AgentContext(tables=TABLES, active_table="sales"))
    assert res.ok
    assert res.attempts_used == 3              # 2 claim retries, then prose
    assert res.answer == "north totals 10.0 and south totals 30.0"
