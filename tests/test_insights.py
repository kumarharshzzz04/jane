"""Tests for the insights module: profile reports + deterministic pattern hunt."""
import numpy as np
import pandas as pd

from jane.insights import find_insights, profile_summary_text, profile_table


def _sample_df() -> pd.DataFrame:
    rng = np.random.RandomState(7)
    n = 200
    amounts = rng.normal(100, 15, n - 1).tolist() + [900.0]
    return pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n, freq="h"),
        "region": rng.choice(["north", "south", "east"], n, p=[0.75, 0.15, 0.10]),
        "amount": amounts,
        "units": rng.randint(1, 10, n),
    })


def test_profile_shape_and_stats():
    r = profile_table("s", _sample_df())
    assert r["rows"] == 200 and r["cols"] == 4
    amount = next(c for c in r["columns"] if c["name"] == "amount")
    assert amount["outliers_iqr"] >= 1  # the planted 900
    assert amount["median"] is not None


def test_profile_correlations_found():
    df = _sample_df()
    df["triple"] = df["units"] * 3
    r = profile_table("s", df)
    assert any({c["a"], c["b"]} == {"units", "triple"} for c in r["correlations"])


def test_summary_text_mentions_key_facts():
    text = profile_summary_text([profile_table("s", _sample_df())])
    assert "TABLE s" in text
    assert "nulls" in text
    assert "outliers" in text


def test_insights_detect_dominance_and_outlier():
    ins = find_insights({"s": _sample_df()})
    kinds = {i.kind for i in ins}
    assert "dominance" in kinds   # region is 75% north
    assert "outlier" in kinds     # planted 900


def test_insights_detect_spike_in_series():
    n = 30
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n, freq="D"),
        "sales": [100.0] * 15 + [300.0] * 15,
    })
    ins = find_insights({"s": df})
    assert any(i.kind in ("spike", "drop") for i in ins)


def test_insights_detect_missing():
    df = pd.DataFrame({"a": range(20), "empty": [None] * 20})
    df.loc[:15, "b"] = range(16)
    ins = find_insights({"s": df})
    assert any(i.kind == "missing" and "empty" in i.title for i in ins)


def test_small_table_is_skipped():
    assert find_insights({"tiny": pd.DataFrame({"a": [1, 2, 3]})}) == []


def test_insights_are_ranked_and_capped():
    ins = find_insights({"s": _sample_df()})
    scores = [i.score for i in ins]
    assert scores == sorted(scores, reverse=True)
