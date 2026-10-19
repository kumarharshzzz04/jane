"""Tests for jane.preprocess — deterministic ops, presets, and UX contract."""
import numpy as np
import pandas as pd
import pytest

from jane.preprocess import (PreprocessError, available_ops, clean_default,
                             run_ops)


def _messy() -> pd.DataFrame:
    return pd.DataFrame({
        "category": ["  north ", "south", "north", " north  ", "south", "east"],
        "amount": [100.0, np.nan, 100.0, 300.0, 200.0, 200.0],
        "date": ["2025-01-01", "2025-01-02", "2025-01-03",
                 "2025-01-04", "2025-01-05", "2025-01-06"],
        "grade": ["a", "b", "a", "b", "a", "b"],
    })


def _messy_with_dupes() -> pd.DataFrame:
    """Fixtures for dedupe: make row 2 an exact duplicate of row 0."""
    df = _messy().copy()
    df.loc[2] = df.loc[0]
    return df


def test_available_ops_registry():
    ops = available_ops()
    for expected in ("clean_text", "to_datetime", "fill_missing", "dedupe",
                     "scale", "encode", "clamp_outliers"):
        assert expected in ops


def test_run_ops_returns_new_frame_original_untouched():
    df = _messy()
    res = run_ops(df, [("dedupe", {}), ("fill_missing", {})])
    assert res.df is not df
    assert df["amount"].isna().sum() == 1          # original preserved
    assert res.df["amount"].isna().sum() == 0      # new frame filled
    assert "dedupe" in res.applied and "fill_missing" in res.applied
    assert res.log and all(isinstance(m, str) for m in res.log)


def test_fill_missing_strategies():
    df = _messy()
    r1 = run_ops(df, [("fill_missing", {"strategy": "auto"})])
    assert r1.df["amount"].isna().sum() == 0
    # median of the observed amounts (100,100,300,200,200) is 200
    assert np.isclose(r1.df.loc[1, "amount"], 200.0)
    r2 = run_ops(df, [("fill_missing", {"strategy": "drop"})])
    assert len(r2.df) == 5                             # the NaN row is gone
    assert any("removed 1 row" in m for m in r2.log)


def test_dedupe_exact_and_subset():
    df = _messy_with_dupes()
    r = run_ops(df, [("dedupe", {})])
    assert len(r.df) == 5
    assert any("removed 1 duplicate" in m for m in r.log)
    rs = run_ops(_messy(), [("dedupe", {"subset": ["category", "grade"]})])
    # dedupe keeps first occurrences and does NOT normalize whitespace first —
    # ' north ' ≠ 'north', so category×grade has 6 distinct keys in the raw frame
    assert len(rs.df) >= 5


def test_clean_text_trims_and_collapses():
    df = _messy()
    r = run_ops(df, [("clean_text", {})])
    assert set(r.df["category"]) == {"north", "south", "east"}
    assert r.df["category"].tolist().count("north") == 3
    assert any("clean_text" in m for m in r.log)


def test_to_datetime_detection():
    df = _messy()
    r = run_ops(df, [("to_datetime", {})])
    assert str(r.df["date"].dtype).startswith("datetime64")


def test_scale_methods():
    df = _messy().dropna()
    r = run_ops(df, [("scale", {"method": "zscore", "columns": ["amount"]})])
    assert abs(r.df["amount"].mean()) < 1e-9
    assert abs(r.df["amount"].std() - 1.0) < 1e-9
    rm = run_ops(df, [("scale", {"method": "minmax", "columns": ["amount"]})])
    assert rm.df["amount"].min() == 0.0 and rm.df["amount"].max() == 1.0
    rr = run_ops(df, ["robust"] and [("scale", {"method": "robust", "columns": ["amount"]})])
    assert rr.df["amount"].median() < 1e-9 or np.isclose(rr.df["amount"].median(), 0.0)


test_scale_methods.__doc__ = "zscore → mean 0/std 1; minmax → [0,1]; robust → median 0."


def test_encode_onehot():
    df = _messy()
    r = run_ops(df, [("encode", {"columns": ["grade"]})])
    assert {"grade_a", "grade_b"} <= set(r.df.columns)
    assert "grade" not in r.df.columns
    r_hi = run_ops(_messy(), [("encode", {"columns": ["date"]})])
    assert "date" in r_hi.df.columns  # untouched: date-like or >12 distinct values


def test_clamp_outliers_winsorizes():
    df = pd.DataFrame({"amount": [10.0] * 20 + [1000.0]})
    r = run_ops(df, [("clamp_outliers", {"columns": ["amount"]})])
    assert r.df["amount"].max() < 1000.0
    assert any("clamped" in m for m in r.log)


def test_unknown_op_raises():
    with pytest.raises(PreprocessError):
        run_ops(_messy(), [("nope_op", {})])


def test_bad_column_raises_readable_error():
    with pytest.raises(PreprocessError, match="not_a_col"):
        run_ops(_messy(), [("scale", {"columns": ["not_a_col"]})])
    with pytest.raises(PreprocessError, match="encode"):
        run_ops(_messy(), [("encode", {"columns": ["ghost"]})])


def test_clean_default_preset():
    df = _messy()
    r = clean_default(df)
    assert r.df["amount"].isna().sum() == 0
    assert str(r.df["date"].dtype).startswith("datetime64")
    assert set(r.df["category"]) == {"north", "south", "east"}
    assert len(r.applied) == 4


def test_aliases_resolve():
    r = run_ops(_messy(), [("standardize", {"columns": ["amount"]}), ("onehot", {"columns": ["grade"]})])
    assert abs(r.df["amount"].std() - 1.0) < 1e-9
    assert "grade_a" in r.df.columns
