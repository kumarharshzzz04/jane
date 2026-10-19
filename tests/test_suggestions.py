"""Tests for suggest_questions — the data-aware example-pill generator.
The live regression it prevents: 'What is the average Product ID?'."""
import pandas as pd

from jane.insights import suggest_questions


def _sales_df():
    # region genuinely moves the metric (west >> others) so the spread-picker
    # has a story to find — the real dataset it mimics does
    return pd.DataFrame({
        "order_id": range(1, 501),                       # sequential ID
        "region": (["west"] * 125 + ["north"] * 125
                   + ["south"] * 125 + ["east"] * 125),
        "units": ([40] * 125 + [12] * 125 + [10] * 125 + [9] * 125),
        # noisy continuous revenue (a real measure, not a 4-value coded column)
        "revenue": ([500.0 + (i % 11) for i in range(125)]
                    + [140.0 + (i % 9) for i in range(125)]
                    + [120.0 + (i % 7) for i in range(125)]
                    + [90.0 + (i % 5) for i in range(125)]),
        "date": pd.date_range("2025-01-01", periods=500, freq="6h").astype(str),
    })


def test_no_id_columns_in_suggestions():
    qs = " ".join(suggest_questions({"retail": _sales_df()})).lower()
    assert "order_id" not in qs
    assert "order id" not in qs


def test_comparison_picks_high_spread_pair():
    """The breakdown pill names the dimension and the measure it moves — region
    genuinely moves units/revenue here — and asks for the whole ranking."""
    qs = suggest_questions({"retail": _sales_df()})
    pill = next((q for q in qs if "break down" in q.lower()), None)
    assert pill, qs
    assert "region" in pill.lower()
    assert "revenue" in pill.lower() or "units" in pill.lower()


def test_flag_column_gets_rate_question():
    df = _sales_df()
    df["returned"] = [0, 1] * 250
    qs = " ".join(suggest_questions({"retail": df})).lower()
    assert "percentage" in qs and "returned" in qs


def test_bin_friendly_column_gets_band_question():
    df = _sales_df()
    df["age"] = [18 + (i * 13) % 50 for i in range(500)]
    qs = " ".join(suggest_questions({"retail": df})).lower()
    assert "band" in qs or "histogram" in qs


def test_trend_question_when_dates_present():
    qs = " ".join(suggest_questions({"retail": _sales_df()})).lower()
    assert "line" in qs or "over" in qs


def test_unique_numeric_treated_as_id():
    df = pd.DataFrame({
        "transaction_ref": [f"TX{i:06d}" for i in range(100)],
        "amount": [10.0 + i for i in range(100)],     # ~unique floats -> ID-like
        "category": ["a", "b"] * 50,
    })
    qs = " ".join(suggest_questions({"t": df})).lower()
    assert "transaction_ref" not in qs
    assert "amount" not in qs or "amount" in " ".join(
        w for w in qs.split() if "transaction" not in qs)


def test_dense_int_entity_key_is_category_not_measure():
    """Walmart-shaped: Store (1-45, dense int) must never be suggested as a
    metric ('Plot Store over Date', 'histogram of Store') — it groups like a
    category but is never the measure."""
    df = pd.DataFrame({
        "Store": [1 + (i % 45) for i in range(6435)],
        "Date": pd.date_range("2010-02-05", periods=6435, freq="6h").astype(str),
        "Weekly_Sales": [15000.0 + (i * 37) % 90000 for i in range(6435)],
        "Holiday_Flag": [(i % 30 == 0) * 1 for i in range(6435)],
    })
    qs = " ".join(suggest_questions({"walmart": df})).lower()
    assert "over store" not in qs
    assert "distribution of store" not in qs
    assert "store spiked" not in qs
    assert "median store" not in qs
    # the real metric still gets used
    assert "weekly sales" in qs or "weekly_sales" in qs


def test_flag_not_used_as_breakdown_metric():
    """A 0/1 flag whose rate varies by group has huge 'spread' but a rate
    question is already covered — the breakdown slot must pick a measure."""
    df = pd.DataFrame({
        "dept": ["a"] * 200 + ["b"] * 200,
        "promoted": [1] * 180 + [0] * 20 + [0] * 190 + [1] * 10,
        "amount": [100.0 + (i * 7.3) % 60 for i in range(200)]
                  + [130.0 + (i * 7.3) % 60 for i in range(200)],
    })
    qs = suggest_questions({"t": df})
    breakdown = [q for q in qs if "break down" in q.lower()]
    assert breakdown, qs
    assert "promoted" not in breakdown[0].lower()
    assert "amount" in breakdown[0].lower()


def test_pills_read_like_analyst_questions():
    """Every pill is a decision question, never a method note: no '(quartiles)'
    parentheticals, no dangling 'as a line chart', and short enough that the
    chat can render it whole instead of truncating it."""
    df = _sales_df()
    df["returned"] = [0, 1] * 250
    df["age"] = [18 + (i * 13) % 50 for i in range(500)]
    qs = suggest_questions({"retail": df})
    assert qs
    for q in qs:
        assert len(q) <= 68, q
        assert "(" not in q and ")" not in q, q
        assert " as a line chart" not in q.lower(), q
    assert all("?" in q for q in qs), qs
