"""Tests for styled chart rendering (Vega-Lite spec -> matplotlib PNG)."""
import pandas as pd
import pytest

from jane.charts import (_leader_callout, _pearson, _pct_mode, _skew_note,
                         _tick_step, _trend_badge, render_all, render_spec)

ROWS = [
    {"region": "north", "total": 120.5},
    {"region": "south", "total": 88.2},
    {"region": "east", "total": 141.9},
]


def test_bar_spec_renders_png():
    spec = {
        "mark": "bar",
        "data": {"values": ROWS},
        "encoding": {"x": {"field": "region"}, "y": {"field": "total"}},
    }
    png = render_spec(spec)
    assert png is not None and png[:8] == b"\x89PNG\r\n\x1a\n"


def test_line_spec_renders_png():
    spec = {
        "mark": "line",
        "data": {"values": [{"month": "2025-01", "rev": 10}, {"month": "2025-02", "rev": 14}]},
        "encoding": {"x": {"field": "month"}, "y": {"field": "rev"}},
    }
    assert render_spec(spec) is not None


def test_spec_without_data_returns_none():
    assert render_spec({"mark": "bar"}) is None
    assert render_spec({"mark": "bar", "data": {"values": []}}) is None


def test_null_values_only_returns_none():
    spec = {"mark": "bar", "data": {"values": [{"a": "x", "b": None}]},
            "encoding": {"x": {"field": "a"}, "y": {"field": "b"}}}
    assert render_spec(spec) is None


def test_field_inference_fallback():
    # no encoding block: infer string col + numeric col
    spec = {"mark": "bar", "data": {"values": ROWS}}
    assert render_spec(spec) is not None


def test_render_all_skips_bad_specs():
    good = {"mark": "bar", "data": {"values": ROWS}}
    out = render_all([good, {"mark": "bar"}, "not-a-dict"])
    assert len(out) == 1


def test_agent_bound_chart_renders():
    from jane.agent import _bind_chart_data
    spec = {"mark": "bar", "encoding": {"x": {"field": "region"},
                                        "y": {"field": "total_revenue"}}}
    bound = _bind_chart_data(spec, ["region", "total_revenue"],
                             [["north", 10.0], ["south", 20.0]])
    assert render_spec(bound) is not None


def test_sandbox_matplotlib_figure_serialized():
    from jane.sandbox import run_code
    tables = {"t": pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})}
    code = (
        "import matplotlib\nmatplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots()\n"
        "ax.bar(['a', 'b', 'c'], df['y'])\n"
        "result_df = df.head(1)\n"
    )
    res = run_code(code, tables, active_table="t", timeout=90)
    assert res.ok, res.error
    assert res.figures and res.figures[0].startswith("data:image/png;base64,")


# ————— new chart types (session 18) —————

def test_long_labels_auto_horizontal():
    spec = {"mark": "bar",
            "data": {"values": [
                {"dept": "Research & Development team", "v": 30},
                {"dept": "Customer Success long label", "v": 22},
                {"dept": "Finance", "v": 11}]},
            "encoding": {"x": {"field": "dept"}, "y": {"field": "v"}}}
    assert render_spec(spec) is not None


def test_histogram_bins_and_renders():
    vals = [{"age": float((i * 7) % 60) + 18} for i in range(150)]
    spec = {"mark": "histogram", "data": {"values": vals},
            "encoding": {"y": {"field": "age"}}}
    assert render_spec(spec) is not None


def test_histogram_needs_enough_values():
    spec = {"mark": "histogram", "data": {"values": [{"v": 1.0}, {"v": 2.0}]},
            "encoding": {"y": {"field": "v"}}}
    assert render_spec(spec) is None


def test_grouped_and_stacked_bars():
    rows = [{"region": "north", "quarter": "Q1", "sales": 10},
            {"region": "north", "quarter": "Q2", "sales": 14},
            {"region": "south", "quarter": "Q1", "sales": 8},
            {"region": "south", "quarter": "Q2", "sales": 11}]
    for mark in ("grouped-bar", "stacked-bar"):
        spec = {"mark": mark, "data": {"values": rows}}
        assert render_spec(spec) is not None, mark


def test_heatmap_long_format():
    rows = [{"a": "x", "b": "p", "c": 0.2}, {"a": "x", "b": "q", "c": 0.5},
            {"a": "y", "b": "p", "c": -0.3}, {"a": "y", "b": "q", "c": 0.8}]
    spec = {"mark": "heatmap", "data": {"values": rows}}
    assert render_spec(spec) is not None


# ————— smart callouts (session 36): computed from the rows, never guessed —————

def test_leader_callout_strict_maximum():
    label, text = _leader_callout([10.0, 8.2, 3.0], ["a", "b", "c"])
    assert label == "a"
    assert text == "22% ahead of #2"


def test_leader_callout_tie_is_silent():
    assert _leader_callout([5.0, 5.0], ["a", "b"]) is None
    assert _leader_callout([5.0, 5.0, 1.0], ["a", "b", "c"]) is None


def test_leader_callout_big_gap_reads_as_multiple():
    label, text = _leader_callout([100.0, 20.0], ["a", "b"])
    assert label == "a" and text == "5.0x #2"


def test_leader_callout_needs_a_real_gap():
    assert _leader_callout([10.0, 9.95], ["a", "b"]) is None


def test_trend_badge_sign_and_floor():
    assert _trend_badge([10.0, 12.0, 15.0]) == "+50% over the window"
    assert _trend_badge([15.0, 12.0, 10.0]) == "-33% over the window"
    assert _trend_badge([10.0, 10.02, 10.05]) is None
    assert _trend_badge([10.0, 14.0]) is None


def test_pearson_matches_known_correlation():
    r = _pearson([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0])
    assert r is not None and abs(r - 1.0) < 1e-9
    assert _pearson([1.0, 2.0], [1.0, 2.0]) is None
    assert _pearson([1.0, 2.0, 3.0], [4.0, 4.0, 4.0]) is None


def test_skew_note_detects_dragged_average():
    assert _skew_note([1.0] * 7 + [10.0]) == "right-skewed"
    assert _skew_note([1.0] * 7 + [-10.0]) == "left-skewed"
    assert _skew_note([float(i) for i in range(10)]) is None
    assert _skew_note([1.0, 2.0, 3.0]) is None


def test_pct_mode_only_for_share_named_columns():
    assert _pct_mode("churn_rate", [0.2, 0.5]) == "fraction"
    assert _pct_mode("attrition_rate", [0.1612]) == "fraction"
    assert _pct_mode("share", [0.4]) == "fraction"
    assert _pct_mode("revenue_share", [22.6, 26.2]) == "percent"
    assert _pct_mode("revenue", [0.2, 0.5]) is None
    assert _pct_mode("share", [120.0, 250.0]) is None


def test_tick_step_keeps_long_labels_apart():
    labels = [f"2024-12-{d:02d}" for d in range(1, 29)]
    step = _tick_step(labels)
    assert 28 / step <= 10  # at most ~10 mono date labels fit 7.6in
    assert _tick_step(["a", "b"]) == 1


def test_mixed_sign_bars_render():
    spec = {"mark": "bar",
            "data": {"values": [{"k": "a", "v": -3.0},
                                 {"k": "b", "v": 5.0},
                                 {"k": "c", "v": 2.0}]},
            "encoding": {"x": {"field": "k"}, "y": {"field": "v"}}}
    assert render_spec(spec) is not None


def test_share_column_bars_render_percent_chips():
    spec = {"mark": "bar",
            "data": {"values": [{"c": "a", "churn_rate": 0.385},
                                 {"c": "b", "churn_rate": 0.12}]},
            "encoding": {"x": {"field": "c"},
                         "y": {"field": "churn_rate"}}}
    assert render_spec(spec) is not None


def test_trending_line_scatter_and_histogram_render():
    line = {"mark": "line",
            "data": {"values": [{"m": f"2025-{i:02d}", "v": 10 + i * 3}
                                 for i in range(1, 7)]},
            "encoding": {"x": {"field": "m"}, "y": {"field": "v"}}}
    assert render_spec(line) is not None
    scatter = {"mark": "point",
               "data": {"values": [{"a": float(i),
                                      "b": float(i) * 2 + (i % 3)}
                                     for i in range(12)]},
               "encoding": {"x": {"field": "a"}, "y": {"field": "b"}}}
    assert render_spec(scatter) is not None
    hist = {"mark": "histogram",
            "data": {"values": [{"age": float((i * 7) % 60) + 18}
                                 for i in range(150)]},
            "encoding": {"y": {"field": "age"}}}
    assert render_spec(hist) is not None
