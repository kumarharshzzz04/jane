"""Chart rendering — editorial-quality matplotlib figures in Jane's palette.

Vega-Lite specs remain the interchange format, but this renderer makes the
calls a designer would make:

- Chart type is *inferred from the data*: dates/many ordered points -> line,
  few categories -> ranked bars, a slice-share -> donut, two numeric columns
  -> scatter. The model's `mark` is a hint, not a command.
- Axes are formatted like a publication: abbreviated money/counts (12k, 1.2M),
  no zero-line clutter, spines removed, faint grid behind the data.
- Bars get a gradient ramp (accent for the max, fading toward the min) and
  rounded tops; value chips sit above each bar in mono.
- Titles are generated when the model omits them ("Y by X"), with a dim
  subtitle showing the aggregate ("sum over N points").
"""
from __future__ import annotations

import io
import math
import re
from datetime import date, datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter

# ————— receipts & rules tokens —————
# Neutral black surfaces: the page, the machine and the charts all sit on the
# same ink, with orange as the one loud colour and verdigris kept for verified.
INK = "#EDF4EE"
DIM = "#9AA1A7"
FAINT = "#5F6368"
GRID = "#23262B"
PANEL = "#0B0C0E"      # ink
ACCENT = "#FF6B2C"     # signal — the one loud color
ACCENT_SOFT = "#FFB35C" # ember
GREEN = "#58C08B"      # verdigris — verified
BLUE = "#5E93B8"       # lake
SERIES = [ACCENT, GREEN, BLUE, ACCENT_SOFT, "#C38FFF", "#FF5C7A"]
AXIS = "#3A3F45"       # the zero baseline, one step brighter than the grid
CHIP_BG = "#14171B"    # value-chip pill on the panel ink
CHIP_EDGE = "#2A2E33"
LEAD_BG = "#2A1A10"    # leader chip: ember-tinted pill
LEAD_TEXT = "#FFD9BF"


def _load_fonts() -> tuple[str, str]:
    """Pick the best available fonts; never emit findfont warnings."""
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    sans = next((f for f in ("Archivo", "Segoe UI", "DejaVu Sans")
                 if f in available), "DejaVu Sans")
    mono = next((f for f in ("IBM Plex Mono", "Consolas", "DejaVu Sans Mono")
                 if f in available), "DejaVu Sans Mono")
    return mono, sans


FONT_MONO, FONT_SANS = _load_fonts()

_DATE_RE = re.compile(
    r"^\d{4}-\d{2}(-\d{2})?([ T]\d{2}:\d{2})?$|^(19|20)\d{2}$|^\d{1,2}/\d{1,2}(/\d{2,4})?$")


def _apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": PANEL,
        "axes.facecolor": PANEL,
        "savefig.facecolor": PANEL,
        "axes.edgecolor": GRID,
        "axes.labelcolor": DIM,
        "text.color": INK,
        "xtick.color": DIM,
        "ytick.color": DIM,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "font.family": FONT_SANS,
        "font.size": 10.5,
        # 132dpi keeps the PNG crisp in the chat column and on retina panels
        # without visibly growing the payload.
        "figure.dpi": 132,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.25,
    })


def _is_temporal(values: list) -> bool:
    """Do these x-values look like dates/months/years?"""
    if not values:
        return False
    hits = sum(1 for v in values if isinstance(v, (datetime, date))
               or (isinstance(v, str) and _DATE_RE.match(v.strip())))
    return hits >= max(2, int(0.7 * len(values)))


def _is_ordinal(values: list) -> bool:
    """Ordered labels like Jan < Feb < Mar or Q1..Q4?"""
    if not values or not all(isinstance(v, str) for v in values):
        return False
    months = ["jan", "feb", "mar", "apr", "may", "jun",
              "jul", "aug", "sep", "oct", "nov", "dec"]
    lows = [v.strip().lower()[:3] for v in values]
    if all(l in months for l in lows) and len(set(lows)) > 1:
        return True
    if all(re.match(r"^[Qq][1-4]$", v.strip()) for v in values):
        return True
    if all(re.match(r"^\d{4}$", str(v)) for v in values):
        return True
    return False


def _fmt_num(v) -> str:
    """Abbreviate money/counts for axes, chips and callouts (12k, 1.2M)."""
    if v is None or (isinstance(v, float) and (np.isnan(v))):
        return ""
    av = abs(v)
    if av >= 1_000_000:
        s = f"{v/1_000_000:.1f}".rstrip("0").rstrip(".") + "M"
    elif av >= 1_000:
        s = f"{v/1_000:.1f}".rstrip("0").rstrip(".") + "k"
    elif av >= 10 or av == int(av):
        s = f"{v:,.0f}"
    else:
        s = f"{v:.2f}".rstrip("0").rstrip(".")
    return s


def _num_formatter() -> FuncFormatter:
    return FuncFormatter(lambda v, _pos: _fmt_num(v))


def _humanize(name: str) -> str:
    return str(name).replace("_", " ").strip().title()


def _title_from(spec: dict, xf: str, yf: str, agg: str, n: int) -> str:
    t = spec.get("title")
    if t:
        return str(t)
    return f"{_humanize(yf)} by {_humanize(xf)}"


def _subtitle(agg: str, n: int) -> str:
    return f"{agg} · {n} {'points' if n != 1 else 'point'}"


# ————— insight helpers: callouts are computed from the rows, never guessed —————

def _leader_callout(values, labels):
    """(leader_label, text) for a STRICT maximum only — None on ties or when
    the lead is under 1%. Reads "18% ahead of #2" up to ~2x, then "5.0x #2"
    (a large multiple parses better as a ratio)."""
    if not values or len(values) < 2:
        return None
    top = max(values)
    if sum(1 for v in values if v == top) != 1:
        return None
    i = values.index(top)
    second = max(v for j, v in enumerate(values) if j != i)
    if top <= 0 or second <= 0:
        return None
    pct = (top - second) / second * 100.0
    if pct < 1.0:
        return None
    if pct <= 120.0:
        return labels[i], f"{pct:.0f}% ahead of #2"
    return labels[i], f"{top / second:.1f}x #2"


def _trend_badge(values, n_min: int = 3) -> str | None:
    """"+38% over the window" when the endpoints actually moved (>=1%)."""
    if len(values) < n_min or values[0] == 0:
        return None
    pct = (values[-1] - values[0]) / abs(values[0]) * 100.0
    if abs(pct) < 1.0:
        return None
    return f"{'+' if pct > 0 else '-'}{abs(pct):.0f}% over the window"


def _pearson(xs, ys) -> float | None:
    """Correlation for the scatter callout; None when undefined."""
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    den = math.sqrt(sum(a * a for a in dx) * sum(b * b for b in dy))
    if den == 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / den


def _skew_note(values) -> str | None:
    """mean-median drift in std units: which tail drags the average."""
    if len(values) < 8:
        return None
    std = float(np.std(values))
    if std == 0:
        return None
    z = (sum(values) / len(values) - float(np.median(values))) / std
    if z > 0.35:
        return "right-skewed"
    if z < -0.35:
        return "left-skewed"
    return None


def _pct_mode(yf, values) -> str | None:
    """Share-named columns get percent chips: 0-1 values are fractions
    (0.385 → 38.5%), 0-100 values keep their number and just gain the sign.
    A plain count column never gets one."""
    if not re.search(r"percent|pct|share|proportion|(?:^|_)rate$",
                     str(yf or ""), re.I):
        return None
    if not values:
        return None
    if all(0 <= v <= 1.0000001 for v in values):
        return "fraction"
    if max(values) <= 100.0000001:
        return "percent"
    return None


def _chip_text(v, mode: str | None) -> str:
    if mode == "fraction":
        return f"{v * 100:.1f}".rstrip("0").rstrip(".") + "%"
    if mode == "percent":
        return f"{v:,.1f}".rstrip("0").rstrip(".") + "%"
    return _fmt_num(v)


_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _short_labels(labels) -> list[str]:
    """Date-ish tick labels the reader can scan: 2024-12-23 → Dec 23."""
    out = []
    for l in labels:
        s = str(l).strip()
        m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", s)
        if m:
            out.append(f"{_MONTHS[int(m.group(2)) - 1]} {int(m.group(3))}")
            continue
        m = re.match(r"^(\d{4})-(\d{2})$", s)
        if m:
            out.append(f"{_MONTHS[int(m.group(2)) - 1]} {m.group(1)}")
            continue
        out.append(s)
    return out


def _axis_formatter(mode: str | None) -> FuncFormatter:
    """A share axis must read in the same units as its chips."""
    if mode == "fraction":
        return FuncFormatter(lambda v, _pos: f"{v * 100:.0f}%")
    if mode == "percent":
        return FuncFormatter(lambda v, _pos: f"{v:,.0f}%")
    return _num_formatter()


def _tick_step(labels, fig_in: float = 7.6, fontsize: float = 8.8) -> int:
    """A step that keeps mono tick labels from colliding on a 7.6in axes."""
    n = len(labels)
    if n <= 1:
        return 1
    widest = max((len(str(l)) for l in labels), default=4)
    per_label_pt = fontsize * 0.72 * max(4, widest)
    fit = max(2, int(fig_in * 72 / per_label_pt))
    return max(1, math.ceil(n / fit))


def _chip(ax, x, y, text, *, leader: bool = False, va: str = "bottom",
          dy: int = 4) -> None:
    """A value chip with an ink pill behind it, so it stays readable over
    gridlines; the leader chip gets the ember treatment."""
    ax.annotate(
        text, (x, y), ha="center", va=va, zorder=6,
        fontsize=8.6, family=FONT_MONO,
        color=LEAD_TEXT if leader else INK,
        bbox=dict(boxstyle="round,pad=0.28",
                  fc=LEAD_BG if leader else CHIP_BG,
                  ec=ACCENT if leader else CHIP_EDGE, lw=0.8),
        xytext=(0, dy if va == "bottom" else -dy),
        textcoords="offset points")


def _rounded_bars(ax, xs, vals, color_max, color_min):
    """Bars with rounded tops and a value->color gradient."""
    vmin, vmax = min(vals), max(vals)
    span = (vmax - vmin) or 1.0
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "ramp", [color_min, color_max])
    for x, v in zip(xs, vals):
        t = (v - vmin) / span
        c = cmap(0.25 + 0.75 * t)  # even the min keeps some color
        bar = FancyBboxPatch(
            (x - 0.34, 0), 0.68, v,
            boxstyle="round,pad=0,rounding_size=0.06",
            mutation_aspect=0.35 / max(abs(vmax), 1e-9),
            linewidth=0, facecolor=c, zorder=3)
        ax.add_patch(bar)
    ax.set_xlim(min(xs) - 0.62, max(xs) + 0.62)


def render_spec(spec: dict) -> bytes | None:
    """Render one Vega-Lite-ish spec to PNG bytes; None if unrenderable."""
    _apply_style()
    try:
        mark = spec.get("mark", "bar")
        if isinstance(mark, dict):
            mark = mark.get("type", "bar")
        mark = str(mark).lower()
        rows = spec.get("data", {}).get("values", [])
        enc = spec.get("encoding", {})
        if not enc and isinstance(spec.get("x"), dict):
            enc = {k: spec[k] for k in ("x", "y", "color") if isinstance(spec.get(k), dict)}
        if not rows or not isinstance(rows, list):
            return None

        xf = enc.get("x", {}).get("field") if isinstance(enc.get("x"), dict) else None
        yf = enc.get("y", {}).get("field") if isinstance(enc.get("y"), dict) else None
        keys = list(rows[0].keys()) if isinstance(rows[0], dict) else []
        if not xf:
            str_cols = [k for k in keys if isinstance(rows[0][k], str)]
            xf = str_cols[0] if str_cols else (keys[0] if keys else None)
        if not yf:
            num_cols = [k for k in keys if isinstance(rows[0][k], (int, float))
                        and not isinstance(rows[0][k], bool)]
            yf = num_cols[0] if num_cols else None
        if not xf or not yf:
            return None

        labels = [str(r.get(xf, "")) for r in rows]
        values: list[float] = []
        for r in rows:
            v = r.get(yf)
            try:
                values.append(float(v) if v is not None else np.nan)
            except (TypeError, ValueError):
                values.append(np.nan)
        if len(values) < 2 or not any(not np.isnan(v) for v in values):
            return None

        pairs = [(l, v) for l, v in zip(labels, values) if not np.isnan(v)]
        labels = [p[0] for p in pairs]
        values = [p[1] for p in pairs]
        n = len(values)

        # ————— smart type selection (model's mark is only a hint) —————
        temporal = _is_temporal(labels)
        ordinal = _is_ordinal(labels)
        avg = sum(abs(v) for v in values) / n
        total_share = sum(abs(v) for v in values)
        # auto-donut ONLY when the column is itself share-named (percent/
        # share/%) and sums to ~100 — a count of 96 employees is NOT a share.
        y_is_share = bool(re.search(r"percent|pct|share|proportion|%",
                                    str(yf or "").lower()))
        looks_share = (y_is_share and total_share > 0
                       and all(0 <= v for v in values)
                       and 2 <= n <= 6 and abs(sum(values) - 100) < 12)

        if mark in ("pie", "arc", "donut") or looks_share:
            return _render_donut(spec, labels, values, xf, yf)
        if temporal or (mark in ("line", "area") and n >= 3):
            return _render_line(spec, labels, values, xf, yf, mark == "area", temporal)
        if mark == "scatter" or (mark == "point" and n > 6):
            return _render_scatter(spec, rows, xf, yf)
        if mark in ("histogram", "hist", "bins"):
            return _render_hist(spec, values, yf or xf)
        if mark in ("heatmap", "corr", "correlation"):
            return _render_heatmap(spec, rows)
        if mark in ("grouped-bar", "grouped", "grouped bar"):
            return _render_grouped(spec, rows, stacked=False)
        if mark in ("stacked-bar", "stacked", "stacked bar"):
            return _render_grouped(spec, rows, stacked=True)
        # long-label or many-category bars read better horizontally
        long_labels = any(len(str(l)) > 14 for l in labels)
        if mark in ("bar-h", "hbar") or ((long_labels or n > 8)
                                          and mark in ("bar", "bar-h")):
            return _render_bars_h(spec, labels, values, xf, yf)
        # default: ranked bars (sorted, capped for legibility)
        return _render_bars(spec, labels, values, xf, yf, ordinal)
    except Exception:
        return None
    finally:
        plt.close("all")


def _finish(fig, ax, spec, xf, yf, agg, pct_mode: str | None = None) -> bytes:
    ax.yaxis.set_major_formatter(_axis_formatter(pct_mode))
    # Horizontal rules only: they let you read a bar's height against an axis
    # without the vertical clutter. Ticks off — the labels carry the position.
    ax.grid(True, axis="y", color=GRID, linewidth=0.7)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, pad=6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(AXIS)   # the baseline should read
    title = _title_from(spec, xf, yf, agg, len(spec.get("data", {}).get("values", [])))
    ax.set_title(title, color=INK, fontsize=14, fontweight="bold",
                 loc="left", pad=16)
    ax.text(0, 1.028, _subtitle(agg, len(spec.get("data", {}).get("values", []))),
            transform=ax.transAxes, color=FAINT, fontsize=8.5, family=FONT_MONO)
    ax.set_xlabel(_humanize(xf), fontsize=9.5, family=FONT_MONO, color=DIM)
    ax.set_ylabel(_humanize(yf), fontsize=9.5, family=FONT_MONO, color=DIM)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def _render_bars(spec, labels, values, xf, yf, ordinal) -> bytes:
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    if not ordinal:  # rank categories by value, biggest first
        order = sorted(range(len(values)), key=lambda i: -values[i])
        labels = [labels[i] for i in order]
        values = [values[i] for i in order]
    xs = np.arange(len(labels))
    as_pct = _pct_mode(yf, values)
    if min(values) < 0:  # rounded patches cannot cross the baseline
        ax.bar(xs, values, width=0.68, color=ACCENT, zorder=3)
    else:
        _rounded_bars(ax, xs, values, ACCENT, "#7A4630")
    wrap = ["\n".join(l[i:i + 12] for i in range(0, len(l), 12)) or " "
            for l in labels]
    ax.set_xticks(xs)
    ax.set_xticklabels(wrap, fontsize=8.8, family=FONT_MONO)
    leader = _leader_callout(list(values), list(labels))
    lo, hi = min(values), max(values)
    if hi > 0:
        ax.set_ylim(min(0.0, lo * 1.15), hi * (1.28 if leader else 1.18))
    else:
        ax.set_ylim(lo * 1.28, 0.0)
    ax.yaxis.set_major_formatter(_axis_formatter(as_pct))
    for x, v in zip(xs, values):
        _chip(ax, x, v, _chip_text(v, as_pct),
              va="bottom" if v >= 0 else "top", dy=4)
    if leader:
        i = values.index(hi)
        ax.annotate(leader[1], (i, hi), ha="center",
                    va="bottom" if hi >= 0 else "top", zorder=6,
                    fontsize=8.4, family=FONT_MONO, color=ACCENT_SOFT,
                    xytext=(0, 22 if hi >= 0 else -22),
                    textcoords="offset points")
    return _finish(fig, ax, spec, xf, yf, "sum", as_pct)


def _render_line(spec, labels, values, xf, yf, area: bool, temporal: bool) -> bytes:
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    xs = np.arange(len(labels))
    if area:
        ax.fill_between(xs, values, color=ACCENT, alpha=0.18, zorder=2)
        ax.plot(xs, values, color=ACCENT_SOFT, linewidth=2.4, zorder=3)
    else:
        # A soft wash under the line reads as volume; dense series drop the
        # markers so the trend itself stays legible.
        ax.fill_between(xs, values, color=ACCENT, alpha=0.10, zorder=2)
        dense = len(values) > 24
        ax.plot(xs, values, color=ACCENT, linewidth=2.4, zorder=3,
                marker=None if dense else "o", markersize=5,
                markerfacecolor=ACCENT,
                markeredgecolor=PANEL, markeredgewidth=1.2)
    best = int(np.argmax(values))
    ax.scatter([best], [values[best]], s=110, color=ACCENT_SOFT, zorder=4,
               edgecolors=INK, linewidths=1.0)
    _chip(ax, best, values[best], _fmt_num(values[best]), leader=True, dy=9)
    trough = int(np.argmin(values))
    if 0 < trough < len(values) - 1:
        ax.scatter([trough], [values[trough]], s=90, color=BLUE, zorder=4,
                   edgecolors=PANEL, linewidths=1.0)
        ax.annotate(_fmt_num(values[trough]), (trough, values[trough]),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.6, color=BLUE,
                    family=FONT_MONO, zorder=4)
    badge = _trend_badge(list(values))
    if badge:
        ax.text(0.99, 1.015, badge, transform=ax.transAxes, ha="right",
                va="bottom", color=ACCENT_SOFT, fontsize=8.6,
                family=FONT_MONO)
    step = _tick_step(labels)
    ax.set_xticks(xs[::step])
    ax.set_xticklabels(_short_labels([labels[i] for i in range(0, len(labels), step)]),
                       fontsize=8.8, family=FONT_MONO)
    lo, hi = min(values), max(values)
    pad = (hi - lo) * 0.25 or hi * 0.1 or 1
    ax.set_ylim(max(0, lo - pad), hi + pad)
    return _finish(fig, ax, spec, xf, yf, "trend" if temporal else "line")


def _render_scatter(spec, rows, xf, yf) -> bytes:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    xs, ys = [], []
    for r in rows:
        try:
            xs.append(float(r.get(xf)))
            ys.append(float(r.get(yf)))
        except (TypeError, ValueError):
            continue
    if len(xs) < 3:
        return None
    ax.scatter(xs, ys, s=52, color=ACCENT, alpha=0.75, zorder=3,
               edgecolors=PANEL, linewidths=0.8)
    if len(xs) >= 4:  # faint trend line
        z = np.polyfit(xs, ys, 1)
        xr = np.linspace(min(xs), max(xs), 50)
        ax.plot(xr, np.polyval(z, xr), color=GREEN, linewidth=1.6,
                alpha=0.65, zorder=2, linestyle="--")
    r = _pearson(xs, ys)
    callout = f"r = {r:.2f}" if r is not None else "scatter"
    ax.text(0.015, 0.97, f"{callout} · n = {len(xs)}", transform=ax.transAxes,
            ha="left", va="top", color=DIM, family=FONT_MONO, fontsize=8.8)
    ax.xaxis.set_major_formatter(_num_formatter())
    return _finish(fig, ax, spec, xf, yf, "correlation")


def _render_bars_h(spec, labels, values, xf, yf) -> bytes:
    """Horizontal ranked bars — for long labels or many categories."""
    fig, ax = plt.subplots(figsize=(7.6, max(3.2, 0.42 * len(labels) + 1.2)))
    order = sorted(range(len(values)), key=lambda i: -values[i])
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    ys = np.arange(len(labels))
    # barh plots bottom-up: reverse so the biggest value sits at the top
    ys = ys[::-1]
    vmin, vmax = min(values), max(values)
    span = (vmax - vmin) or 1.0
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "ramp", ["#4A3325", ACCENT])
    as_pct = _pct_mode(yf, values)
    for y, v in zip(ys, values):
        t = (v - vmin) / span
        ax.barh(y, v, height=0.62, color=cmap(0.25 + 0.75 * t), zorder=3)
        ax.annotate(_chip_text(v, as_pct), (v, y), ha="left", va="center",
                    fontsize=8.6, color=INK, family=FONT_MONO, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.28", fc=CHIP_BG,
                              ec=CHIP_EDGE, lw=0.8),
                    xytext=(5, 0), textcoords="offset points")
    leader = _leader_callout(list(values), list(labels)) if len(values) >= 2 \
        else None
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8.8, family=FONT_MONO)
    ax.set_xlim(0, vmax * 1.34)
    ax.xaxis.set_major_formatter(_axis_formatter(as_pct))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    title = _title_from(spec, xf, yf, "sum", len(labels))
    ax.set_title(title, color=INK, fontsize=14, fontweight="bold", loc="left", pad=16)
    ax.text(0, 1.03, _subtitle("sum", len(labels)), transform=ax.transAxes,
            color=FAINT, fontsize=8.5, family=FONT_MONO)
    if leader:
        ax.text(0.99, 1.015, leader[1], transform=ax.transAxes, ha="right",
                va="bottom", color=ACCENT_SOFT, fontsize=8.6,
                family=FONT_MONO)
    ax.set_xlabel(_humanize(yf), fontsize=9.5, family=FONT_MONO, color=DIM)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def _render_hist(spec, values, col) -> bytes:
    """Distribution histogram with freedman-style binning. Bins the raw
    numbers the sandbox passed through; median/mean marked."""
    clean = [v for v in values if not np.isnan(v)]
    if len(clean) < 8:
        return None
    arr = np.asarray(clean)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1 or (arr.max() - arr.min()) or 1.0
    bin_w = 2 * iqr / (len(arr) ** (1 / 3))          # Freedman–Diaconis
    n_bins = int(np.clip(round((arr.max() - arr.min()) / bin_w), 6, 30)) \
        if arr.max() > arr.min() else 8
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    counts, edges, patches = ax.hist(
        arr, bins=n_bins, color=ACCENT, alpha=0.88, zorder=3,
        edgecolor=PANEL, linewidth=1.0)
    med = float(np.median(arr))
    mean = float(np.mean(arr))
    ax.axvspan(q1, q3, color=BLUE, alpha=0.07, zorder=1)
    ax.axvline(med, color=GREEN, linewidth=1.8, zorder=4)
    ax.axvline(mean, color=ACCENT_SOFT, linewidth=1.4, linestyle="--", zorder=4)
    # Labels live on whichever half the bars leave open: a right-skewed
    # distribution piles up on the left, so its stats read on the right.
    mid = (float(arr.min()) + float(arr.max())) / 2
    if med <= mid:
        tx, ha = 0.985, "right"
    else:
        tx, ha = 0.015, "left"
    ax.text(tx, 0.97, f"median {_fmt_num(med)}", transform=ax.transAxes,
            ha=ha, va="top", fontsize=8.4, color=GREEN, family=FONT_MONO)
    ax.text(tx, 0.91, f"mean {_fmt_num(mean)}", transform=ax.transAxes,
            ha=ha, va="top", fontsize=8.4, color=ACCENT_SOFT,
            family=FONT_MONO)
    ax.xaxis.set_major_formatter(_num_formatter())
    skew = _skew_note(list(clean))
    agg = f"{n_bins} bins" + (f" · {skew}" if skew else "")
    return _finish(fig, ax, spec, col, "count", agg)


def _render_grouped(spec, rows, stacked: bool) -> bytes:
    """Two-categorical bars: rows carry <cat1>, <cat2>, <value>.
    Auto-detects the fields; legend = second category."""
    if not rows or not isinstance(rows[0], dict):
        return None
    keys = list(rows[0].keys())
    str_keys = [k for k in keys if isinstance(rows[0][k], str)]
    num_keys = [k for k in keys if isinstance(rows[0][k], (int, float))
                and not isinstance(rows[0][k], bool)]
    if len(str_keys) < 2 or not num_keys:
        return None
    c1, c2, val = str_keys[0], str_keys[1], num_keys[0]
    groups: dict[str, dict[str, float]] = {}
    for r in rows:
        try:
            groups.setdefault(str(r.get(c1)), {})[str(r.get(c2))] = float(r.get(val))
        except (TypeError, ValueError):
            continue
    if len(groups) < 2:
        return None
    g1 = list(groups.keys())
    g2 = sorted({k for inner in groups.values() for k in inner})
    if len(g1) * len(g2) > 60 or len(g2) > 10:
        return None
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    xs = np.arange(len(g1))
    width = 0.8 / len(g2)
    bottom = np.zeros(len(g1))
    for j, s2 in enumerate(g2):
        vals = np.array([groups[g].get(s2, 0.0) for g in g1])
        color = SERIES[j % len(SERIES)]
        if stacked:
            ax.bar(xs, vals, width * 0.92, bottom=bottom, color=color,
                   label=_humanize(s2), zorder=3,
                   edgecolor=PANEL, linewidth=0.8)
            bottom += vals
        else:
            off = (j - (len(g2) - 1) / 2) * width
            ax.bar(xs + off, vals, width * 0.9, color=color,
                   label=_humanize(s2), zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([_humanize(g) for g in g1], fontsize=8.8,
                       family=FONT_MONO)
    ax.legend(frameon=False, labelcolor=INK, fontsize=8.4,
              ncols=len(g2), loc="lower right", bbox_to_anchor=(1.0, 1.0),
              handlelength=1.1, columnspacing=1.2,
              prop={"family": FONT_MONO, "size": 8.4})
    ax.yaxis.set_major_formatter(_num_formatter())
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    agg = "stacked" if stacked else "grouped"
    ax.set_title(f"{_humanize(val)} by {_humanize(c1)} and {_humanize(c2)}",
                 color=INK, fontsize=14, fontweight="bold", loc="left", pad=16)
    ax.text(0, 1.03, agg, transform=ax.transAxes, color=FAINT,
            fontsize=8.5, family=FONT_MONO)
    ax.set_xlabel(_humanize(c1), fontsize=9.5, family=FONT_MONO, color=DIM)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def _render_heatmap(spec, rows) -> bytes:
    """Correlation matrix: rows carry numeric columns (wide format), or
    <row>,<col>,<value> long format (two string keys + one numeric).
    Verdigris-to-signal diverging ramp."""
    if not rows or not isinstance(rows[0], dict):
        return None
    keys = [k for k, v in rows[0].items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)]
    str_keys = [k for k, v in rows[0].items() if isinstance(v, str)]
    long_mode = len(str_keys) >= 2 and len(keys) == 1 and len(rows) >= 4
    if long_mode:
        sk = str_keys[:2]
        mat: dict[str, dict[str, float]] = {}
        for r in rows:
            mat.setdefault(str(r.get(sk[0])), {})[str(r.get(sk[1]))] = float(r.get(keys[0]))
        labels_m = list(mat.keys())
        col_names = sorted({b for inner in mat.values() for b in inner})
        data = np.array([[mat[a].get(b, np.nan) for b in col_names] for a in labels_m])
    else:
        import pandas as _pd
        data = _pd.DataFrame([{k: r.get(k) for k in keys} for r in rows],
                             columns=keys).corr().to_numpy()
        col_names = keys
    if data.size == 0 or np.isnan(data).all():
        return None
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "corr", [BLUE, PANEL, ACCENT])
    im = ax.imshow(data, cmap=cmap, vmin=-1 if long_mode else None,
                   vmax=1 if long_mode else None)
    n = data.shape[0]
    ax.set_xticks(range(len(col_names)))
    ax.set_yticks(range(n))
    short = [k[:14] for k in col_names]
    ax.set_xticklabels(short, rotation=38, ha="right", fontsize=8,
                       family=FONT_MONO, color=DIM)
    ax.set_yticklabels(short, fontsize=8, family=FONT_MONO, color=DIM)
    for i in range(n):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}".rstrip("0").rstrip("."),
                        ha="center", va="center", fontsize=7.6,
                        family=FONT_MONO, color=INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(spec.get("title") or "Correlation heatmap", color=INK,
                 fontsize=14, fontweight="bold", loc="left", pad=14)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def _render_donut(spec, labels, values, xf, yf) -> bytes:
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    colors = [SERIES[i % len(SERIES)] for i in range(len(values))]
    wedges, _ = ax.pie(
        [abs(v) for v in values], colors=colors, startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.42, edgecolor=PANEL, linewidth=2.5))
    total = sum(abs(v) for v in values)
    ax.text(0, 0.08, _num_formatter()(total, None), ha="center",
            fontsize=21, fontweight="bold", color=INK)
    ax.text(0, -0.14, f"total {_humanize(yf).lower()}", ha="center",
            fontsize=8.5, color=FAINT, family=FONT_MONO)
    top_i = int(np.argmax([abs(v) for v in values]))
    ax.text(0, -0.32,
            f"leader {_humanize(labels[top_i])} · "
            f"{abs(values[top_i]) / total:.0%}",
            ha="center", fontsize=8.6, color=ACCENT_SOFT, family=FONT_MONO)
    legend_labels = [
        f"{_humanize(l)}  ·  {_num_formatter()(abs(v), None)}  ({abs(v)/total:.0%})"
        for l, v in zip(labels, values)]
    ax.legend(wedges, legend_labels, loc="center left",
              bbox_to_anchor=(1.02, 0.5), frameon=False,
              labelcolor=INK, fontsize=9.2)
    ax.set_title(_title_from(spec, xf, yf, "share", len(values)),
                 color=INK, fontsize=14, fontweight="bold", loc="left", pad=14)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def render_all(charts: list[dict]) -> list[bytes]:
    out = []
    for spec in charts or []:
        if isinstance(spec, dict):
            png = render_spec(spec)
            if png:
                out.append(png)
    return out
