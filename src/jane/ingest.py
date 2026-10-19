"""File ingestion → pandas DataFrames. Excel (multi-sheet, merged cells), CSV, TSV, JSON, Parquet."""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from .logging_setup import get_logger

log = get_logger("jane.ingest")

EXCEL_EXTS = {".xlsx", ".xls", ".xlsm", ".xlsb", ".ods", ".odt"}
CSV_EXTS = {".csv", ".tsv", ".txt"}
TABULAR_EXTS = EXCEL_EXTS | CSV_EXTS | {".json", ".parquet", ".pq"}


@dataclass
class Sheet:
    name: str
    df: pd.DataFrame


@dataclass
class LoadedFile:
    name: str
    sheets: list[Sheet] = field(default_factory=list)

    @property
    def primary(self) -> pd.DataFrame:
        return self.sheets[0].df


def _unmerge_excel_sheets(raw_bytes: bytes, ext: str) -> dict[str, pd.DataFrame]:
    """Read every sheet, forward-filling values lost to merged cells."""
    out: dict[str, pd.DataFrame] = {}
    if ext in {".ods", ".odt"}:
        frames = pd.read_excel(io.BytesIO(raw_bytes), sheet_name=None, engine="odf")
        for name, df in frames.items():
            out[str(name)] = df.ffill().bfill()
        return out
    if ext == ".xlsb":
        frames = pd.read_excel(io.BytesIO(raw_bytes), sheet_name=None, engine="pyxlsb")
        for name, df in frames.items():
            out[str(name)] = df.ffill().bfill()
        return out

    # xlsx/xls/xlsm via openpyxl for merged-cell awareness
    from openpyxl import load_workbook

    wb = load_workbook(io.BytesIO(raw_bytes), read_only=False, data_only=True)
    for ws in wb.worksheets:
        merged = list(ws.merged_cells.ranges)
        data = ws.values
        try:
            first = next(data)
        except StopIteration:
            out[ws.title] = pd.DataFrame()
            continue
        rows = [list(first)] + [list(r) for r in data]
        width = max(len(r) for r in rows)
        grid = [r + [None] * (width - len(r)) for r in rows]

        if merged:
            # paint the top-left value across every merged range, then ffill per-column
            for rng in merged:
                tl = ws.cell(rng.min_row, rng.min_col).value
                for r in range(rng.min_row, min(rng.max_row, len(grid)) + 1):
                    for c in range(rng.min_col, min(rng.max_col, len(grid[r - 1]) if grid else 0) + 1):
                        if r - 1 < len(grid) and c - 1 < len(grid[r - 1]):
                            grid[r - 1][c - 1] = tl

        # reports often start with a title/metadata block: sniff the real header row
        raw = pd.DataFrame(grid, columns=[f"col_{i}" for i in range(width)])
        hidx = sniff_header_row(raw)
        header = grid[hidx]
        body = grid[hidx + 1:]
        df = pd.DataFrame(body, columns=[str(h) if h is not None else f"col_{i}"
                                         for i, h in enumerate(header)])

        # basic cleanup: forward-fill horizontal+vertical gaps the merge left behind
        if merged:
            df = df.ffill(axis=0).bfill(axis=0)
        out[ws.title] = df
    return out


def sniff_header_row(df_raw: pd.DataFrame, max_scan: int = 10) -> int:
    """Return index of the row most likely to be the real header (0 if first row)."""
    best_idx, best_score = 0, -1.0
    for i in range(min(max_scan, len(df_raw))):
        row = df_raw.iloc[i]
        vals = row.dropna().astype(str)
        if len(vals) == 0:
            continue
        non_numeric = sum(1 for v in vals if not _is_number(v))
        uniq = vals.nunique()
        score = non_numeric * 2 + uniq
        # header rows are usually short-ish; penalize rows that are basically data
        if i > 0 and non_numeric < len(vals) * 0.5:
            score -= 5
        if score > best_score:
            best_idx, best_score = i, score
    return best_idx


def _is_number(v: str) -> bool:
    try:
        float(str(v).replace(",", ""))
        return True
    except (TypeError, ValueError):
        return False


def _modal_width(df: pd.DataFrame) -> int:
    """Most common non-null count per row — the real width when stray title rows exist."""
    if not len(df):
        return 0
    widths = df.notna().sum(axis=1)
    return int(widths.mode().iat[0])


def _header_like(row: pd.Series) -> bool:
    """A plausible header: mostly filled, mostly non-numeric cells."""
    vals = [str(v) for v in row.dropna() if str(v).strip()]
    if not vals:
        return False
    non_numeric = sum(1 for v in vals if not _is_number(v))
    return non_numeric >= 0.8 * len(vals)


def _find_header_row(df2: pd.DataFrame, max_scan: int = 10) -> int:
    """Index of the row where the real table starts (0 == row 0 / no reslice).

    Reslicing is warranted only when leading rows are *narrower* than the
    modal row width (title/metadata lines). The table then starts at the first
    full-width, header-like row that has data beneath it.
    """
    if len(df2) < 2:
        return 0
    widths = df2.notna().sum(axis=1)
    modal = _modal_width(df2)
    if modal < 2:
        return 0
    for i in range(min(max_scan, len(df2) - 1)):
        if widths.iat[i] >= 0.9 * modal and _header_like(df2.iloc[i]) \
                and widths.iloc[i + 1:].max() >= 0.6 * modal:
            return i
    return 0


def _detect_delimiter(raw: bytes) -> str:
    """Pick the delimiter whose per-line count is most consistent.

    csv.Sniffer can be fooled by title lines with spaces; counting candidate
    delimiters directly is more robust for messy exports.
    """
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:
        return ","
    lines = [ln for ln in text.splitlines() if ln.strip()][:200]
    best, best_score = ",", -1.0
    for d in (",", "\t", ";", "|"):
        if not lines:
            break
        counts = [ln.count(d) for ln in lines]
        modal = max(set(counts), key=counts.count)
        if modal == 0:
            continue
        frac = sum(1 for c in counts if c == modal) / len(counts)
        score = frac + min(modal, 5) / 100
        if score > best_score + 1e-9:
            best, best_score = d, score
    return best


def _leading_junk_lines(raw: bytes, enc: str, sample: int = 65536) -> int:
    """Count leading lines with no delimiter (title/metadata lines).

    Reads only the first `sample` bytes — O(1) regardless of file size.
    A clean CSV's first line (the header) always contains the delimiter; a
    title line like 'Monthly Export Report' does not. Quote-safe via csv.reader.
    """
    try:
        text = raw[:sample].decode(enc, errors="replace")
    except Exception:
        return 0
    delims = [",", "\t", ";", "|"]
    junk = 0
    for row in csv.reader(io.StringIO(text)):
        line = ",".join(row)
        if not line.strip():
            continue  # blank
        if not any(d in line for d in delims):
            junk += 1
        else:
            break
    return junk


def _split_rows(raw: bytes, enc: str, sep: str, max_lines: int) -> pd.DataFrame:
    """Ragged-tolerant row split of the first max_lines via csv.reader."""
    text = raw.decode(enc, errors="replace")
    rows = []
    for i, row in enumerate(csv.reader(io.StringIO(text), delimiter=sep)):
        rows.append([c if c.strip() else None for c in row])
        if i + 1 >= max_lines:
            break
    width = max((len(r) for r in rows), default=1)
    rows = [r + [None] * (width - len(r)) for r in rows]
    return pd.DataFrame(rows)


def _read_csv_like(raw: bytes, name: str) -> pd.DataFrame:
    sep = "\t" if name.lower().endswith(".tsv") else None  # None -> pandas sniffs via engine='python'
    last_err: Exception | None = None
    for enc in ("utf-8-sig", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=sep, encoding=enc, engine="python")
            # cheap gates first: leading junk lines (raw text scan of the file top),
            # then column-level checks on the *head* only (big-frame safety)
            junk = _leading_junk_lines(raw, enc)
            head = df.head(20)
            mixed_or_empty = 0
            for c in head.columns:
                col = head[c].dropna()
                if col.empty:
                    mixed_or_empty += 1
                elif col.size >= 2:
                    nums = sum(1 for v in col if _is_number(str(v)))
                    if 0 < nums < col.size:
                        mixed_or_empty += 1
            single_col_mangled = (
                len(df.columns) == 1
                and head[head.columns[0]].astype(str)
                    .str.contains(r"[,;\t]", regex=True).mean() > 0.5
            )
            suspicious = (any(str(c).startswith("Unnamed:") for c in df.columns)
                          or mixed_or_empty >= 1 or single_col_mangled
                          or junk > 0)
            if suspicious:
                sep2 = sep or _detect_delimiter(raw)
                probe = _split_rows(raw, enc, sep2, max_lines=100)
                hidx = max(_find_header_row(probe), _leading_junk_lines(raw, enc))
                if hidx > 0:
                    try:
                        df = pd.read_csv(io.BytesIO(raw), sep=sep2, encoding=enc,
                                         engine="python", skiprows=hidx)
                    except pd.errors.ParserError:
                        # genuinely ragged body: rebuild from a padded row grid
                        df2 = _split_rows(raw, enc, sep2, max_lines=10**9)
                        header = [str(h) if not pd.isna(h) else f"col_{i}"
                                  for i, h in enumerate(df2.iloc[hidx])]
                        df = pd.DataFrame(df2.iloc[hidx + 1:].to_numpy(), columns=header)
            return df
        except UnicodeDecodeError as e:
            last_err = e
        except pd.errors.ParserError:
            # retry with tab/semicolon fallbacks before giving up
            for alt in ("\t", ";", "|"):
                try:
                    return pd.read_csv(io.BytesIO(raw), sep=alt, encoding=enc, engine="python")
                except Exception:
                    continue
            raise
    raise last_err or ValueError(f"Could not decode {name}")


def _safe_table_name(name: str) -> str:
    """SQL-safe table identifier: DuckDB reads bare hyphens as minus signs
    ('FROM bank-additional-full' = arithmetic), so they become underscores.
    Collapses repeats and trims leading digits/dots."""
    import re as _re
    out = _re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_")
    if not out:
        out = "data"
    if out[0].isdigit():
        out = "t_" + out
    return out


def load_file(path: str | Path, sheet: Optional[str] = None,
              progress: Optional[Callable[[str], None]] = None) -> LoadedFile:
    """Load any supported file into sheets of DataFrames (merged cells handled)."""
    p = Path(path)
    raw = p.read_bytes()
    ext = p.suffix.lower()
    say = progress or (lambda s: None)
    say(f"reading {p.name} ({len(raw)/1e6:.1f} MB)")

    if ext in EXCEL_EXTS:
        frames = _unmerge_excel_sheets(raw, ext)
    elif ext in CSV_EXTS:
        frames = {p.stem: _read_csv_like(raw, p.name)}
    elif ext == ".json":
        try:
            frames = {p.stem: pd.json_normalize(json.loads(raw.decode("utf-8")))}
        except Exception:
            frames = {p.stem: pd.read_json(io.BytesIO(raw))}
    elif ext in {".parquet", ".pq"}:
        frames = {p.stem: pd.read_parquet(io.BytesIO(raw))}
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    sheets = [Sheet(name=_safe_table_name(k), df=v) for k, v in frames.items()]
    if sheet:
        sheets = [s for s in sheets if s.name == sheet]
    if not sheets:
        raise ValueError(f"No sheets to load from {p.name}")
    say(f"loaded {len(sheets)} sheet(s): {', '.join(s.name for s in sheets)}")
    return LoadedFile(name=p.name, sheets=sheets)


def load_uploaded(raw: bytes, filename: str, sheet: Optional[str] = None,
                  progress: Optional[Callable[[str], None]] = None) -> LoadedFile:
    """Load from in-memory bytes (Streamlit upload path) using a temp file."""
    import tempfile

    suffix = Path(filename).suffix or ".csv"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir=None) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    try:
        loaded = load_file(tmp_path, sheet=sheet, progress=progress)
        # temp files get random names — restore the user's filename as sheet name
        stem = Path(filename).stem or "data"
        if len(loaded.sheets) == 1:
            loaded.sheets[0].name = _safe_table_name(stem)
        return loaded
    finally:
        Path(tmp_path).unlink(missing_ok=True)
