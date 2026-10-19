"""FastAPI bridge for Jane — serves the React frontend from the existing package.

Zero changes inside src/jane: this file imports the existing modules (agent,
ingest, engine, sandbox, charts, insights, history, config) and translates
HTTP/SSE to them, per docs/superpowers/specs/2026-10-01-react-frontend-3d-landing-design.md.

Run: .venv/Scripts/python.exe -m uvicorn server:app --port 8000
"""
from __future__ import annotations

import base64
import json
import queue
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from jane.agent import AnalystAgent, AgentContext
from jane.charts import render_all
from jane.config import settings
from jane.history import HistoryStore
from jane.ingest import load_uploaded
from jane.ollama_client import OllamaClient, OllamaError
from jane.types import AnalysisCancelled, AnalysisResult, ChatMessage
from jane.validator import extract_sql  # noqa: F401  (kept: parity helpers may use it)

app = FastAPI(title="Jane bridge", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:5174", "http://127.0.0.1:5174",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Session state: a single-user local app keeps everything in this process.
# ---------------------------------------------------------------------------

SAMPLE_DIR = Path(__file__).parent / "src" / "jane" / "sample_data"

state_lock = threading.RLock()
# One analysis runs at a time (state_lock), so one cancel flag is enough: the
# user stops the run in flight, and the agent loop notices between attempts.
_cancel = threading.Event()
_sess: dict[str, Any] = {
    "tables": {},            # name -> DataFrame
    "active": None,          # active table name
    "file_name": None,
    "history": [],           # [{role, content, ...payload}] like app.py
    "session_id": None,      # HistoryStore session
    "model_choice": None,    # OllamaClient.model_override
}
_client = OllamaClient(settings.ollama_host, settings.request_timeout)
_agent = AnalystAgent(_client)
_store = HistoryStore(settings.history_db)


def _tables_payload() -> list[dict]:
    out = []
    for name, df in _sess["tables"].items():
        out.append({
            "name": name,
            "rows": int(len(df)),
            "columns": [str(c) for c in df.columns],
        })
    return out


def _agent_with_choice() -> AnalystAgent:
    _agent.client.model_override = _sess["model_choice"]
    return _agent


def _installed_models() -> list[str]:
    """Model names Ollama actually has, in the order it lists them."""
    return [str(t.get("name")) for t in _client.tags()]


def _active_model() -> str:
    choice = _sess["model_choice"]
    if choice:
        return choice
    installed = _installed_models()
    if settings.primary_model in installed:
        return settings.primary_model
    if settings.fallback_model in installed:
        return settings.fallback_model
    return installed[0] if installed else settings.primary_model


def _open_question(question: str) -> AgentContext:
    """Record the user turn and build the context for it.

    Shared by /api/ask and /api/brief, so both paths see the same history and a
    brief's later sections can build on its earlier ones.
    """
    with state_lock:
        ctx = AgentContext(
            tables=_sess["tables"],
            active_table=_sess["active"],
            history=[ChatMessage(m["role"], m["content"])
                     for m in _sess["history"] if m["role"] in ("user", "assistant")],
        )
        if _sess["session_id"] is None:
            _sess["session_id"] = _store.new_session(title=question[:60])
        _store.add_message(_sess["session_id"], "user", question)
        _sess["history"].append({"role": "user", "content": question})
    return ctx


def _jsonable_rows(rows: list) -> list:
    """Rows may hold numpy/pandas scalars; make them JSON-safe (default=str)."""
    return json.loads(json.dumps(rows, default=str))


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str


class ModelRequest(BaseModel):
    model: str


class BriefRequest(BaseModel):
    """Optional override; the default brief picks its own questions."""
    questions: list[str] | None = None


class SheetRequest(BaseModel):
    name: str


class SampleRequest(BaseModel):
    file: str  # a filename inside src/jane/sample_data — never a path


class PrepRequest(BaseModel):
    action: str  # a key of _PREP_ACTIONS — never client-supplied arguments


class FixerRequest(BaseModel):
    code: str
    language: str  # "sql" | "python"


# ---------------------------------------------------------------------------
# Health / state / suggestions
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    try:
        return {"ollama": True, "models": _installed_models(), "model": _active_model(),
                "primary": settings.primary_model, "fallback": settings.fallback_model}
    except OllamaError:
        return {"ollama": False, "models": [],
                "model": _sess["model_choice"] or settings.primary_model,
                "primary": settings.primary_model, "fallback": settings.fallback_model}


@app.post("/api/model")
def set_model(req: ModelRequest) -> dict:
    """Switch the model every later question runs on.

    Validated against Ollama so the picker can never select something that
    isn't installed (the agent would silently fall back and look broken).
    """
    try:
        installed = _installed_models()
    except OllamaError as e:
        raise HTTPException(503, f"Ollama unreachable: {e}") from e
    if req.model not in installed:
        raise HTTPException(400, f"Model not installed: {req.model}")
    with state_lock:
        _sess["model_choice"] = req.model
    return {"model": req.model, "models": installed,
            "primary": settings.primary_model, "fallback": settings.fallback_model}


@app.get("/api/state")
def get_state() -> dict:
    with state_lock:
        return {
            "has_data": bool(_sess["tables"]),
            "file_name": _sess["file_name"],
            "active": _sess["active"],
            "tables": _tables_payload(),
            "model": _sess["model_choice"] or settings.primary_model,
        }


@app.get("/api/suggestions")
def suggestions() -> dict:
    with state_lock:
        if not _sess["tables"]:
            return {"questions": []}
        from jane.insights import suggest_questions
        return {"questions": suggest_questions(_sess["tables"], _sess["active"])}


# ---------------------------------------------------------------------------
# Dashboard: deterministic insight cards + a chart grid for the active table
# ---------------------------------------------------------------------------

def _png_data_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


def _tc(text: str) -> str:
    """Title Case a column-derived label, keeping short acronyms uppercase."""
    words = []
    for w in str(text).split():
        words.append(w.upper() if len(w) <= 3 and w.isalpha() and w.islower()
                     and w not in ("and", "vs", "by", "of") else w[:1].upper() + w[1:])
    return " ".join(words)


def _panel(title: str, question: str, spec: dict | None, note: str = "") -> dict | None:
    """One chart panel: the drawn figure plus the question that regenerates it."""
    if not spec:
        return None
    try:
        png = render_all([spec])[0]
    except Exception:
        return None
    return {"title": title, "question": question, "note": note,
            "chart": _png_data_uri(png)}


def _dashboard_panels(name: str, df: pd.DataFrame) -> list[dict]:
    """A small, deterministic chart grid — the shape of the table at a glance."""
    import numpy as np

    from jane.insights import (_entity_key_like, _is_id_like, _mostly_dates,
                               _shorten, yname)

    cols = [str(c) for c in df.columns]
    id_cols = {c for c in cols if _is_id_like(c, df[c])}
    nums = [c for c in df.select_dtypes("number").columns if c not in id_cols]
    flags = [c for c in nums if df[c].dropna().isin([0, 1]).all()
             and df[c].dropna().nunique() <= 2]
    measures = [c for c in nums if c not in flags]
    cats = [c for c in cols
            if c not in id_cols and c not in nums
            and 2 <= df[c].nunique(dropna=True) <= 12]
    keys = [c for c in cols if c not in id_cols and _entity_key_like(df[c])]
    dates = [c for c in cols
             if c not in id_cols
             and (pd.api.types.is_datetime64_any_dtype(df[c])
                  or (_is_text_col(df[c]) and _mostly_dates(df[c])))]

    panels: list[dict] = []
    values = lambda xs: [  # noqa: E731 — tiny row builder for specs
        {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()} for r in xs
    ]

    def spread(cnum: str, ccat: str) -> float:
        g = df.groupby(ccat)[cnum].mean()
        if len(g) < 2 or g.isna().all():
            return 0.0
        return float(g.max() - g.min()) / (abs(float(g.mean())) + 1e-9)

    # 1 — the metric over time (the story most tables are about)
    if dates and measures:
        d, m = dates[0], measures[0]
        try:
            agg = (df[[d, m]].dropna().assign(_d=lambda x: x[d].astype(str).str[:10])
                   .groupby("_d", as_index=False)[m].sum().sort_values("_d"))
            panel = _panel(
                f"{_tc(yname(m))} over time",
                f"Plot {yname(m)} over {yname(d)} as a line chart",
                {"mark": "line", "title": f"{_tc(yname(m))} over {_tc(yname(d))}",
                 "data": {"values": values(agg.tail(60).to_dict("records"))},
                 "encoding": {"x": {"field": "_d"}, "y": {"field": m}}},
                note=f"summed by {yname(d)}")
            if panel:
                panels.append(panel)
        except Exception:
            pass

    # 2 — the measure broken down by the category that moves it most
    if measures and (cats or keys):
        groups = (cats + keys)[:6]
        best = max(((m, c, spread(m, c)) for m in measures[:5] for c in groups),
                   key=lambda t: t[2], default=None)
        if best and best[2] > 0.05:
            m, c = best[0], best[1]
            agg = (df.groupby(c, as_index=False)[m].sum()
                   .sort_values(m, ascending=False).head(12))
            panels.append(_panel(
                f"{_tc(yname(m))} by {_tc(yname(c))}",
                f"Break down {yname(m)} by {yname(c)}, with a bar chart",
                {"mark": "bar", "title": f"{_tc(yname(m))} by {_tc(yname(c))}",
                 "data": {"values": values(agg.to_dict("records"))},
                 "encoding": {"x": {"field": c}, "y": {"field": m}}},
                note=f"top {min(12, df[c].nunique())} of {df[c].nunique()} {yname(c)} values") or None)

    # 3 — the distribution of the primary measure
    if measures:
        cont = measures[0]
        raw = df[cont].dropna()
        if len(raw) > 20 and raw.nunique() > 8:
            rows = [{"value": float(v)} for v in raw.head(400)]
            panels.append(_panel(
                f"Distribution of {_tc(yname(cont))}",
                f"Show the distribution of {yname(cont)} as a histogram",
                {"mark": "histogram", "title": f"Distribution of {_tc(yname(cont))}",
                 "data": {"values": rows},
                 "encoding": {"x": {"field": "value"}}}) or None)

    # 4 — the relationship between the strongest-measures pair
    if len(measures) >= 2:
        cand = measures[:5]
        best_pair, best_r = None, 0.0
        for i, a in enumerate(cand):
            for b in cand[i + 1:]:
                pair = df[[a, b]].dropna()
                if len(pair) < 8:
                    continue
                try:
                    r = abs(float(np.corrcoef(pair[a], pair[b])[0, 1]))
                except Exception:
                    continue
                if r > best_r:
                    best_pair, best_r = (a, b), r
        # a scatter of two unrelated columns is noise: show it only when the
        # correlation carries some signal, unless nothing else got built.
        if best_pair and (best_r > 0.15 or not panels):
            a, b = best_pair
            pairs = df[[a, b]].dropna()
            sample = pairs.sample(min(300, len(pairs)), random_state=0)
            try:
                r = float(np.corrcoef(sample[a], sample[b])[0, 1])
            except Exception:
                r = 0.0
            panels.append(_panel(
                f"{_tc(yname(a))} vs {_tc(yname(b))}",
                f"Is there a relationship between {yname(a)} and {yname(b)}? Plot it.",
                {"mark": "point", "title": f"{_tc(yname(a))} vs {_tc(yname(b))}",
                 "data": {"values": values(sample.to_dict("records"))},
                 "encoding": {"x": {"field": a}, "y": {"field": b}}},
                note=f"r = {r:+.2f} · {len(pairs):,} pairs") or None)

    # 5 — where the gaps are
    missing = {c: int(df[c].isna().sum()) for c in cols}
    holes = {c: n for c, n in missing.items() if n}
    if holes:
        rows = [{"column": c, "missing": int(n)} for c, n in
                sorted(holes.items(), key=lambda kv: -kv[1])[:12]]
        panels.append(_panel(
            "Missing values by column",
            "Which columns have missing data, and how much does it matter?",
            {"mark": "bar", "title": "Missing values by column",
             "data": {"values": rows},
             "encoding": {"x": {"field": "column"}, "y": {"field": "missing"}}},
            note=f"{sum(holes.values())} empty cells in {len(holes)} column(s)") or None)

    # 6 — composition: how the rows divide up
    if cats:
        c = cats[0]
        agg = df[c].value_counts().head(8)
        rows = [{"group": _shorten(str(k)), "rows": int(v)} for k, v in agg.items()]
        panels.append(_panel(
            f"Rows by {_tc(yname(c))}",
            f"Which {yname(c)} appears most often?",
            {"mark": "bar", "title": f"Rows by {_tc(yname(c))}",
             "data": {"values": rows},
             "encoding": {"x": {"field": "group"}, "y": {"field": "rows"}}}) or None)

    return [p for p in panels if p]


def _is_text_col(s: pd.Series) -> bool:
    return s.dtype == object or pd.api.types.is_string_dtype(s)


@app.get("/api/dashboard")
def dashboard() -> dict:
    """What the data looks like: ranked insight cards + a deterministic chart grid."""
    with state_lock:
        tables = dict(_sess["tables"])
        active = _sess["active"]
    if not tables:
        return {"table": None, "rows": 0, "cards": [], "panels": [], "columns": []}
    import numpy as np

    from jane.insights import find_insights, insight_question, profile_table

    tname = active if active in tables else next(iter(tables))
    df = tables[tname]
    cards = []
    for ins in find_insights({tname: df}, max_per_table=8):
        cards.append({
            "kind": ins.kind,
            "title": ins.title,
            "detail": ins.detail,
            "table": ins.table,
            "score": round(float(ins.score), 3),
            "question": insight_question(ins),
        })
    try:
        profile = profile_table(tname, df)
    except Exception:
        profile = {"columns": [], "correlations": []}
    columns = []
    for col in profile.get("columns", []):
        columns.append({
            "name": str(col.get("name")),
            "dtype": str(col.get("dtype", "")),
            "nulls": int(col.get("nulls", 0) or 0),
            "missing_pct": float(col.get("null_pct", 0.0) or 0.0),
            "distinct": int(col.get("unique", 0) or 0),
            "min": col.get("min"),
            "max": col.get("max"),
            "median": col.get("median"),
            "outliers": int(col.get("outliers_iqr", 0) or 0),
        })
    return {
        "table": tname,
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "memory_mb": round(float(df.memory_usage(deep=True).sum()) / 1e6, 2),
        "cards": cards,
        "panels": _dashboard_panels(tname, df),
        "columns": columns,
        "correlations": profile.get("correlations", []),
    }



# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload(file: UploadFile = File(...), sheet: str | None = None) -> dict:
    raw = await file.read()
    limit = settings.max_upload_mb * 1024 * 1024
    if len(raw) > limit:
        raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB limit")
    return _load_bytes(raw, file.filename or "upload.csv", sheet=sheet)


@app.get("/api/samples")
def samples() -> dict:
    """The datasets that ship with Jane — the first screen is not a dead end."""
    return {"samples": [{"file": p.name, "name": p.stem.replace("_", " ").title()}
                        for p in sorted(SAMPLE_DIR.glob("*.csv"))]}


@app.post("/api/sample")
def load_sample(req: SampleRequest) -> dict:
    path = (SAMPLE_DIR / req.file).resolve()
    if path.parent != SAMPLE_DIR.resolve() or not path.is_file():
        raise HTTPException(404, f"No bundled sample named {req.file!r}")
    return _load_bytes(path.read_bytes(), path.name)


def _load_bytes(raw: bytes, filename: str, sheet: str | None = None) -> dict:
    """Parse bytes and install them as this session's tables.

    One path for a user upload and a bundled sample: the two can never drift in
    how tables, history or suggestions get set up.
    """
    try:
        loaded = load_uploaded(raw, filename, sheet=sheet)
    except Exception as e:
        raise HTTPException(400, f"Could not read file: {e}") from e
    if not loaded.sheets:
        raise HTTPException(400, "No readable sheets found")
    with state_lock:
        _sess["tables"] = {s.name: s.df for s in loaded.sheets}
        _sess["active"] = loaded.sheets[0].name
        _sess["file_name"] = filename
        _sess["history"] = []
        _sess["session_id"] = None
        active = _sess["active"]
    from jane.insights import suggest_questions
    return {
        "sheets": [s.name for s in loaded.sheets],
        "active": active,
        "tables": _tables_payload(),
        "suggestions": suggest_questions({s.name: s.df for s in loaded.sheets}, active),
    }


# ---------------------------------------------------------------------------
# Sheet switching (multi-sheet Excel: all sheets stay in memory)
# ---------------------------------------------------------------------------

@app.post("/api/sheet")
def pick_sheet(req: SheetRequest) -> dict:
    with state_lock:
        if req.name not in _sess["tables"]:
            raise HTTPException(404, f"No such sheet: {req.name}")
        # A table switch KEEPS the conversation — Streamlit's own table selector
        # (app.py:636) never cleared it, and only upload replaces the data outright.
        # Clearing here would destroy the chat the moment you stepped back from a
        # prep table to its original.
        _sess["active"] = req.name
        return {"active": _sess["active"], "tables": _tables_payload()}


# ---------------------------------------------------------------------------
# Prep — deterministic cleaning; every action writes a NEW table
# (spec: docs/superpowers/specs/2026-10-02-prep-panel-design.md §2)
# ---------------------------------------------------------------------------

# One source of truth for the keys the Prep panel may send. The client passes only a
# key: the op + its arguments are fixed here, next to the engine they feed, so there
# is nothing to validate or abuse. "tidy" runs the multi-op preset below instead.
_PREP_ACTIONS: dict[str, tuple[str, dict]] = {
    "fill_missing":   ("fill_missing", {}),
    "drop_missing":   ("fill_missing", {"strategy": "drop"}),
    "dedupe":         ("dedupe", {}),
    "clean_text":     ("clean_text", {}),
    "to_datetime":    ("to_datetime", {}),
    "scale_zscore":   ("scale", {"method": "zscore"}),
    "scale_minmax":   ("scale", {"method": "minmax"}),
    "scale_robust":   ("scale", {"method": "robust"}),
    "encode":         ("encode", {}),
    "clamp_outliers": ("clamp_outliers", {}),
    "tidy":           ("tidy", {}),   # clean_default: text → dates → fill → dedupe
}


@app.post("/api/prep")
def prep(req: PrepRequest) -> dict:
    """Run one deterministic cleaning action on the active table.

    Writes a NEW table (`{active}_prep[_N]`, or `{active}_tidy`) and switches to it;
    the original frame is never touched. A no-op is a success reported through the
    log — `PrepResult.skipped` is never populated by `run_ops`, so there is no
    `skipped` field to lie with. Nothing is written unless the run succeeds.
    """
    from jane.preprocess import PreprocessError, clean_default, run_ops

    if req.action not in _PREP_ACTIONS:
        raise HTTPException(400,
                            f"Unknown action '{req.action}'. "
                            f"Valid: {', '.join(_PREP_ACTIONS)}")
    with state_lock:
        active = _sess["active"]
        if not active or active not in _sess["tables"]:
            raise HTTPException(400, "No active table — load a file first")
        try:
            if req.action == "tidy":
                result = clean_default(_sess["tables"][active])
                base = f"{active}_tidy"
            else:
                op_name, args = _PREP_ACTIONS[req.action]
                result = run_ops(_sess["tables"][active], [(op_name, args)])
                base = f"{active}_prep"
        except PreprocessError as e:
            raise HTTPException(400, str(e)) from e
        except Exception as e:  # readable message, never a traceback in the body
            raise HTTPException(500, str(e)) from e

        # byte-for-byte the Streamlit naming rule: base, then _2, _3, …
        new_name, k = base, 2
        while new_name in _sess["tables"]:
            new_name, k = f"{base}_{k}", k + 1
        _sess["tables"][new_name] = result.df
        _sess["active"] = new_name
        return {
            "ok": True,
            "active": new_name,
            "tables": _tables_payload(),
            "new_table": {"name": new_name,
                          "rows": int(len(result.df)),
                          "cols": int(result.df.shape[1])},
            "log": list(result.log),
        }


# ---------------------------------------------------------------------------
# Ask — SSE stream of stage events + final result
# ---------------------------------------------------------------------------

def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _build_final(res: AnalysisResult) -> dict:
    """AnalysisResult -> the frontend's final payload (mirrors app.py rendering)."""
    figures: list[str] = list(res.figures)  # sandbox matplotlib, already data-URIs
    try:
        for png in render_all(res.charts):
            figures.append("data:image/png;base64," + base64.b64encode(png).decode())
    except Exception:
        pass
    code_lang = "python" if res.code and ("import" in res.code or "result_df" in res.code) else "sql"
    with state_lock:
        _sess["history"].append({
            "role": "assistant", "content": res.answer or res.error or "",
            "code": res.code, "tables": res.tables, "charts": res.charts,
            "ok": res.ok, "error": res.error,
        })
        if _sess["session_id"] is not None:
            _store.add_message(_sess["session_id"], "assistant", json.dumps({
                "content": res.answer or res.error or "",
                "code": res.code, "tables": _jsonable_rows(res.tables),
                "charts": res.charts, "ok": res.ok, "error": res.error,
            }, default=str))
        tables = [{"name": t["name"], "columns": t["columns"],
                   "rows": _jsonable_rows(t["rows"])} for t in res.tables]
    return {
        "ok": res.ok,
        "answer": res.answer or "",
        "error": res.error,
        "code": res.code,
        "language": code_lang,
        "stdout": res.stdout or "",
        "tables": tables,
        "charts": res.charts,
        "figures": figures,
        "attempts_used": res.attempts_used,
        "model_used": res.model_used,
        # the receipt is what the agent measured, not what we hope happened
        "guards": [asdict(g) for g in res.guards],
        "verified": res.verified,
        "repairs": res.repairs,
    }


@app.post("/api/ask")
def ask(req: AskRequest) -> StreamingResponse:
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Empty question")
    with state_lock:
        if not _sess["tables"]:
            raise HTTPException(409, "Upload a dataset first")
        ctx = _open_question(question)

    def generate() -> Iterator[str]:
        events: queue.Queue = queue.Queue()

        def progress(step: str, detail: str = "") -> None:
            events.put(("stage", {"step": step, "detail": detail}))

        def worker() -> None:
            try:
                with state_lock:  # one analysis at a time
                    _cancel.clear()          # this run owns the flag
                    try:
                        res = _agent_with_choice().analyze(
                            question, ctx, progress=progress, cancel=_cancel)
                    finally:
                        _cancel.clear()
                events.put(("final", _build_final(res)))
            except AnalysisCancelled:
                events.put(("error", {"message": "Stopped."}))
            except OllamaError as e:
                events.put(("error", {"message": f"Model error: {e}"}))
            except Exception as e:  # noqa: BLE001 — the bridge reports, never crashes
                events.put(("error", {"message": f"Unexpected error: {e}"}))
            finally:
                events.put(None)

        threading.Thread(target=worker, daemon=True).start()
        yield _sse("stage", {"step": "queued", "detail": question})
        while True:
            item = events.get()
            if item is None:
                break
            yield _sse(item[0], item[1])

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/api/cancel")
def cancel() -> dict:
    """Stop the analysis in flight.

    Only one runs at a time (state_lock), so one flag is enough. The agent loop
    checks it between attempts and after each model call, so the worst case is
    the current model call finishing — the lock is then released and a queued
    ask proceeds. Nothing is shipped or stored for a stopped run.
    """
    _cancel.set()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Brief — one ask, a whole analyst read: the three sharpest questions, each
# answered and verified on its own, streamed one section at a time.
# ---------------------------------------------------------------------------

def _brief_questions() -> list[str]:
    """The questions worth a full memo on this table (the pill list is already
    ranked by the deterministic insight sweep)."""
    with state_lock:
        if not _sess["tables"]:
            return []
        from jane.insights import suggest_questions
        return suggest_questions(_sess["tables"], _sess["active"])[:3]


@app.post("/api/brief")
def brief(req: BriefRequest) -> StreamingResponse:
    with state_lock:
        if not _sess["tables"]:
            raise HTTPException(409, "Upload a dataset first")
        questions = [q.strip() for q in (req.questions or _brief_questions())
                     if q and q.strip()][:4]
    if not questions:
        raise HTTPException(400, "Nothing readable in this table yet")

    def generate() -> Iterator[str]:
        yield _sse("brief", {"total": len(questions), "questions": questions})
        fatal = False
        for i, q in enumerate(questions, 1):
            events: queue.Queue = queue.Queue()

            def progress(step: str, detail: str = "", index: int = i) -> None:
                events.put(("stage", {"step": step, "detail": detail, "index": index}))

            def worker(question: str = q, index: int = i) -> None:
                try:
                    ctx = _open_question(question)
                    with state_lock:  # one analysis at a time
                        _cancel.clear()          # this section owns the flag
                        try:
                            res = _agent_with_choice().analyze(
                                question, ctx, progress=progress, cancel=_cancel)
                        finally:
                            _cancel.clear()
                    events.put(("section", {"index": index, "question": question,
                                            "final": _build_final(res)}))
                except AnalysisCancelled:
                    events.put(("error", {"message": "Stopped.",
                                          "index": index, "cancelled": True}))
                except OllamaError as e:
                    events.put(("error", {"message": f"Model error: {e}",
                                          "index": index, "fatal": True}))
                except Exception as e:  # noqa: BLE001 — the bridge reports, never crashes
                    events.put(("error", {"message": f"Unexpected error: {e}",
                                          "index": index}))
                finally:
                    events.put(None)

            threading.Thread(target=worker, daemon=True).start()
            yield _sse("stage", {"step": "queued", "detail": q, "index": i})
            while True:
                item = events.get()
                if item is None:
                    break
                if item[0] == "error" and item[1].get("fatal"):
                    fatal = True
                if item[0] == "error" and item[1].get("cancelled"):
                    fatal = True          # a stopped brief stops its remaining sections
                yield _sse(item[0], item[1])
            if fatal:
                break
        yield _sse("done", {"total": len(questions)})

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------
# Fixer — run edited SQL/Python from the chat view
# ---------------------------------------------------------------------------

@app.post("/api/fixer")
def fixer(req: FixerRequest) -> dict:
    with state_lock:
        if not _sess["tables"]:
            raise HTTPException(409, "Upload a dataset first")
        tables = dict(_sess["tables"])
        active = _sess["active"]
    if req.language == "sql":
        from jane.engine import SQLEngine
        eng = SQLEngine(tables)
        try:
            r = eng.execute(req.code)
            return {"ok": r.ok, "error": r.error,
                    "tables": [{"name": "query_result", "columns": r.columns,
                                "rows": _jsonable_rows(r.rows)}] if r.ok else [],
                    "truncated": r.truncated, "figures": [], "stdout": ""}
        finally:
            eng.close()
    from jane.sandbox import run_code
    r = run_code(req.code, tables, active, timeout=90)
    return {"ok": r.ok, "error": r.error or "; ".join(r.safety_errors),
            "tables": [{"name": t["name"], "columns": t["columns"],
                        "rows": _jsonable_rows(t["rows"])} for t in r.tables],
            "truncated": False, "figures": r.figures, "stdout": r.stdout}


# ---------------------------------------------------------------------------
# Static hosting: serve the built React app so the DEFAULT launch is one
# process (start_jane.bat), the way Streamlit used to be.
#
# Registered LAST, on purpose — Starlette matches routes in registration order,
# so every /api/* route above is matched before this catch-all mount.
# ---------------------------------------------------------------------------

_DIST = Path(__file__).parent / "frontend" / "dist"

if (_DIST / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="frontend")
else:
    @app.get("/", include_in_schema=False)
    def _frontend_not_built() -> dict:
        raise HTTPException(
            503,
            "The React frontend is not built. Run: "
            "cd frontend && npm install && npm run build",
        )
