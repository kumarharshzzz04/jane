"""API tests for the bridge's prep endpoint (POST /api/prep).

Spec: docs/superpowers/specs/2026-10-02-prep-panel-design.md

server.py opens HistoryStore(settings.history_db) AT MODULE IMPORT, so the setting is
redirected to a throwaway directory before `import server` and restored immediately
after — the suite must never touch the real data/ history db. server._sess is a module
global, so every test resets it.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from starlette.testclient import TestClient

from jane.config import settings
from jane.preprocess import PreprocessError

# --- import the bridge without touching the real history db -------------------
_tmp_history = Path(tempfile.mkdtemp(prefix="jane_test_history_"))
_real_history_db = settings.history_db
settings.history_db = str(_tmp_history / "history.db")

import server  # noqa: E402  (must follow the settings redirect)

settings.history_db = _real_history_db

client = TestClient(server.app)

ACTIONS = [
    "fill_missing", "drop_missing", "dedupe", "clean_text", "to_datetime",
    "scale_zscore", "scale_minmax", "scale_robust", "encode", "clamp_outliers",
]


def _messy_df() -> pd.DataFrame:
    """Every op has something real to do here."""
    return pd.DataFrame({
        "date": ["2025-06-01", "2025-06-02", "2025-06-01", "2025-06-03"],
        "region": [" east ", "east", "WEST", "west"],
        "product": ["tea", "coffee", "tea", "pastry"],
        "units": [3.0, None, 3.0, 9.0],
        "revenue": [12.5, 40.0, 12.5, None],
    })


def _load(df: pd.DataFrame | None = None, name: str = "sales") -> pd.DataFrame:
    df = _messy_df() if df is None else df
    with server.state_lock:
        server._sess["tables"][name] = df
        server._sess["active"] = name
    return df


@pytest.fixture(autouse=True)
def clean_session():
    """server._sess is a module global — no table may leak between tests."""
    with server.state_lock:
        server._sess["tables"].clear()
        server._sess["active"] = None
        server._sess["history"].clear()
        server._sess["session_id"] = None
        server._sess["file_name"] = None
    yield
    with server.state_lock:
        server._sess["tables"].clear()
        server._sess["active"] = None
        server._sess["history"].clear()
        server._sess["session_id"] = None
        server._sess["file_name"] = None


# --------------------------------------------------------------------------- #
# Happy path: a new table, switched active, original untouched
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("action", ACTIONS)
def test_each_action_creates_a_new_table_and_switches(action: str):
    original = _load()
    r = client.post("/api/prep", json={"action": action})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["active"] == "sales_prep"
    assert body["new_table"]["name"] == "sales_prep"
    # rows/cols describe the frame that was actually stored — drop ops shrink it,
    # encode expands it, the rest keep the shape.
    stored = server._sess["tables"]["sales_prep"]
    assert body["new_table"]["rows"] == len(stored)
    assert body["new_table"]["cols"] == int(stored.shape[1])
    assert len(stored) <= len(original)
    assert "sales_prep" in server._sess["tables"]
    # the original is kept and byte-identical — every action writes a NEW table
    assert "sales" in server._sess["tables"]
    assert server._sess["tables"]["sales"].equals(original)
    assert server._sess["active"] == "sales_prep"
    assert isinstance(body["log"], list) and body["log"], body


def test_drop_missing_actually_drops_rows():
    """The one action allowed to shrink the frame: 2 of 4 rows carry a NaN."""
    _load()
    r = client.post("/api/prep", json={"action": "drop_missing"})
    assert r.status_code == 200, r.text
    assert r.json()["new_table"]["rows"] == 2


def test_tidy_preset_uses_tidy_suffix_and_logs_the_chain():
    _load()
    r = client.post("/api/prep", json={"action": "tidy"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["active"] == "sales_tidy"
    # clean_text -> to_datetime -> fill_missing -> dedupe
    assert len(body["log"]) == 4, body["log"]


def test_name_collision_increments_the_suffix():
    _load()
    server._sess["tables"]["sales_prep"] = pd.DataFrame({"a": [9]})
    r = client.post("/api/prep", json={"action": "clean_text"})
    assert r.status_code == 200, r.text
    assert r.json()["active"] == "sales_prep_2"


def test_noop_is_a_success_whose_log_says_so():
    """No-ops never fail: run_ops reports them in the log (there is no `skipped`
    field to populate — that PrepResult attribute is never written)."""
    _load(pd.DataFrame({"region": ["east", "west"], "units": [1.0, 2.0]}))
    r = client.post("/api/prep", json={"action": "fill_missing"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert "nothing to fill" in body["log"][0], body["log"]


def test_state_and_suggestions_follow_the_new_active_table():
    _load()
    assert client.post("/api/prep", json={"action": "dedupe"}).status_code == 200
    state = client.get("/api/state").json()
    assert state["active"] == "sales_prep"
    assert {t["name"] for t in state["tables"]} == {"sales", "sales_prep"}
    assert isinstance(client.get("/api/suggestions").json()["questions"], list)


def test_prep_keeps_the_conversation():
    """Prep refines the same working set: only upload resets the chat."""
    _load()
    server._sess["history"].append({"role": "user", "content": "how many rows?"})
    assert client.post("/api/prep", json={"action": "dedupe"}).status_code == 200
    assert len(server._sess["history"]) == 1


# --------------------------------------------------------------------------- #
# Failures: 4xx with a readable message, and NOTHING written
# --------------------------------------------------------------------------- #

def test_unknown_action_is_400_and_names_valid_keys():
    _load()
    r = client.post("/api/prep", json={"action": "drop_table"})
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "drop_table" in detail and "fill_missing" in detail
    assert "sales_prep" not in server._sess["tables"]
    assert server._sess["active"] == "sales"


def test_no_active_table_is_400():
    r = client.post("/api/prep", json={"action": "dedupe"})
    assert r.status_code == 400
    assert "load a file" in r.json()["detail"].lower()


def test_engine_failure_writes_nothing(monkeypatch):
    _load()
    server._sess["history"].append({"role": "user", "content": "first?"})

    def boom(*_a, **_k):
        raise PreprocessError("dedupe: column 'nope' not in table")

    import jane.preprocess as preprocess
    monkeypatch.setattr(preprocess, "run_ops", boom)

    r = client.post("/api/prep", json={"action": "dedupe"})
    assert r.status_code == 400
    assert "nope" in r.json()["detail"]
    # no partial table, no active switch, history untouched
    assert set(server._sess["tables"]) == {"sales"}
    assert server._sess["active"] == "sales"
    assert len(server._sess["history"]) == 1


def test_missing_action_field_is_a_validation_error():
    _load()
    r = client.post("/api/prep", json={})
    assert r.status_code == 422


# --------------------------------------------------------------------------- #
# M2 — history semantics: a table switch keeps it, only upload resets it
# --------------------------------------------------------------------------- #

def test_table_switch_keeps_the_conversation():
    """Regression for the targeted fix: /api/sheet used to clear history and
    session_id, which would destroy the chat when stepping back from a prep
    table to its original."""
    _load()
    _load(pd.DataFrame({"a": [1, 2]}), name="sales_tidy")
    server._sess["history"].append({"role": "user", "content": "who leads?"})
    server._sess["session_id"] = 42

    r = client.post("/api/sheet", json={"name": "sales"})
    assert r.status_code == 200, r.text
    assert r.json()["active"] == "sales"
    assert server._sess["active"] == "sales"
    assert len(server._sess["history"]) == 1
    assert server._sess["session_id"] == 42


def test_switch_to_unknown_table_is_404():
    _load()
    r = client.post("/api/sheet", json={"name": "no_such_table"})
    assert r.status_code == 404


def test_upload_still_resets_history_and_session():
    _load()
    server._sess["history"].append({"role": "user", "content": "old question"})
    server._sess["session_id"] = 42
    csv = b"date,units\n2025-06-01,3\n2025-06-02,4\n"
    r = client.post("/api/upload", files={"file": ("tiny.csv", csv, "text/csv")})
    assert r.status_code == 200, r.text
    assert r.json()["active"] == server._sess["active"]
    assert server._sess["history"] == []
    assert server._sess["session_id"] is None


# ---- POST /api/cancel: stop the run in flight ------------------------------

def test_cancel_endpoint_sets_the_flag_and_is_idempotent():
    """Stop is a flag the agent loop reads between attempts — it must answer ok
    even when nothing is running (the user can hit Stop twice, or after the run
    already finished)."""
    assert server._cancel.is_set() is False
    r = client.post("/api/cancel")
    assert r.status_code == 200, r.text
    assert r.json() == {"ok": True}
    assert server._cancel.is_set() is True
    assert client.post("/api/cancel").json() == {"ok": True}
    server._cancel.clear()          # leave the module as the tests found it
