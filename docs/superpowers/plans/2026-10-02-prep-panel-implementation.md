# Implementation Plan — Data Workspace: Prep Panel

Spec: `docs/superpowers/specs/2026-10-02-prep-panel-design.md` (approved 2026-10-02).
The writing-plans skill is not available in this environment; this plan follows the
precedent of `2026-10-01-react-frontend-implementation.md` — written and executed
directly by the same session, with the spec approval as the gate to proceed.

## M1 — Endpoint + tests (TDD)

1. `pyproject.toml`: `dev = ["pytest>=8.0", "httpx>=0.27"]`, then
   `.venv/Scripts/python.exe -m pip install -e ".[dev]"`.
2. `tests/test_prep_api.py` — written first, red until step 3. The file must set
   `settings.history_db` to a temp path **before** `import server`, because `server.py`
   opens `HistoryStore(settings.history_db)` at module import and must not touch the
   real `data/` history DB. A fixture resets `_sess` (tables/active/history/session_id)
   between tests so tables cannot leak from case to case.
   - each of the 11 keys creates a new table and sets it active
   - the original frame is untouched afterwards (shape + values)
   - name collision yields `retail_sales_prep_2`
   - `log` is returned; a no-op logs "nothing to fill" and still returns `ok: true`
     (`PrepResult.skipped` is never populated by `run_ops`, so the response has no
     `applied`/`skipped` fields — the log is the contract)
   - unknown action → 400 naming the valid keys; no active table → 400
   - `PreprocessError` → 400 carrying its message
   - a failure writes nothing (tables dict unchanged)
3. `server.py`: `PrepRequest(BaseModel)`, the `_PREP_ACTIONS` table (spec §1, 11 keys),
   and `POST /api/prep` per spec §2 — one `state_lock` acquisition (it is an `RLock`,
   and the work is pandas-bound), the `_prep` / `_prep_2` naming loop, the `_sess` write,
   `_tables_payload()` in the response. Errors per spec §2.
4. Full `pytest` green (257 existing + new).

## M2 — Targeted fixes

5. `/api/sheet`: drop `_sess["history"] = []` and `_sess["session_id"] = None` — only
   `/api/upload` resets. Add the test: switching tables keeps the conversation, upload
   still clears it.
6. Switcher over **all** tables: `ChatView` renders `state.tables` instead of the
   `sheets` prop, and the checked state keys off `state.active`; drop the now-redundant
   `activeSheet` (and `sheets`) props through `App → ChatView`. `onPickSheet` still
   posts `/api/sheet`. Remove any state this leaves unused in `App.tsx`.
7. Gates: `pytest` + `tsc -b`.

## M3 — The panel

8. `frontend/src/lib/types.ts`: `PrepAction` (union of the 11 keys) + `PrepResult`.
9. `frontend/src/lib/api.ts`: `prepAction(action)`.
10. `frontend/src/App.tsx`: `handlePrep` alongside `handleFile`/`handlePickSheet` —
    call the endpoint, merge the response's `active` + `tables` into the existing
    `BridgeState` (pills refresh via ChatView's existing `state.active` effect).
11. `frontend/src/components/jane/PrepView.tsx`: book-rail header, empty state, the ten
    buttons (3/2/1 grid) + full-width tidy preset, disabled while in flight, success
    strip with the collapsible prep log, inline error panel.
    `DataView.tsx`: renders PrepView, owns the (future) panel selection.
    `ChatView.tsx`: tab union gains `'data'`, third pill, composer and pills hidden for
    it exactly as for the dashboard.
12. Gates: `tsc -b`, oxlint 0 warnings, `npm run build`.

## M4 — Verify + ship

13. Live click-through in the preview panel: upload `retail_sales.csv` → Data → tidy →
    header chip, table switcher and pills all follow the new table → switch back to the
    original → conversation still present → ask about the cleaned table. Repeat with a
    no-op case (titanic `to_datetime` on an already-clean column) to see the
    "nothing to fill"-style log line.
14. Full `pytest`; append the session entry to `PROJECT_LOG.md`; commit and push.

## Standing rules during implementation

- `src/jane/` is untouched — `preprocess.py` is the engine and is already tested by
  `tests/test_preprocess.py`.
- Tests stay green at every milestone (257 baseline).
- Frontend files: components in `frontend/src/components/{ui,jane}/`, lib in
  `frontend/src/lib/`; every module keeps a one-line docstring comment per repo style.
- Identity unchanged: poppy `#FF6B2C` for action, verdigris `#58C08B` for verified only.
- UI verification runs through the app's own preview panel (`agent-browser open` times
  out in this environment); restart the bridge with the documented recipe on :8100.
