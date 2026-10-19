# Jane Data Workspace: Prep Panel — Design Spec

Date: 2026-10-02
Status: Approved design (sections 1–2 approved in chat); implementation plan to follow via writing-plans.

## 0. Context and goals

The React frontend (`frontend/` + `server.py`) currently offers Chat and Dashboard.
The Streamlit UI (`src/jane/app.py`, five tabs: Chat, Manual analysis, Profile, Prep,
Schema) still has capabilities the React app never received. This spec restores the
first of them: **Prep** — deterministic data cleaning, no model involved.

User decisions made during brainstorming:

- **Classified architectural** (a new subsystem: several views, new bridge endpoints,
  and a second writer to the shared active-table state), so it is decomposed into three
  sub-projects: **Prep** (this spec), Manual SQL, and Profile/Schema — each specced and
  planned separately.
- **Structure:** a permanent top-level `Chat | Dashboard | Data` nav. `Data` is created
  by this spec and hosts Prep first; Manual/Profile/Schema later arrive as sub-panels
  without touching top-level nav again.
- **Depth: one-click parity.** Same ten fixed action buttons + the tidy preset the
  Streamlit tab had, whole-table, fixed defaults. No column pickers, no pipeline
  builder, no parameters from the client.

Constraints:

- `src/jane/preprocess.py` is the engine and is **not modified** — it is already tested
  (`tests/test_preprocess.py`) and its contract says errors raise, no-ops report.
- Identity unchanged: black `#0B0C0E` ink, panel `#15161A`, poppy `#FF6B2C` action,
  verdigris `#58C08B` verified only; Archivo UI, IBM Plex Mono for machine text.
- Everything stays local and in-memory, as today: prep tables live in `_sess["tables"]`
  and disappear on bridge restart or new upload (no persistence layer).

Out of scope: column-level parameters, preview/diff before applying, undo or delete-a-table,
persisting tables to disk, Profile/Schema/Manual panels (separate specs), and any model
involvement in prep.

## 1. Scope and structure

```
frontend/src/components/jane/
├── ChatView.tsx        tab union gains 'data'; third pill "Data"
├── DashboardView.tsx   untouched
├── DataView.tsx        NEW — root of the Data tab; renders the prep panel.
│                       When Manual/Profile/Schema land, a sub-pill row is added
│                       HERE only; the top-level nav never changes again.
└── PrepView.tsx        NEW — the prep panel (header, button grid, result, log, errors)

server.py              + _PREP_ACTIONS table, + POST /api/prep,
                       + targeted fix to /api/sheet (see §4)
frontend/src/lib/api.ts     + prepAction()
frontend/src/lib/types.ts   + PrepResult / PrepAction types
```

The action keys are declared **once**, in `server.py`, mapping to the exact
`(op, args)` pairs the Streamlit tab used:

| key | maps to |
|---|---|
| `fill_missing` | `("fill_missing", {})` |
| `drop_missing` | `("fill_missing", {"strategy": "drop"})` |
| `dedupe` | `("dedupe", {})` |
| `clean_text` | `("clean_text", {})` |
| `to_datetime` | `("to_datetime", {})` |
| `scale_zscore` | `("scale", {"method": "zscore"})` |
| `scale_minmax` | `("scale", {"method": "minmax"})` |
| `scale_robust` | `("scale", {"method": "robust"})` |
| `encode` | `("encode", {})` |
| `clamp_outliers` | `("clamp_outliers", {})` |
| `tidy` | `clean_default(df)` — the preset: clean text → dates → fill missing → dedupe |

The client sends only a key. No client-supplied arguments, so there is nothing to
validate or abuse, and button labels stay with the UI.

## 2. Bridge contract

```
POST /api/prep            body: {"action": "fill_missing"}
```

Flow, under one `state_lock` acquisition (the work is pandas-bound and takes seconds;
`/api/ask` only holds the lock for bookkeeping, so no deadlock, and holding it prevents
a concurrent ask from reading a half-written table):

1. Read `_sess["active"]`; if there is none → 400 `No active table — load a file first`.
2. Look the key up in `_PREP_ACTIONS`; unknown → 400 naming the valid keys.
3. Run `run_ops(df, [(op, args)])`, or `clean_default(df)` for `tidy`. These return a
   **new** frame and raise before anything is written, so a failure leaves `_sess`
   untouched — there is no partial table to clean up.
4. Name the result: base is `f"{active}_prep"` (`f"{active}_tidy"` for the preset), then
   `_2`, `_3`, … while it collides — byte-for-byte the Streamlit rule.
5. Insert into `_sess["tables"]`, set `_sess["active"]` to the new name.
6. Return.

```jsonc
{
  "ok": true,
  "active": "retail_sales_prep",       // BridgeState.active after the change
  "tables": [ /* _tables_payload() — same as /api/state */ ],
  "new_table": { "name": "retail_sales_prep", "rows": 500, "cols": 6 },
  "log": ["clean_text: normalized whitespace in 500 cell(s)"]
}
```

A single action yields one `log` line; `tidy` runs four ops, so its `log` holds the
chain. **No `applied` / `skipped` fields.** The engine defines `PrepResult.skipped`, but
`run_ops` never populates it (every op it executes goes to `applied`, change or not), so
an empty `skipped` would be a lie waiting to be believed. The `log` is the source of
truth: a no-op is a **success** whose line says so — `fill_missing: nothing to fill —
no missing values found`, `to_datetime: no date-like columns found` — never an error.

Errors: `PreprocessError` → 400 with its message; unknown action → 400; no active table
→ 400; unexpected exception → 500 with `str(e)` (never a traceback in the body).

**The conversation is preserved.** Upload replaces the data, so it resets history;
prep refines the same working set, so it does not. The next question's context reads
`_sess["active"]` live, so it simply starts using the cleaned table.

## 3. Frontend

**Nav.** `ChatView`'s tab union becomes `'chat' | 'dashboard' | 'data'`; a third pill
renders beside Dashboard using the existing pill styling. The pinned composer and
suggestion pills stay chat-only (they already hide for the dashboard).

**`DataView.tsx`** renders `PrepView` today; it owns no state beyond the (future) panel
selection. **`PrepView.tsx`** renders:

- **Header (book-rail):** `PREP — retail_sales` left; `500 rows × 6 cols · original
  untouched` right, from `state.tables` / `state.active`.
- **Empty state:** when `!state.has_data`, the dashboard's "Load a file first" style.
- **Button grid:** the ten labels exactly as Streamlit wrote them — `Fill missing
  (median / Unknown / ffill)`, `Drop rows with missing`, `Drop duplicate rows`,
  `Clean whitespace (cells + headers)`, `Convert date-like text to datetime`,
  `Scale numeric (z-score)`, `Scale numeric (min-max)`, `Scale numeric (robust)`,
  `One-hot encode low-cardinality text`, `Clamp outliers (IQR fence)` — responsive 3/2/1
  columns, existing `Button` primitive in the subtle variant, mono labels. The key each
  button sends is the one from §1's table.
- **Tidy preset:** full-width poppy primary button —
  `✨ Tidy preset: clean text → dates → fill missing → dedupe`.
- **In-flight:** every button disables while the request runs (same convention as the
  model picker during streaming).
- **Success strip:** `Saved as retail_sales_prep · 500 rows × 6 cols`, plus `log` as a
  collapsible mono list. A no-op is *success* — its line reads "nothing to fill" — never
  an error.
- **Error:** inline panel beneath the grid with the server message; buttons re-enable;
  no state was touched.

**State flow.** `App.tsx` gains `handlePrep`, alongside `handleFile`/`handlePickSheet`:
call `prepAction(key)`, then merge the response's `active` and `tables` into the existing
`BridgeState` (the response carries everything that changes; `model`/`file_name` do not).
Because `ChatView` already refetches suggestions whenever `state.active` changes
(`useEffect([... state.has_data, state.active])`), the pills refresh for free.

## 4. Targeted fixes this feature requires

1. **The switcher must list every table, not just uploaded sheets.** It currently renders
   `sheets` (set only by upload), so after a prep run you would be stranded on
   `retail_sales_prep` with no way back to the original. Render `state.tables` instead.
2. **The checked state must key off `state.active`**, dropping the separate `activeSheet`
   prop — prep changes `state.active` without touching that prop, which would otherwise
   show a stale sheet as active.
3. **`/api/sheet` stops resetting history and `session_id`; only upload resets.** The
   bridge over-applied Streamlit's *file-load* reset to table switching: Streamlit's own
   table selector (`app.py:636`) switches tables and keeps the conversation. Without this,
   switching back to the original table after a prep silently destroys the chat — which
   would defeat the history-preservation decision in §2.

## 5. Testing and rollout

- **New dev dependency: `httpx`** (installed into `.venv`), which is what
  `starlette.testclient.TestClient` requires. It enables the **first server-side test
  file in this repo** — `server.py` currently has zero coverage despite carrying the
  ask/brief/model/fixer endpoints — and is reusable for those later.
- **`tests/test_prep_api.py` (new):** each of the 11 keys creates a new table and sets it
  active; the original frame is unchanged afterwards (shape + values); a name collision
  yields `_prep_2`; `log` comes back and a no-op logs "nothing to fill" while still
  returning `ok: true`; unknown action → 400; no active
  table → 400; a failure writes nothing;
  `GET /api/suggestions` then reflects the new table (which is what refreshes the pills);
  history survives prep and table switches but is cleared by upload.
- `tests/test_preprocess.py` is untouched — the engine is already covered.
- **Gates:** full `pytest` (257 existing + new), `tsc -b`, oxlint 0 warnings,
  `npm run build`, and a live click-through in the preview: upload → Data → run tidy →
  header chip and pills update → switch back to the original → conversation still there →
  ask about the cleaned table.

## 6. Milestones

Implementation proceeds via writing-plans after this spec is approved:

1. `httpx` install + `POST /api/prep` with `_PREP_ACTIONS`, TDD against `tests/test_prep_api.py`.
2. Targeted fixes: `/api/sheet` history behavior, switcher over `state.tables`, `state.active` keying.
3. `PrepView` + `DataView` + the third pill + `api.ts`/`types.ts`.
4. Gates + live verification + `PROJECT_LOG.md` entry.
