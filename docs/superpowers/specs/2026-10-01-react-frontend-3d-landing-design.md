# Jane React Frontend + FastAPI Bridge — Design Spec

Date: 2026-10-01
Status: Approved design (sections 1–3 approved in chat); implementation plan to follow via writing-plans.

## 0. Context and goals

Jane is a local-first AI data analyst (Ollama + DuckDB, schema injection, sandboxed
execution) whose UI is Streamlit (`src/jane/app.py`, ~1,237 lines). The user wants the
frontend rebuilt as a modern React app with an interactive 3D "machine" landing page,
inspired by the Agentic Factory 3D component (a three.js machine with modes, cameras,
hover/click stations). The landing carries a "Start asking" button that transitions into
a chat view where the analysis happens.

Constraints agreed with the user:

- Keep Jane's visual identity: canopy ink `#0C1512`, bone `#EDF4EE`, signal orange
  `#FF6B2C` (action), verdigris `#58C08B` (verified), ember `#FFB35C` (warnings);
  Fraunces display / Archivo UI / IBM Plex Mono machine voice; dot-grid backdrop.
- Monorepo: new `frontend/` folder + `server.py` at repo root.
- FastAPI bridge importing the existing `src/jane` package — **zero changes inside
  `src/jane/`** (app.py stays; Streamlit remains usable until parity).
- Full feature parity is the goal; the first milestone is the hero flow (landing →
  upload → ask → answer with code artifact, receipt, pills), then the remaining tabs.
- Machine is a custom three.js scene (not a retheme of the provided video-factory
  component): a "data factory" — the CSV packet rides a conveyor through stations.

Out of scope for this spec: model fine-tuning, cloud anything, multi-user auth.

## 1. Architecture

```
jane/  (repo root)
├── src/jane/          existing package — untouched
├── server.py          NEW FastAPI bridge (~400 lines, one file)
└── frontend/          NEW Vite + React 19 + TypeScript + Tailwind v4 + shadcn
    ├── src/components/ui/        shadcn primitives + data-factory-3d.tsx
    ├── src/components/jane/      app-specific components (Landing, Chat, …)
    ├── src/lib/api.ts            typed fetch + SSE client
    └── src/lib/types.ts          mirrors bridge contracts
```

Single-user local app → the bridge holds one in-process session (tables dict, active
table, agent instance). All state lives in `src/jane` modules as today; the bridge is a
thin translator. Chat history persists via existing `history.py` (SQLite).

Dev: `uvicorn server:app --port 8000` + `vite` dev server proxying `/api` → :8000.
Prod/launcher: `start_jane_react.bat` runs both windows (bridge + frontend); the old
`start_jane.bat` (Streamlit) keeps working until parity, then is retired.

Dependencies to add to the venv: `fastapi`, `uvicorn`, `python-multipart`.
Node ≥ 20 already present.

## 2. FastAPI bridge contract

CORS for localhost:5173/5174. All responses JSON; errors are `{"detail": "..."}` with
proper status codes. Upload limit 200 MB enforced by config (`limits.max_upload_mb`).

- `GET /api/health` → `{ollama: bool, model: str|null, version: str}`
- `GET /api/state` → `{tables: [{name, rows, cols, columns}], active, model, samples: [...]}`
- `POST /api/upload` (multipart) → ingest via `load_uploaded` / `load_file`; multi-sheet
  Excel returns sheet choices; sets active table → `{tables, active, suggestions}`.
- `POST /api/ask` body `{question, history?}` → **SSE stream**:
  - `event: stage` data `{step, detail, attempt?}` — wired to the orchestrator's
    existing `progress` callback (schema loaded → SQL attempt 2/5 → verifying guards → …)
  - `event: final` data `{ok, answer, code, language, columns, tables, charts,
    figures (PNG data-URIs), attempts_used, model_used, error?, guards: string[]}`
    — `charts` are raw Vega-Lite specs; the bridge ALSO calls `jane.charts.render_all()`
    exactly as app.py does and returns the PNG data-URIs in `figures` (rendering stays
    server-side, the frontend just shows `<img>`). `guards` is the same static manifest
    app.py uses — `["truth", "contradiction", "winner", "direction", "profit",
    "denominator"]`, shown as 6/6 on success — honest today, replaceable later if the
    orchestrator ever reports per-guard outcomes.
  - `event: error` data `{message}` for fatal failures (Ollama down, no table loaded)
- `POST /api/fixer` body `{code, language}` → runs edited SQL via `SQLEngine` or Python
  via the sandbox → same shape as `final` (fresh result, not streamed).
- `GET /api/suggestions` → `suggest_questions()` output for the active table.
- Parity endpoints (built after hero flow): `/api/manual` (fetch table grid, apply
  edits, ad-hoc SQL), `/api/prep` (ops registry + tidy), `/api/schema`,
  `/api/profile` (insights), `/api/sessions` (history CRUD), `/api/model`
  (list/pull/switch with pull progress as SSE or polling).

The `progress(step, detail)` callback signature already exists in the orchestrator —
the bridge wraps it to enqueue SSE events; the agent runs in a worker thread.

## 3. Frontend: landing and the data-factory machine

`components/ui/data-factory-3d.tsx` — custom three.js scene, fully procedural (no
models, no images; canvas-texture screens as in the reference component). Props mirror
the reference API: `height`, `className`, `embed`, `onStation`, `onReady`, plus
`window.__machine` (setMode/focusStation/setCamera/play/pause) and `__machineDebug`.

Scene: dark ink plate + conveyor loop; a CSV packet (bone card with mono type) travels
the loop through five stations: **Ingest → Schema → Agent → Verify → Chart**.

- Ingest: hopper/roller accepting the packet.
- Schema: board with a printed column grid (canvas texture: real column-name rows).
- Agent: printhead writing SQL (mono text on screen; glowing amber).
- Verify: station with the receipt printing `VERIFIED ✓ guards 6/6` in verdigris.
- Chart: screen painting an animated bar/line chart (canvas texture).
- Palette: ink `#0C1512` surfaces, bone `#EDF4EE` type, orange `#FF6B2C` actions/energy,
  verdigris `#58C08B` verification accents. Fonts painted on textures: Plex Mono.
- Modes: Assembled / Cutaway (shell clipping at deck height) / Stations (spread) /
  One question (single packet, journey HUD). Cameras: Overview / Side / Top / Station /
  Flight. Hover tooltip + click focuses station; labels are DOM overlays projected
  per-frame, clamped to viewport (reference implementation pattern).
- Loading overlay + WebGL-unavailable error card with retry; context-loss recovery;
  `prefers-reduced-motion` pauses autoplay; pixel-ratio step-down on sustained low FPS;
  full disposal on unmount (geometries, materials, textures, renderer context).

Landing composition (`components/jane/Landing.tsx`): machine fills the right ~58% on
wide frames (`embed` mode with view-offset like the reference); left column carries the
Fraunces headline **"Answers you can audit."**, the six-guard list (live from a static
manifest initially), sample chips, and the **Start asking** button (orange CTA). Dot-grid
backdrop, notebook rail header ("Jane — local data analyst" / file formats right),
"made by Kumar Harsh" credit. Transition to chat: landing blurs/scales out, chat fades in.

## 4. Frontend: chat and panels

`App.tsx` state machine: `landing → chat`, keyed by active table.

Chat view (hero milestone):
- Message list: user questions as quiet right-aligned bubbles; Jane answers open-layout.
  Answer blocks: prose answer, SQL/pandas code artifact (mono, copyable), result table,
  figures as `<img src=data:URI>` (server-rendered PNGs via `jane.charts.render_all`),
  verification receipt strip (model · tries · guards 6/6), stage indicator while
  streaming (from SSE `stage`), stdout expander when present.
- Suggestion pills under the composer (from `/api/suggestions`).
- In-chat fixer: on `ok:false`, an editable code box + Run → `/api/fixer`, result
  rendered in place.
- Composer pinned to viewport bottom; Enter to send; Shift+Enter newline.
- Header: table name + row count, model pill, Ollama health dot, "New data" (upload).
- Sidebar/panel: table switcher (radio cards), sessions later.

Parity panels (after hero flow, same shell): Manual analysis (editable grid + ad-hoc
SQL), Prep (op buttons + tidy + log), Profile (insights), Schema (what the model sees).
Grid component choice deferred to the implementation plan (candidate: TanStack Table).

`lib/api.ts`: typed client; SSE reader over fetch `ReadableStream` (events: stage/
final/error); `VITE_USE_MOCK=1` mock mode for offline UI work (mirrors the HadeS
pattern; scripted answers with fake figures).

## 5. Errors, testing, rollout

- Errors: SSE `error` events → retryable banner; Ollama down → banner with
  `ollama pull qwen2.5-coder:7b` hint; no table loaded → composer disabled with hint;
  WebGL failure → static hero fallback (machine replaced by dot-grid poster), Start
  still works.
- Testing: existing pytest suite untouched and green; bridge smoke via curl for every
  endpoint; frontend `tsc -b` + lint clean; browser click-through of the hero flow
  (landing → start → upload sample → ask → receipt) on 1440×900 and 390px widths.
- Rollout: `start_jane_react.bat` new; `start_jane.bat` unchanged until parity, then
  repointed; PROJECT_LOG.md session entry per repo contract; AGENTS.md updated only if
  an architectural decision changes (it does: add the React/bridge note at parity).

## 6. Milestones

1. **Bridge + hero flow**: server.py (health/state/upload/ask/fixer/suggestions) with
   curl-verified SSE; frontend scaffold; Landing with working Start (machine can be a
   poster placeholder); chat with upload/ask/receipt/pills/fixer.
2. **The machine**: data-factory-3d.tsx full scene, modes/cameras/labels, wired into
   Landing (embed), perf + reduced-motion + disposal pass.
3. **Parity panels**: Manual, Prep, Profile, Schema, sessions, model picker + pull.
4. **Cutover**: repoint start_jane.bat, retire Streamlit from the launcher path, update
   PROJECT_LOG/AGENTS.
