# Implementation Plan — React Frontend + FastAPI Bridge

Spec: `docs/superpowers/specs/2026-10-01-react-frontend-3d-landing-design.md` (approved 2026-10-01).
The writing-plans skill is not available in this environment; this plan is written and
executed directly by the same session, with the user's standing approval to proceed.

## M1 — Bridge + hero flow

1. Install venv deps: `fastapi`, `uvicorn`, `python-multipart` (via the project venv pip).
2. `server.py` (repo root): FastAPI bridge importing `src/jane` untouched.
   - `GET /api/health`, `GET /api/state`, `GET /api/suggestions`
   - `POST /api/upload` (multipart → `load_uploaded`, multi-sheet info, active table)
   - `POST /api/ask` (SSE: `stage` events from the orchestrator `progress` callback,
     `final` with charts→`render_all()` PNG data-URIs, receipt guard manifest,
     history persisted via `HistoryStore`)
   - `POST /api/fixer` (SQL via `SQLEngine`, Python via `sandbox.run_code`)
   - CORS for localhost:5173/5174; agent + OllamaClient module-level singletons;
     ask runs in a worker thread; check `AnalysisResult`/`QueryResult`/`SandboxResult`
     field names before wiring (types.py verified: ok/answer/code/tables/columns/charts/
     figures/attempts_used/model_used/error).
3. Frontend scaffold: `npm create vite@latest frontend -- --template react-ts`,
   Tailwind v4 (`@tailwindcss/vite`), path aliases `@/*`, shadcn init + button/card/badge/
   input/separator, fonts via @fontsource-variable (Archivo, Fraunces unavailable →
   use Fraunces from @fontsource if packaged, else fallback serif stack; Plex Mono from
   @fontsource/ibm-plex-mono), theme tokens from app.py (ink/bone/poppy/moss/ember),
   `class="dark"` on html, Vite proxy `/api` → :8000.
4. `lib/types.ts` + `lib/api.ts` (SSE parser over fetch ReadableStream; upload; state).
5. Chat view: header (table + model + health dot), message list (user bubble / open-layout
   answer, code artifact, table, figures, receipt, stage indicator), suggestion pills,
   fixer (editable code + Run), pinned composer. `App.tsx` landing→chat switch.
6. Landing v1: notebook rail, Fraunces headline "Answers you can audit.", guard list,
   Start asking CTA (3D machine placeholder poster), transition to chat.
7. Verify: curl every endpoint incl. SSE ask on a real sample; `tsc -b` clean; browser
   click-through 1440×900 + 390px; pytest suite still green.

## M2 — The machine (data-factory-3d.tsx)

8. Scene skeleton: renderer/camera/OrbitControls/lights/env, ink plate + conveyor loop
   (CatmullRom closed curve, instanced slats), packet mesh riding the path.
9. Five stations (Ingest/Schema/Agent/Verify/Chart) procedural builds + canvas-texture
   screens (column grid, SQL text, receipt VERIFIED, animated bar chart).
10. Modes (assembled/cutaway clipping/stations spread/one-question journey HUD),
    cameras (overview/side/top/station/flight), hover tooltip + click focus, DOM labels,
    `window.__machine` + `__machineDebug`.
11. Landing embed: right-58% view offset, headline left, Start asks → chat.
    Perf: pixel-ratio step-down, visibility pause, reduced-motion, full disposal.

## M3 — Parity panels

12. `/api/manual` + Manual panel (grid via TanStack Table, edit, ad-hoc SQL).
13. `/api/prep` + Prep panel; `/api/profile` + Profile panel; `/api/schema` panel.
14. `/api/sessions` + session list; `/api/model` + model picker with pull progress.

## M4 — Cutover

15. `start_jane_react.bat` (bridge + frontend); repoint `start_jane.bat` at parity;
    PROJECT_LOG.md entry; AGENTS.md architecture note at cutover.

## Standing rules during implementation

- No changes inside `src/jane/` except none planned; tests stay green (147).
- Frontend files: components in `frontend/src/components/{ui,jane}/`, lib in
  `frontend/src/lib/`; every module gets a one-line docstring comment per repo style.
- Verify each milestone end-to-end before moving on.
