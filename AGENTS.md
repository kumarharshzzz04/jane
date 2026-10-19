# AGENTS.md — standing rules for AI agents on this repo

## Read order (every session)
1. `PROJECT_LOG.md` — what happened, what's next
2. `AGENTS.md` (this file) — how we work here
3. `docs/research-report.md` — why the architecture is what it is

## Hard rules
- **No vector RAG for tabular numbers. Ever.** Schema injection + SQL/code execution for numbers. RAG (`rag.py`, stdlib TF-IDF, fully local) is allowed ONLY for context: the user's notes/glossary + auto column profiles. Retrieved context is background — every figure must still come from executing code.
- Charts = Vega-Lite JSON dicts as the interchange format (model/sandbox emit specs), but **rendering is matplotlib** (`charts.py`, alpine-dusk styled). The sandbox may also return matplotlib figures directly via a `fig` variable (serialized to PNG data-URIs). Never let the LLM emit matplotlib code on the SQL path.
- Never execute LLM code in-process. Sandbox subprocess + AST whitelist only.
- **DuckDB runs with `enable_external_access=false`.** External access is ON by
  default, so a bare `connect()` let read-only SQL reach the local filesystem
  through table functions with no `read_`/`write_` prefix (`glob()` listed a
  real directory, `sniff_csv()` returned a file's contents). Every new DuckDB
  connection — engine, sandbox child, scripts — must pass that config; the
  validator's token list is the second barrier, not the only one.
- **Statement safety lives in position, not in a keyword blacklist.**
  `validator.validate_sql` allows only a single statement starting with
  SELECT/WITH, and `_FORBIDDEN` holds just the names reachable *after* that check
  (file/network readers, plus insert/update/delete/merge, which a `WITH` clause
  can prefix). Statement keywords added "to be safe" make real data
  unqueryable — a column named `load`, `call`, `set` or `copy` was refused for
  months. No token list can see inside a string literal, so in-expression
  executors like DuckDB's `query('…')` are held by the connection config alone.
- **The interpreter must run the checkout you are editing.** An editable
  install bakes in the path it was built at, so a project copied with its
  `.venv` silently imports the *other* tree — this happened: the venv resolved
  `jane` to `C:\Users\kingh\Downloads\ai data analyist\src` while the repo
  lived elsewhere, so the app ran a different checkout's engine and none of the
  fixes here applied. `tests/test_env_portability.py` now fails loudly on that,
  and the editable `.pth` is self-locating (`Path(sys.prefix).parent / "src"`) so
  a copy keeps working. If `pip install -e .` is unavailable (no setuptools in
  the venv, no network), fix that `.pth` directly rather than skipping it.
- **Never hardcode a path.** Derive it: `config.PROJECT_ROOT` for data and the
  history db, `Path(__file__).parent` for package data. Anything read at runtime
  that ships with the package (sample datasets, `assets/`) must be declared in
  `[tool.setuptools.package-data]` or a non-editable install drops it.
- Never round-trip user Excel through openpyxl writes. Outputs go to new files.
- All user data stays local. The only permitted network hop is localhost → Ollama.
- Respect `config.yaml` + env overrides; no hardcoded endpoints/models in new code.
- New modules go in `src/jane/`, tests in `tests/`, named `test_*.py`.

## Conventions
- Type hints everywhere; dataclasses over dicts for internal contracts.
- Every module has a module docstring stating its job in one line.
- Result containers (`AnalysisResult`, `QueryResult`, `SandboxResult`) are the API between layers — don't leak pandas between UI and agent.
- Tests must not require Ollama or network. Mock `OllamaClient` for agent tests.

## When you finish a session
1. Run the test suite (`python -m pytest -q`).
2. Append one bullet to `PROJECT_LOG.md` → Session log, and update "Open / next ideas".
3. If you changed an architectural decision, update the "Architecture (agreed)" section — and say why in the log.

## Architecture (agreed)
- **The React UI is private and not in this repository.** It is served by the
  FastAPI bridge (`server.py`) from `frontend/dist` on **http://localhost:8100**
  when that folder exists; the static mount is registered LAST, so every `/api/*`
  route above it always wins. `start_jane.bat` degrades to API-only when it does not.
- **Streamlit (`src/jane/app.py`) is the public interface**, via
  `start_jane_streamlit.bat` on :8501 — same engine, same guards. Keep it working;
  visual work belongs in the private React repo.
- Parity status: chat, upload + sheets, sample quick-load, dashboard, model
  switcher, brief, fixer, Prep, per-answer JSON export and cancel are ported;
  Manual analysis, Schema and sessions are NOT.
- **Cancel is cooperative, not a kill.** `POST /api/cancel` sets one flag; the
  agent loop checks it between attempts and after each model call, so a stopped
  run ships nothing and releases `state_lock` after at most one model call. Do
  not report a stopped run as an answer.
- **The verification receipt is measured, never assembled.** `AnalysisResult`
  carries `guards` (list[GuardOutcome], built by `verification.guard_report()
  from the answer being returned vs the rows it came from) plus a derived
  `.verified` that demands at least one `passed` and no `failed`. Never report a
  guard name that did not run: a check with nothing to look at is `skipped`, and
  a skipped-only report is NOT verified. Server, Streamlit and the React receipt
  all read those fields — one source of truth.
- **Multi-table answers are judged against the rows they came from.** The schema
  block marks the active table, and the table-derived guards resolve their source
  through `verification._source_frame(ctx, code)` — the table the executed code
  names, falling back to `ctx.active_table` when it names none. A query that reads
  two tables (a JOIN) has no single source, so those guards `skipped` rather than
  report on rows the answer did not come from. (Fixed 2026-10-10; before this the
  agent could query any listed table while the guards only ever looked at
  `ctx.active_table` — session 29's finding.)
