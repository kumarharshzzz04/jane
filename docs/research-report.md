# Local AI Data Analyst — Research Report

## Executive Summary

Your instinct was correct: **classic vector RAG is the wrong primary engine for tabular/numeric analysis.** The research converges on a different architecture — **schema injection + SQL/code execution**, not retrieval. Embedding rows into a vector store degrades numeric precision, bloats indexes, and invites hallucinated values (ACL 2026 findings paper, TableRAG NeurIPS'24, TabRAG 2025). RAG belongs in this project as a *secondary* capability (glossary, data dictionary, doc search) — not as the analyst.

The recommended architecture is:

1. Load files → **DuckDB** (in-process SQL, no server)
2. Inject **schema + sample rows** into the LLM prompt
3. LLM generates **SQL or pandas code** — never a retrieval query
4. Execute safely in a **subprocess sandbox** with timeouts
5. Validate result shape/dtype; on failure, **inject traceback and retry (max 3)**
6. Emit charts as **Vega-Lite JSON** (not matplotlib code)
7. Fall back to a **manual analysis pane** (editable DataFrame + ad-hoc SQL) when the AI fails
8. RAG only for docs/glossary — never for the numbers

---

## 1. Local Models (Ollama)

The decisive evidence came from a real Q1 2026 benchmark study (pooyagolchian.com) that ran four frameworks against five local Ollama models on three standard tool-use benchmarks, with GPT-4o as ceiling:

| Model size ↕▾ | Tool-use capability ↕▾ | Verdict ↕▾ |
|---|---|---|
| −**< 7B** | Reliability collapses | Unusable |
| −**7B** (Qwen 3.5 7B) | 71.2% BFCL v3, 45 tok/s on M4 | Single-tool queries only |
| −**14B** | Crosses 65% on multi-step ReAct | Chained ops, marginal |
| −**32B** (Qwen 2.5 32B) | 82.6% BFCL v3 — **beats Mistral Large 2** | Production-grade |
| −**70B supervisor + 7B workers** | Beats four uniform 32B agents | Multi-agent coordinator |
⚙

**Recommendations:**

- **Primary:** `qwen2.5-coder:32b` (or newest Qwen 3.x coder equivalent)
- **Fallback (8–16 GB VRAM):** `qwen2.5-coder:7b`
- **Embeddings (glossary only):** `nomic-embed-text`

**Hardware budgets (Q4 quantized):** 7B ≈ 6 GB VRAM · 14B ≈ 10 GB · 32B ≈ 20–24 GB.

Ollama's stable API runs at `http://127.0.0.1:11434` and exposes `/api/chat`, `/api/generate`, `/api/tags`, `/api/pull`, `/api/ps`. Structured outputs (JSON-schema-constrained decoding) and tool calling are both supported (confirmed in release notes).

---

## 2. Data Ingestion

- `pandas.read_excel` handles **xls, xlsx, xlsm, xlsb, odf, ods, odt** in one call.
- **Merged cells are the #1 gotcha:** pandas reads only the top-left value and leaves siblings blank. Must detect via `openpyxl.load_workbook(...).merged_cells.ranges` and **forward-fill** before handing to the LLM.
- **Multi-sheet:** `pd.read_excel(..., sheet_name=None)` returns a dict — expose a sheet picker in the UI.
- **Header-on-row-N detection:** sniff first ~10 rows for the row with the most non-null, non-numeric, unique strings.
- **Never round-trip through openpyxl** — it drops charts/images. Always write outputs to a *new* file.
- CSV → DuckDB `read_csv_auto()` with schema auto-detect. Parquet and JSON are native.

---

## 3. Analysis Engine — Explicitly NOT RAG

The evidence is emphatic (ChatNexus structured-data RAG guide, TableRAG NeurIPS'24, TabRAG 2025, ACL 2026 RAG-over-tables paper):

**The correct hybrid pipeline:**

1. **Schema Discovery** — read column names, types, null counts, sample values
2. **Schema Injection** — put that into the prompt
3. **NL → SQL / pandas** — LLM generates a query
4. **Whitelist validation** — only `SELECT / WHERE / GROUP BY / JOIN / ORDER / LIMIT`
5. **Execute** in sandbox
6. **Result-shape check** — dtype, nulls, row count
7. **Fallback to vector retrieval** only if execution fails

RAG is used for: data dictionaries, glossary terms, business rules, supporting documents. **Never for the numbers.**

If you ever load tables too wide for context (1000+ columns), TableRAG's hierarchical/schema-aware embeddings are the escape hatch — but they are not the default.

---

## 4. Agent Framework — Smolagents

**Recommended: [Smolagents](https://github.com/huggingface/smolagents) (HuggingFace).** It's **code-first** — instead of JSON tool calls, it emits executable Python. For a data analyst this is perfect: the LLM's "tool call" **is** the pandas/DuckDB code, eliminating a translation layer and leveraging the model's strongest capability (code generation > JSON emission for most LLMs).

**Architecture patterns:**

- **ReAct** — covers ~80% of real tasks
- **Plan-and-Execute** — needed for 3–4+ chained operations (load → join → clean → groupby → plot)
- **Coordinator (32B) + workers (7B)** — for multi-agent scale

Hand-rolling a ReAct loop (~40 lines) is a valid zero-dependency alternative.

---

## 5. Self-Correction — 3-Attempt Retry Loop

Industry-standard pattern (artvandelay/agentic-backtesting, CorrectBench, teacherpeterpan self-correction survey, SelfCorrect-Agent in Neurocomputing 2026):

```
generate → execute → capture traceback
   ↓
inject error + prior code + schema → regenerate
   ↓
max 3 attempts → surface failure to user with code + traceback for manual takeover
```

Pair with explicit validators ("four contracts"): **input, output, execution, ground-truth.** Blind retries can "fix" a bug by producing a plausible-but-wrong answer — validators prevent that.

The 3-attempt cap is the industry norm. Beyond it, hand off to the user's manual pane — this directly satisfies your "place where I can analyze stuff if it missed some" requirement.

---

## 6. Charts — Declarative JSON, Not Code

**Vega-Lite is the de-facto LLM chart-spec target:**

- `hyungkwonko/chart-llm` — 1,981 human-authored Vega-Lite specs (usable as few-shot examples)
- `VegaChat` (arXiv 2601.15385) — NL → Vega-Lite generation framework
- `nyanp/chat2plot` — explicitly avoids code-gen, produces JSON specs

**Higher-level alternative: [Flint](https://microsoft.github.io/flint-chart/)** (Microsoft Research, Aug 2026) — one intermediate JSON spec compiles to Vega-Lite / ECharts / Chart.js / Plotly. 50 chart types, 123 backend examples, explicitly designed for AI agents.

**Rendering:** `vega-embed` in the browser, or `st.vega_lite_chart` in Streamlit. Never let the LLM generate matplotlib Python — that's precisely the sandbox hazard you avoid.

---

## 7. UI — Streamlit

Prior art is unambiguous (Step 7 evidence):

- `sahilsawant-da/Secure-Local-Analyst` — PandasAI + Streamlit + Ollama
- `Shakorly/CSV_AI_Agent_Using_Ollama` — CSV agent, Streamlit
- `deepanshu88/ollama-excel` — Excel add-in (`=Ollama(A2)` UDF pattern)
- `georgekhananaev/excel-ai-assistant` — Python desktop Excel transform

**Streamlit primitives to use:**

- `st.chat_message` / `st.chat_input` — conversation
- `st.data_editor` — **the manual analysis fallback pane** ("place where I analyze stuff if it missed some")
- `st.vega_lite_chart` — inline charts
- `st.file_uploader` — ingestion
- `st.session_state` — memory

**Upgrade path:** Next.js + FastAPI (the `e2b-dev/ai-analyst` pattern) if the tool grows beyond a single user.

---

## 8. Safety / Sandboxing

DuckDB SQL is inherently safer than arbitrary Python — **restrict to SELECT/WHERE/GROUP BY/JOIN**. For the pandas code path:

- **AST-level static analysis** — whitelist imports (`pandas, numpy, duckdb, plotly, altair`)
- **Block:** `os`, `subprocess`, `sys`, `socket`, `requests`, `open(mode='w')`, `exec`, `eval`, `__import__`
- **Subprocess with timeout** (default 30s) + `resource.setrlimit`
- **Optional Docker/gVisor isolation** for paranoid mode
- **Read-only mounts** for data
- **Network egress blocked** by default

---

## 9. Reference Projects to Fork/Study

| Project ↕▾ | What to steal ↕▾ |
|---|---|
| −`sahilsawant-da/Secure-Local-Analyst` | PandasAI + Streamlit + Ollama integration |
| −`Shakorly/CSV_AI_Agent_Using_Ollama` | Minimal CSV agent loop |
| −`deepanshu88/ollama-excel` | Excel UDF pattern for later add-in |
| `georgekhananaev/excel-ai-assistant` | Desktop Excel transform UX |
| `e2b-dev/ai-analyst` | UX quality bar (Next.js reference) |
| `nyanp/chat2plot` | LLM → JSON chart spec pattern |
| `hyungkwonko/chart-llm` | 1,981 Vega-Lite few-shot examples |
| `microsoft/flint-chart` | Higher-level chart spec language |
⚙

---

## 10. Recommended Stack — At a Glance

| Layer ↕▾ | Choice ↕▾ | Why ↕▾ |
|---|---|---|
| −Runtime | **Ollama** | Local, free, structured outputs |
| −Primary model | `qwen2.5-coder:32b` | 82.6% BFCL v3, best local tool-use |
| −Fallback model | `qwen2.5-coder:7b` | Runs on modest GPUs |
| −Embeddings | `nomic-embed-text` | Glossary/docs RAG only |
| −Analytics engine | **DuckDB** | In-process, CSV/Excel/Parquet/JSON native |
| −Dataframe lib | pandas + openpyxl | Merged-cell handling, Excel interop |
| −Agent framework | **Smolagents** | Code-first, HF native, minimal |
| −Chart spec | **Vega-Lite** | LLM-friendly JSON, browser embeddable |
| −Renderer | `vega-embed` / `st.vega_lite_chart` | Interactive, no code-gen |
| −UI | **Streamlit** | Matches prior art, fastest to build |
| −Memory | `st.session_state` + SQLite history | Simple, local |
| −Self-correction | 3-attempt traceback loop | Industry standard |
⚙

---

## Skipped / Uncovered

Steps 14–15 (conversational memory design, PandasAI 3.0 architecture) were skipped due to context budget. Steps 2, 9, 10 (framework comparison, UI comparison, sandbox deep-dive) failed on search-provider anti-bot challenges — mitigated by prior art and general best-practice knowledge. **The biggest remaining unknown is conversational memory design** — that's a build-phase decision the project scaffold below handles with a simple SQLite message-log approach that can be upgraded later.

