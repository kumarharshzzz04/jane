"""Jane — Streamlit UI.

Chat with your data: upload → schema injected → model writes SQL/pandas →
sandboxed execution → tables + Vega-Lite charts. Manual analysis pane as the
fallback when the agent misses.

Visual identity: alpine dusk. A full-bleed mountain-meadow photograph anchors
the app; glass panels float over it. Alpenglow white type, poppy accents,
meadow-night ink. IBM Plex Mono marks machine artifacts (SQL, stats, code);
Bricolage Grotesque carries the UI voice.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from jane.agent import AnalystAgent, AgentContext
from jane.config import settings
from jane.history import HistoryStore
from jane.rag import load_notes, save_notes
from jane.ingest import load_file, load_uploaded
from jane.ollama_client import OllamaClient, OllamaError
from jane.types import ChatMessage

st.set_page_config(
    page_title="Jane — made by Kumar Harsh",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ASSETS = Path(__file__).parent / "assets"
SAMPLES = Path(__file__).parent / "sample_data"

# ---------------------------------------------------------------------------
# Theme tokens — receipts & rules (flat lab-notebook skin)
# Two functional colors: signal orange = action, verdigris = verified.
# ---------------------------------------------------------------------------

CANVAS = "#0B0C0E"          # ink (neutral black)
GLASS = "#15161A"           # flat panel (no blur, no green cast)
GLASS_LIGHT = "#EDF4EE"     # bone (light chip fill)
POPPY = "#FF6B2C"           # signal — actions
EMBER = "#FFB35C"           # ember — warnings
MOSS = "#58C08B"            # verdigris — verified
ALPINE = "#EDF4EE"          # bone — text
DIM = "rgba(237, 244, 238, 0.60)"

MONO = "'IBM Plex Mono', 'Consolas', monospace"
SANS = "'Archivo', 'Segoe UI', system-ui, sans-serif"
DISPLAY = "'Fraunces', 'Georgia', serif"


def _backdrop_data_uri() -> str:
    try:
        raw = (ASSETS / "backdrop.jpg").read_bytes()
        return "data:image/jpeg;base64," + base64.b64encode(raw).decode()
    except Exception:
        return ""


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

        :root {{
          --glass: {GLASS}; --glass-light: {GLASS_LIGHT};
          --poppy: {POPPY}; --ember: {EMBER}; --moss: {MOSS};
          --alpine: {ALPINE}; --dim: {DIM};
          --mono: {MONO}; --sans: {SANS}; --display: {DISPLAY};
          --rule: rgba(237, 244, 238, 0.14);
          --verified: {MOSS};
        }}

        /* the field: canopy ink + fine dot grid - a lab notebook, not a photo */
        .stApp {{
          background: {CANVAS};
          background-image: radial-gradient(rgba(237, 244, 238, 0.075) 1px, transparent 1.2px);
          background-size: 24px 24px;
          color: var(--alpine);
        }}

        /* landing field: the living canvas sits between the background and
           the raised content; harmless on the chat view (canvas unmounts) */
        section.stMain {{ position: relative; z-index: 1; }}

        /* - notebook rail: ruled sheet header anchoring the page - */
        .book-rail {{
          display: flex; justify-content: space-between; align-items: baseline;
          gap: 1rem; flex-wrap: wrap;
          padding-bottom: 0.55rem; border-bottom: 2px solid rgba(237, 244, 238, 0.30);
          position: relative;
          font-family: var(--mono); font-size: 0.72rem; color: var(--dim);
        }}
        .book-rail::after {{
          content: ''; position: absolute; left: 0; right: 0; bottom: -5px; height: 1px;
          background: rgba(237, 244, 238, 0.14);
        }}
        .book-rail b {{ color: var(--alpine); font-weight: 600; }}
        .book-rail b.brand {{
          color: var(--poppy); font-size: 0.95rem; letter-spacing: 0.02em;
        }}
        #MainMenu, footer {{ visibility: hidden; }}
        [data-testid="stToolbar"] {{ visibility: visible; background: transparent; }}
        [data-testid="stToolbar"] [data-testid="stStatusWidget"],
        [data-testid="stToolbar"] a,
        [data-testid="stToolbar"] button:not([data-testid="stExpandSidebarButton"]) {{
          visibility: hidden;
        }}
        [data-testid="stExpandSidebarButton"] {{
          background: var(--glass); border: 1px solid var(--rule);
          border-radius: 6px; color: var(--alpine);
        }}
        [data-testid="stExpandSidebarButton"]:hover {{
          border-color: var(--poppy); color: var(--poppy);
        }}

        header[data-testid="stHeader"] {{
          background: transparent; height: 3.2rem;
          /* the header spans the top ~50px; since section.stMain became a
             stacking context (z-index: 1, for the particle field), the header
             would sit above the Jane chip and swallow its clicks */
          z-index: 0 !important; pointer-events: none;
        }}
        header[data-testid="stHeader"] [data-testid="stToolbar"] {{
          pointer-events: auto;   /* expand/toolbar controls stay clickable */
        }}

        html, body, .stApp, .stMarkdown, p, li {{ font-family: var(--sans); }}
        h1, h2, h3 {{ font-family: var(--display) !important; font-weight: 600; letter-spacing: -0.015em; }}
        code {{ font-family: var(--mono) !important; }}

        /* - wordmark: the popover chip IS the brand mark, pinned top-right - */
        section.stMain [data-testid="stPopover"] {{
          position: fixed; top: 14px; right: 18px; z-index: 999998;
          width: fit-content; left: auto;
        }}
        section.stMain [data-testid="stPopover"] button {{
          background: var(--glass) !important;
          border: 1px solid var(--rule) !important;
          border-radius: 6px !important; color: var(--poppy) !important;
          font-family: var(--mono) !important;
          font-weight: 700 !important; font-size: 1.05rem !important;
          letter-spacing: 0.01em; padding: 0.45rem 1.15rem !important;
          transition: border-color .15s, color .15s;
        }}
        section.stMain [data-testid="stPopover"] button::before {{
          content: '\\25CF'; color: var(--poppy); margin-right: 0.55rem; font-size: 0.7rem;
        }}
        /* the wordmark is a mark, not a dropdown control: the chevron lives
           in an aria-hidden div beside the label (nested one level below the
           button's flex wrapper) — hide it */
        section.stMain [data-testid="stPopover"] button div[aria-hidden="true"] {{
          display: none;
        }}
        /* the panel is a control panel, not a second page: cap its width.
           Streamlit portals the body to document.body — it is NOT inside
           section.stMain, so no descendant prefix here. */
        [data-testid="stPopoverBody"] {{
          width: 380px !important;
        }}
        section.stMain [data-testid="stPopover"] button:hover {{
          border-color: var(--poppy) !important;
          background: rgba(255, 107, 44, 0.10) !important;
        }}

        section[data-testid="stSidebar"] {{
          background: {CANVAS};
          border-left: 1px solid var(--rule);
        }}
        section[data-testid="stSidebar"] * {{ color: var(--alpine); }}
        section[data-testid="stSidebar"] hr {{
          border-color: var(--rule); margin: 0.4rem 0;
        }}
        section[data-testid="stSidebar"] .stButton > button {{
          width: 100%; border: 1px solid var(--rule);
          border-radius: 6px; background: transparent;
          color: var(--alpine); font-weight: 500; font-size: 0.85rem;
          transition: background .15s, border-color .15s;
        }}
        section[data-testid="stSidebar"] .stButton > button:hover {{
          background: rgba(237, 244, 238, 0.07); border-color: rgba(237, 244, 238, 0.35);
        }}
        section[data-testid="stSidebar"] .stButton > button:focus-visible {{
          outline: 2px solid var(--poppy); outline-offset: 2px;
        }}
        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{
          background: rgba(237, 244, 238, 0.05); border: 1px dashed rgba(237, 244, 238, 0.3);
          border-radius: 6px;
        }}

        .file-stats {{ font-family: var(--mono); font-size: 0.72rem; color: var(--dim);
                       white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}

        /* - tabs: ruled index, not pills - */
        .stTabs [data-baseweb="tab-list"] {{
          gap: 0; border-bottom: 1px solid var(--rule); width: 100%;
        }}
        .stTabs [data-baseweb="tab"] {{
          background: transparent; border: none;
          border-bottom: 2px solid transparent;
          border-radius: 0; padding: 0.5rem 1.1rem; margin: 0 0 -1px;
          color: var(--dim); font-weight: 600; font-size: 0.9rem;
          transition: color .15s, border-color .15s;
        }}
        .stTabs [data-baseweb="tab"]:hover {{ color: var(--alpine); }}
        .stTabs [aria-selected="true"] {{
          background: transparent; color: var(--alpine);
          border-bottom: 2px solid var(--poppy);
        }}

        /* - buttons - */
        .stButton > button {{
          border: 1px solid var(--rule); border-radius: 6px;
          background: var(--glass); color: var(--alpine);
          font-weight: 500; font-size: 0.88rem;
          transition: border-color .15s, color .15s;
        }}
        .stButton > button:hover {{
          border-color: var(--poppy); color: var(--poppy);
        }}
        .stButton > button[kind="primary"] {{
          background: var(--poppy); border-color: var(--poppy); color: {CANVAS};
          font-weight: 700;
        }}
        .stButton > button[kind="primary"]:hover {{
          filter: brightness(1.08); color: {CANVAS};
        }}
        .stButton > button:focus-visible {{ outline: 2px solid var(--poppy); outline-offset: 2px; }}

        /* - landing: editorial left rail - */
        .hero-title {{
          font-family: var(--display) !important; font-weight: 700; letter-spacing: -0.02em;
          font-size: clamp(3rem, 6vw, 5.1rem); line-height: 1.02;
          color: var(--alpine); margin: 0;
        }}
        .hero-sub {{
          font-family: var(--sans); font-weight: 400; font-size: clamp(1.05rem, 1.4vw, 1.22rem);
          color: var(--dim); margin: 1.1rem 0 0; max-width: 32em; line-height: 1.55;
        }}

        /* - landing sample chips: ledger-strength invitations - */
        .land-actions .stButton > button {{
          border: 1px solid rgba(237, 244, 238, 0.38); background: var(--glass);
          font-family: var(--mono); font-size: 0.85rem; font-weight: 500;
          padding: 0.62rem 1.35rem; border-radius: 6px;
          transition: border-color .15s, color .15s;
        }}
        .land-actions .stButton > button:hover {{
          border-color: var(--poppy); color: var(--poppy);
        }}

        /* - specimen receipt: an example answer, typeset like the real thing - */
        .spec {{
          background: #0A120E; border: 1px solid var(--rule); border-radius: 8px;
          padding: 1.05rem 1.25rem 0.9rem;
        }}
        .spec-tag {{
          display: flex; justify-content: space-between; align-items: baseline;
          font-family: var(--mono); font-size: 0.7rem; color: var(--dim);
          padding-bottom: 0.5rem; border-bottom: 1px solid var(--rule); margin-bottom: 0.85rem;
        }}
        .spec-tag .spec-kind {{ color: var(--moss); font-weight: 600; letter-spacing: 0.08em; }}
        .spec-q {{
          font-family: var(--sans); font-weight: 600; color: var(--alpine);
          font-size: 0.95rem; line-height: 1.35; margin: 0 0 0.7rem;
        }}
        .spec-q span {{
          color: var(--dim); font-family: var(--mono); font-size: 0.68rem;
          font-weight: 400; margin-right: 0.5rem;
        }}
        .spec-a {{
          font-family: var(--display); font-size: 1.8rem; font-weight: 600;
          color: var(--alpine); letter-spacing: -0.01em; line-height: 1.1; margin: 0 0 0.25rem;
        }}
        .spec-sub {{
          font-family: var(--mono); font-size: 0.72rem; color: var(--dim); margin: 0 0 0.95rem;
        }}
        .spec-table {{
          width: 100%; border-collapse: collapse; font-family: var(--mono);
          font-size: 0.72rem; color: var(--alpine); margin: 0 0 0.95rem;
        }}
        .spec-table th {{
          color: var(--dim); font-weight: 500; text-align: left;
          border-bottom: 1px solid var(--rule); padding: 0.26rem 0.7rem 0.26rem 0;
        }}
        .spec-table td {{
          border-bottom: 1px solid var(--rule); padding: 0.3rem 0.7rem 0.3rem 0;
        }}
        .spec-table th.num, .spec-table td.num {{ text-align: right; padding-right: 0; }}
        .spec-guards {{ border-top: 1px solid var(--rule); padding-top: 0.5rem; margin-bottom: 0.75rem; }}
        .spec-guards .g {{
          display: flex; justify-content: space-between; gap: 0.8rem;
          padding: 0.2rem 0; font-size: 0.74rem;
        }}
        .spec-guards .g .chk {{ color: var(--alpine); }}
        .spec-guards .g .chk::before {{ content: '\\2713  '; color: var(--moss); }}
        .spec-guards .g .nm {{
          font-family: var(--mono); font-size: 0.68rem; color: var(--moss);
          white-space: nowrap;
        }}
        .spec-foot {{
          display: flex; align-items: baseline;
          border-top: 1px solid var(--rule); padding-top: 0.55rem;
          font-family: var(--mono); font-size: 0.68rem; color: var(--dim);
        }}
        .spec-foot span {{ padding: 0 0.9rem; border-left: 1px solid var(--rule); }}
        .spec-foot span:first-child {{ padding-left: 0; border-left: none; }}
        .spec-foot .ok {{
          margin-left: auto; color: var(--moss); font-weight: 600;
          border-left: none; padding-right: 0;
        }}

        /* - chat - */
        [data-testid="stChatMessage"] {{ background: transparent; padding: 0.35rem 0; }}
        div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) > div {{
          background: var(--glass);
          border: 1px solid var(--rule);
          border-left: 3px solid var(--poppy);
          border-radius: 0 8px 8px 0; padding: 0.85rem 1.1rem;
        }}
        div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) > div {{
          background: transparent;
        }}
        div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"] {{
          font-weight: 600;
        }}

        [data-testid="stChatInput"] textarea {{
          font-family: var(--sans); font-size: 0.95rem; color: var(--alpine);
        }}
        [data-testid="stElementContainer"]:has(> [data-testid="stChatInput"]) {{
          position: fixed; bottom: 12px; left: 0; right: 0; z-index: 999990;
          padding: 0 max(1.2rem, calc(50vw - 430px));
        }}
        [data-testid="stElementContainer"]:has(> [data-testid="stChatInput"])::before {{
          content: ''; position: absolute; inset: -14px 0 -12px 0; z-index: -1;
          background: linear-gradient(180deg, rgba(12, 21, 18, 0) 0%, rgba(12, 21, 18, 0.94) 45%);
          pointer-events: none;
        }}
        [data-testid="stChatInput"] {{
          border-radius: 8px;
          border: 1px solid var(--rule) !important;
          background: var(--glass) !important;
          overflow: hidden;
        }}
        [data-testid="stChatInput"]:focus-within {{
          border-color: var(--poppy) !important;
        }}
        [data-testid="stChatInput"] > div {{
          background: transparent !important;
          padding: 4px 6px 4px 14px;
          align-items: center;
        }}
        [data-testid="stChatInput"] textarea {{
          background: transparent !important;
          border: none !important;
          box-shadow: none !important;
          padding: 0.55rem 0.5rem 0.55rem 0;
          max-height: 120px;
          overflow-y: auto;
        }}
        [data-testid="stChatInput"] button {{
          background: var(--poppy); color: {CANVAS}; border-radius: 6px;
          width: 36px; height: 36px; min-height: 36px;
          flex-shrink: 0; margin: 0;
        }}

        /* - receipts: the verification strip under every answer - */
        .receipt {{
          font-family: var(--mono); font-size: 0.72rem; color: var(--dim);
          border-top: 1px solid var(--rule); margin-top: 0.8rem; padding-top: 0.45rem;
          display: flex; gap: 1.1rem; flex-wrap: wrap;
        }}
        .receipt b {{ color: var(--moss); font-weight: 600; }}
        .receipt .r-model {{ color: var(--alpine); }}

        /* - machine artifacts - */
        div[data-testid="stCode"] {{
          background: #0A120E;
          border: 1px solid var(--rule);
          border-left: 3px solid var(--poppy);
          border-radius: 0 6px 6px 0;
        }}
        div[data-testid="stCode"] pre, div[data-testid="stCode"] code {{
          background: transparent !important; color: #CBE3D2 !important;
        }}

        [data-testid="stDataFrame"] {{
          border: 1px solid var(--rule); border-radius: 6px;
          overflow: hidden; background: var(--glass);
        }}
        [data-testid="stDataFrame"] [role="columnheader"] {{
          font-family: var(--mono); font-size: 0.72rem; color: var(--dim);
        }}

        div[data-testid="stAlert"] {{
          background: var(--glass); border: 1px solid var(--rule);
          border-left: 3px solid var(--ember); border-radius: 0 6px 6px 0; color: var(--alpine);
        }}

        .stCaption, [data-testid="stCaptionContainer"] {{
          font-family: var(--mono); font-size: 0.72rem; color: var(--dim);
        }}

        /* example pills: bone chips, dark text - invitation to click */
        .ex-pill button {{
          background: var(--glass-light) !important; color: {CANVAS} !important;
          border: none !important; border-radius: 6px !important;
          font-weight: 600 !important; font-size: 0.84rem !important;
          padding: 0.45rem 1rem !important;
          transition: transform .12s, box-shadow .12s;
        }}
        .ex-pill button:hover {{
          transform: translateY(-1px); background: #FFFFFF !important; color: {CANVAS} !important;
        }}
        .ex-pill button:focus-visible {{
          outline: 2px solid var(--poppy); outline-offset: 2px;
          background: #FFFFFF !important; color: {CANVAS} !important;
        }}
        .ex-pill button:focus {{
          background: var(--glass-light) !important; color: {CANVAS} !important;
        }}

        @media (prefers-reduced-motion: reduce) {{
          * {{ transition: none !important; animation: none !important; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

for key, default in {
    "tables": {},
    "active_table": None,
    "file_name": None,
    "history": [],
    "session_id": None,
    "agent": None,
    "pending_question": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def get_store() -> HistoryStore:
    if "store" not in st.session_state:
        st.session_state.store = HistoryStore(settings.history_db)
    return st.session_state.store


def get_agent() -> AnalystAgent:
    chosen = st.session_state.get("model_choice")
    if st.session_state.agent is None:
        st.session_state.agent = AnalystAgent(OllamaClient(settings.ollama_host, settings.request_timeout))
    st.session_state.agent.client.model_override = chosen
    return st.session_state.agent


def load_tables_from_path(path: Path) -> None:
    loaded = load_file(path)
    st.session_state.tables = {s.name: s.df for s in loaded.sheets}
    st.session_state.active_table = loaded.sheets[0].name
    st.session_state.file_name = path.name
    st.session_state.history = []
    st.session_state.session_id = None


def _surprise_question(tables: dict, active: str) -> str:
    """Turn the strongest deterministic insight into a follow-up question."""
    from jane.insights import find_insights
    ins = find_insights({active: tables[active]})
    if not ins:
        return f"What is the most interesting pattern in {active}?"
    top = ins[0]
    prompts = {
        "spike": f"Why did {top.title.split(' in ')[0]}? Break the months around it down.",
        "drop": f"Why did {top.title.split(' in ')[0]}? Show the numbers behind it.",
        "outlier": f"Show the rows with the most extreme {top.title.split(' in ')[-1]} values.",
        "dominance": f"How does the {top.title.split(' of ')[-1]} skew affect the averages? Compare groups.",
        "correlation": f"Plot the relationship between {top.title.split(' (')[0]} and explain it.",
        "missing": f"How much does the missing data in {top.title.split(' is ')[0]} limit analysis?",
    }
    return prompts.get(top.kind, f"What is the most interesting pattern in {active}?")


def _followup_questions(history: list, tables: dict, active: str) -> list[str]:
    """Suggest next questions from the last result's shape.

    Rows may be dicts (sandbox path) or positional lists (SQL path) —
    normalize before inspecting.
    """
    if not history or history[-1]["role"] != "assistant":
        return []
    last = history[-1]
    questions: list[str] = []
    if last.get("tables"):
        t = last["tables"][0]
        cols = list(t.get("columns") or [])
        rows = t.get("rows") or []
        if rows and isinstance(rows[0], list):
            rows = [dict(zip(cols, r)) for r in rows]
        if len(rows) >= 2 and cols and isinstance(rows[0], dict):
            cat_col = next((c for c in cols if isinstance(rows[0].get(c), str)), None)
            num_cols = [c for c in cols
                        if isinstance(rows[0].get(c), (int, float))
                        and not isinstance(rows[0].get(c), bool)]
            num_col = num_cols[0] if num_cols else None
            if cat_col and num_col:
                try:
                    top_row = max(rows, key=lambda r: r.get(num_col) or 0)
                    questions.append(f"Drill into {str(top_row.get(cat_col))[:20]} only")
                    questions.append(f"Plot {num_col} by {cat_col} as a pie share")
                except (TypeError, ValueError):
                    pass
            elif len(num_cols) >= 2:
                # two numeric columns (e.g. class -> fare): suggest the plot + extremes
                questions.append(f"Plot {num_cols[1]} against {num_cols[0]}")
                try:
                    top_row = max(rows, key=lambda r: r.get(num_cols[1]) or 0)
                    questions.append(f"Which row has the highest {num_cols[1]}?")
                except (TypeError, ValueError):
                    pass
    if questions:
        questions.append("✦ What else is surprising in this data?")
    return questions[:3]


def _render_fixer(msg: dict, mi: int) -> None:
    """The in-chat fixer: a FAILED answer's code is editable and runnable right
    here — no detour to another tab (the old give-up message pointed at the
    Manual analysis tab, which is not where the user is). Keyed by message
    index so edits survive reruns."""
    _lang = "python" if "import" in msg["code"] else "sql"
    # on_click fires BEFORE the script reruns, so the expander is already
    # open on the rerun that renders the results — without this, the Run
    # click's rerun collapses the expander and the result is invisible
    _open_key = f"fix_open_{mi}"
    with st.expander("Fix it yourself — edit and run",
                     expanded=st.session_state.get(_open_key, False)):
        fixed = st.text_area("Code", value=msg["code"], height=220, key=f"fix_code_{mi}")
        st.button("Run fixed code", key=f"fix_run_{mi}", type="primary",
                  on_click=lambda k=_open_key: st.session_state.__setitem__(k, True))
        if st.session_state.get(f"fix_run_{mi}"):
            if _lang == "sql":
                from jane.engine import SQLEngine
                eng = SQLEngine(st.session_state.tables)
                try:
                    r = eng.execute(fixed)
                    if r.ok:
                        st.dataframe(pd.DataFrame(r.rows, columns=r.columns),
                                     use_container_width=True, hide_index=True)
                        if r.truncated:
                            st.caption("Result truncated — refine the query.")
                    else:
                        st.error(r.error)
                finally:
                    eng.close()
            else:
                from jane.sandbox import run_code as _run_code
                r = _run_code(fixed, st.session_state.tables,
                              st.session_state.active_table, timeout=90)
                if r.ok:
                    for _t in r.tables:
                        st.dataframe(pd.DataFrame(_t["rows"], columns=_t["columns"]),
                                     use_container_width=True, hide_index=True)
                    for _f in r.figures:
                        st.markdown(
                            f'<img src="{_f}" style="max-width:100%;border-radius:12px;">',
                            unsafe_allow_html=True)
                    if r.stdout and r.stdout.strip():
                        with st.expander("Execution output"):
                            st.text(r.stdout)
                else:
                    st.error(r.error)


# ---------------------------------------------------------------------------
# Data & model popover button — fixed top-right; opens the control panel
# ---------------------------------------------------------------------------

st.markdown('<div class="data-pop"></div>', unsafe_allow_html=True)
with st.popover("Jane", use_container_width=False):
    st.caption("made by Kumar Harsh — everything runs locally, nothing leaves this machine.")

    uploaded = st.file_uploader(
        "Upload a file",
        type=["csv", "tsv", "txt", "xlsx", "xls", "xlsm", "xlsb", "ods", "json", "parquet"],
    )
    if uploaded is not None and uploaded.name != st.session_state.file_name:
        size_mb = uploaded.size / 1e6
        if size_mb > settings.max_upload_mb:
            st.error(f"File too large ({size_mb:.0f} MB > {settings.max_upload_mb} MB limit).")
        else:
            with st.spinner("Ingesting…"):
                try:
                    raw = uploaded.getvalue()
                    loaded = load_uploaded(raw, uploaded.name, progress=lambda s: None)
                    st.session_state.tables = {s.name: s.df for s in loaded.sheets}
                    st.session_state.active_table = loaded.sheets[0].name
                    st.session_state.file_name = uploaded.name
                    st.session_state.history = []
                    st.session_state.session_id = None
                    st.success(f"Loaded **{uploaded.name}** — {len(loaded.sheets)} sheet(s)")
                except Exception as e:
                    st.session_state["ingest_error"] = f"{uploaded.name}: {e}"
                    st.error(f"Ingest failed: {e}")

    samples = sorted(SAMPLES.glob("*.csv")) if SAMPLES.exists() else []
    if samples:
        st.caption("Or try a sample:")
        scols = st.columns(len(samples))
        for col, sample in zip(scols, samples):
            with col:
                if st.button(sample.stem.replace('_', ' ').title(), key=f"pop_{sample.name}",
                             use_container_width=True):
                    load_tables_from_path(sample)
                    st.rerun()

    if st.session_state.tables:
        names = list(st.session_state.tables.keys())
        if len(names) > 1:
            st.session_state.active_table = st.selectbox(
                "Active table", names,
                index=names.index(st.session_state.active_table) if st.session_state.active_table in names else 0,
            )
        else:
            st.session_state.active_table = names[0]
        df_active = st.session_state.tables[st.session_state.active_table]
        st.markdown(
            f'<div class="file-stats">{st.session_state.file_name} · '
            f'{len(df_active):,} rows × {len(df_active.columns)} cols</div>',
            unsafe_allow_html=True,
        )
        if st.button("Clear data", use_container_width=True):
            st.session_state.tables = {}
            st.session_state.active_table = None
            st.session_state.file_name = None
            st.session_state.history = []
            st.rerun()

    st.divider()
    st.subheader("My notes")
    st.caption("Jane reads these for context — your domain knowledge, abbreviations, business rules.")
    _notes = load_notes()
    _notes_text = st.text_area(
        "Notes", value="\n\n".join(_notes), height=110,
        key="notes_editor", label_visibility="collapsed",
        placeholder="e.g. 'EMEA = Europe/Middle East/Africa'. 'net revenue excludes refunds'.",
    )
    _c1, _c2 = st.columns(2)
    with _c1:
        if st.button("Save notes", use_container_width=True, type="primary"):
            save_notes([n.strip() for n in _notes_text.split("\n\n") if n.strip()])
            st.toast("Notes saved — Jane will use them.")
    with _c2:
        if _notes and st.button("Clear notes", use_container_width=True):
            save_notes([])
            st.rerun()

    st.divider()
    st.subheader("Model")
    try:
        client = OllamaClient(settings.ollama_host, 5)
        models = [m["name"] for m in client.tags()]
        st.selectbox("Model", models, key="model_choice", label_visibility="collapsed",
                     index=models.index(settings.primary_model) if settings.primary_model in models else 0)
        st.caption("Runs locally. Nothing leaves this machine.")
        if "qwen2.5-coder" not in " ".join(models):
            st.info("A coder model gives noticeably better SQL/Python. Pull one?")
            if st.button("Pull qwen2.5-coder:7b (~4.7 GB)", use_container_width=True):
                import urllib.request
                req = urllib.request.Request(
                    settings.ollama_host.rstrip("/") + "/api/pull",
                    data=json.dumps({"name": "qwen2.5-coder:7b"}).encode(),
                    headers={"Content-Type": "application/json"}, method="POST")
                progress_bar = st.progress(0.0, text="Pulling qwen2.5-coder:7b…")
                try:
                    with urllib.request.urlopen(req, timeout=3600) as resp:
                        for line in resp:
                            obj = json.loads(line)
                            if "total" in obj and "completed" in obj and obj["total"]:
                                progress_bar.progress(min(1.0, obj["completed"] / obj["total"]),
                                                      text=f"Pulling… {obj.get('status', '')}")
                    progress_bar.progress(1.0, text="Pulled! Re-open the model picker.")
                except Exception as e:
                    st.error(f"Pull failed: {e}")
    except OllamaError:
        st.error(f"Ollama not reachable at {settings.ollama_host} — start it with `ollama serve`.")

    st.divider()
    st.subheader("Sessions")
    store = get_store()
    if st.button("New session", use_container_width=True):
        st.session_state.session_id = None
        st.session_state.history = []
        st.rerun()
    for sid, title, _ts in store.list_sessions(15):
        if st.button(f"{title} (#{sid})", key=f"sess_{sid}", use_container_width=True):
            st.session_state.session_id = sid
            st.session_state.history = [
                {"role": r.role, "content": r.content} for r in store.get_messages(sid)
            ]
            st.rerun()

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Landing — the photo is the hero
# ---------------------------------------------------------------------------

if not st.session_state.tables:
    # surface upload failures on the main page — the popover closes on rerun
    if st.session_state.get("ingest_error"):
        st.error(f"Upload failed — {st.session_state['ingest_error']}")
        del st.session_state["ingest_error"]
    # ——— the living field: cursor-reactive particle web behind the landing ———
    # ported from the spider-particles three.js component (grid + cursor
    # spotlight + web lines + lerp/snap/fade + reduced-motion damping),
    # re-tuned to the receipts palette. Canvas only — no dependencies.
    # st.markdown strips <script> tags — deliver the field through a
    # zero-height same-origin iframe whose ONLY job is to install the field
    # script into the PARENT document. Running in the parent context means
    # the loop survives the iframe's death (Streamlit destroys the iframe on
    # every rerun/unmount) and can tear itself down deterministically: the
    # landing's .specimen block is the aliveness signal.
    from streamlit.components.v1 import html as _component_html
    _component_html(
        """
        <script>
        (function () {
          var pw = window.parent, pd = pw.document;
          var old = pd.getElementById('jane-field-installer');
          if (old) old.remove();
          if (pw.__janeFieldDie) pw.__janeFieldDie();   // kill any previous field
          var s = pd.createElement('script');
          s.id = 'jane-field-installer';
          s.textContent = [
            "(function(){",
            "var BONE=[237,244,238],EMBER=[255,107,44],VERD=[88,192,139];",
            "var SPOTLIGHT=260,CONNECT=150,FADE=0.045,LERP=0.12,RM_FACTOR=0.6;",
            "var pw=window,pd=document;",
            "if(pw.matchMedia('(pointer: coarse)').matches)return;",
            "if(pw.__janeFieldDie)pw.__janeFieldDie();",
            "var gen=(pw.__janeFieldGen=(pw.__janeFieldGen||0)+1);",
            "var stale=pd.getElementById('jane-field-canvas');if(stale)stale.remove();",
            "var app=pd.querySelector('.stApp')||pd.body;",
            "var cv=pd.createElement('canvas');cv.id='jane-field-canvas';",
            "cv.style.cssText='position:fixed;inset:0;z-index:0;pointer-events:none;';",
            "app.insertBefore(cv,app.firstChild);",
            "var ctx=cv.getContext('2d');",
            "var dpr=Math.min(pw.devicePixelRatio||1,2);",
            "var W=0,H=0,pts=[],mx=-9999,my=-9999,sx=-9999,sy=-9999,present=false,alpha=0;",
            "function resize(){W=pw.innerWidth;H=pw.innerHeight;",
            "cv.width=W*dpr;cv.height=H*dpr;cv.style.width=W+'px';cv.style.height=H+'px';",
            "ctx.setTransform(dpr,0,0,dpr,0,0);",
            "var gap=24,cols=Math.max(2,Math.floor(W/gap)),rows=Math.max(2,Math.floor(H/gap));",
            "pts=[];for(var r=0;r<rows;r++)for(var c=0;c<cols;c++)",
            "pts.push([(c+0.5)*(W/cols),(r+0.5)*(H/rows)]);}",
            "resize();",
            "var rmMq=pw.matchMedia('(prefers-reduced-motion: reduce)');",
            "function onMove(e){mx=e.clientX;my=e.clientY;",
            "if(!present){present=true;sx=mx;sy=my;}}",
            "function onLeave(){present=false;}",
            "pw.addEventListener('resize',resize);",
            "pw.addEventListener('mousemove',onMove);",
            "pd.addEventListener('mouseleave',onLeave);",
            "function die(){",
            "pw.removeEventListener('resize',resize);",
            "pw.removeEventListener('mousemove',onMove);",
            "pd.removeEventListener('mouseleave',onLeave);",
            "if(cv.isConnected)cv.remove();",
            "var ins=pd.getElementById('jane-field-installer');if(ins)ins.remove();",
            "pw.__janeFieldDie=null;}",
            "pw.__janeFieldDie=die;",
            "function frame(){",
            "if(pw.__janeFieldDie!==die||!pd.querySelector('.specimen')){die();return;}",
            "requestAnimationFrame(frame);",
            "var rm=rmMq.matches;",
            "var fade=rm?FADE*RM_FACTOR:FADE,lerp=rm?LERP*RM_FACTOR:LERP;",
            "var spot=rm?SPOTLIGHT*RM_FACTOR:SPOTLIGHT,conn=rm?CONNECT*RM_FACTOR:CONNECT;",
            "alpha=present?Math.min(1,alpha+fade):Math.max(0,alpha-fade);",
            "if(present){sx+=(mx-sx)*lerp;sy+=(my-sy)*lerp;}",
            "ctx.clearRect(0,0,W,H);if(!present&&alpha<=0)return;",
            "if(present){ctx.lineWidth=1;",
            "for(var i=0;i<pts.length;i++){",
            "var dx=pts[i][0]-sx,dy=pts[i][1]-sy,d=Math.sqrt(dx*dx+dy*dy);",
            "if(d>=conn)continue;",
            "var a=(1-d/conn)*0.5*alpha;",
            "ctx.strokeStyle='rgba('+VERD[0]+','+VERD[1]+','+VERD[2]+','+a+')';",
            "ctx.beginPath();ctx.moveTo(sx,sy);ctx.lineTo(pts[i][0],pts[i][1]);ctx.stroke();}}",
            "for(var j=0;j<pts.length;j++){",
            "var ddx=pts[j][0]-sx,ddy=pts[j][1]-sy,dd=Math.sqrt(ddx*ddx+ddy*ddy);",
            "if(dd>=spot)continue;",
            "var t=1-dd/spot,rr=0.6+t*2.1,warm=Math.min(1,t*1.4);",
            "var cr=BONE[0]+(EMBER[0]-BONE[0])*warm,cg=BONE[1]+(EMBER[1]-BONE[1])*warm,cb=BONE[2]+(EMBER[2]-BONE[2])*warm;",
            "ctx.beginPath();",
            "ctx.fillStyle='rgba('+(cr|0)+','+(cg|0)+','+(cb|0)+','+(t*0.85*alpha)+')';",
            "ctx.arc(pts[j][0],pts[j][1],rr,0,6.2832);ctx.fill();}",
            "if(present){",
            "ctx.beginPath();",
            "ctx.fillStyle='rgba('+EMBER[0]+','+EMBER[1]+','+EMBER[2]+','+(0.12*alpha)+')';",
            "ctx.arc(sx,sy,14,0,6.2832);ctx.fill();",
            "ctx.beginPath();",
            "ctx.fillStyle='rgba('+EMBER[0]+','+EMBER[1]+','+EMBER[2]+','+(0.9*alpha)+')';",
            "ctx.arc(sx,sy,3.2,0,6.2832);ctx.fill();}}",
            "requestAnimationFrame(frame);",
            "})();"
          ].join("\\n");
          (pd.head || pd.documentElement).appendChild(s);
        })();
        </script>
        """,
        height=0,
    )

    st.markdown("<div style='height: 5vh'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="book-rail">
          <span><b class="brand">Jane</b> — local data analyst · field book · made by Kumar Harsh</span>
          <span>CSV / Excel / JSON / Parquet</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height: 7vh'></div>", unsafe_allow_html=True)
    hleft, hgap, hspec = st.columns([1.35, 0.25, 1.1])
    with hleft:
        st.markdown(
            """
            <div>
              <h1 class="hero-title">Answers you can<br>audit, not just read.</h1>
              <p class="hero-sub">Jane writes real SQL and pandas against your file,
              then checks her own work with deterministic guards before she answers.
              Every reply ships with its verification receipt.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)
        # quick-start chips — every supported sample is one click
        samples = sorted(SAMPLES.glob("*.csv")) if SAMPLES.exists() else []
        if samples:
            st.markdown("<div class='land-actions'>", unsafe_allow_html=True)
            # narrow columns + one wide spacer: chips sit together, not at
            # opposite ends of the column
            pill_cols = st.columns([*[1.75] * len(samples), 4])
            for col, sample in zip(pill_cols, samples):
                with col:
                    if st.button(f"try {sample.stem.title()}", key=f"land_{sample.name}",
                                 use_container_width=True):
                        load_tables_from_path(sample)
                        st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
    with hspec:
        # the artifact itself: a real verified answer, typeset as a receipt
        st.markdown(
            f"""
            <div class="spec specimen">   <!-- .specimen = field aliveness signal -->
              <div class="spec-tag">
                <span>RECEIPT Nº 0247 — retail_sales.csv</span>
                <span class="spec-kind">SPECIMEN</span>
              </div>
              <p class="spec-q"><span>Q</span>Do discounted orders generate more revenue
              per order than non-discounted ones?</p>
              <p class="spec-a">Yes — 12.5% more</p>
              <p class="spec-sub">54.21 vs 48.15 average revenue per order</p>
              <table class="spec-table">
                <tr><th>group</th><th class="num">orders</th><th class="num">per order</th></tr>
                <tr><td>discounted</td><td class="num">260</td><td class="num">54.21</td></tr>
                <tr><td>full price</td><td class="num">240</td><td class="num">48.15</td></tr>
              </table>
              <div class="spec-guards">
                <div class="g"><span class="chk">numbers match the result</span><span class="nm">truth guard</span></div>
                <div class="g"><span class="chk">text agrees with the table</span><span class="nm">contradiction guard</span></div>
                <div class="g"><span class="chk">percentages use the right base</span><span class="nm">denominator sanity</span></div>
                <div class="g"><span class="chk">winners really win</span><span class="nm">winner check</span></div>
                <div class="g"><span class="chk">empty results get re-asked</span><span class="nm">inspection ladder</span></div>
              </div>
              <div class="spec-foot">
                <span>qwen2.5-coder:7b</span>
                <span>1 try</span>
                <span>4 rows traced</span>
                <span class="ok">VERIFIED ✓</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.stop()

# ---------------------------------------------------------------------------
# Main — chat
# ---------------------------------------------------------------------------

_active_df = st.session_state.tables.get(st.session_state.active_table)
_rows_label = f", {_active_df.shape[0]:,} rows" if _active_df is not None else ""
st.markdown(
    f"""
    <div class="book-rail" style="margin-bottom:0.55rem;">
      <span><b class="brand">Jane</b> · <b>{st.session_state.file_name}</b>{_rows_label}</span>
      <span>table: {st.session_state.active_table} — every answer checked by 6 guards · made by Kumar Harsh</span>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_chat, tab_manual, tab_profile, tab_prep, tab_schema = st.tabs(
    ["Chat", "Manual analysis", "Profile", "Prep", "Schema"])

with tab_chat:
    # example pills: data-aware, ID-proof, driven by the insight sweep
    from jane.insights import suggest_questions
    df_active = st.session_state.tables[st.session_state.active_table]
    examples = suggest_questions(st.session_state.tables, st.session_state.active_table)

    ex_cols = st.columns(len(examples) + 1)
    for col, ex in zip(ex_cols, examples):
        with col:
            if st.button(ex, key=f"ex_{hash(ex)}", use_container_width=True):
                st.session_state.pending_question = ex
    with ex_cols[-1]:
        if st.button("Surprise me", key="surprise_me", use_container_width=True):
            st.session_state.pending_question = _surprise_question(
                st.session_state.tables, st.session_state.active_table)

    # follow-up suggestions based on the last answer
    followups = _followup_questions(st.session_state.history, st.session_state.tables,
                                    st.session_state.active_table)
    if followups:
        st.caption("Try next:")
        fu_cols = st.columns(len(followups))
        for col, fq in zip(fu_cols, followups):
            with col:
                if st.button(fq, key=f"fu_{hash(fq)}", use_container_width=True):
                    st.session_state.pending_question = fq

    # clear space so the last message isn't hidden behind the pinned input bar
    st.markdown('<div style="height: 96px"></div>', unsafe_allow_html=True)

    # auto-scroll: an anchor at the bottom of the chat stream
    st.markdown('<div id="jane_chat_end"></div>', unsafe_allow_html=True)
    if st.session_state.get("_scroll_to_end"):
        st.session_state["_scroll_to_end"] = False
        st.markdown(
            """<script>(function tryScroll(n){ if(n<=0) return; const el = document.getElementById('jane_chat_end'); if (el) { el.scrollIntoView({behavior:'smooth', block:'end'}); } else { setTimeout(() => tryScroll(n-1), 120); } })(15);</script>""",
            unsafe_allow_html=True)

    for _mi, msg in enumerate(st.session_state.history):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("code"):
                st.code(msg["code"], language="python" if "import" in msg["code"] else "sql")
            if msg.get("ok") is False and msg.get("code"):
                _render_fixer(msg, _mi)
            for chart in msg.get("charts", []):
                try:
                    st.vega_lite_chart({}, chart, use_container_width=True)
                except Exception as e:
                    st.warning(f"chart render failed: {e}")
            for t in msg.get("tables", []):
                st.dataframe(pd.DataFrame(t["rows"], columns=t["columns"]),
                             use_container_width=True, hide_index=True)
                import base64 as _b64
                _csv = pd.DataFrame(t["rows"], columns=t["columns"]).to_csv(index=False)
                st.download_button(
                    label=f"Export {t['name']} as CSV", data=_csv,
                    mime="text/csv", file_name=f"jane_{t['name']}.csv",
                    key=f"dl_{hash(str(t['rows'][:2]))}")
            for fig_uri in msg.get("figures", []):
                try:
                    st.markdown(f'<img src="{fig_uri}" style="max-width:100%;border-radius:12px;">',
                                unsafe_allow_html=True)
                except Exception:
                    pass

    question = st.chat_input("Ask about your data…")
    if question is None and st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None

    if question:
        st.session_state.history.append({"role": "user", "content": question})
        store = get_store()
        if st.session_state.session_id is None:
            st.session_state.session_id = store.new_session(title=question[:60])
        store.add_message(st.session_state.session_id, "user", question)
        with st.chat_message("user"):
            st.markdown(question)

        agent_ctx = AgentContext(
            tables=st.session_state.tables,
            active_table=st.session_state.active_table,
            history=[ChatMessage(m["role"], m["content"]) for m in st.session_state.history[:-1]
                     if m["role"] in ("user", "assistant")],
        )
        # the user message is already appended (history[:-1] above feeds the
        # agent context) — the exchange persists before the fresh render, so
        # the fresh fixer's widget key (fix_code_{len-1}) matches the index
        # the history loop uses on the next rerun
        _fresh_mi = len(st.session_state.history)   # index of the reply to come

        with st.chat_message("assistant"):
            status = st.status("Thinking…", expanded=True)
            response = None
            try:
                response = get_agent().analyze(
                    question, agent_ctx,
                    progress=lambda step, detail="": status.update(
                        label=f"{step}: {detail}" if detail else step) or None,
                )
            except OllamaError as e:
                st.error(f"Model error: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
            finally:
                status.update(state="complete", label="done")

            if response is not None:
                if response.ok:
                    st.markdown(response.answer or "(no explanation returned)")
                else:
                    st.error(response.error)
                # the receipt: what ran, how many tries, what actually checked it
                if response.ok:
                    glyph = {"passed": "✓", "failed": "✗"}
                    checks = "".join(
                        f'<span>{glyph.get(g.status, "")} {g.name}</span>'
                        for g in response.guards)
                    if response.verified:
                        verdict = "verified ✓"
                    elif any(g.status == "failed" for g in response.guards):
                        verdict = "unverified — a check objected"
                    else:
                        verdict = "unverified — nothing was checked"
                    st.markdown(
                        f'<div class="receipt"><span class="r-model">model '
                        f'{response.model_used or "?"}</span>'
                        f'<span>tries <b>{response.attempts_used}</b></span>'
                        f'{checks}'
                        f'<span>{verdict}</span></div>',
                        unsafe_allow_html=True)
                if response.code:
                    lang = "python" if "import" in response.code or "result_df" in response.code else "sql"
                    st.code(response.code, language=lang)
                if response.stdout and response.stdout.strip():
                    with st.expander("Execution output"):
                        st.text(response.stdout)
                # styled matplotlib render of every chart spec
                from jane.charts import render_all
                try:
                    pngs = render_all(response.charts)
                except Exception:
                    pngs = []
                for i, png in enumerate(pngs):
                    b64 = __import__("base64").b64encode(png).decode()
                    st.markdown(
                        f'<img src="data:image/png;base64,{b64}" style="max-width:100%;border-radius:12px;" alt="chart">',
                        unsafe_allow_html=True)
                    st.download_button(label="Download chart as PNG", data=png,
                                       mime="image/png", file_name=f"jane_chart_{i+1}.png",
                                       key=f"dl_chart_{st.session_state.session_id}_{len(st.session_state.history)}_{i}")
                if response.charts and not pngs:
                    st.caption("(chart spec could not be rendered)")
                for fig_uri in response.figures:
                    st.markdown(f'<img src="{fig_uri}" style="max-width:100%;border-radius:12px;">',
                                unsafe_allow_html=True)
                for t in response.tables:
                    st.dataframe(pd.DataFrame(t["rows"], columns=t["columns"]),
                                 use_container_width=True, hide_index=True)

                store.add_message(st.session_state.session_id, "assistant",
                                  json.dumps({
                                      "content": response.answer or response.error or "",
                                      "code": response.code, "tables": response.tables,
                                      "charts": response.charts, "ok": response.ok,
                                      "error": response.error,
                                  }))
                st.session_state.history.append({
                    "role": "assistant",
                    "content": response.answer if response.ok else (response.error or ""),
                    "code": response.code,
                    "ok": response.ok,
                    "tables": response.tables,
                    "charts": response.charts,
                    "figures": response.figures,
                })
                # the fixer renders in THIS rerun too, not just the next one:
                # _fresh_mi was computed before the reply was appended, so the
                # key matches what the history loop will use after rerun
                if not response.ok and response.code:
                    _render_fixer(st.session_state.history[-1], _fresh_mi)
                st.session_state["_scroll_to_end"] = True

with tab_manual:
    st.caption("Edit the data or run ad-hoc SQL when the agent misses something.")
    if st.session_state.tables:
        left, right = st.columns(2)
        with left:
            st.subheader("Editable data")
            edited = st.data_editor(
                st.session_state.tables[st.session_state.active_table],
                key="manual_editor", use_container_width=True, hide_index=True, height=420,
            )
            if st.button("Save edits as new active table", use_container_width=True):
                name = st.session_state.active_table + "_edited"
                st.session_state.tables[name] = edited
                st.session_state.active_table = name
                st.success(f"Saved as **{name}** — now the active table.")
        with right:
            st.subheader("Ad-hoc SQL")
            default_sql = f"SELECT * FROM \"{st.session_state.active_table}\" LIMIT 50"
            sql = st.text_area("Query", value=st.session_state.get("adhoc_sql", default_sql), height=180,
                               key="adhoc_sql_input")
            if st.button("Run SQL", type="primary", use_container_width=True):
                from jane.engine import SQLEngine
                eng = SQLEngine(st.session_state.tables)
                try:
                    res = eng.execute(sql)
                    if res.ok:
                        st.session_state["adhoc_result"] = {
                            "columns": res.columns, "rows": res.rows, "sql": sql,
                            "truncated": res.truncated,
                        }
                    else:
                        st.error(res.error)
                finally:
                    eng.close()
            res = st.session_state.get("adhoc_result")
            if res and res["sql"] == sql:
                st.dataframe(pd.DataFrame(res["rows"], columns=res["columns"]),
                             use_container_width=True, hide_index=True)
                if res.get("truncated"):
                    st.caption("Result truncated — refine the query.")

with tab_profile:
    st.caption("An analyst's first briefing — computed locally, no model involved.")
    if st.session_state.tables:
        from jane.insights import find_insights, profile_summary_text, profile_table

        reports = [profile_table(n, df) for n, df in st.session_state.tables.items()]
        insights = find_insights(st.session_state.tables)

        st.subheader("Notable patterns")
        if insights:
            for ins in insights[:8]:
                icon = {"missing": "◌", "spike": "▲", "drop": "▼", "outlier": "◆",
                        "dominance": "▣", "correlation": "⇄"}.get(ins.kind, "•")
                st.markdown(
                    f"**{icon} {ins.title}**  \n{ins.detail}")
        else:
            st.info("No standout patterns detected — this dataset is quite even.")

        st.subheader("Full profile")
        st.code(profile_summary_text(reports), language=None)
        if st.button("Ask Jane to interpret this profile", use_container_width=True):
            st.session_state.pending_question = (
                f"Interpret this data profile and tell me the 3 most important things "
                f"to know before analyzing {st.session_state.active_table}:\n\n"
                + profile_summary_text(reports))

with tab_prep:
    st.caption("Standard preprocessing — deterministic pandas, no model. "
               "Every action writes a NEW table; the original stays untouched.")
    if st.session_state.tables:
        from jane.preprocess import clean_default, run_ops

        dfp = st.session_state.tables[st.session_state.active_table]
        actions: dict[str, tuple[str, dict]] = {
            "Fill missing (median / Unknown / ffill)": ("fill_missing", {}),
            "Drop rows with missing": ("fill_missing", {"strategy": "drop"}),
            "Drop duplicate rows": ("dedupe", {}),
            "Clean whitespace (cells + headers)": ("clean_text", {}),
            "Convert date-like text to datetime": ("to_datetime", {}),
            "Scale numeric (z-score)": ("scale", {"method": "zscore"}),
            "Scale numeric (min-max)": ("scale", {"method": "minmax"}),
            "Scale numeric (robust)": ("scale", {"method": "robust"}),
            "One-hot encode low-cardinality text": ("encode", {}),
            "Clamp outliers (IQR fence)": ("clamp_outliers", {}),
        }
        flat = list(actions.items())
        for start in range(0, len(flat), 3):
            row = st.columns(3)
            for col, (label, spec) in zip(row, flat[start:start + 3]):
                with col:
                    if st.button(label, key=f"prep_{label}", use_container_width=True):
                        try:
                            pr = run_ops(dfp, [spec])
                            base = f"{st.session_state.active_table}_prep"
                            new_name, k = base, 2
                            while new_name in st.session_state.tables:
                                new_name = f"{base}_{k}"
                                k += 1
                            st.session_state.tables[new_name] = pr.df
                            st.session_state.active_table = new_name
                            st.session_state["prep_log"] = (new_name, pr.log)
                            st.toast(f"Saved as {new_name}", icon="✅")
                            st.rerun()
                        except Exception as e:
                            st.error(f"{label} failed: {e}")

        if st.button("✨ Tidy preset: clean text → dates → fill missing → dedupe",
                     use_container_width=True):
            pr = clean_default(dfp)
            base = f"{st.session_state.active_table}_tidy"
            new_name, k = base, 2
            while new_name in st.session_state.tables:
                new_name = f"{base}_{k}"
                k += 1
            st.session_state.tables[new_name] = pr.df
            st.session_state.active_table = new_name
            st.session_state["prep_log"] = (new_name, pr.log)
            st.toast(f"Saved as {new_name}", icon="✨")
            st.rerun()

        log_entry = st.session_state.get("prep_log")
        if log_entry:
            log_name, log_lines = log_entry
            with st.expander(f"Prep log — {log_name}", expanded=False):
                for line in log_lines:
                    st.markdown(f"- {line}")

with tab_schema:
    if st.session_state.tables:
        for tname, df in st.session_state.tables.items():
            st.subheader(tname)
            st.caption(f"{len(df):,} rows × {len(df.columns)} cols")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(df.dtypes.rename("dtype").to_frame(), use_container_width=True)
            with col2:
                st.dataframe(df.head(20), use_container_width=True, hide_index=True)
