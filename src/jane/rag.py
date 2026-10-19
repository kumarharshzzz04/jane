"""Lightweight local RAG — retrieval over user notes and data profiles.

Scope (per the architecture contract): RAG may supply *context* — the user's
own notes, glossaries, and column documentation — to help the model write
better SQL. It never supplies numbers: every figure still comes from
executing code against the tables.

Pure stdlib TF-IDF (no network, no embeddings server): good enough at the
note-corpus scale a single analyst produces, zero moving parts.
"""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import settings
from .logging_setup import get_logger

log = get_logger("jane.rag")

_WORD_RE = re.compile(r"[a-z0-9_]{2,}")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was", "be",
    "for", "on", "with", "as", "by", "at", "it", "this", "that", "from", "we",
    "you", "i", "not", "but", "has", "have", "had", "do", "does", "did",
}


@dataclass
class Doc:
    doc_id: str
    text: str
    source: str  # "note" | "profile"


@dataclass
class RagIndex:
    docs: list[Doc] = field(default_factory=list)
    _tf: list[dict[str, float]] = field(default_factory=list)
    _df: dict[str, int] = field(default_factory=dict)
    _built_at: float = 0.0

    def add(self, doc_id: str, text: str, source: str) -> None:
        self.docs.append(Doc(doc_id=doc_id, text=text, source=source))

    def build(self) -> None:
        self._tf = []
        self._df = {}
        for doc in self.docs:
            tf: dict[str, float] = {}
            for tok in _WORD_RE.findall(doc.text.lower()):
                if tok in _STOP:
                    continue
                tf[tok] = tf.get(tok, 0.0) + 1.0
            # sublinear scaling: long notes shouldn't dominate
            for k in tf:
                tf[k] = 1.0 + math.log(tf[k])
            self._tf.append(tf)
            for tok in tf:
                self._df[tok] = self._df.get(tok, 0) + 1
        self._built_at = time.time()

    def search(self, query: str, k: int = 3) -> list[tuple[Doc, float]]:
        if not self.docs:
            return []
        if not self._tf:
            self.build()
        q = {}
        for tok in _WORD_RE.findall(query.lower()):
            if tok in _STOP:
                continue
            q[tok] = q.get(tok, 0.0) + 1.0
        if not q:
            return []
        n = len(self.docs)
        scored: list[tuple[float, int]] = []
        for i, tf in enumerate(self._tf):
            s = 0.0
            for tok, qf in q.items():
                if tok in tf:
                    idf = math.log((n + 1) / (self._df.get(tok, 0) + 0.5))
                    s += qf * tf[tok] * idf
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        return [(self.docs[i], s) for s, i in scored[:k]]

    def stale_for(self, tables: dict, notes: list[str]) -> bool:
        """True when docs don't match current tables/notes."""
        return len(self.docs) != len(notes) + sum(1 for _ in tables)


def build_index(tables: dict, notes: list[str]) -> RagIndex:
    """Index user notes + auto-generated column profiles per table."""
    from .schema import describe_table

    idx = RagIndex()
    for i, note in enumerate(notes):
        if note.strip():
            idx.add(f"note-{i}", note, "note")
    for tname, df in tables.items():
        try:
            info = describe_table(tname, df)
        except Exception:
            continue
        col_lines = [f"Table {tname} has {info.row_count} rows."]
        for c in info.columns:
            col_lines.append(
                f"Column {c['name']} in {tname}: type {c['dtype']}, "
                f"{c['nulls']} nulls, sample values {c['sample'][:3]}"
            )
        idx.add(f"profile-{tname}", " ".join(col_lines), "profile")
    idx.build()
    return idx


def retrieve_context(question: str, tables: dict, notes: list[str],
                     k: int = 4) -> str:
    """Return a compact context block for the prompt (may be empty)."""
    try:
        idx = build_index(tables, notes)
        hits = idx.search(question, k=k)
    except Exception as e:
        log.warning("rag retrieval failed: %s", e)
        return ""
    if not hits:
        return ""
    lines = []
    for doc, score in hits:
        body = doc.text if doc.source == "profile" else doc.text
        if len(body) > 700:
            body = body[:700] + "…"
        tag = "USER NOTE" if doc.source == "note" else "DATA PROFILE"
        lines.append(f"[{tag} | relevance {score:.2f}]\n{body}")
    return "\n\n".join(lines)


def notes_path() -> Path:
    p = Path(settings.history_db).parent / "jane_notes.json"
    return p


def load_notes() -> list[str]:
    p = notes_path()
    if not p.exists():
        return []
    try:
        return [str(n) for n in json.loads(p.read_text(encoding="utf-8"))]
    except Exception:
        return []


def save_notes(notes: list[str]) -> None:
    p = notes_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(notes, ensure_ascii=False, indent=1), encoding="utf-8")
