"""Shared typed results for the whole pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


class AnalysisCancelled(Exception):
    """The user stopped this run. Nothing is shipped and nothing is stored."""


@dataclass
class GuardOutcome:
    """One check's verdict on one answer — the receipt is built from these."""

    name: str
    status: str          # "passed" | "failed" | "skipped"
    detail: str = ""     # why, in plain words — shown under the receipt


@dataclass
class AnalysisResult:
    ok: bool
    answer: str = ""
    code: Optional[str] = None
    stdout: str = ""
    tables: list = field(default_factory=list)      # list[list[list[Any]]]
    columns: list = field(default_factory=list)     # list[list[str]]
    charts: list = field(default_factory=list)      # list[dict] Vega-Lite specs
    figures: list = field(default_factory=list)     # list[str] png data-URIs (sandbox matplotlib)
    attempts_used: int = 0
    error: Optional[str] = None
    model_used: Optional[str] = None
    guards: list = field(default_factory=list)      # list[GuardOutcome] what actually ran
    repairs: int = 0                                # model calls spent re-writing the prose

    def chart_payload(self) -> list[dict]:
        """Charts as {spec, chart_id} dicts, specs already JSON-compatible."""
        return self.charts or []

    @property
    def verified(self) -> bool:
        """Checked against real rows and nothing objected — never a vacuous tick.

        A guard that could not run reports "skipped", and a skipped-only report
        is not verification: with no rows there is nothing to trace a number to.
        """
        return bool(self.ok
                    and any(g.status == "passed" for g in self.guards)
                    and not any(g.status == "failed" for g in self.guards))
