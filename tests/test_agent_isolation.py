"""Agent isolation (spec §5): two agents, interleaved tables, no
cross-contamination. Uses a stub client so no Ollama is needed — pins a
property the Phase-1 refactor must not break (agent state lives per-call).
"""
import threading

import pandas as pd
import pytest

from jane.agent import AgentContext, AnalystAgent
from jane.types import AnalysisCancelled, AnalysisResult


class StubClient:
    """Replies from a per-instance script — any state leakage between agents
    shows up as the wrong scripted answer coming back."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.model_override = None

    def has_model(self, name):
        return True

    def tags(self):
        return [{"name": "stub:latest"}]

    def chat(self, messages, model, temperature=None):
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        if self.on_call is not None:
            self.on_call(self.calls)      # lets a test cancel mid-run
        import json
        return json.dumps(reply)

    on_call = None


def _reply(table_name):
    return {
        "plan": f"count rows of {table_name}",
        "language": "sql",
        "code": f"SELECT COUNT(*) AS n FROM \"{table_name}\"",
        "explanation": f"table {table_name} counted",
    }


def _tables(n_rows, tag):
    return {f"t_{tag}": pd.DataFrame({"x": range(n_rows)})}


def test_two_agents_no_cross_contamination():
    a = AnalystAgent(client=StubClient([_reply("t_a")]))
    b = AnalystAgent(client=StubClient([_reply("t_b")]))

    ctx_a = AgentContext(tables=_tables(10, "a"), active_table="t_a")
    ctx_b = AgentContext(tables=_tables(500, "b"), active_table="t_b")

    ra = a.analyze("how many rows in t_a", ctx_a)
    rb = b.analyze("how many rows in t_b", ctx_b)

    assert ra.ok and rb.ok, (ra.error, rb.error)
    rows_a = dict(zip(ra.tables[0]["columns"], ra.tables[0]["rows"][0]))
    rows_b = dict(zip(rb.tables[0]["columns"], rb.tables[0]["rows"][0]))
    assert rows_a["n"] == 10, f"agent A saw B's table: {rows_a}"
    assert rows_b["n"] == 500, f"agent B saw A's table: {rows_b}"


def test_same_agent_sequential_calls_independent():
    """One agent, two different contexts back-to-back — results stay per-call."""
    agent = AnalystAgent(client=StubClient([_reply("t_x"), _reply("t_y")]))
    r1 = agent.analyze("count", AgentContext(
        tables=_tables(7, "x"), active_table="t_x"))
    r2 = agent.analyze("count", AgentContext(
        tables=_tables(77, "y"), active_table="t_y"))
    rows1 = dict(zip(r1.tables[0]["columns"], r1.tables[0]["rows"][0]))
    rows2 = dict(zip(r2.tables[0]["columns"], r2.tables[0]["rows"][0]))
    assert rows1["n"] == 7 and rows2["n"] == 77


def test_shared_table_dict_not_mutated():
    """analyze() must not mutate the caller's tables dict (Prep/fixer rely on it)."""
    tables = _tables(5, "s")
    before = {k: len(v) for k, v in tables.items()}
    agent = AnalystAgent(client=StubClient([_reply("t_s")]))
    agent.analyze("count", AgentContext(tables=tables, active_table="t_s"))
    after = {k: len(v) for k, v in tables.items()}
    assert before == after and list(tables.keys()) == ["t_s"]


# ---- cancel: a stopped run ships nothing and stops retrying ---------------

def test_cancel_before_the_run_calls_the_model_zero_times():
    client = StubClient([_reply("t_a")])
    agent = AnalystAgent(client=client)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(AnalysisCancelled):
        agent.analyze("how many rows", AgentContext(
            tables=_tables(10, "a"), active_table="t_a"), cancel=cancel)
    assert client.calls == 0


def test_cancel_during_the_run_stops_the_retry_budget():
    """The model is asked once; the user stops it; no second attempt is made
    and no result is shipped."""
    cancel = threading.Event()
    client = StubClient([_reply("t_a")])
    client.on_call = lambda n: cancel.set()      # stop the moment the model answers
    agent = AnalystAgent(client=client)
    with pytest.raises(AnalysisCancelled):
        agent.analyze("how many rows", AgentContext(
            tables=_tables(10, "a"), active_table="t_a"), cancel=cancel)
    assert client.calls == 1


def test_no_cancel_event_runs_normally():
    agent = AnalystAgent(client=StubClient([_reply("t_a")]))
    res = agent.analyze("how many rows", AgentContext(
        tables=_tables(10, "a"), active_table="t_a"), cancel=threading.Event())
    assert res.ok, res.error
