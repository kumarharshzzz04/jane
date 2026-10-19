"""The receipt must be measured, never asserted.

Two failures this pins, both seen live on retail_sales:
  * an answer with NO result rows was stamped "guards passed 6/6 · verified ✓"
    while its prose carried an invented figure;
  * the receipt's fraction was len(guards)/len(guards) — always 100%, whatever
    ran.
"""
from jane.agent.verification import guard_report
from jane.types import AnalysisResult

QUESTION = "Break down revenue by region - who leads, and by how much?"
# the real totals: the true margin is 1.5%, the answer below invents 12%
ROWS = [["south", 22663.59], ["east", 22333.27]]
COLUMNS = ["region", "total_revenue"]


def _result(answer: str, rows) -> AnalysisResult:
    return AnalysisResult(ok=True, answer=answer,
                          tables=[{"name": "query_result",
                                   "columns": COLUMNS, "rows": rows}])


def test_no_rows_means_nothing_was_checked():
    """The vacuous case: no rows ⇒ skipped, and skipped is not verified."""
    res = _result("VERDICT - the south region leads by 12%.", [])
    res.guards = guard_report(res.answer, res, QUESTION)
    assert [g.status for g in res.guards] == ["skipped"]
    assert res.verified is False


def test_verified_requires_a_real_check():
    """With rows to trace against, the same claim is caught and not verified."""
    res = _result("VERDICT - the south region leads by 12%.", ROWS)
    res.guards = guard_report(res.answer, res, QUESTION)
    truth = next(g for g in res.guards if g.name == "truth")
    assert truth.status == "failed"
    assert "no source" in truth.detail
    assert res.verified is False      # a failed check can never be a tick


def test_clean_answer_is_verified_and_says_what_it_traced():
    res = _result("South leads with 22663.59 revenue.", ROWS)
    res.guards = guard_report(res.answer, res, QUESTION)
    assert res.verified is True
    truth = next(g for g in res.guards if g.name == "truth")
    assert truth.status == "passed"
    assert truth.detail == "1 number(s), each found in the result"


def test_numberless_answer_claims_no_tracing():
    """A pass must describe what it checked — 'no numbers' is not 'rows traced'."""
    res = _result("South has the largest share of the business.", ROWS)
    res.guards = guard_report(res.answer, res, QUESTION)
    truth = next(g for g in res.guards if g.name == "truth")
    assert truth.status == "passed"
    assert "no numbers" in truth.detail


def test_report_never_names_a_guard_it_could_not_run():
    """Only checks that actually executed may appear in the receipt."""
    res = _result("", [])
    res.guards = guard_report("", res, QUESTION)
    assert {g.name for g in res.guards} <= {"truth", "contradiction"}


def test_give_up_result_is_never_verified():
    assert AnalysisResult(ok=False, error="gave up").verified is False
