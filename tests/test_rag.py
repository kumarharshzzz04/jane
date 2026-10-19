"""Tests for the local RAG module: TF-IDF retrieval over notes + profiles."""
import pandas as pd

from jane.rag import build_index, load_notes, retrieve_context, save_notes


TABLES = {
    "orders": pd.DataFrame({
        "order_id": [1, 2, 3],
        "region": ["EMEA", "APAC", "EMEA"],
        "net_amount": [100.0, 250.0, 80.0],
    })
}


def test_note_retrieved_for_matching_query():
    notes = ["EMEA stands for Europe, Middle East and Africa; it excludes the UK."]
    ctx = retrieve_context("How much revenue did EMEA generate?", TABLES, notes)
    assert "EMEA" in ctx
    assert "USER NOTE" in ctx


def test_profile_retrieved_for_column_query():
    ctx = retrieve_context("What does net_amount look like? Any nulls?", TABLES, [])
    assert "DATA PROFILE" in ctx
    assert "net_amount" in ctx


def test_no_noise_for_unrelated_query():
    notes = ["Fiscal year ends in March."]
    ctx = retrieve_context("How many orders are there?", TABLES, notes)
    # the orders profile may match, but the unrelated note must not
    assert "Fiscal year" not in ctx


def test_empty_inputs_return_empty_context():
    assert retrieve_context("anything", {}, []) == ""
    assert retrieve_context("", TABLES, []) == ""


def test_context_never_contains_table_values():
    # RAG supplies schema/notes, not data values — numbers still come from code
    ctx = retrieve_context("net amount totals", TABLES, [])
    assert "250.0" not in ctx


def test_notes_roundtrip(tmp_path, monkeypatch):
    import jane.rag as rag
    monkeypatch.setattr(rag, "notes_path", lambda: tmp_path / "notes.json")
    save_notes(["alpha rule", "beta rule"])
    assert load_notes() == ["alpha rule", "beta rule"]
    save_notes([])
    assert load_notes() == []


def test_build_index_doc_counts():
    idx = build_index(TABLES, ["note one", "note two"])
    # 2 notes + 1 profile (one table)
    assert len(idx.docs) == 3
