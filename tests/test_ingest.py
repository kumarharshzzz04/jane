"""Tests for ingestion: CSV, JSON, header sniffing. (No Excel: needs openpyxl files.)"""
import json

import pandas as pd
import pytest

from jane.ingest import _read_csv_like, load_file, sniff_header_row


def test_read_csv_basic(tmp_path):
    p = tmp_path / "sales.csv"
    p.write_text("region,amount\nnorth,10\nsouth,20\n", encoding="utf-8")
    loaded = load_file(p)
    df = loaded.primary
    assert list(df.columns) == ["region", "amount"]
    assert len(df) == 2
    assert df["amount"].sum() == 30


def test_read_csv_latin1(tmp_path):
    p = tmp_path / "euro.csv"
    p.write_bytes("name,city\nJosé,Paris\n".encode("latin-1"))
    df = load_file(p).primary
    assert df.iloc[0]["name"] == "José"


def test_read_tsv(tmp_path):
    p = tmp_path / "t.tsv"
    p.write_text("a\tb\n1\tx\n2\ty\n", encoding="utf-8")
    df = load_file(p).primary
    assert list(df.columns) == ["a", "b"]


def test_load_json_flat(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps([{"x": 1, "y": "a"}, {"x": 2, "y": "b"}]), encoding="utf-8")
    df = load_file(p).primary
    assert len(df) == 2
    assert set(df.columns) == {"x", "y"}


def test_unsupported_extension(tmp_path):
    p = tmp_path / "f.exe"
    p.write_bytes(b"MZ")
    with pytest.raises(ValueError):
        load_file(p)


def test_sniff_header_row_first():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert sniff_header_row(df) == 0


def test_sniff_header_row_later():
    df = pd.DataFrame(
        [["Report X", "", ""], ["", "", ""], ["region", "amount", "note"], ["north", 10, "ok"]],
        columns=[0, 1, 2],
    )
    assert sniff_header_row(df) == 2


def test_read_csv_sniff_separator_fallback(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("a;b\n1;2\n", encoding="utf-8")
    df = _read_csv_like(p.read_bytes(), "s.csv")
    assert list(df.columns) == ["a", "b"]
