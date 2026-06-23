"""Real-world dirty-data robustness (v0.9 hardening).

Exercises the robust ingestion door (profiler.ingest / read_table) on the messes
real datasets actually have — non-UTF-8 encodings, a BOM, non-comma delimiters,
numbers stored as text (thousands/currency/percent), and stray missing-tokens —
and asserts the engine reads + types them correctly, end-to-end, without crashing.

Regression guards (the conservative half): genuine categoricals / id codes must
NOT be coerced into numbers.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from researchforge.catalog import Catalog
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset
from researchforge.profiler.profile import read_table


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #
def test_utf8_bom_does_not_pollute_first_column(tmp_path: Path) -> None:
    csv = tmp_path / "bom.csv"
    csv.write_bytes("region,value\nnorth,1\nsouth,2\n".encode("utf-8-sig"))
    df = read_table(csv)
    assert list(df.columns) == ["region", "value"]   # no '﻿' on "region"
    assert df.attrs.get("rf_encoding") == "utf-8-sig"


def test_gbk_chinese_file_reads_without_crash(tmp_path: Path) -> None:
    csv = tmp_path / "gbk.csv"
    csv.write_bytes("地区,数值\n华北,10\n华南,20\n".encode("gb18030"))
    df = read_table(csv)
    assert list(df.columns) == ["地区", "数值"]
    assert df["地区"].tolist() == ["华北", "华南"]
    assert df.attrs.get("rf_encoding") == "gb18030"


# --------------------------------------------------------------------------- #
# Delimiter sniffing
# --------------------------------------------------------------------------- #
def test_semicolon_delimited_not_one_column(tmp_path: Path) -> None:
    csv = tmp_path / "semi.csv"
    csv.write_text("a;b;c\n1;2;3\n4;5;6\n", encoding="utf-8")
    df = read_table(csv)
    assert list(df.columns) == ["a", "b", "c"]
    assert df.shape == (2, 3)
    assert df.attrs.get("rf_sep") == ";"


def test_tab_delimited(tmp_path: Path) -> None:
    csv = tmp_path / "tabs.tsv"
    csv.write_text("x\ty\n1\t2\n3\t4\n", encoding="utf-8")
    df = read_table(csv)
    assert list(df.columns) == ["x", "y"]
    assert df.shape == (2, 2)


# --------------------------------------------------------------------------- #
# Numeric coercion of text columns
# --------------------------------------------------------------------------- #
def test_thousands_separator_coerced_to_continuous(tmp_path: Path) -> None:
    csv = tmp_path / "thou.csv"
    # quoted so the inner comma is a thousands sep, not a delimiter
    csv.write_text(
        'city,pop\nA,"1,234"\nB,"5,678"\nC,"9,012"\nD,"3,456"\nE,"7,890"\n',
        encoding="utf-8",
    )
    df = read_table(csv)
    assert pd.api.types.is_numeric_dtype(df["pop"])
    assert df["pop"].tolist() == [1234.0, 5678.0, 9012.0, 3456.0, 7890.0]
    assert "pop" in df.attrs["rf_coercions"]

    fp = profile_dataset(csv)
    assert fp.column("pop").kind in {"continuous", "count", "id"}
    assert any(i.kind == "coerced_numeric" and i.column == "pop" for i in fp.issues)


def test_currency_symbols_coerced(tmp_path: Path) -> None:
    csv = tmp_path / "cur.csv"
    csv.write_text("item,price\na,$10\nb,$20\nc,$30\nd,$40\n", encoding="utf-8")
    df = read_table(csv)
    assert pd.api.types.is_numeric_dtype(df["price"])
    assert df["price"].tolist() == [10.0, 20.0, 30.0, 40.0]


def test_percent_coerced_to_proportion(tmp_path: Path) -> None:
    csv = tmp_path / "pct.csv"
    csv.write_text("grp,rate\na,12%\nb,50%\nc,8%\nd,100%\n", encoding="utf-8")
    df = read_table(csv)
    assert pd.api.types.is_numeric_dtype(df["rate"])
    assert df["rate"].tolist() == [0.12, 0.50, 0.08, 1.00]


def test_stray_missing_tokens_become_nan_in_numeric_col(tmp_path: Path) -> None:
    csv = tmp_path / "na.csv"
    csv.write_text(
        "id,score\n1,10\n2,-\n3,20\n4,missing\n5,30\n6,40\n7,50\n8,60\n",
        encoding="utf-8",
    )
    df = read_table(csv)
    assert pd.api.types.is_numeric_dtype(df["score"])
    assert int(df["score"].isna().sum()) == 2          # "-" and "missing"
    assert df["score"].dropna().tolist() == [10, 20, 30, 40, 50, 60]


# --------------------------------------------------------------------------- #
# Conservative: genuine categoricals / ids must NOT be coerced
# --------------------------------------------------------------------------- #
def test_real_categorical_not_coerced(tmp_path: Path) -> None:
    csv = tmp_path / "cat.csv"
    csv.write_text("g,y\nx,1\ny,2\nz,3\nx,4\ny,5\n", encoding="utf-8")
    df = read_table(csv)
    assert not pd.api.types.is_numeric_dtype(df["g"])
    assert "g" not in df.attrs["rf_coercions"]


def test_id_codes_not_coerced(tmp_path: Path) -> None:
    csv = tmp_path / "ids.csv"
    csv.write_text("code,v\nA1,1\nA2,2\nB7,3\nC9,4\n", encoding="utf-8")
    df = read_table(csv)
    assert not pd.api.types.is_numeric_dtype(df["code"])
    assert "code" not in df.attrs["rf_coercions"]


def test_mostly_text_minority_numbers_not_coerced(tmp_path: Path) -> None:
    # 2/6 numeric (<90%) -> left as categorical, not turned numeric
    csv = tmp_path / "mixed.csv"
    csv.write_text("v\napple\n2\nbanana\ncherry\n5\ndate\n", encoding="utf-8")
    df = read_table(csv)
    assert not pd.api.types.is_numeric_dtype(df["v"])


# --------------------------------------------------------------------------- #
# Disclosure + end-to-end
# --------------------------------------------------------------------------- #
def test_high_cardinality_flagged(tmp_path: Path) -> None:
    csv = tmp_path / "hc.csv"
    df = pd.DataFrame({"note": [f"free text {i}" for i in range(100)],
                       "v": list(range(100))})
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    assert any(i.kind == "high_cardinality" and i.column == "note" for i in fp.issues)


def test_end_to_end_run_on_dirty_csv(tmp_path: Path) -> None:
    # two numbers-as-text columns + a real grouping label; must profile AND run.
    csv = tmp_path / "dirty_end2end.csv"
    rows = ["grp,sales,growth"]
    for i in range(30):
        rows.append(f"{'A' if i % 2 else 'B'},\"{1000 + i * 13:,}\",{i % 25}%")
    csv.write_text("\n".join(rows) + "\n", encoding="utf-8")

    fp = profile_dataset(csv)
    # both messy numeric columns recovered as numeric
    assert fp.column("sales").kind in {"continuous", "count", "id"}
    assert fp.column("growth").kind in {"continuous", "count"}

    entry = Catalog.load().by_id("correlation_matrix")
    res = run_analysis(fp, entry, output_root=str(tmp_path / "out"))
    # ran end-to-end on the dirty file without crashing and produced a summary
    assert isinstance(res.summary, str) and res.summary
    assert Path(res.output_dir).exists()
