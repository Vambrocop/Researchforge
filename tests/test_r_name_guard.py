"""Tests for the centralized R column-name identifier guard (rbridge.is_r_safe_name /
r_names_safe) and the 3 call sites that previously reached R without checking it:
ecology.differential_abundance (aldex2), meta.meta_regression, configurational.panel_qca.

All behavioral tests here assert the guard fires BEFORE any R call, so they run and
pass with or without R/packages installed (no skipif needed).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import rbridge, run_analysis
from researchforge.profiler import profile_dataset


# --- pure helper unit tests -------------------------------------------------


def test_is_r_safe_name_accepts_plain_identifiers() -> None:
    assert rbridge.is_r_safe_name("age") is True
    assert rbridge.is_r_safe_name("ok.name_1") is True


def test_is_r_safe_name_rejects_injection_attempt() -> None:
    assert rbridge.is_r_safe_name('x"]]);system("calc")') is False


def test_is_r_safe_name_rejects_space() -> None:
    assert rbridge.is_r_safe_name("has space") is False


def test_is_r_safe_name_rejects_leading_digit() -> None:
    assert rbridge.is_r_safe_name("2bad") is False


def test_r_names_safe_all_safe() -> None:
    assert rbridge.r_names_safe(["a", "b"]) is True


def test_r_names_safe_one_bad() -> None:
    assert rbridge.r_names_safe(["a", "bad name"]) is False


def test_r_names_safe_skips_empties() -> None:
    assert rbridge.r_names_safe(["a", None, ""]) is True


# --- (a) ecology.differential_abundance, da_method=aldex2 -------------------


def _da_entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="differential_abundance",
        method="Differential abundance (CLR + Wilcoxon)",
        domain="microbiology",
        family="ecology",
        goal="explain",
        preconditions=Precondition(min_count_cols=2, requires_group=True, min_rows=10),
    )


def test_differential_abundance_aldex2_bad_taxon_name_degrades(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    rows = []
    for grp in ("ctrl", "case"):
        for _ in range(18):
            base = rng.integers(8, 20, 3).astype(int)
            rows.append({"group": grp, "bad taxon": int(base[0]), "otu1": int(base[1]), "otu2": int(base[2])})
    df = pd.DataFrame(rows)
    csv = tmp_path / "abund.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _da_entry(), output_root=str(tmp_path / "o"), config={"da_method": "aldex2"}
    )
    assert "标识符式列名" in res.summary
    assert "CLR+Mann-Whitney" in res.summary  # degraded, not crashed


# --- (b) meta.meta_regression -----------------------------------------------


def _mr_entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="meta_regression", method="Meta-regression", domain="statistics",
        family="meta", goal="synthesize",
        preconditions=Precondition(requires_effect_sizes=True, min_rows=5),
    )


def test_meta_regression_bad_moderator_name_degrades(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    k = 10
    df = pd.DataFrame(
        {
            "study": [f"S{i}" for i in range(k)],
            "yi": rng.normal(0.3, 0.1, k),
            "sei": rng.uniform(0.1, 0.2, k),
            "bad name": rng.uniform(0, 10, k),  # space -> R-unsafe moderator
        }
    )
    csv = tmp_path / "mr.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _mr_entry(), output_root=str(tmp_path / "o"))
    assert "Meta 回归失败" in res.summary
    assert "标识符式" in res.summary


# --- (c) configurational.panel_qca ------------------------------------------


def _pqca_entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="panel_qca", method="Panel fsQCA", domain="social science",
        family="configurational", goal="explain",
        preconditions=Precondition(is_panel=True, min_continuous=3, min_rows=20),
    )


def test_panel_qca_bad_unit_name_degrades(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    rows = []
    for i in range(12):
        for p in range(1, 7):
            A, B, C = rng.uniform(0, 10), rng.uniform(0, 10), rng.uniform(0, 10)
            Y = max(A, B) * 0.8 + rng.uniform(0, 2)
            rows.append({"bad unit": f"u{i}", "period": p, "A": round(A, 2),
                         "B": round(B, 2), "C": round(C, 2), "Y": round(Y, 2)})
    df = pd.DataFrame(rows)
    csv = tmp_path / "pq.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    assert fp.is_panel and fp.unit_col == "bad unit"  # sanity: panel detected despite the name
    res = run_analysis(
        fp, _pqca_entry(), output_root=str(tmp_path / "o"),
        config={"outcome": "Y", "conditions": ["A", "B", "C"]},
    )
    assert "面板 QCA 失败" in res.summary
    assert "标识符式" in res.summary
