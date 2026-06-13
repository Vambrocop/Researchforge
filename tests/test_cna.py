"""Tests for the Coincidence Analysis (cna) executor branch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis, rbridge
from researchforge.profiler import profile_dataset

_HAS_CNA = rbridge.r_available() and rbridge.r_package_available("cna")


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="cna", method="CNA", domain="social science", family="configurational",
        goal="explain", preconditions=Precondition(min_rows=10),
    )


@pytest.mark.skipif(not _HAS_CNA, reason="R cna not available")
def test_cna_crisp_recovers_structure(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    n = 50
    A = rng.integers(0, 2, n)
    B = rng.integers(0, 2, n)
    C = rng.integers(0, 2, n)
    D = rng.integers(0, 2, n)
    Y = ((A & B) | C).astype(int)
    flip = rng.uniform(0, 1, n) < 0.05
    Y = np.where(flip, 1 - Y, Y)
    df = pd.DataFrame({"A": A, "B": B, "C": C, "D": D, "Y": Y})
    csv = tmp_path / "c.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "cna" in res.summary
    assert res.estimates["n_solutions"] >= 1
    assert res.estimates["max_consistency"] > 0.7  # a strong configurational solution


@pytest.mark.skipif(not _HAS_CNA, reason="R cna not available")
def test_cna_fuzzy_continuous(tmp_path: Path) -> None:
    rng = np.random.default_rng(8)
    n = 60
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(0, 10, n)
    # outcome high when x1 high (fuzzy)
    y = x1 + 0.2 * x2 + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"cond1": x1, "cond2": x2, "outc": y})
    csv = tmp_path / "f.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"con": 0.75, "cov": 0.75})
    # either solves (fuzzy-calibrated) or honestly reports no structure — never crashes
    assert ("cna" in res.summary) or ("CNA 失败" in res.summary)


def test_cna_too_few_factors_degrades(tmp_path: Path) -> None:
    df = pd.DataFrame({"A": [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1], "B": [1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0]})
    csv = tmp_path / "c.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "CNA 失败" in res.summary  # <3 factors
