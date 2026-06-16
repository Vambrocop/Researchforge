"""Tests for the BART (dbarts) executor branch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis, rbridge
from researchforge.profiler import profile_dataset

_HAS_DBARTS = rbridge.r_available() and rbridge.r_package_available("dbarts")


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="bart", method="BART", domain="machine learning", family="ml",
        goal="predict", preconditions=Precondition(min_continuous=2, min_rows=50),
    )


@pytest.mark.skipif(not _HAS_DBARTS, reason="R dbarts not available")
def test_bart_fits_and_ranks_predictors(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n = 500
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(-3, 3, n)
    x3 = rng.normal(0, 1, n)  # irrelevant
    y = np.sin(x1) + 0.4 * x2**2 + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})
    csv = tmp_path / "b.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "dbarts" in res.summary
    assert res.estimates["r_squared_insample"] > 0.7  # captures the nonlinear signal
    assert res.estimates["sigma"] > 0
    # the irrelevant x3 should not be the most-used split variable
    vi = pd.read_csv(Path(res.output_dir) / "bart_variable_importance.csv")
    assert vi.iloc[0]["predictor"] in ("x1", "x2")


def test_bart_no_predictors_degrades(tmp_path: Path) -> None:
    # a single continuous column -> no predictors -> honest failure (no R needed)
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"y": rng.normal(0, 1, 60)})
    csv = tmp_path / "x.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "BART 失败" in res.summary
