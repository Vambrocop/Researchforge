"""Tests for the GAM (mgcv) executor branch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis, rbridge
from researchforge.profiler import profile_dataset

_HAS_MGCV = rbridge.r_available() and rbridge.r_package_available("mgcv")


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="gam", method="GAM", domain="statistics", family="statistics",
        goal="explain", preconditions=Precondition(min_continuous=2, min_rows=30),
    )


@pytest.mark.skipif(not _HAS_MGCV, reason="R mgcv not available")
def test_gam_detects_nonlinearity(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 250
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(-3, 3, n)
    y = np.sin(x1) + 0.4 * x2**2 + rng.normal(0, 0.4, n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    csv = tmp_path / "g.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "mgcv" in res.summary
    # strongly nonlinear smooths -> effective df well above 1 (a straight line)
    assert res.estimates["edf_s(x1)"] > 2.0
    assert res.estimates["deviance_explained"] > 0.7


@pytest.mark.skipif(not _HAS_MGCV, reason="R mgcv not available")
def test_gam_config_outcome(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n = 120
    a = rng.uniform(0, 5, n)
    target = np.cos(a) + rng.normal(0, 0.3, n)
    # 'a' is the first continuous column; override outcome to 'target'
    df = pd.DataFrame({"a": a, "target": target})
    csv = tmp_path / "g.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"outcome": "target", "predictors": ["a"]},
    )
    assert "mgcv" in res.summary
    assert "edf_s(a)" in res.estimates


def test_gam_no_smoothable_predictor_degrades(tmp_path: Path) -> None:
    # only a binary predictor -> nothing to smooth -> honest skip (no R needed)
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"y": rng.normal(0, 1, 60), "flag": rng.integers(0, 2, 60)})
    csv = tmp_path / "g.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "GAM" in res.summary and ("跳过" in res.summary or "失败" in res.summary)
