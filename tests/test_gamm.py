"""Tests for the GAMM (mgcv) executor branch."""

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
        id="gamm", method="GAMM", domain="statistics", family="statistics",
        goal="explain", preconditions=Precondition(requires_group=True, min_continuous=2, min_rows=50),
    )


def _data(tmp_path: Path) -> Path:
    rng = np.random.default_rng(1)
    ng, per = 20, 20
    grp = np.repeat(np.arange(ng), per)
    re = rng.normal(0, 1.2, ng)
    n = ng * per
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(-3, 3, n)
    y = np.sin(x1) + 0.4 * x2**2 + re[grp] + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"site": [f"s{g}" for g in grp], "y": y, "x1": x1, "x2": x2})
    csv = tmp_path / "g.csv"
    df.to_csv(csv, index=False)
    return csv


@pytest.mark.skipif(not _HAS_MGCV, reason="R mgcv not available")
def test_gamm_smooths_plus_random_intercept(tmp_path: Path) -> None:
    fp = profile_dataset(_data(tmp_path))
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "mgcv" in res.summary
    assert res.estimates["edf_s(x1)"] > 2.0  # sin -> strongly nonlinear
    assert res.estimates["deviance_explained"] > 0.7
    assert res.estimates["random_intercept_sd"] > 0  # a real between-site SD


@pytest.mark.skipif(not _HAS_MGCV, reason="R mgcv not available")
def test_gamm_config_group_and_outcome(tmp_path: Path) -> None:
    fp = profile_dataset(_data(tmp_path))
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"group": "site", "outcome": "y", "predictors": ["x1", "x2"]},
    )
    assert "mgcv" in res.summary and "edf_s(x1)" in res.estimates


def test_gamm_no_group_degrades(tmp_path: Path) -> None:
    # only continuous columns, no grouping variable -> honest failure (no R needed)
    rng = np.random.default_rng(2)
    n = 80
    df = pd.DataFrame({"y": rng.normal(0, 1, n), "x1": rng.uniform(0, 10, n), "x2": rng.uniform(0, 5, n)})
    csv = tmp_path / "x.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "GAMM" in res.summary and ("分组" in res.summary or "失败" in res.summary)
