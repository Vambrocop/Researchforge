"""Tests for the ARDL bounds-test / error-correction executor branch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="ardl_bounds", method="ARDL bounds test", domain="economics",
        family="time-series", goal="explain",
        preconditions=Precondition(min_continuous=2, min_rows=30),
    )


def test_ardl_detects_longrun_and_recovers_coef(tmp_path: Path) -> None:
    # y error-corrects toward a long-run relation y = 1.5*x; x is I(1)
    rng = np.random.default_rng(0)
    n = 200
    x = np.cumsum(rng.normal(0, 1, n))
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = y[t - 1] - 0.3 * (y[t - 1] - 1.5 * x[t - 1]) + rng.normal(0, 0.5)
    df = pd.DataFrame({"t": np.arange(n), "y": y, "x": x})
    csv = tmp_path / "a.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "y", "predictors": ["x"]})
    assert "完成" in res.summary
    assert res.estimates["bounds_F"] > res.estimates["crit_upper_95"]   # long-run relationship
    assert res.estimates["speed_of_adjustment"] < 0                     # error-correcting
    assert abs(res.estimates["longrun_x"] - 1.5) < 0.5                  # recovers ~1.5
    assert "存在长期" in res.summary


def test_ardl_independent_walks_no_longrun(tmp_path: Path) -> None:
    # two independent random walks -> no long-run relationship (F should not exceed the I(1) bound)
    rng = np.random.default_rng(7)
    n = 200
    y = np.cumsum(rng.normal(0, 1, n))
    x = np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame({"t": np.arange(n), "y": y, "x": x})
    csv = tmp_path / "i.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "y", "predictors": ["x"]})
    assert "完成" in res.summary
    assert "存在长期" not in res.summary                                # no spurious long-run verdict


def test_ardl_needs_outcome_and_regressor(tmp_path: Path) -> None:
    df = pd.DataFrame({"t": np.arange(40), "y": np.cumsum(np.random.default_rng(1).normal(0, 1, 40))})
    csv = tmp_path / "x.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "ARDL 边界检验失败" in res.summary
