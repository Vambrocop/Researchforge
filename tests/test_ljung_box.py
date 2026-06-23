"""Tests for the ljung_box (Ljung-Box white-noise test) executor branch.

Known structure: an AR(1) series has strong autocorrelation -> Ljung-Box rejects
white noise (is_white_noise = 0, min_p tiny); independent white noise -> fails to
reject (is_white_noise = 1). Plus a config-lags test and an honest-degrade test.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_HAS_SM = importlib.util.find_spec("statsmodels") is not None


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="ljung_box", method="Ljung-Box test", domain="time series",
        family="time-series", goal="explore",
        preconditions=Precondition(is_timeseries=True, min_rows=20),
    )


def _ar1(n: int, phi: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 1, n)
    y = np.empty(n)
    y[0] = e[0]
    for t in range(1, n):
        y[t] = phi * y[t - 1] + e[t]
    return y


@pytest.mark.skipif(not _HAS_SM, reason="statsmodels not available")
def test_ar1_rejects_white_noise(tmp_path: Path) -> None:
    y = _ar1(400, 0.7, seed=1)
    df = pd.DataFrame({"t": np.arange(len(y)), "x": y})
    csv = tmp_path / "ar1.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"column": "x"})
    assert "完成" in res.summary
    e = res.estimates
    assert e["is_white_noise"] == 0.0
    assert e["min_p"] < 0.05
    assert e["lb_p_lag10"] < 0.05
    assert e["n"] == 400.0
    lb = pd.read_csv(Path(res.output_dir) / "ljung_box.csv")
    assert {"lag", "Q", "p"} == set(lb.columns)


@pytest.mark.skipif(not _HAS_SM, reason="statsmodels not available")
def test_white_noise_not_rejected(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    y = rng.normal(0, 1, 500)
    df = pd.DataFrame({"t": np.arange(len(y)), "x": y})
    csv = tmp_path / "wn.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"column": "x"})
    assert "完成" in res.summary
    assert res.estimates["is_white_noise"] == 1.0
    assert res.estimates["min_p"] >= 0.05


@pytest.mark.skipif(not _HAS_SM, reason="statsmodels not available")
def test_config_multiple_lags(tmp_path: Path) -> None:
    y = _ar1(300, 0.6, seed=4)
    df = pd.DataFrame({"t": np.arange(len(y)), "x": y})
    csv = tmp_path / "ml.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"column": "x", "lags": [5, 10, 20]})
    assert "完成" in res.summary
    assert res.estimates["n_lags_tested"] == 3.0
    lb = pd.read_csv(Path(res.output_dir) / "ljung_box.csv")
    assert sorted(lb["lag"].tolist()) == [5, 10, 20]


def test_degrade_too_short(tmp_path: Path) -> None:
    # below min_n -> honest skip, no estimates fabricated
    df = pd.DataFrame({"t": np.arange(8), "x": np.arange(8, dtype=float)})
    csv = tmp_path / "s.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"column": "x"})
    assert "跳过" in res.summary
    assert "is_white_noise" not in res.estimates
