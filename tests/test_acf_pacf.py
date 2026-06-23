"""Tests for the acf_pacf (ACF + PACF diagnostics) executor branch.

Known structure: an AR(1) series should show a high lag-1 ACF, a PACF that cuts off
after lag 1 (so suggested_ar_order ≈ 1), and many significant ACF lags. We also
independently recompute the lag-1 ACF to pin estimator correctness, and cover the
honest-degrade paths (constant series, non-numeric).
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
        id="acf_pacf", method="ACF/PACF diagnostics", domain="time series",
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


def _manual_acf_lag1(y: np.ndarray) -> float:
    """Independent lag-1 ACF (biased / statsmodels convention: divide by n, full mean)."""
    y = np.asarray(y, dtype=float)
    ybar = y.mean()
    num = np.sum((y[1:] - ybar) * (y[:-1] - ybar))
    den = np.sum((y - ybar) ** 2)
    return float(num / den)


@pytest.mark.skipif(not _HAS_SM, reason="statsmodels not available")
def test_ar1_acf_pacf_structure(tmp_path: Path) -> None:
    y = _ar1(400, 0.7, seed=1)
    df = pd.DataFrame({"t": np.arange(len(y)), "x": y})
    csv = tmp_path / "ar1.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    # small nlags so a clean AR(1) cut-off isn't masked by spurious high-lag PACF spikes
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"column": "x", "nlags": 10})
    assert "完成" in res.summary
    e = res.estimates
    # AR(1) phi=0.7: lag-1 ACF should be clearly positive (~0.7)
    assert e["acf_lag1"] > 0.5
    # PACF of an AR(1) cuts off after lag 1 -> dominant lag-1 PACF, AR order hint ~1
    assert e["pacf_lag1"] > 0.5
    # last significant PACF lag = suggested AR order; for a clean AR(1) this is ~1
    # (allow a small spurious high-lag spike rather than asserting exactly 1)
    assert 1.0 <= e["suggested_ar_order"] <= 2.0
    assert e["n_sig_acf_lags"] >= 2  # AR(1) ACF decays slowly -> several sig lags
    assert e["n"] == 400.0
    # CSV present with the expected columns
    acf_df = pd.read_csv(Path(res.output_dir) / "acf_pacf.csv")
    assert {"lag", "acf", "acf_ci", "pacf", "pacf_ci"} == set(acf_df.columns)


@pytest.mark.skipif(not _HAS_SM, reason="statsmodels not available")
def test_lag1_acf_matches_manual(tmp_path: Path) -> None:
    # independently recompute lag-1 ACF and confirm the branch matches it
    y = _ar1(300, 0.5, seed=7)
    manual = _manual_acf_lag1(y)
    df = pd.DataFrame({"t": np.arange(len(y)), "x": y})
    csv = tmp_path / "m.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"column": "x"})
    assert res.estimates["acf_lag1"] == pytest.approx(manual, abs=1e-3)


@pytest.mark.skipif(not _HAS_SM, reason="statsmodels not available")
def test_white_noise_few_sig_lags(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    y = rng.normal(0, 1, 500)
    df = pd.DataFrame({"t": np.arange(len(y)), "x": y})
    csv = tmp_path / "wn.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"column": "x"})
    assert "完成" in res.summary
    # white noise: low lag-1 ACF, few significant lags (binomial(40, 0.05) chance hits)
    assert abs(res.estimates["acf_lag1"]) < 0.15
    assert res.estimates["n_sig_acf_lags"] <= 8


@pytest.mark.skipif(not _HAS_SM, reason="statsmodels not available")
def test_config_nlags(tmp_path: Path) -> None:
    y = _ar1(200, 0.6, seed=2)
    df = pd.DataFrame({"t": np.arange(len(y)), "x": y})
    csv = tmp_path / "nl.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"column": "x", "nlags": 12})
    assert "完成" in res.summary
    acf_df = pd.read_csv(Path(res.output_dir) / "acf_pacf.csv")
    assert acf_df["lag"].max() == 12


def test_degrade_constant_series(tmp_path: Path) -> None:
    df = pd.DataFrame({"t": np.arange(40), "x": np.ones(40)})
    csv = tmp_path / "c.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"column": "x"})
    assert "跳过" in res.summary
    assert "acf_lag1" not in res.estimates
