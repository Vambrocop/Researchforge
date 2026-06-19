"""Tests for the value_at_risk (VaR + Expected Shortfall) finance branch.

Known-value checks against closed forms:
  * Normal sample: historical VaR ~= parametric Gaussian VaR (large n), and
    Cornish-Fisher VaR ~= Gaussian VaR when skew/kurt ~= 0.
  * Student-t (fat-tailed) sample: ES > Gaussian VaR (tail risk).
Plus config override (column / alpha / is_returns) and a degrade (too-short).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="value_at_risk", method="Value at Risk + Expected Shortfall",
        domain="finance", family="finance", goal="describe",
        preconditions=Precondition(is_timeseries=True, min_rows=30, min_continuous=1),
    )


def _run(df: pd.DataFrame, tmp_path: Path, config=None):
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    return run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config=config or {})


def test_normal_historical_matches_parametric(tmp_path: Path) -> None:
    # iid Normal(0, 1) returns: at large n, the empirical quantile ~= mu + z*sigma,
    # and (skew, excess-kurt) ~= 0 so Cornish-Fisher z_cf ~= z (CF VaR ~= Gaussian VaR).
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0, 0.02, 20000)
    df = pd.DataFrame({"t": np.arange(len(rets)), "ret": rets})
    res = _run(df, tmp_path, {"column": "ret", "is_returns": True, "alpha": [0.95, 0.99]})
    assert "完成" in res.summary
    e = res.estimates
    # closed-form Gaussian VaR on losses: mu_loss + z*sigma; losses = -rets, so
    # mu_loss ~= 0, sigma ~= 0.02 -> VaR_95 ~= 1.645*0.02 ~= 0.0329
    expected = abs(float(stats.norm.ppf(0.95))) * 0.02
    assert abs(e["var_gauss_95"] - expected) < 0.004
    # historical ~= parametric at this n
    assert abs(e["var_hist_95"] - e["var_gauss_95"]) < 0.003
    # Cornish-Fisher ~= Gaussian when skew/kurt ~= 0
    assert abs(e["var_cf_95"] - e["var_gauss_95"]) < 0.003
    assert abs(e["skewness"]) < 0.15 and abs(e["excess_kurtosis"]) < 0.3
    # ES > VaR always (mean beyond the quantile exceeds the quantile)
    assert e["es_hist_95"] > e["var_hist_95"]
    assert e["es_gauss_99"] > e["var_gauss_99"]


def test_fat_tail_es_exceeds_gaussian_var(tmp_path: Path) -> None:
    # Student-t(df=3) is fat-tailed: tail losses are extreme, so the (coherent) ES
    # captures tail mass the Gaussian VaR ignores -> historical ES >> Gaussian VaR.
    rng = np.random.default_rng(7)
    rets = stats.t(df=3).rvs(size=20000, random_state=rng) * 0.01
    df = pd.DataFrame({"t": np.arange(len(rets)), "ret": rets})
    res = _run(df, tmp_path, {"column": "ret", "is_returns": True, "alpha": [0.99]})
    assert "完成" in res.summary
    e = res.estimates
    # fat tails -> positive excess kurtosis
    assert e["excess_kurtosis"] > 1.0
    # historical ES at 99% exceeds the Gaussian VaR at 99% (tail underestimation)
    assert e["es_hist_99"] > e["var_gauss_99"]
    # Cornish-Fisher inflates VaR above Gaussian for fat tails (positive kurt term)
    assert e["var_cf_99"] > e["var_gauss_99"]


def test_cornish_fisher_formula_exact(tmp_path: Path) -> None:
    # Verify the hand-rolled Cornish-Fisher expansion reproduces the engine's z_cf
    # exactly from the reported skew S and excess-kurt K (catch a transcription bug).
    rng = np.random.default_rng(3)
    rets = stats.skewnorm(a=4).rvs(size=8000, random_state=rng) * 0.01  # skewed
    df = pd.DataFrame({"t": np.arange(len(rets)), "ret": rets})
    res = _run(df, tmp_path, {"column": "ret", "is_returns": True, "alpha": [0.99]})
    e = res.estimates
    S, K = e["skewness"], e["excess_kurtosis"]
    mu_l, sd_l = e["mean_loss"], e["sd_loss"]
    z = float(stats.norm.ppf(0.99))
    z_cf = (z + (z**2 - 1) * S / 6 + (z**3 - 3 * z) * K / 24
            - (2 * z**3 - 5 * z) * S**2 / 36)
    var_cf_expected = mu_l + z_cf * sd_l
    assert abs(e["var_cf_99"] - var_cf_expected) < 1e-4


def test_price_series_autoconverts(tmp_path: Path) -> None:
    # A trending all-positive PRICE series should be auto-detected and log-returned.
    rng = np.random.default_rng(1)
    rets = rng.normal(0.0005, 0.02, 2000)
    price = 100.0 * np.cumprod(1 + rets)  # all positive, trending up
    df = pd.DataFrame({"t": np.arange(len(price)), "px": price})
    res = _run(df, tmp_path, {"column": "px"})
    assert "完成" in res.summary
    assert "价格" in res.summary and "对数收益" in res.summary
    # n drops by 1 after differencing
    assert res.estimates["n_obs"] == float(len(price) - 1)


def test_config_alpha_override(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"t": np.arange(5000), "ret": rng.normal(0, 0.01, 5000)})
    res = _run(df, tmp_path, {"column": "ret", "is_returns": True, "alpha": 0.90})
    assert "完成" in res.summary
    assert "var_hist_90" in res.estimates  # single scalar alpha honored
    assert "var_hist_95" not in res.estimates


def test_too_short_degrades(tmp_path: Path) -> None:
    df = pd.DataFrame({"t": np.arange(10), "ret": np.random.default_rng(1).normal(0, 1, 10)})
    res = _run(df, tmp_path, {"column": "ret", "is_returns": True})
    assert "失败" in res.summary
    assert not res.estimates
