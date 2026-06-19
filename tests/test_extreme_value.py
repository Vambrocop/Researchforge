"""Tests for the extreme_value (EVT / Peaks-Over-Threshold / GPD) finance branch.

Known-value checks:
  * Heavy-tailed (Student-t) data recovers a positive GPD shape xi (Frechet tail).
  * The hand-rolled GPD tail-VaR formula reproduces from the reported xi/sigma/u/Nu.
  * Light-tailed (Normal) data gives xi close to 0.
Plus config override (threshold) and a degrade (too-few exceedances).
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
        id="extreme_value", method="Extreme Value Theory (POT / GPD)",
        domain="finance", family="finance", goal="describe",
        preconditions=Precondition(is_timeseries=True, min_rows=50, min_continuous=1),
    )


def _run(df: pd.DataFrame, tmp_path: Path, config=None):
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    return run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config=config or {})


def test_heavy_tail_recovers_positive_xi(tmp_path: Path) -> None:
    # Student-t(df=3) returns: theoretical tail index = 1/df -> xi ~= 0.33 > 0.
    rng = np.random.default_rng(11)
    rets = stats.t(df=3).rvs(size=12000, random_state=rng) * 0.01
    df = pd.DataFrame({"t": np.arange(len(rets)), "ret": rets})
    res = _run(df, tmp_path, {"column": "ret", "is_returns": True})
    assert "完成" in res.summary
    e = res.estimates
    assert e["xi_shape"] > 0.05  # heavy/Frechet tail
    assert e["n_exceedances"] >= 30
    assert e["sigma_scale"] > 0
    # extreme return levels increase with alpha
    assert e["evt_var_999"] > e["evt_var_990"]


def test_gpd_tail_var_formula_exact(tmp_path: Path) -> None:
    # Reproduce the hand-rolled GPD tail-VaR formula from the reported parameters:
    #   VaR_p = u + (sigma/xi)*[((n/Nu)*(1-p))^(-xi) - 1]
    rng = np.random.default_rng(13)
    rets = stats.t(df=4).rvs(size=10000, random_state=rng) * 0.01
    df = pd.DataFrame({"t": np.arange(len(rets)), "ret": rets})
    res = _run(df, tmp_path, {"column": "ret", "is_returns": True, "evt_alpha": [0.99]})
    e = res.estimates
    xi, sigma, u = e["xi_shape"], e["sigma_scale"], e["threshold_u"]
    n, Nu = e["n_obs"], e["n_exceedances"]
    p = 0.99
    ratio = n / Nu
    if abs(xi) < 1e-8:
        expected = u + sigma * np.log(ratio / (1 - p))
    else:
        expected = u + (sigma / xi) * (((ratio * (1 - p)) ** (-xi)) - 1)
    # tolerance covers the engine's 6-dp rounding of xi/sigma/u
    assert abs(e["evt_var_990"] - expected) < 1e-3


def test_light_tail_xi_near_zero(tmp_path: Path) -> None:
    # Normal data has an exponential (xi=0, Gumbel) tail; GPD shape should be ~0.
    rng = np.random.default_rng(17)
    rets = rng.normal(0, 0.02, 12000)
    df = pd.DataFrame({"t": np.arange(len(rets)), "ret": rets})
    res = _run(df, tmp_path, {"column": "ret", "is_returns": True})
    assert "完成" in res.summary
    # Normal MDA -> xi near 0 (allow sampling slack; should be well below the t case)
    assert abs(res.estimates["xi_shape"]) < 0.25


def test_config_threshold_override(tmp_path: Path) -> None:
    rng = np.random.default_rng(19)
    rets = stats.t(df=3).rvs(size=10000, random_state=rng) * 0.01
    df = pd.DataFrame({"t": np.arange(len(rets)), "ret": rets})
    # explicit low threshold -> many more exceedances than the 95th-pct default
    res = _run(df, tmp_path, {"column": "ret", "is_returns": True, "threshold": 0.005})
    assert "完成" in res.summary
    assert abs(res.estimates["threshold_u"] - 0.005) < 1e-9
    assert res.estimates["n_exceedances"] > 500  # ~ heavy right tail mass beyond 0.005


def test_too_few_exceedances_degrades(tmp_path: Path) -> None:
    # A very high threshold leaves < 30 exceedances -> honest skip.
    rng = np.random.default_rng(23)
    rets = rng.normal(0, 0.01, 1000)
    df = pd.DataFrame({"t": np.arange(len(rets)), "ret": rets})
    res = _run(df, tmp_path, {"column": "ret", "is_returns": True, "threshold_quantile": 0.995})
    assert "失败" in res.summary
    assert "30" in res.summary


def test_too_short_degrades(tmp_path: Path) -> None:
    df = pd.DataFrame({"t": np.arange(20), "ret": np.random.default_rng(1).normal(0, 1, 20)})
    res = _run(df, tmp_path, {"column": "ret", "is_returns": True})
    assert "失败" in res.summary
