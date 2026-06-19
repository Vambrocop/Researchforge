"""Tests for the risk_adjusted_return (Sharpe / Sortino / Calmar / drawdown) branch.

Known-value checks: hand-computed Sharpe and maximum drawdown on a crafted path,
plus an independent numpy reimplementation cross-check. Config override + degrade.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="risk_adjusted_return", method="Risk-adjusted performance",
        domain="finance", family="finance", goal="describe",
        preconditions=Precondition(is_timeseries=True, min_rows=20, min_continuous=1),
    )


def _run(df: pd.DataFrame, tmp_path: Path, config=None):
    tmp_path.mkdir(parents=True, exist_ok=True)  # callers may pass a nested subdir
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    return run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config=config or {})


def test_hand_computed_max_drawdown(tmp_path: Path) -> None:
    # Crafted path: a clean -36% peak-to-trough drawdown.
    #   r = [+.10, +.10, -.20, -.20, +.10] repeated; the worst peak->trough within a
    #   block is wealth 1.21 -> 0.7744 = 0.64 of peak -> drawdown = -0.36 exactly.
    # Pad with zeros to satisfy min_rows=20 (zeros never deepen the drawdown below -0.36
    # because after the block wealth recovers above the trough but stays below the peak;
    # we keep the first block as the global peak so the -0.36 trough is the global min).
    block = [0.10, 0.10, -0.20, -0.20, 0.10]
    rets = np.array(block + [0.0] * 20, dtype=float)
    df = pd.DataFrame({"t": np.arange(len(rets)), "ret": rets})
    res = _run(df, tmp_path, {"column": "ret", "is_returns": True})
    assert "完成" in res.summary
    # 0.7744 / 1.21 - 1 = -0.36 exactly
    assert abs(res.estimates["max_drawdown"] - (-0.36)) < 1e-9


def test_matches_independent_numpy(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    rets = rng.normal(0.001, 0.02, 500)
    df = pd.DataFrame({"t": np.arange(len(rets)), "ret": rets})
    ppy, rf = 252.0, 0.0001
    res = _run(df, tmp_path, {"column": "ret", "is_returns": True,
                              "periods_per_year": ppy, "rf": rf})
    e = res.estimates

    mu, sd = rets.mean(), rets.std(ddof=1)
    excess = rets - rf
    exp_sharpe = excess.mean() / sd * np.sqrt(ppy)
    dd = np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2))
    exp_sortino = excess.mean() / dd * np.sqrt(ppy)
    w = np.cumprod(1 + rets)
    exp_max_dd = (w / np.maximum.accumulate(w) - 1).min()
    exp_calmar = (mu * ppy) / abs(exp_max_dd)

    assert abs(e["ann_return"] - mu * ppy) < 1e-6
    assert abs(e["ann_volatility"] - sd * np.sqrt(ppy)) < 1e-6
    assert abs(e["sharpe"] - exp_sharpe) < 1e-3
    assert abs(e["sortino"] - exp_sortino) < 1e-3
    assert abs(e["max_drawdown"] - exp_max_dd) < 1e-6
    assert abs(e["calmar"] - exp_calmar) < 1e-2


def test_sortino_ge_sharpe_when_downside_smaller(tmp_path: Path) -> None:
    # Right-skewed returns: downside deviation < total sigma -> Sortino > Sharpe.
    rng = np.random.default_rng(5)
    rets = np.abs(rng.normal(0, 0.02, 600)) - 0.005  # mostly positive, mild downside
    df = pd.DataFrame({"t": np.arange(len(rets)), "ret": rets})
    res = _run(df, tmp_path, {"column": "ret", "is_returns": True})
    assert "完成" in res.summary
    assert res.estimates["sortino"] > res.estimates["sharpe"]


def test_config_periods_per_year_override(tmp_path: Path) -> None:
    rng = np.random.default_rng(8)
    rets = rng.normal(0.001, 0.02, 300)
    df = pd.DataFrame({"t": np.arange(len(rets)), "ret": rets})
    res12 = _run(df, tmp_path / "a", {"column": "ret", "is_returns": True, "periods_per_year": 12})
    res252 = _run(df, tmp_path / "b", {"column": "ret", "is_returns": True, "periods_per_year": 252})
    # annualized return scales linearly with periods_per_year (mean*ppy). Tolerance is
    # loose because estimates are rounded to 6 dp and the ratio of two small rounded
    # numbers amplifies that rounding (handler math is exact: mu*252 / mu*12 == 21).
    ratio = res252.estimates["ann_return"] / res12.estimates["ann_return"]
    assert abs(ratio - (252 / 12)) < 1e-2
    assert res12.estimates["periods_per_year"] == 12.0


def test_too_short_degrades(tmp_path: Path) -> None:
    df = pd.DataFrame({"t": np.arange(10), "ret": np.random.default_rng(1).normal(0, 1, 10)})
    res = _run(df, tmp_path, {"column": "ret", "is_returns": True})
    assert "失败" in res.summary
    assert not res.estimates
