"""ARIMA 自动定阶在大季节周期上实际挂死——Wave S1 引入的性能回归。

`test_end_to_end[co2_timeseries]` failed with `FileNotFoundError: …analysis_code.py`. That was
a symptom: the run took so long its parallel-worker tmp dir was reclaimed underneath it.
statsmodels' co2 dataset is weekly (n=2225) and the seasonal detector correctly finds sp=52.
A SARIMAX state vector carries `sp` lags per seasonal AR/MA term, so measured on that frame:

    (0,1,1)(0,1,0,52)   P=Q=0, differencing only      0.8s
    (0,1,0)(0,1,1,52)   one seasonal MA             213s
    (1,1,1)(1,1,1,52)                               219s      AICc 2772 (statistically best)

`_GRID_FIT_BUDGET = 48` × ~219s ≈ 2.7 hours. Wave S1's cold review exercised MONTHLY data
(sp=12, whole search 8.6s) and never met this.

Three things had to be true for the fix to work, and each was found by measurement:
  * a wall-clock budget alone was not enough — it only checks BETWEEN fits, so the run still
    paid one 213s fit (2.7h → 216s);
  * the cost model must count only the seasonal AR/MA terms — counting D over-penalised the
    one CHEAP seasonal model (0.8s) and skipped everything, fitting 0 candidates;
  * the FALLBACK order has to respect the budget too — skipping 36 costly candidates saved
    nothing while the caller then fitted (1,d,1)(1,D,1,sp), which is the 219s model.

After: co2 finishes in ~23s and discloses what was skipped and how to opt in; the monthly
airline case still selects (0,1,1)(0,1,1)[12] with nothing skipped.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog import Catalog
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_CAT = Catalog.load()


def _fp(df, tmp_path, name="d.csv"):
    csv = tmp_path / name
    df.to_csv(csv, index=False)
    return profile_dataset(csv)


def _weekly_seasonal(n=2225, seed=0):
    """Weekly data with a yearly cycle — the co2 shape, without the statsmodels dependency."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = 320 + 0.02 * t + 3 * np.sin(2 * np.pi * t / 52) + rng.normal(0, 0.4, n)
    return pd.DataFrame({"date": pd.date_range("1958-03-29", periods=n, freq="W-SAT")
                        .strftime("%Y-%m-%d"),
                         "co2": y.round(3)})


def _monthly_airline(n=60, seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    seas = np.array([-8, -6, -2, 2, 6, 9, 8, 5, 1, 4, 12, 20])[t % 12]
    return pd.DataFrame({"month": pd.date_range("2019-01-01", periods=n, freq="MS")
                        .strftime("%Y-%m"),
                         "revenue": (100 + 0.8 * t + seas + rng.normal(0, 2.5, n)).round(1)})


@pytest.mark.slow
def test_a_large_seasonal_period_does_not_hang(tmp_path):
    """The regression this file exists for: 2.7 hours of grid search on weekly data."""
    fp = _fp(_weekly_seasonal(), tmp_path, name="weekly.csv")
    t0 = time.perf_counter()
    res = run_analysis(fp, _CAT.by_id("arima"), output_root=str(tmp_path / "o"))
    elapsed = time.perf_counter() - t0
    assert elapsed < 180, f"arima took {elapsed:.0f}s on n=2225 weekly data"
    assert res.estimates.get("seasonal_periods") == 52    # detection itself was right
    assert res.estimates.get("forecast_next") is not None


@pytest.mark.slow
def test_the_skipped_candidates_are_disclosed_with_a_way_in(tmp_path):
    """The expensive model is genuinely the better fit (AICc 2772 vs 3669), so dropping it
    silently would trade a hang for a worse answer with no word to the user."""
    fp = _fp(_weekly_seasonal(), tmp_path, name="weekly2.csv")
    res = run_analysis(fp, _CAT.by_id("arima"), output_root=str(tmp_path / "p"))
    assert "被跳过" in res.summary, res.summary[:300]
    assert "fit_cost_budget" in res.summary, "an honest skip names the escape hatch"


def test_the_monthly_airline_case_is_untouched(tmp_path):
    """Wave S1's verified behaviour: n=60, sp=12 selects the airline model and skips nothing."""
    fp = _fp(_monthly_airline(), tmp_path, name="monthly.csv")
    res = run_analysis(fp, _CAT.by_id("arima"), output_root=str(tmp_path / "q"))
    assert res.estimates.get("seasonal_periods") == 12
    assert (res.estimates["p"], res.estimates["q"]) == (0.0, 1.0)
    assert (res.estimates["P"], res.estimates["Q"]) == (0.0, 1.0)
    assert "被跳过" not in res.summary, "a cheap seasonal grid must not be gated"


def test_seasonal_differencing_is_not_priced_like_a_seasonal_arma_term(tmp_path):
    """The cost model counts only the seasonal AR/MA terms. Counting D too over-penalised the
    ONE cheap seasonal model (0.8s measured) and left the search fitting zero candidates."""
    from researchforge.executor.branches.timeseries import _FIT_COST_BUDGET

    n = 2225
    sp = 52

    def cost(order, sorder):
        p, _, q = order
        P, D, Q, s = sorder
        return float(n) * float(max(p, q + 1) + s * (P + Q)) ** 2

    assert cost((0, 1, 1), (0, 1, 0, sp)) < _FIT_COST_BUDGET     # differencing only: cheap
    assert cost((0, 1, 0), (0, 1, 1, sp)) > _FIT_COST_BUDGET     # one seasonal MA: 213s
    # ...and the monthly airline model must stay affordable
    assert 60.0 * float(max(0, 2) + 12 * (0 + 1)) ** 2 < _FIT_COST_BUDGET


def test_config_can_buy_the_expensive_search_back(tmp_path):
    """The gate is a default, not a prohibition."""
    from researchforge.executor.branches.timeseries import _FIT_COST_BUDGET

    fp = _fp(_monthly_airline(), tmp_path, name="cfg.csv")
    res = run_analysis(fp, _CAT.by_id("arima"), output_root=str(tmp_path / "r"),
                       config={"fit_cost_budget": _FIT_COST_BUDGET * 100,
                               "search_seconds": 120})
    assert res.estimates.get("forecast_next") is not None
