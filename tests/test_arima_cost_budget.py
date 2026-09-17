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
    assert "fit_seconds" in res.summary or "search_seconds" in res.summary, (
        "an honest skip names the escape hatch")


def test_the_monthly_airline_case_is_untouched(tmp_path):
    """Wave S1's verified behaviour: n=60, sp=12 selects the airline model and skips nothing."""
    fp = _fp(_monthly_airline(), tmp_path, name="monthly.csv")
    res = run_analysis(fp, _CAT.by_id("arima"), output_root=str(tmp_path / "q"))
    assert res.estimates.get("seasonal_periods") == 12
    assert (res.estimates["p"], res.estimates["q"]) == (0.0, 1.0)
    assert (res.estimates["P"], res.estimates["Q"]) == (0.0, 1.0)
    assert "被跳过" not in res.summary, "a cheap seasonal grid must not be gated"


@pytest.mark.parametrize(
    "order,sorder",
    [((0, 1, 0), (0, 1, 0, 52)), ((0, 1, 1), (0, 1, 0, 52)), ((0, 1, 0), (0, 1, 1, 52)),
     ((1, 1, 1), (1, 1, 1, 52)), ((2, 1, 2), (0, 1, 0, 12)), ((0, 1, 1), (0, 1, 1, 12)),
     ((1, 0, 1), (1, 0, 1, 12)), ((2, 1, 2), (1, 1, 1, 12)), ((3, 0, 0), (0, 0, 0, 0))],
)
def test_the_state_dimension_matches_statsmodels_itself(order, sorder):
    """The cost model is only as good as its state dimension, so compare with the library
    rather than with a restatement of our own formula.

    The version of this test I wrote first RE-IMPLEMENTED the cost function inside the test
    file and asserted against that copy — it would have stayed green no matter what the branch
    computed, and it did stay green while the branch's formula was wrong in two places (cold
    review A5.2). The formula it was defending, `max(p, q+1) + sp*(P+Q)`, gets 1 instead of 54
    on the very first case below.
    """
    import statsmodels.api as sm

    from researchforge.executor.branches.timeseries import _sarimax_k_states

    y = np.random.default_rng(0).normal(0, 1, 400).cumsum()
    real = sm.tsa.SARIMAX(y, order=order, seasonal_order=sorder).k_states
    assert _sarimax_k_states(order, sorder) == real, (order, sorder)


def test_the_projection_tracks_measured_fit_time():
    """Calibration anchors, measured on this machine by the cold review:
    n=600 / sp=52 / (0,1,1)(0,1,1,52) took 51.1s; n=7000 / sp=24 / (0,1,1)(0,1,1,24) 29.3s."""
    from researchforge.executor.branches.timeseries import _fit_seconds

    assert _fit_seconds(600, (0, 1, 1), (0, 1, 1, 52)) == pytest.approx(51.1, rel=0.25)
    assert _fit_seconds(7000, (0, 1, 1), (0, 1, 1, 24)) == pytest.approx(29.3, rel=0.25)
    # the monthly airline case S1 was verified on must stay far inside any sane budget
    assert _fit_seconds(60, (0, 1, 1), (0, 1, 1, 12)) < 0.1


def test_config_can_buy_the_expensive_search_back(tmp_path):
    """The gate is a default, not a prohibition."""
    fp = _fp(_monthly_airline(), tmp_path, name="cfg.csv")
    res = run_analysis(fp, _CAT.by_id("arima"), output_root=str(tmp_path / "r"),
                       config={"fit_seconds": 600, "search_seconds": 120})
    assert res.estimates.get("forecast_next") is not None


def test_the_fallback_respects_the_budget_instead_of_fitting_the_worst_model(tmp_path):
    """Cold review A1 (MUST-FIX): when every candidate was gated out, the CALLER overwrote
    _auto_order's budget-respecting order with (1,d,1)(1,D,1,sp) and fitted it — the very
    shape the gate exists to avoid (219s at sp=52). The protection was void on this path.

    Driven through the public config key, so it is the user-reachable path that is pinned."""
    fp = _fp(_monthly_airline(n=72), tmp_path, name="fb.csv")
    res = run_analysis(fp, _CAT.by_id("arima"), output_root=str(tmp_path / "s"),
                       config={"fit_seconds": 1e-9})       # gate out literally everything
    e = res.estimates
    assert (e["P"], e["Q"]) == (0.0, 0.0), f"fallback must not carry seasonal AR/MA: {e}"
    assert e["p"] <= 1.0 and e["q"] <= 1.0, e
    # ...and it must not be described as the best of anything
    assert "未经 AICc 比较" in res.summary, res.summary[-400:]
    assert "可承受候选中的最优" not in res.summary


def test_the_skip_notice_does_not_invent_a_magnitude(tmp_path):
    """Cold review A4: the old text asserted '单次拟合可达数百秒' irrespective of sp and n,
    and printed it on an sp=12 / n=72 run whose seasonal fits take well under a second."""
    fp = _fp(_monthly_airline(n=72), tmp_path, name="mag.csv")
    res = run_analysis(fp, _CAT.by_id("arima"), output_root=str(tmp_path / "t"),
                       config={"fit_seconds": 1e-9})
    assert "数百秒" not in res.summary
    assert "被跳过" in res.summary
