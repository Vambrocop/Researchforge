"""ARIMA/SARIMA auto-order (AICc grid) + prediction intervals.

Two user-visible methodology gaps closed here:
  * the order was hardcoded (1,1,1) — any time-series reviewer asks "why no order selection?",
    and forcing d=1 OVER-DIFFERENCES an already-stationary series;
  * ARIMA/SARIMA emitted a point forecast with NO interval, which is not reportable.

Design pinned by these tests: `d` is fixed by a unit-root TEST (never by AIC — likelihoods
across different differencing are computed on different effective samples and are not
comparable), then (p,q[,P,Q]) are ranked by AICc on that fixed `d`.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from researchforge.catalog import Catalog
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

warnings.filterwarnings("ignore")

_CAT = Catalog.load()


def _dates(n):
    return pd.date_range("2015-01-01", periods=n, freq="D").strftime("%Y-%m-%d")


def _run(df, tmp_path, cfg=None):
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    return run_analysis(fp, _CAT.by_id("arima"), output_root=str(tmp_path / "o"), config=cfg)


def _stationary_ar1(n=120, seed=7):
    rng = np.random.default_rng(seed)
    v = np.zeros(n)
    for i in range(1, n):
        v[i] = 0.7 * v[i - 1] + rng.normal(0, 1)
    return pd.DataFrame({"date": _dates(n), "v": (50 + v).round(3)})


def _random_walk(n=120, seed=7):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"date": _dates(n), "v": (100 + np.cumsum(rng.normal(0, 1, n))).round(3)})


# ── d comes from the unit-root test, not from AIC ─────────────────────────────────────
def test_stationary_series_is_not_differenced(tmp_path):
    res = _run(_stationary_ar1(), tmp_path)
    assert res.estimates["d"] == 0, "a stationary AR(1) must NOT be differenced (over-differencing)"


def test_random_walk_is_differenced_once(tmp_path):
    res = _run(_random_walk(), tmp_path)
    assert res.estimates["d"] == 1


def test_summary_discloses_selection_rule(tmp_path):
    res = _run(_random_walk(), tmp_path)
    assert "自动定阶" in res.summary and "AICc" in res.summary
    assert "跨 d 不可比" in res.summary  # the AIC-across-differencing trap is disclosed


# ── prediction interval ───────────────────────────────────────────────────────────────
def test_forecast_carries_prediction_interval(tmp_path):
    res = _run(_random_walk(), tmp_path)
    fc = pd.read_csv(f"{res.output_dir}/forecast.csv")
    assert list(fc.columns) == ["step", "forecast", "lower", "upper"]
    assert (fc["lower"] <= fc["forecast"]).all() and (fc["forecast"] <= fc["upper"]).all()
    assert {"pi_lower_next", "pi_upper_next", "forecast_next"} <= set(res.estimates)


def test_interval_widens_with_horizon(tmp_path):
    """A differenced (unit-root) forecast's uncertainty must grow with the horizon."""
    res = _run(_random_walk(), tmp_path)
    fc = pd.read_csv(f"{res.output_dir}/forecast.csv")
    width = (fc["upper"] - fc["lower"]).to_numpy()
    assert width[-1] > width[0]


def test_ci_level_is_configurable(tmp_path):
    wide = _run(_random_walk(), tmp_path, cfg={"ci": 0.99})
    narrow = _run(_random_walk(), tmp_path, cfg={"ci": 0.80})
    w = wide.estimates["pi_upper_next"] - wide.estimates["pi_lower_next"]
    n = narrow.estimates["pi_upper_next"] - narrow.estimates["pi_lower_next"]
    assert w > n, "a 99% interval must be wider than an 80% one"


# ── config overrides + honest disclosure ──────────────────────────────────────────────
def test_config_d_overrides_the_test(tmp_path):
    res = _run(_random_walk(), tmp_path, cfg={"d": 0})
    assert res.estimates["d"] == 0 and "config" in res.summary


def test_grid_bounds_are_configurable_and_disclosed(tmp_path):
    res = _run(_random_walk(), tmp_path, cfg={"max_p": 1, "max_q": 1})
    assert res.estimates["p"] <= 1 and res.estimates["q"] <= 1
    assert "p≤1, q≤1" in res.summary


def test_boundary_pick_is_flagged(tmp_path):
    """When the winner sits ON the grid edge the search was cut short — say so."""
    res = _run(_stationary_ar1(), tmp_path, cfg={"max_p": 1, "max_q": 0})
    if res.estimates["p"] == 1:
        assert "网格边界" in res.summary


def test_aicc_is_reported(tmp_path):
    res = _run(_random_walk(), tmp_path)
    assert np.isfinite(res.estimates["aicc"]) and np.isfinite(res.estimates["aic"])


# ── order SANITY (inference-review: the whole suite passed with two must-fix bugs present,
# because nothing asserted that the SELECTED ORDER is reasonable — only that some order existed)
def test_random_walk_selects_a_parsimonious_order(tmp_path):
    """Truth is (0,1,0). A search whose likelihoods are computed on different effective samples
    stampedes to the grid maximum — pin parsimony so that regression cannot return."""
    res = _run(_random_walk(), tmp_path)
    assert res.estimates["p"] + res.estimates["q"] <= 1, (
        f"random walk should need ~no AR/MA terms, got "
        f"({res.estimates['p']},{res.estimates['d']},{res.estimates['q']})")


def test_seasonal_series_selects_a_parsimonious_order(tmp_path):
    """The monthly DGP is close to the airline model (0,1,1)(0,1,1)[12] — total order should be
    small, not the near-maximum the broken comparison used to pick."""
    rng = np.random.default_rng(7)
    n = 60
    t = np.arange(n)
    seas = np.array([-8, -6, -2, 2, 6, 9, 8, 5, 1, 4, 12, 20])[t % 12]
    df = pd.DataFrame({"month": pd.date_range("2019-01-01", periods=n, freq="MS").strftime("%Y-%m"),
                       "revenue": (100 + 0.8 * t + seas + rng.normal(0, 2.5, n)).round(1)})
    res = _run(df, tmp_path)
    total = (res.estimates["p"] + res.estimates["q"]
             + res.estimates.get("P", 0) + res.estimates.get("Q", 0))
    assert total <= 3, f"seasonal order should stay parsimonious, got total={total}"


def test_undifferenced_series_keeps_its_level(tmp_path):
    """d=0 must fit a CONSTANT: SARIMAX defaults to a mean-zero process, so a series around 500
    would forecast 0.0 with an absurd interval. Pins the intercept."""
    rng = np.random.default_rng(3)
    n = 120
    df = pd.DataFrame({"date": _dates(n), "v": (500 + rng.normal(0, 5, n)).round(3)})
    res = _run(df, tmp_path)
    assert res.estimates["d"] == 0
    assert abs(res.estimates["forecast_next"] - 500) < 25, (
        f"forecast {res.estimates['forecast_next']:.2f} lost the series level (~500)")
    assert res.estimates["pi_lower_next"] > 400  # interval must not be absurdly wide


def test_stationary_ar1_mean_reverts(tmp_path):
    """An AR(1) around 50 must mean-revert toward the sample mean, not drift away."""
    res = _run(_stationary_ar1(), tmp_path)
    assert abs(res.estimates["forecast_next"] - 50) < 5
