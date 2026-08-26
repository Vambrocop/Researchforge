"""Wave M8 — calendar-aware seasonal-period detection for forecasting.

Deep-dogfood finding: on a short MONTHLY series the periodogram gives the seasonal period
too few cycles to certify (or picks a wrong harmonic — 6 instead of 12), so Holt-Winters fits
a trend-only model and misses obvious calendar seasonality. The fix adds the CALENDAR period
(monthly→12, quarterly→4, weekly→52, daily→7) as a strong candidate, USED only when a
seasonal-strength check at that period confirms real seasonal signal — so a non-seasonal
monthly series is not forced into an 11-parameter seasonal model.

These tests pin both directions: clear calendar seasonality is now detected (where the
periodogram alone misses/mis-detects it), and a trend-only series stays non-seasonal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from researchforge.catalog import Catalog
from researchforge.executor import run_analysis
from researchforge.executor._branch_api import Ctx
from researchforge.executor.branches.forecasting import (
    _date_seasonal_period,
    _detect_period,
    _seasonal_strength_ok,
)
from researchforge.executor.branches.timeseries import _periodogram_period
from researchforge.profiler import profile_dataset


def _monthly_seasonal(n: int = 60, seed: int = 7) -> pd.DataFrame:
    # clear December-peak seasonality (amplitude dominates), mild trend, low noise.
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    seasonal = np.array([-8, -6, -2, 2, 6, 9, 8, 5, 1, 4, 12, 20])[t % 12]
    rev = (100 + 0.8 * t + seasonal + rng.normal(0, 2.5, n)).round(1)
    return pd.DataFrame({"month": pd.date_range("2019-01-01", periods=n, freq="MS").strftime("%Y-%m"),
                         "revenue": rev})


def _monthly_trend_only(n: int = 60, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return pd.DataFrame({"month": pd.date_range("2019-01-01", periods=n, freq="MS").strftime("%Y-%m"),
                         "y": (100 + 1.2 * t + rng.normal(0, 3, n)).round(1)})


def _daily_random_walk(n: int = 500, seed: int = 88) -> pd.DataFrame:
    # a daily unit-root series (stock-price-like). GUARD: a random walk keeps strong
    # autocorrelation at every lag after mere linear detrending — it must NOT be handed a
    # spurious weekly (period-7) season (regression the first M8 cut introduced; fixed by
    # first-differencing before the ACF check).
    rng = np.random.default_rng(seed)
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"date": pd.bdate_range("2022-01-03", periods=n).strftime("%Y-%m-%d"),
                         "close": price.round(3)})


def _daily_weekly(n: int = 140, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    v = (50 + 7 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 2, n)).round(2)
    return pd.DataFrame({"date": pd.date_range("2022-01-01", periods=n).strftime("%Y-%m-%d"), "v": v})


def _ctx_and_y(df: pd.DataFrame, tmp_path, cfg=None):
    csv = tmp_path / "t.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    vcol = next(c.name for c in fp.columns if c.kind == "continuous")
    d2 = df.sort_values(fp.time_col) if (fp.time_col and fp.time_col in df.columns) else df
    y = pd.to_numeric(d2[vcol], errors="coerce").dropna().to_numpy(float)
    ctx = Ctx(df=df, fp=fp, entry=None, cfg=cfg or {}, d=None,
              files=[], summary=[], estimates={}, code=[])
    return ctx, y


# ── calendar period is detected where the periodogram misses / mis-detects it ─────────
def test_monthly_seasonality_detected_when_periodogram_misdetects(tmp_path):
    df = _monthly_seasonal()
    ctx, y = _ctx_and_y(df, tmp_path)
    assert _date_seasonal_period(ctx, len(y)) == 12
    assert _detect_period(ctx, y) == 12
    # the periodogram alone gets it wrong here (a short series → wrong harmonic), which is the
    # whole reason the calendar prior is needed.
    assert _periodogram_period(y, len(y)) != 12


def test_trend_only_monthly_stays_non_seasonal(tmp_path):
    df = _monthly_trend_only()
    ctx, y = _ctx_and_y(df, tmp_path)
    assert _date_seasonal_period(ctx, len(y)) == 12  # calendar candidate exists
    assert _seasonal_strength_ok(y, 12) is False     # but no real seasonal signal
    assert _detect_period(ctx, y) is None            # → not forced into a seasonal model


def test_daily_weekly_season_detected(tmp_path):
    df = _daily_weekly()
    ctx, y = _ctx_and_y(df, tmp_path)
    assert _date_seasonal_period(ctx, len(y)) == 7
    assert _detect_period(ctx, y) == 7


def test_random_walk_not_given_spurious_season(tmp_path):
    # regression guard: a unit-root daily series must NOT get a fake weekly period-7 season.
    df = _daily_random_walk()
    ctx, y = _ctx_and_y(df, tmp_path)
    assert _date_seasonal_period(ctx, len(y)) == 7  # calendar candidate exists (daily)
    assert _seasonal_strength_ok(y, 7) is False     # but differencing shows no real season
    assert _detect_period(ctx, y) is None


def test_config_seasonal_periods_overrides(tmp_path):
    df = _monthly_seasonal()
    ctx, y = _ctx_and_y(df, tmp_path, cfg={"seasonal_periods": 4})
    assert _detect_period(ctx, y) == 4  # explicit config always wins


# ── seasonal-strength unit: rejects the white-noise band ──────────────────────────────
def test_seasonal_strength_rejects_noise():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, 60)
    assert _seasonal_strength_ok(noise, 12) is False


def test_seasonal_strength_accepts_strong_season():
    t = np.arange(72)
    y = 10 * np.sin(2 * np.pi * t / 12) + np.random.default_rng(1).normal(0, 1, 72)
    assert _seasonal_strength_ok(y, 12) is True


# ── end-to-end: Holt-Winters fits a seasonal model on clear calendar seasonality ──────
def test_exp_smoothing_fits_seasonal_on_calendar_data(tmp_path):
    fp = profile_dataset(_write(_monthly_seasonal(), tmp_path, "s.csv"))
    res = run_analysis(fp, Catalog.load().by_id("exponential_smoothing"),
                       output_root=str(tmp_path / "o1"))
    assert res.estimates.get("seasonal_periods") == 12
    assert "季节" in res.summary


def test_exp_smoothing_stays_non_seasonal_on_trend_only(tmp_path):
    fp = profile_dataset(_write(_monthly_trend_only(), tmp_path, "n.csv"))
    res = run_analysis(fp, Catalog.load().by_id("exponential_smoothing"),
                       output_root=str(tmp_path / "o2"))
    assert res.estimates.get("seasonal_periods") in (0, 0.0)


def _write(df: pd.DataFrame, tmp_path, name: str) -> str:
    p = tmp_path / name
    df.to_csv(p, index=False)
    return str(p)
