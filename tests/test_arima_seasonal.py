"""SARIMA upgrade — ARIMA seasonalization (twin of the M8 Holt-Winters seasonal fix).

The ARIMA branch was fixed ARIMA(1,1,1), non-seasonal, so on calendar-seasonal data it gave a
flat forecast while Holt-Winters (M8) captured the season (dogfood: retail). This upgrade reuses
the calendar-aware, strength-confirmed period detector (forecasting._detect_period) to lift
ARIMA(1,1,1) to SARIMA(1,1,1)(1,1,1)[P] when — and only when — a real seasonal period is
confirmed, with an honest fall-back to the non-seasonal fit if SARIMAX cannot converge.

Pins: SARIMA is used on clear calendar seasonality (and its forecast actually varies with the
season), a trend-only / random-walk series stays non-seasonal ARIMA, and config can override.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from researchforge.catalog import Catalog
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

warnings.filterwarnings("ignore")  # near-deterministic seasonal data → statsmodels convergence noise


def _monthly_seasonal(n: int = 60, seed: int = 7) -> pd.DataFrame:
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


def _daily_random_walk(n: int = 400, seed: int = 88) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"date": pd.bdate_range("2022-01-03", periods=n).strftime("%Y-%m-%d"),
                         "close": price.round(3)})


def _run(df: pd.DataFrame, tmp_path, cfg=None):
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    return run_analysis(fp, Catalog.load().by_id("arima"), output_root=str(tmp_path / "o"),
                        config=cfg)


def test_sarima_used_on_calendar_seasonality(tmp_path):
    res = _run(_monthly_seasonal(), tmp_path)
    assert res.estimates.get("seasonal_periods") == 12
    assert "SARIMA(1,1,1)(1,1,1)[12]" in res.summary


def test_sarima_forecast_varies_with_season(tmp_path):
    # the whole point: a seasonal forecast is NOT a flat line — its 12-step spread is material.
    res = _run(_monthly_seasonal(), tmp_path)
    fc = pd.read_csv(f"{res.output_dir}/forecast.csv")["forecast"].to_numpy()
    assert fc.std() > 3.0, f"SARIMA forecast should carry seasonal variation, got std={fc.std():.2f}"


def test_trend_only_stays_nonseasonal_arima(tmp_path):
    res = _run(_monthly_trend_only(), tmp_path)
    assert res.estimates.get("seasonal_periods") in (0, 0.0)
    assert "ARIMA(1,1,1)" in res.summary and "SARIMA" not in res.summary


def test_random_walk_stays_nonseasonal_arima(tmp_path):
    res = _run(_daily_random_walk(), tmp_path)
    assert res.estimates.get("seasonal_periods") in (0, 0.0)
    assert "SARIMA" not in res.summary


def test_config_seasonal_none_forces_arima(tmp_path):
    res = _run(_monthly_seasonal(), tmp_path, cfg={"seasonal": "none"})
    assert res.estimates.get("seasonal_periods") in (0, 0.0)
    assert "SARIMA" not in res.summary


def test_config_seasonal_periods_override(tmp_path):
    # explicitly force a (wrong-but-valid) period; the branch must honor it.
    res = _run(_monthly_seasonal(), tmp_path, cfg={"seasonal_periods": 4})
    assert res.estimates.get("seasonal_periods") == 4
    assert "SARIMA(1,1,1)(1,1,1)[4]" in res.summary


def test_small_sample_seasonal_falls_back_to_arima(tmp_path):
    # inference-review MUST-FIX: a season is confirmed but there are < 3 full cycles, so the
    # seasonal-difference model is non-identifiable (SARIMAX would report converged=True with a
    # boundary AIC). The branch must fall back to non-seasonal ARIMA with an honest disclosure —
    # NOT silently emit a degenerate SARIMA.
    rng = np.random.default_rng(1)
    t = np.arange(30)  # 2.5 cycles of a monthly season → < 3*sp=36
    strong = np.array([-20, -15, -5, 5, 15, 25, 20, 12, 2, 10, 30, 45])[t % 12]
    df = pd.DataFrame({"month": pd.date_range("2020-01-01", periods=30, freq="MS").strftime("%Y-%m"),
                       "rev": (100 + 0.5 * t + strong + rng.normal(0, 1.0, 30)).round(1)})
    res = _run(df, tmp_path)
    assert res.estimates.get("seasonal_periods") in (0, 0.0)
    assert "SARIMA" not in res.summary
    assert "样本不足" in res.summary  # honest disclosure that the season was seen but not modeled


def test_convergence_section_excludes_incomparable_aic():
    # inference-review SHOULD-FIX: AIC/SSE/loglik are not comparable across methods (esp. across
    # differencing structures) — a shared key name must not produce a cross-method numeric claim.
    from types import SimpleNamespace

    from researchforge.study_report import _convergence_section

    def _entry(mid, est):
        return {"result": SimpleNamespace(estimates=est),
                "rec": SimpleNamespace(entry=SimpleNamespace(id=mid))}

    entries = [_entry("arima", {"aic": 12.0, "shared_stat": 1.0}),
               _entry("exponential_smoothing", {"aic": 500.0, "shared_stat": 1.1})]
    text = "\n".join(_convergence_section(entries))
    assert "`aic`" not in text          # AIC excluded from the cross-method table
    assert "`shared_stat`" in text      # a genuinely comparable shared key still appears
