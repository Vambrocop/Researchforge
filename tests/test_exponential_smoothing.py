"""Tests for the Holt-Winters exponential-smoothing forecasting branch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="exponential_smoothing", method="Exponential smoothing (Holt-Winters)",
        domain="time series", family="time-series", goal="predict",
        preconditions=Precondition(is_timeseries=True, min_rows=12),
    )


def test_ets_trend_seasonal_recovers_sensible_next_step(tmp_path: Path) -> None:
    # trend + period-12 seasonal + small noise -> next-step forecast continues the level
    rng = np.random.default_rng(0)
    t = np.arange(120)
    y = 10 + 0.3 * t + 5 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.5, 120)
    df = pd.DataFrame({"t": t, "sales": y})
    csv = tmp_path / "s.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"column": "sales"})
    assert "完成" in res.summary
    est = res.estimates
    for k in ("alpha", "aic", "sse", "forecast_next", "n", "seasonal_periods"):
        assert k in est, f"missing estimate {k}"
    assert est["n"] == 120
    # next-step forecast should be in the same ballpark as the recent level (trend up ~46)
    assert 30 < est["forecast_next"] < 70
    # forecast CSV with PI columns
    fc = pd.read_csv(Path(res.output_dir) / "forecast.csv")
    assert {"step", "point", "lower", "upper"} <= set(fc.columns)
    assert len(fc) == 10
    assert (fc["lower"] <= fc["point"]).all() and (fc["point"] <= fc["upper"]).all()


def test_ets_config_h_and_forced_trend(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    t = np.arange(60)
    y = 5 + 0.5 * t + rng.normal(0, 0.4, 60)
    df = pd.DataFrame({"t": t, "v": y})
    csv = tmp_path / "v.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"column": "v", "h": 5, "trend": "add", "seasonal": "none"})
    assert "完成" in res.summary
    assert res.estimates["seasonal_periods"] == 0.0
    fc = pd.read_csv(Path(res.output_dir) / "forecast.csv")
    assert len(fc) == 5
    # upward trend -> last forecast above first
    assert fc["point"].iloc[-1] > fc["point"].iloc[0]


def test_ets_too_short_degrades(tmp_path: Path) -> None:
    df = pd.DataFrame({"t": range(6), "v": [1.0, 2, 3, 2, 1, 2]})
    csv = tmp_path / "short.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"column": "v"})
    assert "跳过" in res.summary and "n=6" in res.summary
    assert res.estimates == {}


def test_ets_constant_series_degrades(tmp_path: Path) -> None:
    df = pd.DataFrame({"t": range(30), "v": [7.0] * 30})
    csv = tmp_path / "c.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"column": "v"})
    assert "跳过" in res.summary and "常数" in res.summary
