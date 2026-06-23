"""Tests for the Theta-method forecasting branch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="theta_method", method="Theta method",
        domain="time series", family="time-series", goal="predict",
        preconditions=Precondition(is_timeseries=True, min_rows=12),
    )


def test_theta_trend_recovers_positive_drift_and_next_step(tmp_path: Path) -> None:
    # clean upward linear trend -> positive drift, next-step continues the line
    rng = np.random.default_rng(0)
    t = np.arange(80)
    y = 20 + 1.2 * t + rng.normal(0, 0.6, 80)
    df = pd.DataFrame({"t": t, "demand": y})
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"column": "demand", "seasonal": "none"})
    assert "完成" in res.summary
    est = res.estimates
    for k in ("forecast_next", "ses_alpha", "drift", "n", "h"):
        assert k in est, f"missing estimate {k}"
    assert est["n"] == 80 and est["h"] == 10
    assert est["drift"] > 0.5  # recovers the ~1.2 slope direction/magnitude
    # next step should continue the line (last obs ~ 20 + 1.2*79 ~ 115)
    assert 105 < est["forecast_next"] < 130
    fc = pd.read_csv(Path(res.output_dir) / "forecast.csv")
    assert {"step", "point"} <= set(fc.columns)
    assert len(fc) == 10
    assert fc["point"].iloc[-1] > fc["point"].iloc[0]  # forecast keeps rising


def test_theta_config_h(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    t = np.arange(50)
    y = 100 - 0.4 * t + rng.normal(0, 0.5, 50)  # downward trend
    df = pd.DataFrame({"t": t, "v": y})
    csv = tmp_path / "v.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"column": "v", "h": 6, "seasonal": "none"})
    assert "完成" in res.summary
    assert res.estimates["drift"] < 0
    fc = pd.read_csv(Path(res.output_dir) / "forecast.csv")
    assert len(fc) == 6


def test_theta_too_short_degrades(tmp_path: Path) -> None:
    df = pd.DataFrame({"t": range(8), "v": [1.0, 2, 3, 4, 5, 4, 3, 2]})
    csv = tmp_path / "short.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"column": "v"})
    assert "跳过" in res.summary
    assert res.estimates == {}
