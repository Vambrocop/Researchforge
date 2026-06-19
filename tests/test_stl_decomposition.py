"""Tests for the STL seasonal-trend decomposition executor branch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="stl_decomposition", method="STL decomposition", domain="time series",
        family="time-series", goal="explain",
        preconditions=Precondition(is_timeseries=True, min_rows=20),
    )


def test_stl_auto_detects_period_and_strength(tmp_path: Path) -> None:
    # trend + period-12 seasonal + noise -> auto-detect period 12, strong Fs/Ft
    rng = np.random.default_rng(0)
    t = np.arange(240)
    y = 0.05 * t + 3 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.6, 240)
    df = pd.DataFrame({"t": t, "sales": y})
    csv = tmp_path / "s.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"value": "sales"})
    assert "完成" in res.summary
    assert res.estimates["period"] == 12
    assert res.estimates["seasonal_strength"] > 0.5
    assert res.estimates["trend_strength"] > 0.5
    comp = pd.read_csv(Path(res.output_dir) / "stl_components.csv")
    assert {"trend", "seasonal", "resid"} <= set(comp.columns)


def test_stl_config_period_override(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    t = np.arange(140)
    y = 2 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 0.5, 140)  # weekly
    df = pd.DataFrame({"t": t, "v": y})
    csv = tmp_path / "w.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"value": "v", "period": 7})
    assert "完成" in res.summary
    assert res.estimates["period"] == 7
    assert res.estimates["seasonal_strength"] > 0.5


def test_stl_no_seasonality_fails(tmp_path: Path) -> None:
    # pure noise + trend, no seasonality -> no clear periodogram peak -> honest fail
    rng = np.random.default_rng(5)
    t = np.arange(240)
    y = 0.05 * t + rng.normal(0, 1, 240)
    df = pd.DataFrame({"t": t, "x": y})
    csv = tmp_path / "n.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"value": "x"})
    assert "STL 分解失败" in res.summary and "季节周期" in res.summary
