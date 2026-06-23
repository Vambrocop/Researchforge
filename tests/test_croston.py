"""Tests for the Croston intermittent-demand forecasting branch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="croston", method="Croston's method (intermittent demand)",
        domain="time series", family="time-series", goal="predict",
        preconditions=Precondition(is_timeseries=True, min_rows=12),
    )


def _intermittent_series(n=60, every=4, size=10.0, seed=0):
    """Zero-heavy demand: a nonzero demand of ~`size` roughly every `every` periods."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    pos = np.arange(every - 1, n, every)
    y[pos] = size + rng.normal(0, 0.5, len(pos))
    return y


def test_croston_intermittent_rate_matches_size_over_interval(tmp_path: Path) -> None:
    # demand ~10 every 4 periods -> rate ~ 10/4 = 2.5
    y = _intermittent_series(n=60, every=4, size=10.0, seed=0)
    df = pd.DataFrame({"t": range(len(y)), "demand": y})
    csv = tmp_path / "i.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"column": "demand"})
    assert "完成" in res.summary
    est = res.estimates
    for k in ("forecast_rate", "sba_forecast", "pct_zero", "mean_interval", "n"):
        assert k in est, f"missing estimate {k}"
    assert est["n"] == 60
    assert est["pct_zero"] > 50  # genuinely intermittent
    assert abs(est["mean_interval"] - 4.0) < 0.6
    assert abs(est["forecast_rate"] - 2.5) < 1.0  # size/interval ~ 2.5
    # SBA correction reduces the (positively biased) Croston rate
    assert est["sba_forecast"] < est["forecast_rate"]
    fc = pd.read_csv(Path(res.output_dir) / "forecast.csv")
    assert {"step", "forecast_rate", "sba_forecast"} <= set(fc.columns)
    assert len(fc) == 10


def test_croston_flags_non_intermittent_series(tmp_path: Path) -> None:
    # dense (non-intermittent) series -> still runs but flags the low %zero
    rng = np.random.default_rng(1)
    y = 5 + rng.normal(0, 1, 40).clip(min=0.1)  # no zeros
    df = pd.DataFrame({"t": range(40), "v": y})
    csv = tmp_path / "dense.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"column": "v"})
    assert "完成" in res.summary
    assert res.estimates["pct_zero"] == 0.0
    assert "并非典型间断需求" in res.summary  # the non-intermittent flag


def test_croston_config_alpha(tmp_path: Path) -> None:
    y = _intermittent_series(n=50, every=5, size=8.0, seed=2)
    df = pd.DataFrame({"t": range(len(y)), "demand": y})
    csv = tmp_path / "a.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"column": "demand", "alpha": 0.3, "h": 4})
    assert "完成" in res.summary and "α=0.3" in res.summary
    fc = pd.read_csv(Path(res.output_dir) / "forecast.csv")
    assert len(fc) == 4


def test_croston_too_short_degrades(tmp_path: Path) -> None:
    df = pd.DataFrame({"t": range(8), "v": [0.0, 0, 5, 0, 0, 3, 0, 0]})
    csv = tmp_path / "short.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"column": "v"})
    assert "跳过" in res.summary
    assert res.estimates == {}
