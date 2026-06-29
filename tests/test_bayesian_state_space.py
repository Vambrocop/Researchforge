"""Tests for the bayesian_state_space executor branch (Bayesian structural TS).

Known trending series y = 2 + 0.1*t + noise -> the local-linear-trend model should
recover the rising level (high correlation with the truth), report a positive slope,
and forecast upward with widening predictive intervals. Plus local-level config and
honest degrade. PyMC-backed -> slow (see conftest).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_HAS_PYMC = (importlib.util.find_spec("pymc") is not None
             and importlib.util.find_spec("arviz") is not None)
_FAST = {"draws": 400, "tune": 700, "chains": 2}


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="bayesian_state_space", method="Bayesian structural time series",
        domain="time-series", family="time-series", goal="forecast",
        preconditions=Precondition(min_rows=20, requires_timeseries=True),
    )


def _trend(tmp_path: Path, T: int = 70, slope: float = 0.1) -> Path:
    rng = np.random.default_rng(0)
    t = np.arange(T)
    y = 2.0 + slope * t + rng.normal(0, 0.4, T)
    csv = tmp_path / "trend.csv"
    pd.DataFrame({"y": y.round(5)}).to_csv(csv, index=False)
    return csv


@pytest.mark.skipif(not _HAS_PYMC, reason="pymc/arviz not available")
def test_recovers_trend_and_forecasts(tmp_path: Path) -> None:
    fp = profile_dataset(_trend(tmp_path))
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"value": "y", **_FAST})
    assert "完成" in res.summary
    assert res.estimates["n"] == 70
    assert res.estimates["local_linear_trend"] == 1.0
    assert res.estimates["final_slope"] > 0          # positive trend detected
    # smoothed level recovers the rising true level
    states = pd.read_csv(Path(res.output_dir) / "state_space_states.csv")
    truth = 2.0 + 0.1 * states["t"].to_numpy()
    assert np.corrcoef(states["level_mean"], truth)[0, 1] > 0.95
    # forecast continues upward, ordered band, widening intervals
    fc = pd.read_csv(Path(res.output_dir) / "state_space_forecast.csv")
    assert (fc["lo"] <= fc["forecast"]).all() and (fc["forecast"] <= fc["hi"]).all()
    assert fc["forecast"].iloc[-1] > res.estimates["final_level"]
    width = fc["hi"] - fc["lo"]
    assert width.iloc[-1] > width.iloc[0]            # predictive interval widens
    assert "state_space.png" in res.files


@pytest.mark.skipif(not _HAS_PYMC, reason="pymc/arviz not available")
def test_local_level_model(tmp_path: Path) -> None:
    fp = profile_dataset(_trend(tmp_path))
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"value": "y", "model": "local_level", **_FAST})
    assert res.estimates["local_linear_trend"] == 0.0
    assert "final_slope" not in res.estimates       # no slope state in local-level
    assert "sigma_level" in res.estimates


def test_degrade_too_short(tmp_path: Path) -> None:
    df = pd.DataFrame({"y": np.arange(10, dtype=float)})
    csv = tmp_path / "short.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"value": "y"})
    assert "跳过" in res.summary
    assert "final_level" not in res.estimates
