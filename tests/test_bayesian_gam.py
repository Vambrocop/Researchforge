"""Tests for the bayesian_gam executor branch (PyMC Bayesian penalized-spline GAM).

Known nonlinear signal y = sin(2*pi*x) + noise -> the smooth should recover the
curve (high correlation with the true function on the grid) with a sensible credible
band. Plus config override and honest-degrade. PyMC-backed -> slow (see conftest).
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
        id="bayesian_gam", method="Bayesian GAM", domain="statistics",
        family="statistics", goal="model", preconditions=Precondition(min_rows=10),
    )


def _sine(tmp_path: Path, n: int = 150) -> Path:
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 1, n))
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.2, n)
    csv = tmp_path / "sine.csv"
    pd.DataFrame({"y": y.round(5), "x": x.round(5)}).to_csv(csv, index=False)
    return csv


@pytest.mark.skipif(not _HAS_PYMC, reason="pymc/arviz not available")
def test_recovers_nonlinear_smooth(tmp_path: Path) -> None:
    fp = profile_dataset(_sine(tmp_path))
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "y", "predictors": ["x"], **_FAST})
    assert "完成" in res.summary
    assert res.estimates["n"] == 150
    assert res.estimates["bayesian_r2"] > 0.6           # captures the sine signal
    assert res.estimates["smooth_range_x"] > 0.5        # genuinely wiggly (not a flat line)
    # smooth recovers the true sine on the grid
    sm = pd.read_csv(Path(res.output_dir) / "gam_smooths.csv").sort_values("x")
    truth = np.sin(2 * np.pi * sm["x"].to_numpy())
    truth = truth - truth.mean()
    assert np.corrcoef(sm["smooth"], truth)[0, 1] > 0.9
    assert (sm["lo"] <= sm["smooth"]).all() and (sm["smooth"] <= sm["hi"]).all()
    assert "gam_smooth.png" in res.files


@pytest.mark.skipif(not _HAS_PYMC, reason="pymc/arviz not available")
def test_config_outcome_override(tmp_path: Path) -> None:
    fp = profile_dataset(_sine(tmp_path, n=120))
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "y", "predictors": ["x"], **_FAST})
    assert res.estimates["n"] == 120
    assert "linear_slope_x" in res.estimates and "smoothing_sd_x" in res.estimates


def test_degrade_too_few_rows(tmp_path: Path) -> None:
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0], "x": [1.0, 2.0, 1.5, 3.0, 2.5]})
    csv = tmp_path / "tiny.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "y", "predictors": ["x"]})
    # either pymc missing (skip msg) or too-few-rows skip; never a crash, no estimates
    assert "跳过" in res.summary
    assert "bayesian_r2" not in res.estimates
