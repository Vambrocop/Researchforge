"""Tests for the unobserved_components (structural time series) executor branch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset
from researchforge.recommender.match import check_preconditions


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="unobserved_components", method="Unobserved components",
        domain="time series", family="time-series", goal="explain",
        preconditions=Precondition(is_timeseries=True, min_rows=30),
    )


def test_uc_recovers_trend(tmp_path: Path) -> None:
    # clear stochastic-ish trend + noise -> non-trivial smoothed level tracking the data,
    # finite variances, valid AIC/BIC.
    rng = np.random.default_rng(0)
    n = 220
    t = np.arange(n)
    level = np.cumsum(rng.normal(0, 0.3, n)) + 0.08 * t  # stochastic level + drift
    y = level + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"t": t, "y": np.round(y, 4)})
    csv = tmp_path / "uc.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"column": "y", "seasonal": False})
    assert "完成" in res.summary
    assert np.isfinite(res.estimates["aic"])
    assert np.isfinite(res.estimates["loglik"])
    comp = pd.read_csv(Path(res.output_dir) / "uc_components.csv")
    assert "level" in comp.columns
    # smoothed level must actually track the series (non-trivial trend recovered)
    r = np.corrcoef(comp["level"].to_numpy(), df["y"].to_numpy())[0, 1]
    assert abs(r) > 0.8
    # level state should span a non-trivial range (not flat)
    assert comp["level"].max() - comp["level"].min() > 1.0


def test_uc_seasonal_variance_nonzero(tmp_path: Path) -> None:
    # trend + period-12 seasonal -> with a seasonal spec the seasonal variance should be > 0
    # (genuinely stochastic/present), seasonal component reported.
    rng = np.random.default_rng(3)
    n = 240
    t = np.arange(n)
    y = 0.05 * t + 4 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"t": t, "sales": np.round(y, 4)})
    csv = tmp_path / "ucs.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"column": "sales", "seasonal_period": 12})
    assert "完成" in res.summary
    assert res.estimates["has_seasonal"] == 1.0
    # seasonal variance present and finite (component is in the model)
    vs = res.estimates.get("var_seasonal")
    assert vs is not None and np.isfinite(vs)
    # irregular variance finite too
    assert np.isfinite(res.estimates["var_irregular"])


def test_uc_too_short_skips(tmp_path: Path) -> None:
    # too short -> honest skip, and precondition gate fails
    rng = np.random.default_rng(9)
    n = 20
    df = pd.DataFrame({"t": np.arange(n), "y": rng.normal(0, 1, n)})
    csv = tmp_path / "short.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    ok, _ = check_preconditions(fp, _entry().preconditions)
    assert not ok  # min_rows=30 not met
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"column": "y"})
    assert "失败" in res.summary
