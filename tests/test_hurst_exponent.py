"""Tests for the hurst_exponent (rescaled-range / R/S analysis) executor branch.

Known structure: a persistent / trending series (cumulative random walk -> strongly
trending levels) gives Hurst > 0.5; independent white noise gives Hurst ≈ 0.5.
Plus the log-log fit R², the regime encoding, CSV shape, and an honest-degrade test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="hurst_exponent", method="Hurst exponent (R/S)", domain="time series",
        family="time-series", goal="explore",
        preconditions=Precondition(is_timeseries=True, min_rows=32),
    )


def test_persistent_series_high_hurst(tmp_path: Path) -> None:
    # a random walk (cumulative sum of noise) is strongly persistent/trending -> H > 0.5
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(0, 1, 2000))
    df = pd.DataFrame({"t": np.arange(len(y)), "x": y})
    csv = tmp_path / "rw.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"column": "x"})
    assert "完成" in res.summary
    e = res.estimates
    assert e["hurst"] > 0.5
    assert e["regime"] == 2.0          # persistent / trending
    assert 0.0 <= e["rs_fit_r2"] <= 1.0
    assert e["rs_fit_r2"] > 0.8        # random walk scales cleanly
    assert e["n"] == 2000.0
    rs = pd.read_csv(Path(res.output_dir) / "hurst_rs.csv")
    assert {"window_size", "log_window", "log_rs"} == set(rs.columns)
    assert len(rs) >= 3


def test_white_noise_hurst_near_half(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    y = rng.normal(0, 1, 2000)
    df = pd.DataFrame({"t": np.arange(len(y)), "x": y})
    csv = tmp_path / "wn.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"column": "x"})
    assert "完成" in res.summary
    # white noise -> Hurst around 0.5 (R/S has a mild small-sample upward bias)
    assert 0.40 < res.estimates["hurst"] < 0.62
    assert res.estimates["regime"] in (0.0, 1.0, 2.0)


def test_degrade_too_short(tmp_path: Path) -> None:
    # below min_n=32 -> honest skip, no estimates fabricated
    rng = np.random.default_rng(2)
    y = rng.normal(0, 1, 20)
    df = pd.DataFrame({"t": np.arange(len(y)), "x": y})
    csv = tmp_path / "s.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"column": "x"})
    assert "跳过" in res.summary
    assert "hurst" not in res.estimates


def test_degrade_constant_series(tmp_path: Path) -> None:
    df = pd.DataFrame({"t": np.arange(200), "x": np.full(200, 3.0)})
    csv = tmp_path / "c.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"column": "x"})
    assert "跳过" in res.summary
    assert "hurst" not in res.estimates
