"""Tests for the structural-break (change-point) executor branch."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

pytestmark = pytest.mark.skipif(importlib.util.find_spec("ruptures") is None, reason="ruptures not installed")


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="structural_breaks", method="Structural break detection", domain="economics",
        family="time-series", goal="explain",
        preconditions=Precondition(is_timeseries=True, min_rows=30),
    )


def test_structural_breaks_finds_mean_shift(tmp_path: Path) -> None:
    # mean shifts 0 -> 5 at t=120; PELT should find ~1 break near 120
    rng = np.random.default_rng(0)
    y = np.r_[rng.normal(0, 1, 120), rng.normal(5, 1, 180)]
    df = pd.DataFrame({"t": np.arange(len(y)), "level": y})
    csv = tmp_path / "b.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"value": "level"})
    assert "完成" in res.summary
    assert res.estimates["n_breaks"] == 1
    seg = pd.read_csv(Path(res.output_dir) / "segments.csv")
    # the single detected break should sit close to the true location (120)
    assert abs(int(seg.iloc[0]["end"]) - 120) <= 10


def test_structural_breaks_none_on_noise(tmp_path: Path) -> None:
    # pure iid noise -> no real breaks -> 0 detected (BIC penalty suppresses false positives)
    rng = np.random.default_rng(5)
    y = rng.normal(0, 1, 300)
    df = pd.DataFrame({"t": np.arange(len(y)), "x": y})
    csv = tmp_path / "n.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"value": "x"})
    assert "完成" in res.summary
    assert res.estimates["n_breaks"] == 0


def test_structural_breaks_fixed_n_and_trend_flag(tmp_path: Path) -> None:
    # strong linear trend -> n_breaks fixed to 2 via config; trend confound flagged
    rng = np.random.default_rng(2)
    y = 0.1 * np.arange(300) + rng.normal(0, 1, 300)
    df = pd.DataFrame({"t": np.arange(len(y)), "g": y})
    csv = tmp_path / "tr.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"value": "g", "n_breaks": 2})
    assert "完成" in res.summary
    assert res.estimates["n_breaks"] == 2
    assert res.estimates["trend_abs_corr"] > 0.7
    assert "强线性趋势" in res.summary


def test_structural_breaks_needs_obs(tmp_path: Path) -> None:
    df = pd.DataFrame({"t": np.arange(15), "x": np.random.default_rng(1).normal(0, 1, 15)})
    csv = tmp_path / "x.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"value": "x"})
    assert "结构突变检测失败" in res.summary
