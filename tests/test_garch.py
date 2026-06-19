"""Tests for the GARCH(1,1) volatility executor branch."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

pytestmark = pytest.mark.skipif(importlib.util.find_spec("arch") is None, reason="arch not installed")


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="garch", method="GARCH(1,1) volatility", domain="economics",
        family="time-series", goal="explain",
        preconditions=Precondition(is_timeseries=True, min_rows=50),
    )


def _garch_series(seed: int, n: int = 800) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 1, n)
    h = np.ones(n)
    y = np.zeros(n)
    for t in range(1, n):
        h[t] = 0.05 + 0.1 * y[t - 1] ** 2 + 0.85 * h[t - 1]  # persistence 0.95, clustering
        y[t] = np.sqrt(h[t]) * e[t]
    return y


def test_garch_recovers_volatility_clustering(tmp_path: Path) -> None:
    y = _garch_series(0)
    df = pd.DataFrame({"t": np.arange(len(y)), "ret": y})
    csv = tmp_path / "g.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"value": "ret"})
    assert "完成" in res.summary
    # ARCH effects present -> LM significant; persistence high (true alpha+beta = 0.95)
    assert res.estimates["arch_lm_pvalue"] < 0.05
    assert 0.6 < res.estimates["persistence"] < 1.05
    assert res.estimates["alpha1"] >= 0 and res.estimates["beta1"] >= 0


def test_garch_flags_no_arch_effects(tmp_path: Path) -> None:
    # iid normal -> no volatility clustering -> ARCH-LM not significant -> flagged
    rng = np.random.default_rng(5)
    y = rng.normal(0, 1, 800)
    df = pd.DataFrame({"t": np.arange(len(y)), "x": y})
    csv = tmp_path / "n.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"value": "x"})
    assert "完成" in res.summary
    assert res.estimates["arch_lm_pvalue"] > 0.05
    assert "GARCH 或非必要" in res.summary


def test_garch_needs_enough_obs(tmp_path: Path) -> None:
    df = pd.DataFrame({"t": np.arange(20), "x": np.random.default_rng(1).normal(0, 1, 20)})
    csv = tmp_path / "x.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"value": "x"})
    assert "GARCH 失败" in res.summary
