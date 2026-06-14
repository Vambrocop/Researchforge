"""Tests for the regression discontinuity (rdrobust) executor branch."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_HAS_RDROBUST = importlib.util.find_spec("rdrobust") is not None


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="rdd", method="RDD", domain="economics", family="causal",
        goal="explain", preconditions=Precondition(min_continuous=2, min_rows=100),
    )


def _data(tmp_path: Path) -> Path:
    rng = np.random.default_rng(1)
    n = 1500
    score = rng.uniform(-1, 1, n)
    outcome = 2.0 * score + 3.0 * (score >= 0) + rng.normal(0, 0.5, n)  # true jump = 3
    df = pd.DataFrame({"score": score, "outcome": outcome, "covar": rng.normal(0, 1, n)})
    csv = tmp_path / "rdd.csv"
    df.to_csv(csv, index=False)
    return csv


@pytest.mark.skipif(not _HAS_RDROBUST, reason="rdrobust not available")
def test_rdd_recovers_jump(tmp_path: Path) -> None:
    fp = profile_dataset(_data(tmp_path))
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"running": "score", "cutoff": 0, "outcome": "outcome"},
    )
    assert "rdrobust" in res.summary
    # true RD jump is 3.0; robust CI should cover it and the point be close
    assert abs(res.estimates["rd_effect"] - 3.0) < 0.5
    assert res.estimates["ci_lb"] <= 3.0 <= res.estimates["ci_ub"]
    assert res.estimates["p_value"] < 0.05
    assert res.estimates["bandwidth"] > 0


def test_rdd_needs_running_config(tmp_path: Path) -> None:
    # rdrobust may or may not be installed; either way, no running -> honest message
    fp = profile_dataset(_data(tmp_path))
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "RDD" in res.summary and ("running" in res.summary or "rdrobust" in res.summary)


@pytest.mark.skipif(not _HAS_RDROBUST, reason="rdrobust not available")
def test_rdd_cutoff_out_of_range_degrades(tmp_path: Path) -> None:
    fp = profile_dataset(_data(tmp_path))
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"running": "score", "cutoff": 99, "outcome": "outcome"},
    )
    assert "RDD 失败" in res.summary and "范围" in res.summary
