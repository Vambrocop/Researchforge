"""Tests for the split conformal prediction executor branch (pure Python)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_HAS_SK = importlib.util.find_spec("sklearn") is not None


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="conformal_prediction", method="Conformal prediction", domain="machine learning",
        family="ml", goal="predict", preconditions=Precondition(min_continuous=2, min_rows=40),
    )


@pytest.mark.skipif(not _HAS_SK, reason="scikit-learn not available")
def test_conformal_coverage_near_target(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 600
    x1, x2 = rng.uniform(0, 10, n), rng.uniform(-3, 3, n)
    y = np.sin(x1) + 0.4 * x2**2 + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    csv = tmp_path / "c.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"alpha": 0.1})
    assert "conformal" in res.summary
    assert res.estimates["target_coverage"] == 0.9
    # empirical coverage should land near the 90% target (finite-sample variance allowed)
    assert 0.82 <= res.estimates["empirical_coverage"] <= 0.98
    assert res.estimates["mean_interval_width"] > 0
    assert res.estimates["conformal_q"] > 0


@pytest.mark.skipif(not _HAS_SK, reason="scikit-learn not available")
def test_conformal_alpha_changes_width(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n = 400
    x = rng.uniform(0, 5, n)
    df = pd.DataFrame({"y": x + rng.normal(0, 1, n), "x": x, "z": rng.normal(0, 1, n)})
    csv = tmp_path / "c.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    wide = run_analysis(fp, _entry(), output_root=str(tmp_path / "w"), config={"alpha": 0.05})
    narrow = run_analysis(fp, _entry(), output_root=str(tmp_path / "n"), config={"alpha": 0.2})
    # higher confidence (smaller alpha) -> wider intervals
    assert wide.estimates["mean_interval_width"] >= narrow.estimates["mean_interval_width"]


@pytest.mark.skipif(not _HAS_SK, reason="scikit-learn not available")
def test_conformal_cal_too_small_disclosed(tmp_path: Path) -> None:
    # n=48 -> n_cal=12; alpha=0.05 needs ceil((12+1)*0.95)=13 > 12, so the 1-alpha
    # finite-sample guarantee is UNATTAINABLE. The void must be surfaced to the user
    # (not just left as a boolean in the CSV) — inference-reviewer must-fix regression guard.
    rng = np.random.default_rng(3)
    n = 48
    x = rng.uniform(0, 5, n)
    df = pd.DataFrame({"y": x + rng.normal(0, 1, n), "x": x, "z": rng.normal(0, 1, n)})
    csv = tmp_path / "c.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"alpha": 0.05})
    assert res.estimates["cal_too_small"] == 1.0
    assert "无法达到" in res.summary       # the voided guarantee is stated in plain text
    assert "达标" not in res.summary        # and must NOT claim the target was met


def test_conformal_too_few_rows(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"y": rng.normal(0, 1, 25), "x": rng.normal(0, 1, 25)})
    csv = tmp_path / "x.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "保形预测失败" in res.summary
