"""Tests for the meta-regression (metafor) executor branch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis, rbridge
from researchforge.profiler import profile_dataset

_HAS_METAFOR = rbridge.r_available() and rbridge.r_package_available("metafor")


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="meta_regression", method="Meta-regression", domain="statistics",
        family="meta", goal="synthesize",
        preconditions=Precondition(requires_effect_sizes=True, min_rows=5),
    )


def _data(tmp_path: Path) -> Path:
    rng = np.random.default_rng(5)
    k = 30
    dose = rng.uniform(1, 10, k)
    year = rng.integers(2005, 2024, k).astype(float)
    yi = 0.1 + 0.06 * dose + rng.normal(0, 0.08, k)  # effect grows with dose; year irrelevant
    sei = rng.uniform(0.08, 0.20, k)
    df = pd.DataFrame({"study": [f"S{i}" for i in range(k)], "yi": yi, "sei": sei, "dose": dose, "year": year})
    csv = tmp_path / "mr.csv"
    df.to_csv(csv, index=False)
    return csv


@pytest.mark.skipif(not _HAS_METAFOR, reason="R metafor not available")
def test_meta_regression_detects_moderator(tmp_path: Path) -> None:
    fp = profile_dataset(_data(tmp_path))
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"moderators": ["dose", "year"]},
    )
    assert "metafor" in res.summary
    assert res.estimates["QM_pvalue"] < 0.05  # moderators jointly significant
    # true dose slope is 0.06; recovered close
    assert abs(res.estimates["beta_dose"] - 0.06) < 0.04
    assert res.estimates["k_studies"] == 30.0


@pytest.mark.skipif(not _HAS_METAFOR, reason="R metafor not available")
def test_meta_regression_needs_moderator(tmp_path: Path) -> None:
    # effect sizes but no moderator column -> honest failure
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {"study": [f"S{i}" for i in range(8)], "yi": rng.normal(0.3, 0.1, 8), "sei": rng.uniform(0.1, 0.2, 8)}
    )
    csv = tmp_path / "nomod.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "Meta 回归失败" in res.summary and "调节变量" in res.summary


def test_meta_regression_no_effect_sizes_degrades(tmp_path: Path) -> None:
    # no recognizable effect-size columns -> honest failure (no R call)
    df = pd.DataFrame({"foo": [1.0, 2, 3, 4, 5], "bar": [5.0, 4, 3, 2, 1]})
    csv = tmp_path / "x.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "Meta 回归失败" in res.summary
