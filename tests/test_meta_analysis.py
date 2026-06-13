"""Tests for the meta-analysis (metafor) executor branch."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.executor import rbridge
from researchforge.profiler import profile_dataset

_HAS_METAFOR = rbridge.r_available() and rbridge.r_package_available("metafor")


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="meta_analysis", method="Meta-analysis", domain="statistics",
        family="meta", goal="synthesize",
        preconditions=Precondition(requires_effect_sizes=True, min_rows=2),
    )


def test_meta_unrecognized_columns_fails_honestly(tmp_path: Path) -> None:
    df = pd.DataFrame({"foo": [1.0, 2.0, 3.0], "bar": [4.0, 5.0, 6.0]})
    csv = tmp_path / "x.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "未识别到效应量数据" in res.summary


@pytest.mark.skipif(not _HAS_METAFOR, reason="R metafor not available")
def test_meta_precomputed_effects(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "study": [f"S{i}" for i in range(8)],
            "yi": [0.4, 0.3, 0.55, 0.28, 0.61, 0.39, 0.12, 0.47],
            "sei": [0.12, 0.18, 0.15, 0.10, 0.20, 0.14, 0.22, 0.13],
        }
    )
    csv = tmp_path / "m.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "metafor" in res.summary
    assert res.estimates["k_studies"] == 8.0
    assert "pooled_effect" in res.estimates
    assert res.estimates["ci_lb"] <= res.estimates["pooled_effect"] <= res.estimates["ci_ub"]


@pytest.mark.skipif(not _HAS_METAFOR, reason="R metafor not available")
def test_meta_raw_two_group_means_smd(tmp_path: Path) -> None:
    rng = pd.DataFrame(
        {
            "study": [f"T{i}" for i in range(8)],
            "m1": [11.2, 10.5, 12.1, 9.8, 11.9, 10.2, 12.5, 11.0],
            "sd1": [2.0, 1.8, 2.2, 1.5, 2.1, 1.9, 2.3, 2.0],
            "n1": [30, 25, 40, 35, 28, 33, 22, 31],
            "m2": [9.8, 9.5, 10.1, 9.0, 10.0, 9.1, 10.5, 9.6],
            "sd2": [1.9, 1.7, 2.0, 1.6, 2.0, 1.8, 2.1, 1.9],
            "n2": [29, 26, 38, 34, 27, 32, 23, 30],
        }
    )
    csv = tmp_path / "smd.csv"
    rng.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "metafor" in res.summary
    assert res.estimates["k_studies"] == 8.0
    # treatment means are higher -> positive standardized mean difference
    assert res.estimates["pooled_effect"] > 0


@pytest.mark.skipif(not _HAS_METAFOR, reason="R metafor not available")
def test_meta_config_fixed_effect(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "study": [f"S{i}" for i in range(6)],
            "yi": [0.4, 0.3, 0.5, 0.28, 0.6, 0.39],
            "vi": [0.02, 0.03, 0.025, 0.01, 0.04, 0.02],
        }
    )
    csv = tmp_path / "fe.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"), config={"method": "FE"}
    )
    assert "固定效应" in res.summary
