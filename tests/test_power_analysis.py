"""Tests for the power / sample-size DoE-advisory branch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="power_analysis", method="Power / sample-size", domain="experimental design",
        family="experimental_design", goal="plan",
        preconditions=Precondition(requires_group=True, min_continuous=1, min_rows=6),
    )


def test_power_sample_size_monotonicity(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    rows = []
    for g, m in zip(["A", "B", "C"], [10.0, 12.0, 14.0]):
        rows += [{"y": m + rng.normal(0, 3), "grp": g} for _ in range(20)]
    csv = tmp_path / "p.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "y", "group": "grp"})
    assert "完成" in res.summary
    assert res.estimates["k_groups"] == 3
    # bigger effect -> fewer subjects; higher power -> more subjects
    assert res.estimates["n_per_group_large_p80"] < res.estimates["n_per_group_small_p80"]
    assert res.estimates["n_per_group_medium_p90"] > res.estimates["n_per_group_medium_p80"]
    assert res.estimates["observed_f"] > 0  # real pilot effect present


def test_power_needs_group(tmp_path: Path) -> None:
    df = pd.DataFrame({"y": np.arange(10.0), "z": np.arange(10.0)})  # no categorical group
    csv = tmp_path / "p.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"outcome": "y"})
    assert "功效/样本量失败" in res.summary
