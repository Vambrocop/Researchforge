"""Tests for the fuzzy regression discontinuity (fuzzy RDD) executor branch."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("rdrobust") is None, reason="rdrobust not installed"
)


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="fuzzy_rdd", method="Fuzzy regression discontinuity", domain="economics",
        family="causal", goal="explain",
        preconditions=Precondition(min_continuous=2, min_rows=100),
    )


def test_fuzzy_rdd_recovers_late(tmp_path: Path) -> None:
    # running x ~ U(-1,1), cutoff 0; treatment probability jumps 0.25 -> 0.85 at the cutoff
    # (fuzzy take-up). outcome smooth in x, with a constant complier effect LATE=3 via treatment.
    rng = np.random.default_rng(0)
    n = 3000
    x = rng.uniform(-1.0, 1.0, n)
    p_treat = np.where(x >= 0.0, 0.85, 0.25)
    t = (rng.uniform(size=n) < p_treat).astype(int)
    late = 3.0
    y = 1.0 + late * t + 0.8 * x + rng.normal(0, 1.0, n)  # smooth in x; jump enters only via t
    csv = tmp_path / "f.csv"
    pd.DataFrame({"y": y, "x": x, "got": t}).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"running": "x", "cutoff": 0.0, "treatment": "got", "outcome": "y"})
    assert "完成" in res.summary
    assert abs(res.estimates["late"] - late) < 1.5          # fuzzy LATE is noisy; loose band around 3
    assert res.estimates["first_stage_jump"] > 0.3          # real first stage (~0.6 jump)
    assert res.estimates["n_effective"] > 0


def test_fuzzy_rdd_needs_treatment_column(tmp_path: Path) -> None:
    # running + outcome given but NO config['treatment'] -> honest fail (this is what makes it fuzzy)
    rng = np.random.default_rng(1)
    x = rng.uniform(-1.0, 1.0, 200)
    df = pd.DataFrame({"y": rng.normal(0, 1, 200), "x": x})
    csv = tmp_path / "x.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"running": "x", "cutoff": 0.0, "outcome": "y"})
    assert "模糊 RDD 失败" in res.summary


def test_fuzzy_rdd_flags_weak_first_stage(tmp_path: Path) -> None:
    # treatment probability barely changes at the cutoff -> weak first stage -> flagged, LATE unreliable
    rng = np.random.default_rng(2)
    n = 2000
    x = rng.uniform(-1.0, 1.0, n)
    p_treat = np.where(x >= 0.0, 0.52, 0.48)  # ~0.04 jump = weak
    t = (rng.uniform(size=n) < p_treat).astype(int)
    y = 1.0 + 2.0 * t + 0.5 * x + rng.normal(0, 1.0, n)
    csv = tmp_path / "w.csv"
    pd.DataFrame({"y": y, "x": x, "got": t}).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"running": "x", "cutoff": 0.0, "treatment": "got", "outcome": "y"})
    assert "第一阶段弱" in res.summary
