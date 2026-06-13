"""Tests for pls_sem: honest-degrade (needs user measurement model) + gate."""

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
        id="pls_sem",
        method="PLS-SEM (partial least squares SEM)",
        domain="statistics",
        family="sem",
        goal="explain",
        preconditions=Precondition(min_continuous=4, min_rows=30),
    )


def test_pls_sem_honest_guidance(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({f"v{i}": rng.normal(0, 1, 40) for i in range(5)})
    csv = tmp_path / "pls.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    # honest-degrade: no fabricated output, surfaces the measurement-model requirement
    assert "测量模型" in res.summary
    assert "SEM" in res.summary or "EFA" in res.summary


def test_pls_sem_precondition_unmet(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(0, 1, 40), "b": rng.normal(0, 1, 40)})  # <4 continuous
    csv = tmp_path / "two.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)

    assert not ok
    assert any("连续" in u for u in unmet)
