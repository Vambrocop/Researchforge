"""Tests for the bayesian_sem honest-degrade branch (blavaan backend not auto-runnable)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="bayesian_sem", method="Bayesian SEM", domain="statistics", family="sem",
        goal="explain", preconditions=Precondition(min_continuous=3, min_rows=50),
    )


def test_bayesian_sem_honest_degrade(tmp_path: Path) -> None:
    # >=3 continuous indicators -> honest-degrade guidance (blavaan backend / alternative)
    rng = np.random.default_rng(0)
    f = rng.normal(0, 1, 80)
    df = pd.DataFrame({f"q{i}": f + rng.normal(0, 0.5, 80) for i in range(4)})
    csv = tmp_path / "s.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "贝叶斯 SEM" in res.summary
    # points to the runnable frequentist alternative + names blavaan/backend
    assert "sem" in res.summary and ("blavaan" in res.summary or "替代" in res.summary)


def test_bayesian_sem_too_few_indicators(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(0, 1, 60), "b": rng.normal(0, 1, 60)})
    csv = tmp_path / "x.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "贝叶斯 SEM 跳过" in res.summary
