"""Tests for hierarchical_clustering: gate + Ward linkage clustering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset
from researchforge.recommender.match import check_preconditions


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="hierarchical_clustering",
        method="Hierarchical clustering (Ward)",
        domain="machine learning",
        family="ml",
        goal="explore",
        preconditions=Precondition(min_continuous=2, min_rows=5),
    )


def test_hierarchical_executor(tmp_path: Path) -> None:
    X, _ = make_blobs(n_samples=60, centers=3, n_features=3, random_state=0)
    df = pd.DataFrame(X, columns=["f0", "f1", "f2"])
    csv = tmp_path / "blobs.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    assert (out / "cluster_assignments.csv").exists()
    assert res.estimates["n_clusters"] >= 2
    assert "cophenetic_corr" in res.estimates


def test_hierarchical_precondition_unmet(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(0, 1, 30), "g": ["a", "b"] * 15})  # only 1 continuous
    csv = tmp_path / "one.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)

    assert not ok
    assert any("连续" in u for u in unmet)
