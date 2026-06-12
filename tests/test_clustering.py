"""Tests for kmeans_clustering — catalog + executor (blobs, degenerate, NaN alignment)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from researchforge.catalog import Catalog
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _km_entry():
    return Catalog.load().by_id("kmeans_clustering")


# ---------------------------------------------------------------------------
# 1. Catalog
# ---------------------------------------------------------------------------

def test_catalog_loads_kmeans():
    entry = _km_entry()
    assert entry is not None
    assert entry.goal == "explore"
    assert entry.family == "ml"
    assert entry.preconditions.min_continuous == 2


# ---------------------------------------------------------------------------
# 2. Recommender — feasible and infeasible
# ---------------------------------------------------------------------------

def test_recommender_feasible_and_infeasible(tmp_path):
    from researchforge.recommender import recommend

    # Feasible: 60 rows, 3 continuous features
    X_blobs, _ = make_blobs(n_samples=60, centers=3, n_features=3, random_state=0)
    df_good = pd.DataFrame(X_blobs, columns=["f0", "f1", "f2"])
    csv_good = tmp_path / "good.csv"
    df_good.to_csv(csv_good, index=False)
    fp_good = profile_dataset(csv_good)
    recs_good = recommend(fp_good)
    km_rec = next((r for r in recs_good if r.entry.id == "kmeans_clustering"), None)
    assert km_rec is not None
    assert km_rec.feasible is True

    # Infeasible: 60 rows, ONE continuous col + one categorical string col
    df_bad = pd.DataFrame({
        "value": np.random.default_rng(0).normal(0, 1, 60),
        "category": ["A", "B"] * 30,
    })
    csv_bad = tmp_path / "bad.csv"
    df_bad.to_csv(csv_bad, index=False)
    fp_bad = profile_dataset(csv_bad)
    recs_bad = recommend(fp_bad)
    km_bad = next((r for r in recs_bad if r.entry.id == "kmeans_clustering"), None)
    assert km_bad is not None
    assert km_bad.rigor.light == "red"
    assert km_bad.feasible is False


# ---------------------------------------------------------------------------
# 3. Executor — blobs (happy path)
# ---------------------------------------------------------------------------

def test_executor_blobs(tmp_path):
    X_blobs, _ = make_blobs(n_samples=150, centers=3, n_features=3, random_state=0)
    df = pd.DataFrame(X_blobs, columns=["f0", "f1", "f2"])
    csv = tmp_path / "blobs.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _km_entry(), output_root=str(tmp_path / "outputs"))
    out = Path(res.output_dir)

    assert (out / "cluster_assignments.csv").exists()
    assert (out / "cluster_profile.csv").exists()
    assert (out / "report.md").exists()
    assert "silhouette" in res.estimates
    k = int(res.estimates["k"])
    assert 2 <= k <= 149

    profile_df = pd.read_csv(out / "cluster_profile.csv")
    assert len(profile_df) == k


# ---------------------------------------------------------------------------
# 4. Executor — degenerate (constant columns, zero variance)
# ---------------------------------------------------------------------------

def test_executor_degenerate_no_crash(tmp_path):
    # Two continuous columns (non-integer so profiler sees them as continuous)
    # with zero variance — silhouette should fail or skip, not crash.
    df = pd.DataFrame({
        "c1": [0.5] * 40,
        "c2": [1.5] * 40,
    })
    csv = tmp_path / "degenerate.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    # Must not raise
    res = run_analysis(fp, _km_entry(), output_root=str(tmp_path / "outputs"))

    assert (Path(res.output_dir) / "report.md").exists()
    assert "silhouette" not in res.estimates
    # summary should contain a note about failure/skip
    assert any(
        kw in res.summary for kw in ("未能形成", "跳过", "失败")
    )


# ---------------------------------------------------------------------------
# 5. Executor — NaN alignment
# ---------------------------------------------------------------------------

def test_executor_nan_alignment(tmp_path):
    rng = np.random.default_rng(7)
    X_blobs, _ = make_blobs(n_samples=80, centers=3, n_features=3, random_state=0)
    df = pd.DataFrame(X_blobs, columns=["f0", "f1", "f2"])

    # Introduce NaNs in f0 on 10 random rows
    nan_idx = rng.choice(df.index, size=10, replace=False)
    df.loc[nan_idx, "f0"] = float("nan")

    csv = tmp_path / "nan_blobs.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _km_entry(), output_root=str(tmp_path / "outputs"))

    out = Path(res.output_dir)
    assert (out / "cluster_assignments.csv").exists()

    assign = pd.read_csv(out / "cluster_assignments.csv")
    # Number of assigned rows == rows with no NaN in ANY feature
    expected_rows = int(df[["f0", "f1", "f2"]].notna().all(axis=1).sum())
    assert len(assign) == expected_rows

    # Every 'row' value is a valid index in the original dataframe
    assert set(assign["row"]).issubset(set(df.index))
