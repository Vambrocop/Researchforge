"""Tests for dbscan_clustering: density clustering with eps from the k-distance knee.

Known structure: 2 tight dense blobs + a handful of scattered far-flung points ->
  * DBSCAN finds 2 clusters,
  * the scattered points are flagged as NOISE (the -1 label),
  * the dense points are (almost) all assigned to a cluster.
Plus eps/min_samples config override + a degrade (too few features / rows). Fixed seed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="dbscan_clustering",
        method="DBSCAN (density-based clustering)",
        domain="machine learning",
        family="mixture",
        goal="explore",
        preconditions=Precondition(min_continuous=2, min_rows=10),
    )


def _two_blobs_plus_noise(seed: int = 0):
    """Two dense Gaussian blobs far apart + uniformly scattered noise in between/around."""
    rng = np.random.default_rng(seed)
    per = 100
    b1 = rng.normal([0.0, 0.0], 0.3, (per, 2))
    b2 = rng.normal([12.0, 12.0], 0.3, (per, 2))
    n_noise = 12
    noise = rng.uniform(low=-4, high=16, size=(n_noise, 2))  # sparse, spread out
    XY = np.vstack([b1, b2, noise])
    # noise rows are the LAST n_noise rows of XY (we keep order, no shuffle, so the
    # injected-noise indices are known for the assertion).
    df = pd.DataFrame({"f1": XY[:, 0], "f2": XY[:, 1]})
    noise_idx = list(range(2 * per, 2 * per + n_noise))
    return df, noise_idx


def _run(df: pd.DataFrame, tmp_path: Path, config=None):
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    return run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config=config or {})


def test_finds_two_clusters_and_noise(tmp_path: Path) -> None:
    df, noise_idx = _two_blobs_plus_noise(seed=0)
    res = _run(df, tmp_path)
    assert "完成" in res.summary
    # two dense blobs -> exactly 2 clusters
    assert res.estimates["n_clusters"] == 2.0
    # at least some of the scattered points become noise
    assert res.estimates["n_noise"] >= 5.0
    # but the dense core (200 pts) shouldn't be mostly noise
    assert res.estimates["noise_frac"] < 0.2


def test_injected_noise_points_flagged(tmp_path: Path) -> None:
    df, noise_idx = _two_blobs_plus_noise(seed=1)
    res = _run(df, tmp_path)
    out = Path(res.output_dir)
    assign = pd.read_csv(out / "dbscan_assignments.csv")
    labels = dict(zip(assign["row"], assign["cluster"]))
    # most injected sparse points are labelled noise (-1)
    flagged = sum(1 for i in noise_idx if labels.get(i) == -1)
    assert flagged >= len(noise_idx) * 0.6
    # dense-blob points (first 200) are overwhelmingly NOT noise
    dense_noise = sum(1 for i in range(200) if labels.get(i) == -1)
    assert dense_noise <= 20


def test_outputs_written(tmp_path: Path) -> None:
    df, _ = _two_blobs_plus_noise(seed=2)
    res = _run(df, tmp_path)
    out = Path(res.output_dir)
    sizes = pd.read_csv(out / "dbscan_cluster_sizes.csv")
    assert {"cluster", "size"} <= set(sizes.columns)
    # -1 (noise) appears as its own group in the size table
    assert -1 in set(sizes["cluster"])
    assert "eps" in res.estimates and res.estimates["eps"] > 0
    assert "min_samples" in res.estimates


def test_config_override_eps_min_samples(tmp_path: Path) -> None:
    df, _ = _two_blobs_plus_noise(seed=3)
    # hand-set eps small enough to still separate the two far blobs
    res = _run(df, tmp_path, {"eps": 1.0, "min_samples": 5, "features": ["f1", "f2"]})
    assert res.estimates["eps"] == 1.0
    assert res.estimates["min_samples"] == 5.0
    assert res.estimates["n_clusters"] == 2.0


def test_degrade_one_feature(tmp_path: Path) -> None:
    df = pd.DataFrame({"f1": np.random.default_rng(0).normal(0, 1, 50)})
    res = _run(df, tmp_path)
    assert "跳过" in res.summary
    assert not res.estimates


def test_degrade_too_few_rows(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"f1": rng.normal(0, 1, 6), "f2": rng.normal(0, 1, 6)})
    res = _run(df, tmp_path)
    assert "跳过" in res.summary
    assert not res.estimates
