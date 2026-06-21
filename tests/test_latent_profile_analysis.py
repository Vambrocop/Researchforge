"""Tests for the latent profile analysis (LPA) executor branch (stepmix).

We GENERATE well-separated Gaussian clusters (known profiles) and assert the
branch recovers k, class means near truth (label-agnostic), and high entropy.
EM is stochastic — the branch pins random_state (default 42); we assert on
robust quantities (k, |mean| separation, entropy, ARI), never exact values.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_HAS_STEPMIX = importlib.util.find_spec("stepmix") is not None

pytestmark = pytest.mark.skipif(not _HAS_STEPMIX, reason="stepmix not available")


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="latent_profile_analysis",
        method="Latent profile analysis",
        domain="psychometrics",
        family="latent_class",
        goal="explore",
        preconditions=Precondition(min_continuous=2, min_rows=50),
    )


def _make_gaussian_clusters(centers, n_per=150, sd=0.6, seed=0):
    """centers: list of mean-vectors. Returns (df, truth)."""
    rng = np.random.default_rng(seed)
    blocks, truth = [], []
    n_dim = len(centers[0])
    for ci, mu in enumerate(centers):
        blk = rng.normal(loc=mu, scale=sd, size=(n_per, n_dim))
        blocks.append(blk)
        truth += [ci] * n_per
    X = np.vstack(blocks)
    truth = np.array(truth)
    cols = [f"x{j+1}" for j in range(n_dim)]
    df = pd.DataFrame(X, columns=cols)
    perm = rng.permutation(len(df))
    return df.iloc[perm].reset_index(drop=True), truth[perm]


def _ari(truth, pred):
    from sklearn.metrics import adjusted_rand_score

    return adjusted_rand_score(truth, pred)


def test_lpa_recovers_two_profiles(tmp_path: Path) -> None:
    # two well-separated Gaussian profiles in 3D
    df, truth = _make_gaussian_clusters([[0, 0, 0], [5, 5, 5]], n_per=150, sd=0.6, seed=0)
    csv = tmp_path / "lpa2.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    kinds = {c.name: c.kind for c in fp.columns}
    assert all(kinds[f"x{j+1}"] == "continuous" for j in range(3))

    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "完成" in res.summary, res.summary
    assert res.estimates["n_classes"] == 2.0
    assert res.estimates["entropy"] >= 0.8
    mem = pd.read_csv(Path(res.output_dir) / "class_membership.csv")
    assert _ari(truth, mem["class"].values) > 0.8

    # recovered means near truth (original scale). The two true centers are 0 and 5;
    # one class should sit near 0 and the other near 5 on every indicator.
    prof = pd.read_csv(Path(res.output_dir) / "class_profiles.csv")
    for j in range(3):
        row = prof[prof["indicator"] == f"x{j+1}"].iloc[0]
        means = sorted([row["class0_mean"], row["class1_mean"]])
        assert abs(means[0] - 0.0) < 1.0
        assert abs(means[1] - 5.0) < 1.0


def test_lpa_recovers_three_profiles(tmp_path: Path) -> None:
    df, truth = _make_gaussian_clusters(
        [[0, 0], [6, 0], [0, 6]], n_per=130, sd=0.7, seed=1
    )
    csv = tmp_path / "lpa3.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "完成" in res.summary, res.summary
    assert res.estimates["n_classes"] == 3.0
    assert res.estimates["entropy"] >= 0.8
    mem = pd.read_csv(Path(res.output_dir) / "class_membership.csv")
    assert _ari(truth, mem["class"].values) > 0.8


def test_lpa_config_n_classes_override(tmp_path: Path) -> None:
    df, _ = _make_gaussian_clusters([[0, 0], [6, 0], [0, 6]], n_per=130, sd=0.7, seed=2)
    csv = tmp_path / "lpa_cfg.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"n_classes": 2})
    assert res.estimates["n_classes"] == 2.0
    assert "config 指定 k" in res.summary


def test_lpa_classes_ordered_by_size(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    big = rng.normal([0, 0, 0], 0.6, size=(260, 3))
    small = rng.normal([5, 5, 5], 0.6, size=(70, 3))
    df = pd.DataFrame(np.vstack([big, small]), columns=["x1", "x2", "x3"])
    csv = tmp_path / "imb.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    sizes = pd.read_csv(Path(res.output_dir) / "class_sizes.csv")
    props = sizes["mixing_proportion"].values
    assert all(props[i] >= props[i + 1] - 1e-9 for i in range(len(props) - 1))


def test_lpa_too_few_indicators(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"x1": rng.normal(0, 1, 120)})
    csv = tmp_path / "one.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary and "≥2" in res.summary


def test_lpa_too_few_rows(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    df = pd.DataFrame(rng.normal(0, 1, (15, 3)), columns=["x1", "x2", "x3"])
    csv = tmp_path / "few.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
