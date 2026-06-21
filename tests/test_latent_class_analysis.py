"""Tests for the latent class analysis (LCA) executor branch (stepmix).

We GENERATE data with KNOWN latent classes (clearly different item-probability
profiles) and assert that the branch recovers k, well-separated profiles, high
entropy, and membership matching truth (adjusted Rand index after label-agnostic
comparison). EM is stochastic — the branch pins random_state (default 42, also
overridable via config seed); we assert on robust quantities (k, separation,
entropy, ARI), never exact parameter values.
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
_HAS_SK = importlib.util.find_spec("sklearn") is not None

pytestmark = pytest.mark.skipif(not _HAS_STEPMIX, reason="stepmix not available")


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="latent_class_analysis",
        method="Latent class analysis",
        domain="psychometrics",
        family="latent_class",
        goal="explore",
        preconditions=Precondition(min_categorical_cols=2, min_rows=50),
    )


def _make_two_class_binary(seed: int = 0, n_per: int = 150):
    """Two latent classes with clearly opposite binary item profiles.

    Class A answers mostly 1 (p=0.9 on every item); class B mostly 0 (p=0.1).
    6 binary indicators -> classes are very well separated.
    """
    rng = np.random.default_rng(seed)
    n_items = 6
    pA, pB = 0.9, 0.1
    A = (rng.random((n_per, n_items)) < pA).astype(int)
    B = (rng.random((n_per, n_items)) < pB).astype(int)
    X = np.vstack([A, B])
    truth = np.array([0] * n_per + [1] * n_per)
    cols = [f"item{j+1}" for j in range(n_items)]
    df = pd.DataFrame(X, columns=cols)
    # shuffle rows so class order in the file is not informative
    perm = rng.permutation(len(df))
    return df.iloc[perm].reset_index(drop=True), truth[perm]


def _make_three_class_binary(seed: int = 1, n_per: int = 120):
    """Three latent classes with distinct binary profiles over 6 items."""
    rng = np.random.default_rng(seed)
    profiles = [
        [0.9, 0.9, 0.9, 0.1, 0.1, 0.1],   # class 0: high on first half
        [0.1, 0.1, 0.1, 0.9, 0.9, 0.9],   # class 1: high on second half
        [0.9, 0.1, 0.9, 0.1, 0.9, 0.1],   # class 2: alternating
    ]
    blocks, truth = [], []
    for ci, p in enumerate(profiles):
        blk = (rng.random((n_per, len(p))) < np.array(p)).astype(int)
        blocks.append(blk)
        truth += [ci] * n_per
    X = np.vstack(blocks)
    truth = np.array(truth)
    cols = [f"item{j+1}" for j in range(6)]
    df = pd.DataFrame(X, columns=cols)
    perm = rng.permutation(len(df))
    return df.iloc[perm].reset_index(drop=True), truth[perm]


def _ari(truth, pred):
    from sklearn.metrics import adjusted_rand_score

    return adjusted_rand_score(truth, pred)


def test_lca_recovers_two_classes(tmp_path: Path) -> None:
    df, truth = _make_two_class_binary(seed=0)
    csv = tmp_path / "lca2.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    # indicators must profile as binary (0/1)
    kinds = {c.name: c.kind for c in fp.columns}
    assert all(kinds[f"item{j+1}"] == "binary" for j in range(6))

    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "完成" in res.summary, res.summary
    # BIC selects k=2 (the truth)
    assert res.estimates["n_classes"] == 2.0
    # well separated -> high entropy
    assert res.estimates["entropy"] >= 0.8
    # recovery: membership matches truth (label-agnostic)
    mem = pd.read_csv(tmp_path / "o" / "latent_class_analysis" / "class_membership.csv")
    assert _ari(truth, mem["class"].values) > 0.8

    # profiles separate: each item's P(=1) differs sharply between class0 and class1
    prof = pd.read_csv(tmp_path / "o" / "latent_class_analysis" / "class_profiles.csv")
    gaps = (prof["class0"] - prof["class1"]).abs()
    assert gaps.mean() > 0.5


def test_lca_recovers_three_classes(tmp_path: Path) -> None:
    df, truth = _make_three_class_binary(seed=1)
    csv = tmp_path / "lca3.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "完成" in res.summary, res.summary
    assert res.estimates["n_classes"] == 3.0
    assert res.estimates["entropy"] >= 0.8
    mem = pd.read_csv(tmp_path / "o" / "latent_class_analysis" / "class_membership.csv")
    assert _ari(truth, mem["class"].values) > 0.8


def test_lca_config_n_classes_override(tmp_path: Path) -> None:
    """config n_classes forces k even when BIC would pick differently."""
    df, _ = _make_three_class_binary(seed=2)
    csv = tmp_path / "lca_cfg.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"n_classes": 2})
    assert res.estimates["n_classes"] == 2.0
    assert "config 指定 k" in res.summary


def test_lca_classes_ordered_by_size(tmp_path: Path) -> None:
    """Label-switching fix: class0 must be the largest (mixing proportion desc)."""
    rng = np.random.default_rng(7)
    # imbalanced: 250 in class A, 80 in class B
    A = (rng.random((250, 6)) < 0.9).astype(int)
    B = (rng.random((80, 6)) < 0.1).astype(int)
    df = pd.DataFrame(np.vstack([A, B]), columns=[f"item{j+1}" for j in range(6)])
    csv = tmp_path / "imb.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    sizes = pd.read_csv(tmp_path / "o" / "latent_class_analysis" / "class_sizes.csv")
    # mixing proportions are non-increasing (class0 >= class1 >= ...)
    props = sizes["mixing_proportion"].values
    assert all(props[i] >= props[i + 1] - 1e-9 for i in range(len(props) - 1))
    assert res.estimates["largest_class_share"] == pytest.approx(props[0], abs=1e-6)


def test_lca_too_few_indicators(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"item1": (rng.random(120) < 0.5).astype(int)})
    csv = tmp_path / "one.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary and "≥2" in res.summary


def test_lca_too_few_rows(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    df = pd.DataFrame(
        (rng.random((15, 6)) < 0.5).astype(int),
        columns=[f"item{j+1}" for j in range(6)],
    )
    csv = tmp_path / "few.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
