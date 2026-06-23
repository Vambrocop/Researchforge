"""Tests for linear_discriminant — LDA supervised dim reduction + classification.

Known structure: 3 well-separated blobs with a categorical target -> LDA should
achieve high CV accuracy and recover discriminant axes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from researchforge.catalog import Catalog
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry():
    return Catalog.load().by_id("linear_discriminant")


def _blob_df(seed: int = 0) -> pd.DataFrame:
    X, y = make_blobs(n_samples=120, centers=3, n_features=4, cluster_std=1.0,
                      random_state=seed)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["species"] = [f"c{int(lbl)}" for lbl in y]
    return df


# ---------------------------------------------------------------------------
# 1. Catalog
# ---------------------------------------------------------------------------

def test_catalog_loads_linear_discriminant():
    entry = _entry()
    assert entry is not None
    assert entry.executor_ref == "py::linear_discriminant"
    assert entry.goal == "predict"
    assert isinstance(entry.biases, list) and entry.biases
    assert isinstance(entry.produces, list) and entry.produces


# ---------------------------------------------------------------------------
# 2. Executor — separable blobs -> high CV accuracy
# ---------------------------------------------------------------------------

def test_executor_blobs_high_cv(tmp_path):
    csv = tmp_path / "blobs.csv"
    _blob_df().to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    assert (out / "lda_scores.csv").exists()
    assert (out / "lda_class_means_ld1.csv").exists()
    assert "cv_accuracy" in res.estimates
    assert res.estimates["n_classes"] == 3.0
    assert res.estimates["n_components"] == 2.0  # min(3-1, 4)
    # well-separated blobs -> CV accuracy should be high
    assert res.estimates["cv_accuracy"] > 0.8, f"cv_acc={res.estimates['cv_accuracy']}"

    scores = pd.read_csv(out / "lda_scores.csv", index_col=0)
    assert "class" in scores.columns
    assert {"LD1", "LD2"}.issubset(scores.columns)


# ---------------------------------------------------------------------------
# 3. Config override — explicit outcome
# ---------------------------------------------------------------------------

def test_config_outcome(tmp_path):
    csv = tmp_path / "blobs.csv"
    _blob_df().to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "species"})
    assert res.estimates["n_classes"] == 3.0
    assert res.estimates["cv_accuracy"] > 0.8


# ---------------------------------------------------------------------------
# 4. Honest degrade — no categorical target
# ---------------------------------------------------------------------------

def test_degrade_no_target(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "a": rng.normal(0, 1, 50),
        "b": rng.normal(0, 1, 50),
        "c": rng.normal(0, 1, 50),
    })
    csv = tmp_path / "cont.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
    assert "cv_accuracy" not in res.estimates


def test_degrade_too_few_rows(tmp_path):
    df = pd.DataFrame({
        "a": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "b": [1.0, 1.1, 0.9, 1.2, 0.8, 1.3],
        "grp": ["x", "y", "x", "y", "x", "y"],
    })
    csv = tmp_path / "tiny.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
