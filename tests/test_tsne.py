"""Tests for tsne — t-SNE 2-D embedding for visualization (pure Python, sklearn).

Known structure: 3 well-separated blobs with a label column. t-SNE should embed,
report a KL divergence and (with a label) a positive silhouette of the embedding.
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
    return Catalog.load().by_id("tsne")


def _blob_df(seed: int = 0) -> pd.DataFrame:
    X, y = make_blobs(n_samples=90, centers=3, n_features=4, random_state=seed)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["group"] = [f"g{int(lbl)}" for lbl in y]
    return df


# ---------------------------------------------------------------------------
# 1. Catalog
# ---------------------------------------------------------------------------

def test_catalog_loads_tsne():
    entry = _entry()
    assert entry is not None
    assert entry.family == "ml"
    assert entry.executor_ref == "py::tsne"
    # biases / produces must be lists (not folded strings) — else Catalog.load would crash.
    assert isinstance(entry.biases, list) and len(entry.biases) >= 3
    assert isinstance(entry.produces, list) and entry.produces


# ---------------------------------------------------------------------------
# 2. Executor — blobs with a label (happy path)
# ---------------------------------------------------------------------------

def test_executor_blobs_with_label(tmp_path):
    csv = tmp_path / "blobs.csv"
    _blob_df().to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    assert (out / "tsne_embedding.csv").exists()
    assert "kl_divergence" in res.estimates
    assert "perplexity" in res.estimates
    assert res.estimates["n_features"] >= 2
    assert res.estimates["n"] >= 5
    # well-separated blobs -> the label's silhouette in the STANDARDIZED INPUT
    # feature space (an input-space separability descriptor, not from the
    # non-metric embedding) is positive.
    assert res.estimates["silhouette_by_label"] == res.estimates["silhouette_by_label"]
    assert res.estimates["silhouette_by_label"] > 0.0

    emb = pd.read_csv(out / "tsne_embedding.csv")
    assert {"tsne1", "tsne2", "label"}.issubset(emb.columns)


# ---------------------------------------------------------------------------
# 3. Executor — no label column (silhouette NaN, still runs)
# ---------------------------------------------------------------------------

def test_executor_no_label(tmp_path):
    X, _ = make_blobs(n_samples=60, centers=3, n_features=4, random_state=1)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    csv = tmp_path / "nolabel.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))

    assert "kl_divergence" in res.estimates
    # no label -> silhouette is NaN, not a crash
    assert np.isnan(res.estimates["silhouette_by_label"])


# ---------------------------------------------------------------------------
# 4. Honest degrade — too few features / rows
# ---------------------------------------------------------------------------

def test_degrade_too_few_features(tmp_path):
    df = pd.DataFrame({"x": np.random.default_rng(0).normal(0, 1, 30)})
    csv = tmp_path / "one.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    joined = res.summary
    assert "跳过" in joined
    assert "kl_divergence" not in res.estimates


def test_degrade_too_few_rows(tmp_path):
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "a": rng.normal(0, 1, 4),
        "b": rng.normal(0, 1, 4),
        "c": rng.normal(0, 1, 4),
    })
    csv = tmp_path / "tiny.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
