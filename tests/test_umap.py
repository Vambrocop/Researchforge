"""Tests for umap — UMAP 2-D embedding for visualization (OPTIONAL backend umap-learn).

umap-learn may not be installed. The catalog/degrade tests run unconditionally; the
happy-path embedding test is skipped (pytest.importorskip) when umap is absent.

Known structure: 3 well-separated blobs with a label column. UMAP should embed to 2-D
with n rows, and (with a label) report a positive INPUT-space silhouette.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from researchforge.catalog import Catalog
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry():
    return Catalog.load().by_id("umap")


def _blob_df(seed: int = 0) -> pd.DataFrame:
    X, y = make_blobs(n_samples=90, centers=3, n_features=4, random_state=seed)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["group"] = [f"g{int(lbl)}" for lbl in y]
    return df


# ---------------------------------------------------------------------------
# 1. Catalog (always runs — entry must be valid regardless of backend)
# ---------------------------------------------------------------------------

def test_catalog_loads_umap():
    entry = _entry()
    assert entry is not None
    assert entry.family == "ml"
    assert entry.executor_ref == "py::umap"
    assert isinstance(entry.biases, list) and len(entry.biases) >= 3
    assert isinstance(entry.produces, list) and entry.produces
    assert {"features", "label", "n_neighbors", "min_dist"} <= {p.name for p in entry.params}


# ---------------------------------------------------------------------------
# 2. Honest degrade — too few features / rows (runs even without umap, because
#    these guards fire only AFTER the umap-present check; so guard them too)
# ---------------------------------------------------------------------------

def test_degrade_too_few_features(tmp_path):
    df = pd.DataFrame({"x": np.random.default_rng(0).normal(0, 1, 30)})
    csv = tmp_path / "one.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    # Either "needs umap-learn" (backend absent) or "needs >=2 features" — both honest skips.
    assert "跳过" in res.summary
    assert "n_neighbors" not in res.estimates


def test_degrade_backend_or_too_few_rows(tmp_path):
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


def test_skip_message_when_backend_absent(tmp_path):
    """If umap-learn is NOT installed, a normal blob run must honestly skip and name
    the substitute backends (never crash / fabricate)."""
    import importlib.util

    if importlib.util.find_spec("umap") is not None:
        pytest.skip("umap-learn is installed; degrade-message path not exercised here")
    csv = tmp_path / "blobs.csv"
    _blob_df().to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
    assert "umap-learn" in res.summary
    assert "n_neighbors" not in res.estimates


# ---------------------------------------------------------------------------
# 3. Happy path — only when umap-learn is available
# ---------------------------------------------------------------------------

def test_executor_blobs_with_label(tmp_path):
    pytest.importorskip("umap")
    csv = tmp_path / "blobs.csv"
    _blob_df().to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    assert (out / "umap_embedding.csv").exists()
    assert "n_neighbors" in res.estimates
    assert "min_dist" in res.estimates
    assert res.estimates["n_features"] >= 2
    assert res.estimates["n"] >= 5

    emb = pd.read_csv(out / "umap_embedding.csv")
    # embedding is 2-D (umap1/umap2) + label, with one row per input sample.
    assert {"umap1", "umap2", "label"}.issubset(emb.columns)
    assert len(emb) == int(res.estimates["n"])

    # well-separated blobs -> the label's silhouette in the STANDARDIZED INPUT space
    # (an input-space separability descriptor, not from the non-metric embedding) is positive.
    assert res.estimates["silhouette_by_label"] == res.estimates["silhouette_by_label"]
    assert res.estimates["silhouette_by_label"] > 0.0


def test_executor_no_label(tmp_path):
    pytest.importorskip("umap")
    X, _ = make_blobs(n_samples=60, centers=3, n_features=4, random_state=1)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    csv = tmp_path / "nolabel.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "n_neighbors" in res.estimates
    # no label -> silhouette is NaN, not a crash
    assert np.isnan(res.estimates["silhouette_by_label"])
