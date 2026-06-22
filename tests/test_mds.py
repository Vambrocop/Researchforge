"""Tests for the `mds` executor branch (metric multidimensional scaling).

Known structure: points laid out on a clear 2D grid, then embedded into a higher-D
space by an orthonormal lift plus a little noise. Metric MDS on the standardized
Euclidean distances should recover the 2D layout — low Kruskal stress-1 and a high
Shepard-diagram correlation. Plus degrade (too few features / too few rows -> honest
skip) and a config-override check.

The catalog yaml exists (ordination.yaml) but the AnalysisEntry is built inline so
the test is self-contained.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog import AnalysisEntry
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_ENTRY = AnalysisEntry(
    id="mds",
    method="Multidimensional scaling (MDS, metric)",
    domain="statistics",
    family="ml",
    goal="explore",
    preconditions={"min_continuous": 2, "min_rows": 4},
)


def _structured_csv(tmp_path: Path, n: int = 60, dim_high: int = 6) -> Path:
    """A clear 2D structure lifted into higher-D with light noise.

    Two latent coordinates (z1, z2) drive `dim_high` observed continuous features
    via a random orthonormal map; small isotropic noise is added. MDS should
    recover the relative 2D geometry from the high-D distances.
    """
    rng = np.random.default_rng(0)
    z1 = rng.uniform(-3, 3, n)
    z2 = rng.uniform(-3, 3, n)
    Z = np.column_stack([z1, z2])
    # random orthonormal-ish lift (2 -> dim_high)
    W = rng.normal(0, 1, (2, dim_high))
    W, _ = np.linalg.qr(W.T)  # (dim_high, 2) orthonormal columns
    X = Z @ W.T + rng.normal(0, 0.05, (n, dim_high))
    cols = {f"f{i+1}": X[:, i] for i in range(dim_high)}
    df = pd.DataFrame(cols)
    csv = tmp_path / "mds_structured.csv"
    df.to_csv(csv, index=False)
    return csv


def test_mds_recovers_2d_structure(tmp_path):
    csv = _structured_csv(tmp_path)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "out"))
    out = Path(res.output_dir)

    assert (out / "mds_embedding.csv").exists(), "mds_embedding.csv missing"
    assert (out / "report.md").exists()

    # estimates populated
    for k in ("stress1", "shepard_corr", "n_components"):
        assert k in res.estimates, f"{k} missing from estimates"

    stress = res.estimates["stress1"]
    shep = res.estimates["shepard_corr"]
    # 2D structure embedded into 2D -> excellent fit.
    assert stress < 0.10, f"stress-1 too high for a recoverable 2D layout: {stress}"
    assert shep > 0.95, f"Shepard correlation too low: {shep}"

    emb = pd.read_csv(out / "mds_embedding.csv", index_col=0)
    assert "dim1" in emb.columns and "dim2" in emb.columns
    assert len(emb) == 60


def test_mds_embedding_distances_match_original(tmp_path):
    """The embedding should preserve the rank-order / magnitude of distances."""
    csv = _structured_csv(tmp_path)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "out"))
    out = Path(res.output_dir)

    from scipy.spatial.distance import pdist
    from sklearn.preprocessing import StandardScaler

    raw = pd.read_csv(csv)
    Xs = StandardScaler().fit_transform(raw.values)
    d_orig = pdist(Xs)

    emb = pd.read_csv(out / "mds_embedding.csv", index_col=0)[["dim1", "dim2"]].values
    d_emb = pdist(emb)
    r = np.corrcoef(d_orig, d_emb)[0, 1]
    assert r > 0.95, f"embedding distances do not track original distances (r={r})"


def test_mds_too_few_features_skips(tmp_path):
    rng = np.random.default_rng(1)
    n = 30
    df = pd.DataFrame({
        "value": rng.normal(0, 1, n),
        "category": (["A", "B", "C"] * (n // 3)),
    })
    csv = tmp_path / "one_cont.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "out"))
    assert "stress1" not in res.estimates
    assert "跳过" in res.summary


def test_mds_too_few_rows_skips(tmp_path):
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "f1": rng.normal(0, 1, 3),
        "f2": rng.normal(0, 1, 3),
        "f3": rng.normal(0, 1, 3),
    })
    csv = tmp_path / "few_rows.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "out"))
    assert "stress1" not in res.estimates
    assert "跳过" in res.summary


def test_mds_config_features_override(tmp_path):
    """config features restricts which columns enter the distance matrix."""
    csv = _structured_csv(tmp_path)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _ENTRY, output_root=str(tmp_path / "out"),
        config={"features": ["f1", "f2", "f3"], "n_components": 2},
    )
    out = Path(res.output_dir)
    assert (out / "mds_embedding.csv").exists()
    assert "stress1" in res.estimates
    # f1..f6 lie on a 2D plane -> any 3 of them still embed cleanly.
    assert res.estimates["n_components"] == 2
