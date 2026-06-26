"""Tests for pca (principal component analysis) — executor branch.

The catalog yaml entry has not been promoted yet, so the AnalysisEntry is
constructed inline for all tests.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog import AnalysisEntry
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

# ---------------------------------------------------------------------------
# Shared inline entry (catalog yaml not added yet — per task spec)
# ---------------------------------------------------------------------------

_PCA_ENTRY = AnalysisEntry(
    id="pca",
    method="Principal component analysis (ordination)",
    domain="machine learning",
    family="ml",
    goal="explore",
    preconditions={"min_continuous": 3, "min_rows": 20},
)


# ---------------------------------------------------------------------------
# Helper — build a correlated 4-feature CSV
# ---------------------------------------------------------------------------

def _make_correlated_csv(tmp_path: Path, n: int = 80) -> Path:
    rng = np.random.default_rng(42)
    base = rng.normal(0, 1, n)
    # f1, f2 are strongly correlated with base (first PC dominates)
    f1 = base + rng.normal(0, 0.2, n)
    f2 = base + rng.normal(0, 0.2, n)
    # f3, f4 are independent noise
    f3 = rng.normal(0, 1, n)
    f4 = rng.normal(0, 1, n)
    csv = tmp_path / "correlated.csv"
    pd.DataFrame({"f1": f1, "f2": f2, "f3": f3, "f4": f4}).to_csv(csv, index=False)
    return csv


# ---------------------------------------------------------------------------
# 1. Happy-path executor test
# ---------------------------------------------------------------------------

def test_executor_pca(tmp_path):
    csv = _make_correlated_csv(tmp_path)
    fp = profile_dataset(csv)

    res = run_analysis(fp, _PCA_ENTRY, output_root=str(tmp_path / "outputs"))
    out = Path(res.output_dir)

    # Required output files
    assert (out / "explained_variance.csv").exists(), "explained_variance.csv missing"
    assert (out / "loadings.csv").exists(), "loadings.csv missing"
    assert (out / "report.md").exists(), "report.md missing"

    # Estimates populated
    assert "pc1_explained_ratio" in res.estimates, "pc1_explained_ratio missing from estimates"

    # pc1 ratio is a valid probability
    pc1 = res.estimates["pc1_explained_ratio"]
    assert 0 < pc1 <= 1, f"pc1_explained_ratio out of range: {pc1}"

    # Sanity-check the CSV shapes
    ev_df = pd.read_csv(out / "explained_variance.csv")
    assert "component" in ev_df.columns
    assert "explained_variance_ratio" in ev_df.columns
    assert "cumulative" in ev_df.columns
    assert len(ev_df) >= 2  # at least 2 PCs for 4 features

    load_df = pd.read_csv(out / "loadings.csv", index_col=0)
    assert "PC1" in load_df.columns
    assert load_df.shape[0] == 4  # 4 features as rows


# ---------------------------------------------------------------------------
# 2. Degenerate — too few continuous features (1 continuous col) → skip
# ---------------------------------------------------------------------------

def test_pca_too_few_features(tmp_path):
    # Only 1 continuous column — should trigger the guard and skip silently
    rng = np.random.default_rng(7)
    n = 40
    df = pd.DataFrame({
        "value": rng.normal(0, 1, n),
        "category": ["A", "B"] * (n // 2),
    })
    csv = tmp_path / "one_cont.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    # Must not raise
    res = run_analysis(fp, _PCA_ENTRY, output_root=str(tmp_path / "outputs"))

    # report.md always written
    assert (Path(res.output_dir) / "report.md").exists()

    # PCA was skipped — no pc1_explained_ratio in estimates
    assert "pc1_explained_ratio" not in res.estimates, (
        "estimates should not contain pc1_explained_ratio when PCA is skipped"
    )
    # Summary should mention the skip
    assert "跳过" in res.summary or "PCA" in res.summary
