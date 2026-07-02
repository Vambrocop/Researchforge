"""Tests for beta_diversity (Bray-Curtis): gate + dissimilarity matrix."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset
from researchforge.recommender.match import check_preconditions


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="beta_diversity",
        method="Beta diversity (Bray-Curtis dissimilarity)",
        domain="ecology",
        family="ecology",
        goal="describe",
        preconditions=Precondition(min_count_cols=2, min_rows=2),
    )


def test_beta_diversity_executor(tmp_path: Path) -> None:
    # 6 sites x 4 species; sites 0 and 1 are identical -> Bray-Curtis = 0
    df = pd.DataFrame(
        {
            "sp0": [5, 5, 0, 1, 2, 3],
            "sp1": [0, 0, 5, 2, 1, 3],
            "sp2": [0, 0, 0, 3, 4, 1],
            "sp3": [1, 1, 2, 0, 0, 2],
        }
    )
    csv = tmp_path / "abund.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    assert sum(1 for c in fp.columns if c.kind == "count") >= 2

    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    dm = pd.read_csv(Path(res.output_dir) / "bray_curtis.csv", index_col=0)

    assert dm.shape == (6, 6)
    assert abs(dm.iloc[0, 0]) < 1e-9  # diagonal is 0
    assert abs(dm.iloc[0, 1]) < 1e-9  # identical sites -> 0
    assert 0.0 <= res.estimates["mean_bray_curtis"] <= 1.0


def test_beta_diversity_drops_all_zero_site(tmp_path: Path) -> None:
    """An all-zero site row is undefined for Bray-Curtis (0/0) -> must be dropped
    before the pairwise distance, not left to leak NaN into the matrix/heatmap."""
    df = pd.DataFrame(
        {
            "sp0": [5, 5, 0, 1, 2, 3],
            "sp1": [0, 0, 0, 2, 1, 3],
            "sp2": [0, 0, 0, 3, 4, 1],
            "sp3": [1, 1, 0, 0, 0, 2],
        }
    )
    csv = tmp_path / "abund_zero.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    dm = pd.read_csv(Path(res.output_dir) / "bray_curtis.csv", index_col=0)

    # the all-zero row (index 2, 0-based) is dropped -> 5 remaining sites, no NaN
    assert dm.shape == (5, 5)
    assert not dm.isna().to_numpy().any()
    assert res.estimates["n_sites"] == 5.0


def test_beta_diversity_precondition_unmet(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(0, 1, 20), "y": rng.normal(0, 1, 20)})  # no count cols
    csv = tmp_path / "cont.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)

    assert not ok
    assert any("计数列" in u for u in unmet)
