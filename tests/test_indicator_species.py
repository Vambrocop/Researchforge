"""Tests for indicator_species (IndVal — Dufrene-Legendre indicator value with a
permutation p-value)."""

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
        id="indicator_species",
        method="Indicator species analysis (IndVal)",
        domain="ecology",
        family="ecology",
        goal="explain",
        preconditions=Precondition(min_count_cols=1, requires_group=True, min_rows=6),
    )


def test_indval_strong_indicator(tmp_path: Path) -> None:
    """sp0 occurs (high abundance) only in group A, sp1 only in group B -> both are
    strong, significant indicators of their group; sp2 is everywhere -> not."""
    rng = np.random.default_rng(42)
    n = 15  # per group

    sp0_a = rng.integers(20, 40, n)  # A-specific
    sp0_b = np.zeros(n, dtype=int)
    sp1_a = np.zeros(n, dtype=int)
    sp1_b = rng.integers(20, 40, n)  # B-specific
    sp2 = rng.integers(8, 14, n * 2)  # ubiquitous, no group preference

    df = pd.DataFrame(
        {
            "sp0": np.concatenate([sp0_a, sp0_b]),
            "sp1": np.concatenate([sp1_a, sp1_b]),
            "sp2": sp2,
            "grp": ["A"] * n + ["B"] * n,
        }
    )
    csv = tmp_path / "indic.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))

    out = Path(res.output_dir) / "indicator_species.csv"
    assert out.exists(), f"indicator_species.csv not found; summary={res.summary}"
    tab = pd.read_csv(out).set_index("taxon")

    # sp0 -> A, sp1 -> B, both with high IndVal + significant
    assert tab.loc["sp0", "indicator_group"] == "A"
    assert tab.loc["sp1", "indicator_group"] == "B"
    assert tab.loc["sp0", "indval"] > 80, f"sp0 IndVal too low: {tab.loc['sp0', 'indval']}"
    assert tab.loc["sp1", "indval"] > 80, f"sp1 IndVal too low: {tab.loc['sp1', 'indval']}"
    assert bool(tab.loc["sp0", "significant"]), "sp0 should be a significant indicator"
    assert bool(tab.loc["sp1", "significant"]), "sp1 should be a significant indicator"
    # ubiquitous taxon: not a significant indicator
    assert not bool(tab.loc["sp2", "significant"]), "sp2 (ubiquitous) should not be significant"

    assert res.estimates["n_significant_indicators"] >= 2


def test_indval_no_indicator(tmp_path: Path) -> None:
    """All taxa drawn from the same distribution in both groups -> no significant
    indicators."""
    rng = np.random.default_rng(11)
    n = 20
    df = pd.DataFrame(
        {
            "sp0": rng.integers(5, 15, n * 2),
            "sp1": rng.integers(5, 15, n * 2),
            "sp2": rng.integers(5, 15, n * 2),
            "grp": ["A"] * n + ["B"] * n,
        }
    )
    csv = tmp_path / "null.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    tab = pd.read_csv(Path(res.output_dir) / "indicator_species.csv")
    # under the null, essentially no significant indicators expected
    assert int(tab["significant"].sum()) == 0, f"expected 0 significant, got {tab['significant'].sum()}"


def test_indval_precondition_no_group(tmp_path: Path) -> None:
    """Count cols but no grouping variable -> precondition unmet."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({f"sp{i}": rng.integers(0, 10, 20) for i in range(3)})
    csv = tmp_path / "no_group.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)
    assert not ok
    assert any("分组" in u for u in unmet), f"expected group-related unmet reason, got {unmet}"
