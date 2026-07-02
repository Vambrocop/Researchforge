"""Tests for permanova (PERMANOVA — distance-based permutation test of community composition)."""

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
        id="permanova",
        method="PERMANOVA (group composition test)",
        domain="ecology",
        family="ecology",
        goal="explain",
        preconditions=Precondition(min_count_cols=2, requires_group=True, min_rows=10),
    )


def test_permanova_significant(tmp_path: Path) -> None:
    """Group A dominated by sp0/sp1; group B dominated by sp2/sp3 -> clearly different composition."""
    rng = np.random.default_rng(42)
    n = 15  # sites per group, 30 total

    # Group A: high sp0/sp1, near-zero sp2/sp3
    sp0_a = rng.integers(20, 40, n)
    sp1_a = rng.integers(15, 35, n)
    sp2_a = rng.integers(0, 2, n)
    sp3_a = rng.integers(0, 2, n)

    # Group B: high sp2/sp3, near-zero sp0/sp1
    sp0_b = rng.integers(0, 2, n)
    sp1_b = rng.integers(0, 2, n)
    sp2_b = rng.integers(20, 40, n)
    sp3_b = rng.integers(15, 35, n)

    df = pd.DataFrame({
        "sp0": np.concatenate([sp0_a, sp0_b]),
        "sp1": np.concatenate([sp1_a, sp1_b]),
        "sp2": np.concatenate([sp2_a, sp2_b]),
        "sp3": np.concatenate([sp3_a, sp3_b]),
        "grp": ["A"] * n + ["B"] * n,
    })
    csv = tmp_path / "sig.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))

    result_csv = Path(res.output_dir) / "permanova_result.csv"
    assert result_csv.exists(), f"permanova_result.csv not found; summary={res.summary}"

    assert "pseudo_F" in res.estimates, f"pseudo_F missing; summary={res.summary}"
    assert "p_value" in res.estimates, f"p_value missing; summary={res.summary}"
    assert res.estimates["pseudo_F"] > 1, (
        f"pseudo_F={res.estimates['pseudo_F']} should be > 1 for clearly separated groups"
    )
    assert res.estimates["p_value"] < 0.05, (
        f"p_value={res.estimates['p_value']} should be < 0.05 for clearly separated groups"
    )
    assert (
        ("PERMDISP" in res.summary)
        or ("betadisper" in res.summary)
        or ("离散度" in res.summary)
    ), f"expected a dispersion/PERMDISP disclosure in summary, got: {res.summary}"


def test_permanova_null(tmp_path: Path) -> None:
    """Both groups drawn from the same distribution -> composition independent of group."""
    rng = np.random.default_rng(7)
    n = 20  # sites per group, 40 total — same distribution for both

    sp0 = rng.integers(5, 15, n * 2)
    sp1 = rng.integers(5, 15, n * 2)
    sp2 = rng.integers(5, 15, n * 2)
    sp3 = rng.integers(5, 15, n * 2)

    df = pd.DataFrame({
        "sp0": sp0,
        "sp1": sp1,
        "sp2": sp2,
        "sp3": sp3,
        "grp": ["A"] * n + ["B"] * n,
    })
    csv = tmp_path / "null.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))

    assert "p_value" in res.estimates, f"p_value missing; summary={res.summary}"
    assert res.estimates["p_value"] > 0.05, (
        f"p_value={res.estimates['p_value']} should be > 0.05 for null (same distribution) groups"
    )


def test_permanova_precondition_no_group(tmp_path: Path) -> None:
    """Dataset has count cols but no group variable -> precondition unmet."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({f"sp{i}": rng.integers(0, 10, 20) for i in range(4)})
    csv = tmp_path / "no_group.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)

    assert not ok
    assert any("分组" in u for u in unmet), f"Expected group-related unmet reason, got: {unmet}"


def test_permanova_precondition_continuous_only(tmp_path: Path) -> None:
    """Continuous-only dataset -> both min_count_cols and requires_group unmet."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(0, 1, 20), "y": rng.normal(0, 1, 20)})
    csv = tmp_path / "cont.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)

    assert not ok
    assert any("計數列" in u or "计数列" in u for u in unmet), (
        f"Expected count-cols unmet reason, got: {unmet}"
    )
