"""Tests for mantel_test (Mantel test — correlation between two distance matrices
with a permutation p-value)."""

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
        id="mantel_test",
        method="Mantel test (distance-matrix correlation)",
        domain="ecology",
        family="ecology",
        goal="explain",
        preconditions=Precondition(min_count_cols=2, min_continuous=1, min_rows=4),
    )


def test_mantel_significant(tmp_path: Path) -> None:
    """Community composition tracks an environmental gradient -> correlated distance
    matrices -> significant Mantel r."""
    rng = np.random.default_rng(42)
    n = 30
    # latent environmental gradient drives both env vars and species composition
    grad = np.linspace(0, 10, n)
    env1 = grad + rng.normal(0, 0.3, n)
    env2 = grad * 0.5 + rng.normal(0, 0.3, n)
    # species abundances are (noisy) monotone functions of the gradient
    sp0 = np.clip((30 - 2.5 * grad + rng.normal(0, 1, n)).round(), 0, None).astype(int)
    sp1 = np.clip((2.5 * grad + rng.normal(0, 1, n)).round(), 0, None).astype(int)
    sp2 = np.clip((15 + 0 * grad + rng.normal(0, 1, n)).round(), 0, None).astype(int)
    sp3 = np.clip((grad ** 1.2 + rng.normal(0, 1, n)).round(), 0, None).astype(int)

    df = pd.DataFrame(
        {"sp0": sp0, "sp1": sp1, "sp2": sp2, "sp3": sp3, "env1": env1, "env2": env2}
    )
    csv = tmp_path / "sig.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))

    assert (Path(res.output_dir) / "mantel_result.csv").exists(), (
        f"mantel_result.csv not found; summary={res.summary}"
    )
    assert "mantel_r" in res.estimates, f"mantel_r missing; summary={res.summary}"
    assert "p_value" in res.estimates, f"p_value missing; summary={res.summary}"
    assert res.estimates["mantel_r"] > 0.2, (
        f"mantel_r={res.estimates['mantel_r']} should be clearly positive"
    )
    assert res.estimates["p_value"] < 0.05, (
        f"p_value={res.estimates['p_value']} should be < 0.05 for correlated matrices"
    )


def test_mantel_null(tmp_path: Path) -> None:
    """Community composition independent of the environment -> uncorrelated distance
    matrices -> non-significant Mantel test."""
    rng = np.random.default_rng(7)
    n = 30
    df = pd.DataFrame(
        {
            "sp0": rng.integers(1, 30, n),
            "sp1": rng.integers(1, 30, n),
            "sp2": rng.integers(1, 30, n),
            "sp3": rng.integers(1, 30, n),
            # env drawn independently of the community
            "env1": rng.normal(0, 1, n),
            "env2": rng.normal(0, 1, n),
        }
    )
    csv = tmp_path / "null.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))

    assert "p_value" in res.estimates, f"p_value missing; summary={res.summary}"
    assert res.estimates["p_value"] > 0.05, (
        f"p_value={res.estimates['p_value']} should be > 0.05 for independent matrices"
    )


def test_mantel_precondition_no_env(tmp_path: Path) -> None:
    """Count cols but no continuous env variable -> precondition unmet."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({f"sp{i}": rng.integers(0, 10, 20) for i in range(4)})
    csv = tmp_path / "no_env.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, _unmet = check_preconditions(fp, _entry().preconditions)
    assert not ok
