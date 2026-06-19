"""Tests for the GGE biplot (Genotype + Genotype×Environment) branch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="gge_biplot", method="GGE biplot", domain="experimental design",
        family="experimental_design", goal="explain",
        preconditions=Precondition(requires_group=True, min_continuous=1, min_rows=9),
    )


def test_gge_recovers_structure_and_which_won_where(tmp_path: Path) -> None:
    # Plant a clean crossover: two genotype groups, each winning in a distinct set of
    # environments. After environment-centering, PC1 should dominate and which-won-where
    # should pick up more than one winning genotype.
    rng = np.random.default_rng(0)
    genos = [f"G{i}" for i in range(6)]
    envs = [f"E{j}" for j in range(5)]
    # genotype loading sign splits the panel; environment loading sign splits environments
    a = np.array([3.0, 2.5, 2.0, -2.0, -2.5, -3.0])
    b = np.array([2.0, 1.5, -1.5, -2.0, 1.0])
    rows = []
    for _ in range(3):
        for i, g in enumerate(genos):
            for j, e in enumerate(envs):
                val = 50 + 4.0 * a[i] * b[j] + rng.normal(0, 0.1)
                rows.append({"yield": val, "genotype": g, "environment": e})
    csv = tmp_path / "ge.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "yield", "genotype": "genotype", "environment": "environment"})
    assert "完成" in res.summary
    assert res.estimates["n_genotypes"] == 6
    assert res.estimates["n_environments"] == 5
    assert res.estimates["PC1_pct"] > 80.0               # rank-1 crossover → PC1 dominates
    assert res.estimates["n_winning_genotypes"] >= 2     # crossover → more than one winner
    files = set(res.files)
    assert "gge_variance.csv" in files
    assert "gge_which_won_where.csv" in files

    # verify the which-won-where table content matches the planted crossover
    www = pd.read_csv(Path(res.output_dir) / "gge_which_won_where.csv")
    assert set(www["environment"]) == set(envs)
    assert www["winning_genotype"].nunique() >= 2


def test_gge_needs_enough_levels(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    rows = []
    for g in ("A", "B"):
        for e in ("E1", "E2"):
            for _ in range(3):
                rows.append({"y": rng.normal(0, 1), "geno": g, "env": e})
    csv = tmp_path / "small.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "y", "genotype": "geno", "environment": "env"})
    assert "GGE biplot 失败" in res.summary
    assert "PC1_pct" not in res.estimates
