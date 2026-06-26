"""Tests for the AMMI (Additive Main effects + Multiplicative Interaction) branch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="ammi", method="AMMI", domain="experimental design",
        family="experimental_design", goal="explain",
        preconditions=Precondition(requires_group=True, min_continuous=1, min_rows=9),
    )


def _ge_trial(tmp_path: Path) -> Path:
    # Plant a known rank-1 G×E interaction: inter[i,j] = a_i * b_j (outer product).
    # AMMI's SVD of the interaction residual should recover IPCA1 dominating (~100%).
    rng = np.random.default_rng(0)
    genos = [f"G{i}" for i in range(6)]
    envs = [f"E{j}" for j in range(5)]
    g_main = {g: 5.0 * i for i, g in enumerate(genos)}       # genotype main effects
    e_main = {e: 3.0 * j for j, e in enumerate(envs)}        # environment main effects
    a = np.array([2.0, -1.0, 0.5, -1.5, 1.0, -1.0])         # genotype interaction loadings
    b = np.array([1.5, -2.0, 0.5, 1.0, -1.0])               # environment interaction loadings
    rows = []
    for _r in range(3):  # 3 reps
        for i, g in enumerate(genos):
            for j, e in enumerate(envs):
                val = 10 + g_main[g] + e_main[e] + 4.0 * a[i] * b[j] + rng.normal(0, 0.05)
                rows.append({"yield": val, "genotype": g, "environment": e})
    csv = tmp_path / "ge.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def test_ammi_recovers_rank1_interaction(tmp_path: Path) -> None:
    csv = _ge_trial(tmp_path)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "yield", "genotype": "genotype", "environment": "environment"})
    assert "完成" in res.summary
    assert res.estimates["n_genotypes"] == 6
    assert res.estimates["n_environments"] == 5
    # planted interaction is rank-1 → IPCA1 should explain almost all the interaction SS
    assert res.estimates["IPCA1_pct"] > 95.0
    assert res.estimates["interaction_ss"] > 0
    files = set(res.files)
    assert "ammi_ipca_variance.csv" in files
    assert "ammi_genotype_stability.csv" in files


def test_ammi_two_axes_when_rank2(tmp_path: Path) -> None:
    # rank-2 interaction with two ORTHOGONAL, comparably-scaled components → IPCA1 alone
    # should NOT capture ~all; IPCA1+IPCA2 should. Use mean-centered orthogonal vectors so
    # the planted interaction is exactly rank-2 and the two singular values are balanced.
    rng = np.random.default_rng(2)
    genos = [f"G{i}" for i in range(6)]
    envs = [f"E{j}" for j in range(5)]
    # genotype loadings: two orthogonal, zero-sum patterns
    a1 = np.array([1.0, 1.0, -1.0, -1.0, 1.0, -1.0]); a1 = a1 - a1.mean()
    a2 = np.array([1.0, -1.0, 1.0, -1.0, 0.0, 0.0]); a2 = a2 - a2.mean()
    # environment loadings: two orthogonal, zero-sum patterns
    b1 = np.array([1.0, 1.0, -1.0, -1.0, 0.0]); b1 = b1 - b1.mean()
    b2 = np.array([1.0, -1.0, 1.0, -1.0, 0.0]); b2 = b2 - b2.mean()
    rows = []
    for _ in range(3):
        for i, g in enumerate(genos):
            for j, e in enumerate(envs):
                inter = 5.0 * a1[i] * b1[j] + 5.0 * a2[i] * b2[j]
                val = 10 + 5 * i + 3 * j + inter + rng.normal(0, 0.02)
                rows.append({"yield": val, "genotype": g, "environment": e})
    csv = tmp_path / "ge2.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "yield", "genotype": "genotype", "environment": "environment"})
    assert "完成" in res.summary
    assert res.estimates["IPCA1_pct"] < 95.0           # not a single axis
    assert res.estimates["IPCA1_IPCA2_pct"] > 98.0     # two axes capture it


def test_ammi_needs_enough_levels(tmp_path: Path) -> None:
    # only 2 genotypes → cannot decompose interaction structure
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
    assert "AMMI 失败" in res.summary
    assert "IPCA1_pct" not in res.estimates
