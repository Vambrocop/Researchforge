"""Tests for cohens_d: pooled-SD standardized mean difference + CI + CLES + Glass.

Known-structure: two groups with a designed ~1 SD mean shift -> d near 1 (large).
We INDEPENDENTLY recompute d (mean diff / pooled SD) to pin estimator correctness,
plus identical-groups (d~0), config override, magnitude/CLES sanity, and degrade.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="cohens_d",
        method="Cohen's d",
        domain="statistics",
        family="effect_sizes",
        goal="describe",
        preconditions=Precondition(requires_group=True, min_continuous=1, min_rows=6),
    )


def _pooled_sd(a, b):
    n1, n2 = a.size, b.size
    return np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))


def test_cohens_d_one_sd_shift_is_large(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n = 80
    # group A higher than B by ~1 SD -> d ≈ 1 (large)
    a = rng.normal(5.0, 1.0, n)
    b = rng.normal(4.0, 1.0, n)
    y = np.concatenate([a, b])
    g = np.array(["A"] * n + ["B"] * n)
    df = pd.DataFrame({"y": y, "grp": g})  # y first -> outcome by convention
    csv = tmp_path / "shift.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    assert (out / "cohens_d.csv").exists()
    d = res.estimates["cohens_d"]
    # independent recomputation of Cohen's d from the same samples
    d_indep = (a.mean() - b.mean()) / _pooled_sd(a, b)
    assert abs(d - d_indep) < 1e-4
    # designed ~1 SD shift -> large band, positive (A > B)
    assert 0.6 < d < 1.4
    assert res.estimates["ci_low"] < d < res.estimates["ci_high"]
    # CLES = P(A > B) should be clearly above 0.5 here
    assert res.estimates["cles"] > 0.6
    assert res.estimates["n1"] == float(n) and res.estimates["n2"] == float(n)
    assert "large" in res.summary or "大" in res.summary


def test_cohens_d_identical_groups_near_zero(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    n = 100
    y = rng.normal(0, 1, 2 * n)  # both groups same distribution
    g = np.array(["A"] * n + ["B"] * n)
    df = pd.DataFrame({"y": y, "grp": g})
    csv = tmp_path / "same.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert abs(res.estimates["cohens_d"]) < 0.4  # ~0
    # CI should straddle 0 when there is no real effect
    assert res.estimates["ci_low"] < 0 < res.estimates["ci_high"]
    assert abs(res.estimates["cles"] - 0.5) < 0.1


def test_cohens_d_config_override_outcome_group(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    n = 60
    a = np.concatenate([rng.normal(10, 1, n), rng.normal(8, 1, n)])  # separated by real_g
    noise = rng.normal(0, 1, 2 * n)
    real_g = np.array(["hi"] * n + ["lo"] * n)
    df = pd.DataFrame({"a": a, "noise": noise, "real_g": real_g})
    csv = tmp_path / "ovr.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "a", "group": "real_g"})
    assert abs(res.estimates["cohens_d"]) > 1.0  # ~2 SD separation
    assert res.estimates["n1"] == float(n)


def test_cohens_d_glass_delta_reported(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    n = 70
    a = rng.normal(3, 2.0, n)   # different SDs between groups
    b = rng.normal(0, 1.0, n)
    y = np.concatenate([a, b])
    g = np.array(["A"] * n + ["B"] * n)
    df = pd.DataFrame({"y": y, "grp": g})
    csv = tmp_path / "het.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    glass = res.estimates["glass_delta"]
    # Glass uses control (B) SD only: (mean_a - mean_b) / sd_b
    glass_indep = (a.mean() - b.mean()) / b.std(ddof=1)
    assert abs(glass - glass_indep) < 1e-3


def test_cohens_d_resolver_picks_high_confidence_outcome_not_first(tmp_path: Path) -> None:
    """A high-confidence-named outcome ('target') placed AFTER a decoy continuous
    column must still be resolved as the outcome (shared resolve_outcome, not raw
    cont_cols[0])."""
    rng = np.random.default_rng(9)
    n = 80
    decoy = rng.normal(0, 1, 2 * n)  # first continuous column, no group signal
    a = rng.normal(5.0, 1.0, n)
    b = rng.normal(4.0, 1.0, n)
    target = np.concatenate([a, b])
    g = np.array(["A"] * n + ["B"] * n)
    df = pd.DataFrame({"decoy": decoy, "target": target, "grp": g})
    csv = tmp_path / "resolver.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    assert fp.likely_outcome == "target" and fp.likely_outcome_confidence == "high"
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    d = res.estimates["cohens_d"]
    # only true if 'target' (~1 SD shift) was modeled, not 'decoy' (no shift)
    assert 0.6 < d < 1.4


def test_cohens_d_degrade_three_groups(tmp_path: Path) -> None:
    # three groups -> not a two-group effect size -> honest skip, no crash
    rng = np.random.default_rng(5)
    n = 30
    y = rng.normal(0, 1, 3 * n)
    g = np.array(["A"] * n + ["B"] * n + ["C"] * n)
    df = pd.DataFrame({"y": y, "grp": g})
    csv = tmp_path / "k3.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
    assert "cohens_d" not in res.estimates


def test_cohens_d_degrade_no_group(tmp_path: Path) -> None:
    df = pd.DataFrame({"y": np.arange(20, dtype=float)})
    csv = tmp_path / "nogrp.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
    assert "cohens_d" not in res.estimates
