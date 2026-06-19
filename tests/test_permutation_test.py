"""Tests for permutation_test: distribution-free group difference.

Known-structure checks: clearly-separated groups -> small p; identical groups
-> large/non-significant p (uniform-ish under H0). Plus k-group F-ratio,
config override (group/outcome/n_perm/seed), reproducibility, and degrade.
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
        id="permutation_test",
        method="Permutation test",
        domain="statistics",
        family="nonparametric",
        goal="explain",
        preconditions=Precondition(requires_group=True, min_continuous=1, min_rows=8),
    )


def test_permutation_separated_groups_small_p(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n = 60
    # two clearly separated groups: A ~ N(0,1), B ~ N(5,1)
    y = np.concatenate([rng.normal(0, 1, n), rng.normal(5, 1, n)])
    g = np.array(["A"] * n + ["B"] * n)
    df = pd.DataFrame({"y": y, "grp": g})  # y first -> outcome by convention
    csv = tmp_path / "sep.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"n_perm": 999, "seed": 0})
    out = Path(res.output_dir)

    assert (out / "group_means.csv").exists()
    assert res.estimates["p_value"] < 0.01  # strong separation
    assert abs(res.estimates["observed_stat"]) > 3.0  # mean diff near 5
    assert res.estimates["n_groups"] == 2.0


def test_permutation_identical_groups_large_p(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    n = 80
    # both groups from the SAME distribution -> H0 true -> p should not be small
    y = rng.normal(0, 1, 2 * n)
    g = np.array(["A"] * n + ["B"] * n)
    df = pd.DataFrame({"y": y, "grp": g})
    csv = tmp_path / "same.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"n_perm": 999, "seed": 0})
    # under H0 the permutation p is ~Uniform(0,1); it should comfortably exceed
    # the 0.05 level for this seed (well-separated null check)
    assert res.estimates["p_value"] > 0.05


def test_permutation_kgroup_f_ratio(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    n = 40
    # 3 groups with distinct means -> F-ratio path, small p
    y = np.concatenate([rng.normal(0, 1, n), rng.normal(3, 1, n), rng.normal(6, 1, n)])
    g = np.array(["A"] * n + ["B"] * n + ["C"] * n)
    df = pd.DataFrame({"y": y, "grp": g})
    csv = tmp_path / "k3.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"n_perm": 999, "seed": 0})
    assert res.estimates["n_groups"] == 3.0
    assert res.estimates["observed_stat"] > 0.0  # F-ratio is positive
    assert res.estimates["p_value"] < 0.01


def test_permutation_min_p_floor(tmp_path: Path) -> None:
    # min p = 1/(n_perm+1): with n_perm capped at 99 floor and tiny n_perm,
    # the reported p can never be below that floor.
    rng = np.random.default_rng(4)
    n = 50
    y = np.concatenate([rng.normal(0, 1, n), rng.normal(8, 1, n)])
    g = np.array(["A"] * n + ["B"] * n)
    df = pd.DataFrame({"y": y, "grp": g})
    csv = tmp_path / "floor.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"n_perm": 50, "seed": 0})  # below floor -> raised to 99
    n_perm = res.estimates["n_perm"]
    assert n_perm == 99.0  # clamped up to the minimum
    assert res.estimates["p_value"] >= 1.0 / (n_perm + 1) - 1e-12


def test_permutation_config_override_columns(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    n = 50
    # two continuous + an explicit grouping column; override outcome/group
    a = np.concatenate([rng.normal(0, 1, n), rng.normal(4, 1, n)])
    noise = rng.normal(0, 1, 2 * n)
    g = np.array(["x", "y"] * n)  # alternating -> no signal w.r.t. 'a'
    real_g = np.array(["lo"] * n + ["hi"] * n)
    df = pd.DataFrame({"a": a, "noise": noise, "g": g, "real_g": real_g})
    csv = tmp_path / "ovr.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "a", "group": "real_g", "n_perm": 999, "seed": 0})
    # 'a' is separated by real_g (lo vs hi) -> small p
    assert res.estimates["p_value"] < 0.01


def test_permutation_reproducible(tmp_path: Path) -> None:
    rng = np.random.default_rng(6)
    n = 40
    y = np.concatenate([rng.normal(0, 1, n), rng.normal(1, 1, n)])
    g = np.array(["A"] * n + ["B"] * n)
    df = pd.DataFrame({"y": y, "grp": g})
    csv = tmp_path / "rep.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)

    r1 = run_analysis(fp, _entry(), output_root=str(tmp_path / "o1"),
                      config={"n_perm": 499, "seed": 7})
    r2 = run_analysis(fp, _entry(), output_root=str(tmp_path / "o2"),
                      config={"n_perm": 499, "seed": 7})
    assert r1.estimates["p_value"] == r2.estimates["p_value"]


def test_permutation_degrade_no_group(tmp_path: Path) -> None:
    # only a continuous column, no group -> honest failure, no crash
    df = pd.DataFrame({"y": np.arange(20, dtype=float)})
    csv = tmp_path / "nogrp.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "失败" in res.summary
    assert "p_value" not in res.estimates
