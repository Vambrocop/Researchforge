"""Tests for mann_whitney: Mann-Whitney U + rank-biserial + Hodges-Lehmann.

Known-structure checks: a clear two-group location shift -> significant U with the
rank-biserial sign correct and a positive Hodges-Lehmann shift; a no-shift control
-> non-significant; sign flips when the high group is listed second; config
override; an independent rank-biserial recompute pins the formula; honest degrade
on 3 groups and on too-small groups.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="mann_whitney",
        method="Mann-Whitney U test",
        domain="statistics",
        family="nonparametric",
        goal="explain",
        preconditions=Precondition(requires_group=True, min_continuous=1, min_rows=6),
    )


def test_mwu_two_group_shift_significant(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n = 50
    # group "lo" lower, group "hi" higher -> first-appearing level is "lo"
    y = np.concatenate([rng.normal(0, 1, n), rng.normal(3, 1, n)])
    g = np.array(["lo"] * n + ["hi"] * n)
    df = pd.DataFrame({"y": y, "grp": g})  # y first -> outcome by convention
    csv = tmp_path / "mwu.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    assert (out / "mannwhitney_groups.csv").exists()
    assert (out / "mannwhitney_stats.csv").exists()
    assert res.estimates["n1"] == float(n)
    assert res.estimates["n2"] == float(n)
    assert res.estimates["p_value"] < 0.01
    # group 1 ("lo") is LOWER -> rank-biserial r should be negative
    assert res.estimates["rank_biserial"] < 0
    # Hodges-Lehmann shift (g_lo - g_hi) ~ -3 -> negative
    assert res.estimates["hodges_lehmann"] < -1.5
    # distribution-free CI present and bracketing the point
    assert res.estimates["hl_ci_low"] <= res.estimates["hodges_lehmann"] <= res.estimates["hl_ci_high"]


def test_mwu_rank_biserial_recompute(tmp_path: Path) -> None:
    """Independently recompute r = 1 - 2U/(n1*n2) using scipy's U for group 1."""
    rng = np.random.default_rng(7)
    n1, n2 = 35, 45
    g1 = rng.normal(1.0, 1, n1)
    g2 = rng.normal(0.0, 1, n2)
    y = np.concatenate([g1, g2])
    g = np.array(["A"] * n1 + ["B"] * n2)
    df = pd.DataFrame({"y": y, "grp": g})
    csv = tmp_path / "mwueff.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))

    U1, _ = sps.mannwhitneyu(g1, g2, alternative="two-sided")
    r_expected = 2.0 * float(U1) / (n1 * n2) - 1.0  # rank-biserial, r>0 => g1 higher
    assert abs(res.estimates["u_stat"] - float(U1)) < 1e-6
    assert abs(res.estimates["rank_biserial"] - r_expected) < 1e-4
    assert res.estimates["rank_biserial"] > 0  # group A (n1) is higher


def test_mwu_sign_flips_with_order(tmp_path: Path) -> None:
    # same data, but the HIGH group appears first -> rank-biserial should be positive
    rng = np.random.default_rng(2)
    n = 40
    y = np.concatenate([rng.normal(3, 1, n), rng.normal(0, 1, n)])
    g = np.array(["hi"] * n + ["lo"] * n)  # "hi" first -> group 1
    df = pd.DataFrame({"y": y, "grp": g})
    csv = tmp_path / "mwuflip.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert res.estimates["rank_biserial"] > 0  # group 1 ("hi") dominates
    assert res.estimates["hodges_lehmann"] > 1.5


def test_mwu_no_shift_control(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    n = 60
    y = rng.normal(0, 1, 2 * n)  # same distribution
    g = np.array(["A"] * n + ["B"] * n)
    df = pd.DataFrame({"y": y, "grp": g})
    csv = tmp_path / "mwuflat.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert res.estimates["p_value"] > 0.05
    assert abs(res.estimates["rank_biserial"]) < 0.3  # small effect


def test_mwu_config_override(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    n = 40
    a = np.concatenate([rng.normal(0, 1, n), rng.normal(2.5, 1, n)])
    noise = rng.normal(0, 1, 2 * n)
    g = np.array(["ctrl"] * n + ["treat"] * n)
    df = pd.DataFrame({"target": a, "noise": noise, "arm": g})
    csv = tmp_path / "mwuovr.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "target", "group": "arm"})
    assert res.estimates["p_value"] < 0.01
    assert res.estimates["n1"] == float(n)


def test_mwu_degrade_three_groups(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    n = 20
    y = np.concatenate([rng.normal(0, 1, n), rng.normal(2, 1, n), rng.normal(4, 1, n)])
    g = np.array(["A"] * n + ["B"] * n + ["C"] * n)
    df = pd.DataFrame({"y": y, "grp": g})
    csv = tmp_path / "mwu3.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "y", "group": "grp"})
    assert "跳过" in res.summary
    assert "u_stat" not in res.estimates


def test_mwu_degrade_tiny_group(tmp_path: Path) -> None:
    # one group has a single observation -> skip
    df = pd.DataFrame({
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 99.0],
        "grp": ["A", "A", "A", "A", "A", "B"],
    })
    csv = tmp_path / "mwutiny.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "y", "group": "grp"})
    assert "跳过" in res.summary
    assert "u_stat" not in res.estimates
