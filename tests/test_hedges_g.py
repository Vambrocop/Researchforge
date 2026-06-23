"""Tests for hedges_g: small-sample bias-corrected SMD (g = d × J).

Known-structure: ~1 SD shift -> g near 1 but strictly less than |d| (J<1 shrinks).
Independently recompute J = 1 - 3/(4N-9) and g = J*d. Plus small-n correction is
larger, config override, and honest degrade.
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
        id="hedges_g",
        method="Hedges' g",
        domain="statistics",
        family="effect_sizes",
        goal="describe",
        preconditions=Precondition(requires_group=True, min_continuous=1, min_rows=6),
    )


def _pooled_sd(a, b):
    n1, n2 = a.size, b.size
    return np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))


def test_hedges_g_one_sd_shift(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n = 50
    a = rng.normal(5.0, 1.0, n)
    b = rng.normal(4.0, 1.0, n)
    y = np.concatenate([a, b])
    g = np.array(["A"] * n + ["B"] * n)
    df = pd.DataFrame({"y": y, "grp": g})
    csv = tmp_path / "shift.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)
    assert (out / "hedges_g.csv").exists()

    # independent recomputation
    d_indep = (a.mean() - b.mean()) / _pooled_sd(a, b)
    N = 2 * n
    j_indep = 1.0 - 3.0 / (4.0 * N - 9.0)
    g_indep = j_indep * d_indep

    assert abs(res.estimates["cohens_d"] - d_indep) < 1e-4
    assert abs(res.estimates["correction_j"] - j_indep) < 1e-6
    assert abs(res.estimates["hedges_g"] - g_indep) < 1e-4
    # J < 1 so |g| strictly less than |d| (bias correction shrinks toward 0)
    assert abs(res.estimates["hedges_g"]) < abs(res.estimates["cohens_d"])
    assert res.estimates["ci_low"] < res.estimates["hedges_g"] < res.estimates["ci_high"]
    assert 0.6 < res.estimates["hedges_g"] < 1.4


def test_hedges_g_smaller_n_bigger_correction(tmp_path: Path) -> None:
    # the correction J is further below 1 (more shrinkage) for smaller total n
    rng = np.random.default_rng(2)

    def _run(n):
        a = rng.normal(2.0, 1.0, n)
        b = rng.normal(0.0, 1.0, n)
        y = np.concatenate([a, b])
        g = np.array(["A"] * n + ["B"] * n)
        df = pd.DataFrame({"y": y, "grp": g})
        csv = tmp_path / f"n{n}.csv"
        df.to_csv(csv, index=False)
        fp = profile_dataset(csv)
        return run_analysis(fp, _entry(), output_root=str(tmp_path / f"o{n}"))

    small = _run(6)
    big = _run(120)
    assert small.estimates["correction_j"] < big.estimates["correction_j"]
    assert big.estimates["correction_j"] > 0.99   # J -> 1 as n grows
    assert small.estimates["correction_j"] < 1.0


def test_hedges_g_config_override(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    n = 40
    a = np.concatenate([rng.normal(10, 1, n), rng.normal(7, 1, n)])
    real_g = np.array(["hi"] * n + ["lo"] * n)
    df = pd.DataFrame({"a": a, "real_g": real_g})
    csv = tmp_path / "ovr.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "a", "group": "real_g"})
    assert res.estimates["hedges_g"] > 1.0
    assert abs(res.estimates["hedges_g"]) < abs(res.estimates["cohens_d"])


def test_hedges_g_degrade_one_group(tmp_path: Path) -> None:
    # a single-level group column -> not two groups -> honest skip
    df = pd.DataFrame({"y": np.arange(20, dtype=float),
                       "grp": ["A"] * 20})
    csv = tmp_path / "one.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
    assert "hedges_g" not in res.estimates
