"""Tests for cliffs_delta: nonparametric ordinal-valid dominance effect size.

Known-structure: fully-separated groups -> delta = +1; reversed order -> -1;
identical -> ~0. Independently recompute delta via pairwise dominance, check the
Romano magnitude ordinal, the δ = 2·AUC − 1 identity, config override, and degrade.
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
        id="cliffs_delta",
        method="Cliff's delta",
        domain="statistics",
        family="effect_sizes",
        goal="describe",
        preconditions=Precondition(requires_group=True, min_continuous=1, min_rows=6),
    )


def _cliff_indep(a, b):
    diff = a[:, None] - b[None, :]
    return ((diff > 0).sum() - (diff < 0).sum()) / (a.size * b.size)


def test_cliffs_delta_full_dominance_is_plus_one(tmp_path: Path) -> None:
    # every value in A strictly above every value in B -> delta = +1 (large)
    n = 40
    # +0.5 so values are non-whole floats -> profiler classifies y as continuous
    # (all-distinct WHOLE integers hit the profiler "id trap" -> outcome not found)
    a = np.arange(100.0, 100.0 + n) + 0.5   # 100.5..139.5
    b = np.arange(0.0, float(n)) + 0.5       # 0.5..39.5
    y = np.concatenate([a, b])
    g = np.array(["A"] * n + ["B"] * n)
    df = pd.DataFrame({"y": y, "grp": g})
    csv = tmp_path / "dom.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)
    assert (out / "cliffs_delta.csv").exists()

    assert abs(res.estimates["cliffs_delta"] - 1.0) < 1e-9
    assert abs(res.estimates["cliffs_delta"] - _cliff_indep(a, b)) < 1e-9
    assert res.estimates["magnitude_negligible_to_large"] == 3.0  # large
    # delta = 2*AUC - 1  ->  P(X1>X2) = (delta+1)/2 = 1.0 here
    assert abs(res.estimates["p_x1_gt_x2"] - 1.0) < 1e-9
    assert abs(res.estimates["cliffs_delta"] - (2 * res.estimates["p_x1_gt_x2"] - 1)) < 1e-9


def test_cliffs_delta_reversed_is_minus_one(tmp_path: Path) -> None:
    # first level entirely BELOW the second -> delta = -1
    n = 30
    # +0.5 -> non-whole floats so y is continuous (avoid the profiler id trap)
    a = np.arange(0.0, float(n)) + 0.5         # low group first
    b = np.arange(100.0, 100.0 + n) + 0.5      # high group second
    y = np.concatenate([a, b])
    g = np.array(["lo"] * n + ["hi"] * n)
    df = pd.DataFrame({"y": y, "grp": g})
    csv = tmp_path / "rev.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert abs(res.estimates["cliffs_delta"] + 1.0) < 1e-9
    assert res.estimates["p_x1_gt_x2"] == 0.0


def test_cliffs_delta_identical_near_zero(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    n = 120
    y = rng.normal(0, 1, 2 * n)
    g = np.array(["A"] * n + ["B"] * n)
    df = pd.DataFrame({"y": y, "grp": g})
    csv = tmp_path / "same.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert abs(res.estimates["cliffs_delta"]) < 0.2     # negligible/small
    assert res.estimates["magnitude_negligible_to_large"] <= 1.0
    assert abs(res.estimates["p_x1_gt_x2"] - 0.5) < 0.1


def test_cliffs_delta_one_sd_shift_positive_large(tmp_path: Path) -> None:
    # designed ~1 SD shift: group1 higher -> cliffs delta clearly positive
    rng = np.random.default_rng(8)
    n = 80
    a = rng.normal(5.0, 1.0, n)
    b = rng.normal(4.0, 1.0, n)
    y = np.concatenate([a, b])
    g = np.array(["A"] * n + ["B"] * n)
    df = pd.DataFrame({"y": y, "grp": g})
    csv = tmp_path / "shift.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    delta = res.estimates["cliffs_delta"]
    assert delta > 0.3                       # positive, at least medium-ish
    # (estimates are stored rounded to ~5 dp for display; compare at that precision)
    assert abs(delta - _cliff_indep(a, b)) < 1e-4
    assert res.estimates["p_x1_gt_x2"] > 0.5


def test_cliffs_delta_config_override(tmp_path: Path) -> None:
    n = 30
    a = np.concatenate([np.arange(50.0, 50.0 + n), np.arange(0.0, float(n))])
    real_g = np.array(["hi"] * n + ["lo"] * n)
    df = pd.DataFrame({"a": a, "real_g": real_g})
    csv = tmp_path / "ovr.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "a", "group": "real_g"})
    assert abs(res.estimates["cliffs_delta"] - 1.0) < 1e-9


def test_cliffs_delta_degrade_no_group(tmp_path: Path) -> None:
    df = pd.DataFrame({"y": np.arange(20, dtype=float)})
    csv = tmp_path / "nogrp.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
    assert "cliffs_delta" not in res.estimates
