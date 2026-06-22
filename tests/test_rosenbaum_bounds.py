"""Tests for the Rosenbaum (2002) sensitivity-bounds branch.

Known checks:
  - strong treatment effect -> high breakdown Γ (robust to hidden bias);
  - weak effect -> breakdown Γ near 1 (fragile);
  - helper sanity (p+ >= p- ; p-values rise with Γ);
  - degrade on missing treatment/outcome; config plumbing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.executor.branches.sensitivity import _rosenbaum_signed_rank_pvals
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="rosenbaum_bounds", method="Rosenbaum sensitivity bounds (hidden bias)",
        domain="statistics", family="causal", goal="explain",
        preconditions=Precondition(requires_treatment=True, min_rows=12, min_continuous=1),
    )


def test_rosenbaum_helper_bounds_order_and_monotone() -> None:
    # all-positive differences -> strongly significant; p+ >= p- and p+ rises with Γ.
    diffs = list(range(1, 21))  # 20 positive paired differences
    p1_plus, p1_minus, w1, n1 = _rosenbaum_signed_rank_pvals(diffs, 1.0)
    p2_plus, p2_minus, w2, n2 = _rosenbaum_signed_rank_pvals(diffs, 2.0)
    assert n1 == 20 and n2 == 20
    assert p1_plus <= p1_minus + 1e-12  # at Γ=1 the two bounds coincide
    assert abs(p1_plus - p1_minus) < 1e-9
    assert p2_plus >= p1_plus            # worst-case p grows with Γ
    assert p2_plus >= p2_minus           # upper bound >= lower bound
    assert p1_plus < 0.05                # baseline significant


def test_rosenbaum_strong_effect_high_breakdown(tmp_path: Path) -> None:
    # treated outcomes much higher than controls -> hard to overturn -> high Γ.
    rng = np.random.default_rng(0)
    n = 120
    treat = np.array([0, 1] * (n // 2))
    y = 3.0 * treat + rng.normal(0, 1, n)
    csv = tmp_path / "strong.csv"
    pd.DataFrame({"y": y, "treat": treat}).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "y", "treatment": "treat", "gamma_max": 5.0})
    assert "完成" in res.summary
    assert res.estimates["p_value_gamma1"] < 0.05
    bd = res.estimates["breakdown_gamma"]
    # either a high breakdown Γ, or never broke down within the grid (extremely robust)
    assert (bd != bd) or bd >= 1.8


def test_rosenbaum_weak_effect_low_breakdown(tmp_path: Path) -> None:
    # tiny treatment effect drowned in noise -> breakdown Γ near 1 (fragile).
    rng = np.random.default_rng(1)
    n = 120
    treat = np.array([0, 1] * (n // 2))
    y = 0.15 * treat + rng.normal(0, 1, n)
    csv = tmp_path / "weak.csv"
    pd.DataFrame({"y": y, "treat": treat}).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "y", "treatment": "treat", "gamma_max": 3.0})
    assert "完成" in res.summary
    bd = res.estimates["breakdown_gamma"]
    # fragile: either non-significant at baseline, or breaks down at a small Γ
    assert (bd == bd) and bd <= 1.6


def test_rosenbaum_config_grid_and_files(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    n = 100
    treat = np.array([0, 1] * (n // 2))
    y = 1.0 * treat + rng.normal(0, 1, n)
    csv = tmp_path / "grid.csv"
    pd.DataFrame({"y": y, "treat": treat}).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "y", "treatment": "treat",
                               "gamma_max": 2.0, "gamma_step": 0.25})
    assert "完成" in res.summary
    assert "rosenbaum_bounds.csv" in res.files
    assert abs(res.estimates["gamma_max_tested"] - 2.0) < 1e-6


def test_rosenbaum_needs_treatment(tmp_path: Path) -> None:
    df = pd.DataFrame({"y": np.random.default_rng(3).normal(0, 1, 20)})  # no treatment
    csv = tmp_path / "x.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"outcome": "y"})
    assert "Rosenbaum 界失败" in res.summary
