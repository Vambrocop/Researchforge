"""Tests for the Oster (2019) coefficient-stability sensitivity branch.

Known checks:
  - controls barely move β (β°≈β̃) -> LARGE δ (robust to unobservables);
  - controls (a confounder) kill the effect -> SMALL δ (fragile);
  - degrade on missing treatment/outcome/controls; config plumbing.
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
        id="oster_delta", method="Oster delta (coefficient stability)",
        domain="economics", family="causal", goal="explain",
        preconditions=Precondition(requires_treatment=True, min_rows=20, min_continuous=1),
    )


def test_oster_robust_when_controls_barely_move_beta(tmp_path: Path) -> None:
    # treatment effect is real and the controls are (near-)independent of treatment,
    # so adding them barely changes β -> δ should be large (robust).
    rng = np.random.default_rng(0)
    n = 600
    treat = rng.normal(0, 1, n)
    c1, c2 = rng.normal(0, 1, n), rng.normal(0, 1, n)  # unrelated to treat
    y = 2.0 * treat + 0.4 * c1 + 0.3 * c2 + rng.normal(0, 1, n)
    csv = tmp_path / "robust.csv"
    pd.DataFrame({"y": y, "treat": treat, "c1": c1, "c2": c2}).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "y", "treatment": "treat", "controls": ["c1", "c2"]})
    assert "完成" in res.summary
    # β° ≈ β̃ (controls don't confound) -> δ large in magnitude, β* keeps the sign
    assert abs(res.estimates["beta_short"] - res.estimates["beta_long"]) < 0.15
    assert abs(res.estimates["delta"]) > 5.0
    assert np.sign(res.estimates["beta_star"]) == np.sign(res.estimates["beta_long"])
    assert "稳健" in res.summary


def test_oster_fragile_when_controls_kill_effect(tmp_path: Path) -> None:
    # a confounder drives BOTH treatment and outcome; the naive β° is large but the
    # long-regression β̃ collapses toward 0 -> small δ (fragile).
    rng = np.random.default_rng(1)
    n = 600
    conf = rng.normal(0, 1, n)
    treat = 1.5 * conf + rng.normal(0, 0.5, n)   # treatment driven by the confounder
    y = 2.0 * conf + rng.normal(0, 0.5, n)        # outcome driven by the SAME confounder, not treat
    csv = tmp_path / "fragile.csv"
    pd.DataFrame({"y": y, "treat": treat, "conf": conf}).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "y", "treatment": "treat", "controls": ["conf"]})
    assert "完成" in res.summary
    # β° (spurious) >> β̃ (≈0 once confounder controlled) and δ small
    assert abs(res.estimates["beta_short"]) > abs(res.estimates["beta_long"]) + 0.5
    assert abs(res.estimates["delta"]) < 1.0
    assert "脆弱" in res.summary


def test_oster_rmax_config_override(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    n = 300
    treat = rng.normal(0, 1, n)
    c1 = 0.5 * treat + rng.normal(0, 1, n)
    y = 1.0 * treat + 0.8 * c1 + rng.normal(0, 1, n)
    csv = tmp_path / "rmax.csv"
    pd.DataFrame({"y": y, "treat": treat, "c1": c1}).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "y", "treatment": "treat", "controls": ["c1"],
                               "r_max": 0.95})
    assert "完成" in res.summary
    assert abs(res.estimates["r_max"] - 0.95) < 1e-6  # config R_max honoured
    assert "oster_delta.csv" in res.files


def test_oster_needs_treatment_and_controls(tmp_path: Path) -> None:
    df = pd.DataFrame({"y": np.random.default_rng(3).normal(0, 1, 30)})  # only an outcome
    csv = tmp_path / "x.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"outcome": "y"})
    assert "Oster δ 失败" in res.summary
