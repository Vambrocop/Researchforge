"""Tests for moderated_mediation — first-stage moderated mediation (PROCESS model 7).

Data-generating process plants a TRUE first-stage moderation:
  M = a1*X + a2*W + a3*(X*W) + noise   (a3 != 0)
  Y = b*M + cprime*X + noise           (b != 0)
so the index of moderated mediation (a3*b) is non-null and its bootstrap CI
should exclude 0. Also checks honest degrade and config role override.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog import AnalysisEntry
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_ENTRY = AnalysisEntry(
    id="moderated_mediation",
    method="First-stage moderated mediation (PROCESS model 7)",
    domain="statistics",
    family="conditional_process",
    goal="explain",
    preconditions={"min_continuous": 4, "min_rows": 30},
)


def _make_mod_med(n: int = 400, a3: float = 0.6, b: float = 0.8, seed: int = 7):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, n)
    W = rng.normal(0, 1, n)
    a1, a2, cprime = 0.3, 0.2, 0.25
    M = a1 * X + a2 * W + a3 * (X * W) + rng.normal(0, 0.5, n)
    Y = b * M + cprime * X + rng.normal(0, 0.5, n)
    # column order chosen so auto-assignment gives Y=outcome, then X, M, W.
    # Use floats so the profiler classifies them as continuous.
    return pd.DataFrame({"outcome_y": Y, "pred_x": X, "med_m": M, "mod_w": W})


def test_index_recovered_ci_excludes_zero(tmp_path: Path) -> None:
    df = _make_mod_med()
    csv = tmp_path / "mm.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _ENTRY, output_root=str(tmp_path / "o"),
        config={"x": "pred_x", "m": "med_m", "y": "outcome_y", "w": "mod_w"},
    )
    est = res.estimates
    # all required keys present
    for k in ("index_mod_med", "index_ci_low", "index_ci_high",
              "indirect_lo", "indirect_mean", "indirect_hi", "a3", "b"):
        assert k in est, f"missing estimate {k}"
    # planted a3*b = 0.6*0.8 = 0.48; recovered index should be clearly positive
    assert est["index_mod_med"] > 0.15
    # bootstrap CI brackets the point estimate and excludes 0
    assert est["index_ci_low"] < est["index_mod_med"] < est["index_ci_high"]
    assert est["index_ci_low"] > 0.0
    # a3 and b recovered with correct sign
    assert est["a3"] > 0.0 and est["b"] > 0.0
    # conditional indirect effect grows with W (positive first-stage moderation)
    assert est["indirect_hi"] > est["indirect_lo"]


def test_products_and_disclosure(tmp_path: Path) -> None:
    df = _make_mod_med(seed=11)
    csv = tmp_path / "mm.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _ENTRY, output_root=str(tmp_path / "o"),
        config={"x": "pred_x", "m": "med_m", "y": "outcome_y", "w": "mod_w"},
    )
    out = Path(res.output_dir)
    assert (out / "moderated_mediation_summary.csv").exists()
    assert (out / "moderated_mediation_grid.csv").exists()
    assert (out / "moderated_mediation_plot.png").exists()
    # core disclosures present
    assert "序贯可忽略性" in res.summary
    assert "bootstrap" in res.summary.lower() or "B=" in res.summary
    assert "seed" in res.summary.lower()


def test_default_role_assignment_discloses(tmp_path: Path) -> None:
    # No config -> roles auto-assigned by column order; the disclosure must fire.
    df = _make_mod_med(seed=3)
    csv = tmp_path / "mm.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))
    assert "index_mod_med" in res.estimates
    assert "角色按列序自动指派" in res.summary


def test_too_few_continuous_skips(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.normal(0, 1, n),
        "label": ["x", "y", "z"] * 20,  # not continuous
    })
    csv = tmp_path / "few.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))
    assert "index_mod_med" not in res.estimates
    assert "跳过" in res.summary
    assert (Path(res.output_dir) / "report.md").exists()


def test_too_few_rows_skips(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n = 20  # < 30
    df = pd.DataFrame({
        "y": rng.normal(0, 1, n),
        "x": rng.normal(0, 1, n),
        "m": rng.normal(0, 1, n),
        "w": rng.normal(0, 1, n),
    })
    csv = tmp_path / "small.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))
    assert "index_mod_med" not in res.estimates
    assert "跳过" in res.summary
