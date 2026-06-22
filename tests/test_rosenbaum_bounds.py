"""Tests for the Rosenbaum (2002) matched-pair sensitivity-bounds branch.

Key correctness checks (the earlier withdrawn version paired by OUTCOME rank —
circular; this version matches on COVARIATES):
  - a strong, clean treatment effect with good covariate overlap -> significant
    at Gamma=1 and robust (large / no finite critical Gamma);
  - no treatment effect -> non-significant at Gamma=1 (analysis "not applicable");
  - config plumbing + product files; honest degrade without covariates.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="rosenbaum_bounds",
        method="Rosenbaum (2002) sensitivity bounds (matched-pair signed-rank)",
        domain="statistics", family="causal", goal="explain",
        preconditions=Precondition(requires_treatment=True, min_rows=20, min_continuous=1),
    )


def _design(n: int, effect: float, seed: int) -> pd.DataFrame:
    """y (outcome) first, z (binary treatment) assigned from covariates x1/x2."""
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(0, 1, n), rng.normal(0, 1, n)
    p = 1.0 / (1.0 + np.exp(-(0.8 * x1 - 0.5 * x2)))  # propensity depends on covariates
    z = (rng.uniform(size=n) < p).astype(int)
    y = effect * z + 0.6 * x1 + 0.4 * x2 + rng.normal(0, 1.0, n)
    return pd.DataFrame({"y": y, "z": z, "x1": x1, "x2": x2})


def test_rosenbaum_robust_strong_effect(tmp_path: Path) -> None:
    csv = tmp_path / "robust.csv"
    _design(300, effect=1.0, seed=0).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "完成" in res.summary
    # a real effect with covariate-matched pairs -> significant at Gamma=1 ...
    assert res.estimates["p_value_gamma1"] < 0.05
    assert res.estimates["n_pairs"] >= 5
    # ... and robust: either no finite breaking Gamma within the grid (極稳健)
    # or a sizeable critical Gamma; the verdict says 稳健 either way.
    assert "稳健" in res.summary
    gc = res.estimates["gamma_critical"]
    assert math.isnan(gc) or gc >= 1.5


def test_rosenbaum_nonsignificant_when_no_effect(tmp_path: Path) -> None:
    csv = tmp_path / "null.csv"
    _design(300, effect=0.0, seed=7).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "完成" in res.summary
    # no treatment effect -> matched-pair differences are ~symmetric about 0
    assert res.estimates["p_value_gamma1"] > 0.05
    assert "不显著" in res.summary
    assert math.isnan(res.estimates["gamma_critical"])  # no breaking Gamma reported


def test_rosenbaum_config_and_files(tmp_path: Path) -> None:
    csv = tmp_path / "cfg.csv"
    df = _design(260, effect=0.8, seed=3)
    df = df.rename(columns={"y": "score", "z": "arm", "x1": "age", "x2": "bmi"})
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"treatment": "arm", "outcome": "score", "covariates": ["age", "bmi"]},
    )
    assert "完成" in res.summary
    assert "rosenbaum_bounds.csv" in res.files
    assert "rosenbaum_summary.txt" in res.files
    assert res.estimates["n_pairs"] >= 5


def test_rosenbaum_degrades_without_covariates(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    n = 40
    df = pd.DataFrame({
        "y": rng.normal(0, 1, n),
        "z": rng.integers(0, 2, n),  # binary treatment but NO covariate to match on
    })
    csv = tmp_path / "nocov.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "Rosenbaum 边界失败" in res.summary
