"""Tests for g_computation: parametric g-formula recovers the ATE under confounding
(where the naive difference is biased); binary-outcome risk difference; honest skips."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="g_computation", method="G-computation", domain="epidemiology",
        family="causal", goal="explain", preconditions=Precondition(min_rows=20, min_continuous=1),
    )


def _confounded(seed: int = 0, n: int = 1500, ate: float = 2.0):
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0, 1, n)
    w2 = rng.normal(0, 1, n)
    p = 1 / (1 + np.exp(-(0.8 * w1 - 0.5 * w2)))     # treatment confounded by W
    t = (rng.uniform(size=n) < p).astype(int)
    y = 1.0 + ate * t + 1.5 * w1 - 1.0 * w2 + 0.5 * t * w1 + rng.normal(0, 1, n)
    return pd.DataFrame({"y": y, "treat": t, "w1": w1, "w2": w2})


def test_g_computation_recovers_ate_under_confounding(tmp_path: Path) -> None:
    csv = tmp_path / "g.csv"
    _confounded(ate=2.0).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"treatment": "treat", "outcome": "y",
                               "covariates": ["w1", "w2"], "n_boot": 200})
    out = Path(res.output_dir)
    assert (out / "g_computation_ate.png").exists()
    e = res.estimates
    # g-formula recovers true ATE ~2.0; naive difference is biased upward (confounding)
    assert abs(e["ate"] - 2.0) < 0.3
    assert e["naive_diff"] > e["ate"] + 0.3
    # bootstrap CI excludes 0
    assert e["ate_ci_low"] > 0
    assert e["n_treated"] > 5 and e["outcome_binary"] == 0.0


def test_g_computation_binary_outcome_risk_difference(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n = 1500
    w = rng.normal(0, 1, n)
    t = (rng.uniform(size=n) < 1 / (1 + np.exp(-w))).astype(int)
    # true risk difference ~ positive; logit outcome depends on t and w
    py = 1 / (1 + np.exp(-(-0.2 + 1.0 * t + 0.8 * w)))
    y = (rng.uniform(size=n) < py).astype(int)
    csv = tmp_path / "gb.csv"
    pd.DataFrame({"y": y, "treat": t, "w": w}).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"treatment": "treat", "outcome": "y", "covariates": ["w"], "n_boot": 200})
    e = res.estimates
    assert e["outcome_binary"] == 1.0
    assert 0.0 < e["ate"] < 1.0           # a risk difference in (0,1) here
    assert 0.0 <= e["e_y1"] <= 1.0 and 0.0 <= e["e_y0"] <= 1.0


def test_g_computation_skips_no_treatment(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"y": rng.normal(0, 1, 40), "x": rng.normal(0, 1, 40)})  # no binary treat
    csv = tmp_path / "x.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "ate" not in res.estimates
    assert "g-计算跳过" in res.summary
