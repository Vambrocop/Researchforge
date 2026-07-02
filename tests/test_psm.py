"""Tests for the propensity score matching (PSM) executor branch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, t as t_dist

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="psm", method="Propensity score matching", domain="economics",
        family="causal", goal="explain",
        preconditions=Precondition(requires_treatment=True, min_rows=30),
    )


def test_psm_recovers_att_under_confounding(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 400
    x1, x2 = rng.normal(0, 1, n), rng.normal(0, 1, n)
    ps = 1.0 / (1.0 + np.exp(-(0.8 * x1 + 0.6 * x2)))  # treatment confounded by X
    t = (rng.uniform(size=n) < ps).astype(int)
    att = 2.0
    y = 1.0 + att * t + 1.5 * x1 + 1.0 * x2 + rng.normal(0, 1, n)  # X confounds outcome too
    csv = tmp_path / "c.csv"
    pd.DataFrame({"y": y, "treat": t, "x1": x1, "x2": x2}).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"treatment": "treat", "outcome": "y", "covariates": ["x1", "x2"]})
    assert "完成" in res.summary
    assert abs(res.estimates["att"] - att) < 0.8          # recovers ~2.0 controlling confounders
    assert res.estimates["n_matched_pairs"] >= 30
    assert res.estimates["max_abs_smd_after"] < 0.25      # matching improved covariate balance


def test_psm_pvalue_uses_t_distribution_at_small_n(tmp_path: Path) -> None:
    """With few matched pairs (low df), the ATT p-value must come from the
    t distribution, which is more conservative (>= p) than the normal
    approximation the code used to use."""
    rng = np.random.default_rng(0)
    n = 14  # small sample -> confounded treatment probabilistically assigned, few pairs
    x1 = rng.normal(0, 1, n)
    ps = 1.0 / (1.0 + np.exp(-(1.0 * x1)))
    treat = (rng.uniform(size=n) < ps).astype(int)
    y = 1.0 + 1.5 * treat + 0.8 * x1 + rng.normal(0, 0.5, n)
    csv = tmp_path / "small.csv"
    pd.DataFrame({"y": y, "treat": treat, "x1": x1}).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"treatment": "treat", "outcome": "y", "covariates": ["x1"]})
    assert "完成" in res.summary

    n_pairs = res.estimates["n_matched_pairs"]
    assert 3 <= n_pairs <= 6  # small-n regime: df as low as 2
    att, se, pval = res.estimates["att"], res.estimates["se"], res.estimates["pvalue"]
    assert np.isfinite(pval)

    tstat = att / se
    df = n_pairs - 1
    expected_t_p = 2 * t_dist.sf(abs(tstat), df=df)
    normal_p = 2 * norm.sf(abs(tstat))
    assert pval == pytest.approx(expected_t_p, rel=1e-9)
    assert pval >= normal_p  # t reference distribution is more conservative at low df


def test_psm_needs_treatment_and_covariates(tmp_path: Path) -> None:
    df = pd.DataFrame({"y": np.random.default_rng(1).normal(0, 1, 20)})  # no treatment, no covs
    csv = tmp_path / "x.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"outcome": "y"})
    assert "倾向得分匹配失败" in res.summary
