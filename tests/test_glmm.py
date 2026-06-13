"""Tests for the GLMM (lme4 glmer) executor branch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis, rbridge
from researchforge.profiler import profile_dataset

_HAS_LME4 = rbridge.r_available() and rbridge.r_package_available("lme4")


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="glmm", method="GLMM", domain="statistics", family="statistics",
        goal="explain", preconditions=Precondition(requires_group=True, min_rows=40),
    )


@pytest.mark.skipif(not _HAS_LME4, reason="R lme4 not available")
def test_glmm_binomial_random_intercept(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n, ngrp = 600, 20
    grp = rng.integers(0, ngrp, n)
    re_int = rng.normal(0, 0.8, ngrp)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    eta = -0.3 + 0.9 * x1 - 0.6 * x2 + re_int[grp]
    p = 1 / (1 + np.exp(-eta))
    y = (rng.uniform(0, 1, n) < p).astype(int)
    df = pd.DataFrame({"site": [f"s{g}" for g in grp], "x1": x1, "x2": x2, "success": y})
    csv = tmp_path / "g.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "binomial" in res.summary
    assert res.estimates["group_intercept_var"] > 0
    assert res.estimates["n_groups"] == 20.0
    assert "icc" in res.estimates  # binomial latent-scale ICC reported


@pytest.mark.skipif(not _HAS_LME4, reason="R lme4 not available")
def test_glmm_poisson_overdispersion_check(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    n, ngrp = 500, 15
    grp = rng.integers(0, ngrp, n)
    re_int = rng.normal(0, 0.5, ngrp)
    x1 = rng.normal(0, 1, n)
    lam = np.exp(0.5 + 0.4 * x1 + re_int[grp])
    y = rng.poisson(lam)
    df = pd.DataFrame({"farm": [f"f{g}" for g in grp], "x1": x1, "visits": y})
    csv = tmp_path / "p.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "poisson" in res.summary
    assert "overdispersion" in res.estimates


def test_glmm_no_group_degrades(tmp_path: Path) -> None:
    # binary outcome + continuous predictors but no grouping column -> honest fail
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {"y": rng.integers(0, 2, 50), "x1": rng.normal(0, 1, 50), "x2": rng.normal(0, 1, 50)}
    )
    csv = tmp_path / "g.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "GLMM 失败" in res.summary
