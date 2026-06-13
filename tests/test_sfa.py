"""Tests for sfa: input/output gate + stochastic frontier via R (skips no R)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import rbridge, run_analysis
from researchforge.profiler import profile_dataset
from researchforge.recommender.match import check_preconditions


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="sfa",
        method="Stochastic Frontier Analysis (SFA)",
        domain="economics",
        family="efficiency",
        goal="explain",
        preconditions=Precondition(min_continuous=2, min_rows=30),
    )


def test_sfa_recovers_frontier(tmp_path: Path) -> None:
    if not (rbridge.r_available() and rbridge.r_package_available("frontier")):
        pytest.skip("R frontier package not available")

    rng = np.random.default_rng(0)
    n = 120
    x1 = rng.uniform(1, 5, n)
    x2 = rng.uniform(1, 5, n)
    v = rng.normal(0, 0.1, n)
    u = np.abs(rng.normal(0, 0.3, n))  # technical inefficiency
    y = np.exp(0.5 + 0.6 * np.log(x1) + 0.3 * np.log(x2) + v - u)
    df = pd.DataFrame({"output": np.round(y, 3), "x1": np.round(x1, 3), "x2": np.round(x2, 3)})
    csv = tmp_path / "prod.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    assert (out / "frontier_coefficients.csv").exists()
    te = pd.read_csv(out / "technical_efficiency.csv")
    assert (te["technical_efficiency"] > 0).all() and (te["technical_efficiency"] <= 1.0 + 1e-6).all()
    assert 0.0 < res.estimates["mean_technical_efficiency"] < 1.0  # genuine inefficiency present
    assert res.estimates["gamma"] > 0.5  # variance dominated by inefficiency (DGP: sigma_u>sigma_v)
    assert res.estimates["lr_inefficiency_pvalue"] < 0.05  # LR test detects the inefficiency


def test_sfa_precondition_unmet(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"only_output": rng.uniform(1, 5, 40)})  # < 2 numeric columns
    csv = tmp_path / "one.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)

    assert not ok
    assert any("连续" in u for u in unmet)
