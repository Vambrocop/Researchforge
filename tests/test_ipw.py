"""Tests for the inverse-probability weighting (IPW) executor branch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="ipw", method="Inverse-probability weighting", domain="economics",
        family="causal", goal="explain",
        preconditions=Precondition(requires_treatment=True, min_rows=30),
    )


def test_ipw_recovers_ate_under_confounding(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 500
    x1, x2 = rng.normal(0, 1, n), rng.normal(0, 1, n)
    ps = 1.0 / (1.0 + np.exp(-(0.7 * x1 + 0.5 * x2)))  # confounded treatment
    t = (rng.uniform(size=n) < ps).astype(int)
    ate = 2.0
    y = 1.0 + ate * t + 1.2 * x1 + 0.8 * x2 + rng.normal(0, 1, n)
    csv = tmp_path / "c.csv"
    pd.DataFrame({"y": y, "treat": t, "x1": x1, "x2": x2}).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"treatment": "treat", "outcome": "y", "covariates": ["x1", "x2"]})
    assert "完成" in res.summary
    assert abs(res.estimates["ate"] - ate) < 0.6     # IPW recovers ~2.0 (constant effect -> ATE=ATT)
    assert res.estimates["se"] > 0
    assert 0 < res.estimates["ess"] <= n


def test_ipw_needs_treatment_and_covariates(tmp_path: Path) -> None:
    df = pd.DataFrame({"y": np.random.default_rng(1).normal(0, 1, 20)})
    csv = tmp_path / "x.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"outcome": "y"})
    assert "逆概率加权失败" in res.summary
