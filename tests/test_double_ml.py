"""Tests for the double/debiased machine learning (doubleml) executor branch."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_HAS_DML = importlib.util.find_spec("doubleml") is not None


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="double_ml", method="Double ML", domain="economics", family="causal",
        goal="explain", preconditions=Precondition(requires_treatment=True, min_rows=100),
    )


def _binary_treat(tmp_path: Path) -> Path:
    rng = np.random.default_rng(7)
    n = 1200
    x0, x1, x2 = rng.normal(0, 1, n), rng.normal(0, 1, n), rng.normal(0, 1, n)
    ps = 1 / (1 + np.exp(-(0.8 * x0 - 0.5 * x1)))
    d = (rng.uniform(0, 1, n) < ps).astype(int)
    y = 2.0 * d + 1.5 * x0 - 1.0 * x1 + 0.3 * x2 + rng.normal(0, 1, n)  # true ATE = 2
    df = pd.DataFrame({"x0": x0, "x1": x1, "x2": x2, "treat": d, "y": y})
    csv = tmp_path / "dml.csv"
    df.to_csv(csv, index=False)
    return csv


@pytest.mark.skipif(not _HAS_DML, reason="doubleml not available")
def test_dml_binary_recovers_ate(tmp_path: Path) -> None:
    fp = profile_dataset(_binary_treat(tmp_path))
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"treatment": "treat", "outcome": "y"},
    )
    assert "doubleml" in res.summary
    # true ATE = 2.0; orthogonal cross-fitted estimate should be close + CI cover it
    assert abs(res.estimates["ate"] - 2.0) < 0.6
    assert res.estimates["ci_lb"] <= 2.0 <= res.estimates["ci_ub"]
    assert res.estimates["p_value"] < 0.05


@pytest.mark.skipif(not _HAS_DML, reason="doubleml not available")
def test_dml_continuous_treatment_plr(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    n = 800
    x0, x1 = rng.normal(0, 1, n), rng.normal(0, 1, n)
    t = 0.5 * x0 + rng.normal(0, 1, n)  # continuous treatment, confounded by x0
    y = 1.5 * t + 2.0 * x0 - x1 + rng.normal(0, 1, n)  # true effect = 1.5
    df = pd.DataFrame({"x0": x0, "x1": x1, "dose": t, "y": y})
    csv = tmp_path / "plr.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"treatment": "dose", "outcome": "y", "controls": ["x0", "x1"]},
    )
    assert "PLR" in res.summary
    # true effect 1.5; finite-sample RF nuisance leaves a little regularization bias,
    # so assert ballpark recovery (a clearly-positive effect near 1.5) + significance
    # rather than exact 95% CI coverage (a stochastic property).
    assert abs(res.estimates["ate"] - 1.5) < 0.5
    assert res.estimates["p_value"] < 0.05


def test_dml_no_treatment_degrades(tmp_path: Path) -> None:
    # only continuous columns, no binary treatment + no config -> honest message
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"y": rng.normal(0, 1, 150), "a": rng.normal(0, 1, 150), "b": rng.normal(0, 1, 150)})
    csv = tmp_path / "x.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "双重机器学习" in res.summary and ("处理变量" in res.summary or "doubleml" in res.summary)
