"""Tests for the regularized_regression (lasso / ridge / elastic-net) executor branch.

Known structure: y is a linear function of a few INFORMATIVE predictors plus several
PURE-NOISE predictors. A penalized regression should (a) achieve cross-validated
R-squared > 0.5 (signal is recoverable) and (b) shrink the noise coefficients toward
zero (n_selected < n_predictors for lasso/elastic-net). Plus honest-degrade tests.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_HAS_SK = importlib.util.find_spec("sklearn") is not None
pytestmark = pytest.mark.skipif(not _HAS_SK, reason="scikit-learn not available")


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="regularized_regression", method="Regularized regression", domain="machine learning",
        family="ml", goal="predict", preconditions=Precondition(min_continuous=1, min_numeric_cols=2, min_rows=20),
    )


def _sparse_regression(tmp_path: Path, n: int = 200, n_noise: int = 6) -> Path:
    """y depends strongly on x1,x2,x3; the rest are noise. y is the FIRST column so the
    regression-family convention picks it as the outcome."""
    rng = np.random.default_rng(0)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    y = 3.0 * x1 - 2.0 * x2 + 1.5 * x3 + rng.normal(0, 0.5, n)
    data = {"y": y, "x1": x1, "x2": x2, "x3": x3}
    for j in range(n_noise):
        data[f"noise{j}"] = rng.normal(0, 1, n)
    df = pd.DataFrame(data).round(5)
    csv = tmp_path / "sparse.csv"
    df.to_csv(csv, index=False)
    return csv


def test_recovers_signal_and_shrinks_noise(tmp_path: Path) -> None:
    # exact-zero sparsity is a LASSO property; default elastic-net (l1_ratio<1) may keep
    # tiny nonzeros, so pin the "shrinks noise" claim to lasso + enough noise predictors
    # that the CV-optimal lasso reliably zeros several.
    fp = profile_dataset(_sparse_regression(tmp_path, n_noise=15))
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"method": "lasso"})
    assert "完成" in res.summary
    # honest cross-validated R-squared recovers the strong signal
    assert res.estimates["cv_r2"] > 0.5
    assert res.estimates["cv_rmse"] > 0
    assert res.estimates["alpha"] > 0
    # lasso shrinks: at least one noise coefficient hits exactly zero
    assert res.estimates["n_selected"] < res.estimates["n_predictors"]
    assert res.estimates["n"] == 200
    # coefficients CSV: informative predictors rank above the noise by |coef|
    coef = pd.read_csv(Path(res.output_dir) / "regularized_coefficients.csv")
    assert set(["predictor", "std_coef", "selected"]).issubset(coef.columns)
    top3 = set(coef.head(3)["predictor"])
    assert top3 == {"x1", "x2", "x3"}
    assert "regularized_coefficients.png" in res.files


def test_method_ridge_keeps_all(tmp_path: Path) -> None:
    # ridge has no exact-zero sparsity -> all coefficients selected
    fp = profile_dataset(_sparse_regression(tmp_path))
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"method": "ridge"})
    assert "完成" in res.summary
    assert res.estimates["n_selected"] == res.estimates["n_predictors"]
    assert res.estimates["cv_r2"] > 0.5


def test_config_outcome_predictors(tmp_path: Path) -> None:
    fp = profile_dataset(_sparse_regression(tmp_path))
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"outcome": "y", "predictors": ["x1", "x2", "x3"], "method": "lasso"},
    )
    assert "完成" in res.summary
    assert res.estimates["n_predictors"] == 3
    assert res.estimates["cv_r2"] > 0.5


def test_resolver_picks_named_outcome_not_first(tmp_path: Path) -> None:
    """A decoy continuous column ('decoy0', pure noise) is placed BEFORE 'y' — the
    shared resolver (ml_supervised._resolve_xy's continuous tier) must still pick
    'y', not cont[0]='decoy0'."""
    rng = np.random.default_rng(31)
    n = 200
    decoy0 = rng.normal(0, 1, n)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    y = 3.0 * x1 - 2.0 * x2 + 1.5 * x3 + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"decoy0": decoy0, "x1": x1, "x2": x2, "x3": x3, "y": y}).round(5)
    csv = tmp_path / "resolver.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    assert fp.likely_outcome == "y" and fp.likely_outcome_confidence == "high"
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"method": "lasso"})
    assert "完成" in res.summary
    # strong signal recovered -> confirms 'y' (not the unrelated decoy0) was modeled.
    assert res.estimates["cv_r2"] > 0.5


def test_degrade_too_few_rows(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"y": rng.normal(0, 1, 10), "x1": rng.normal(0, 1, 10), "x2": rng.normal(0, 1, 10)})
    csv = tmp_path / "tiny.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
    assert "cv_r2" not in res.estimates


def test_degrade_no_predictor(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"y": rng.normal(0, 1, 40)})
    csv = tmp_path / "onecol.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
    assert "cv_r2" not in res.estimates
