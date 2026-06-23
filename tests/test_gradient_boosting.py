"""Tests for the gradient_boosting (sklearn GBM) executor branch.

Known structure:
  * regression — y is driven mainly by ONE informative predictor plus noise columns ->
    permutation importance should rank that predictor top, and cross-validated R-squared
    should be > 0.5.
  * classification — a separable 2-class problem -> high cross-validated accuracy.
Plus honest-degrade tests.
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
        id="gradient_boosting", method="Gradient boosting", domain="machine learning",
        family="ml", goal="predict", preconditions=Precondition(min_numeric_cols=2, min_rows=20),
    )


def _one_strong_predictor(tmp_path: Path, n: int = 200, n_noise: int = 4) -> Path:
    """y is a (nonlinear) function of x_signal; the other columns are noise. y first."""
    rng = np.random.default_rng(0)
    x_signal = rng.uniform(-3, 3, n)
    y = 2.0 * x_signal + 0.5 * x_signal ** 2 + rng.normal(0, 0.4, n)
    data = {"y": y.round(5), "x_signal": x_signal.round(5)}
    for j in range(n_noise):
        data[f"noise{j}"] = rng.normal(0, 1, n).round(5)
    df = pd.DataFrame(data)
    csv = tmp_path / "signal.csv"
    df.to_csv(csv, index=False)
    return csv


def _separable_two_class(tmp_path: Path, n: int = 120) -> Path:
    rng = np.random.default_rng(1)
    half = n // 2
    f1 = np.concatenate([rng.normal(-2.0, 0.6, half), rng.normal(2.0, 0.6, half)])
    f2 = np.concatenate([rng.normal(-2.0, 0.6, half), rng.normal(2.0, 0.6, half)])
    label = np.array([0] * half + [1] * half)  # binary, but only non-continuous col
    df = pd.DataFrame({"label": label, "f1": f1.round(5), "f2": f2.round(5)})
    df = df.sample(frac=1.0, random_state=2).reset_index(drop=True)
    csv = tmp_path / "blobs.csv"
    df.to_csv(csv, index=False)
    return csv


def test_regression_ranks_signal_top(tmp_path: Path) -> None:
    fp = profile_dataset(_one_strong_predictor(tmp_path))
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "完成" in res.summary
    assert res.estimates["cv_r2"] > 0.5
    assert res.estimates["cv_rmse"] > 0
    assert res.estimates["n"] == 200
    imp = pd.read_csv(Path(res.output_dir) / "gbm_permutation_importance.csv")
    # sorted descending -> the informative predictor is the top row
    assert imp.iloc[0]["predictor"] == "x_signal"
    assert {"perm_importance_mean", "perm_importance_sd"}.issubset(imp.columns)
    assert "gbm_permutation_importance.png" in res.files


def test_classification_high_accuracy(tmp_path: Path) -> None:
    # name the categorical label as outcome (engine defaults to continuous=regression)
    fp = profile_dataset(_separable_two_class(tmp_path))
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"outcome": "label"})
    assert "完成" in res.summary
    assert res.estimates["cv_accuracy"] > 0.9
    assert res.estimates["cv_f1_macro"] > 0.9
    assert "cv_r2" not in res.estimates  # classification path
    assert "gbm_permutation_importance.csv" in res.files


def test_config_hyperparams(tmp_path: Path) -> None:
    fp = profile_dataset(_one_strong_predictor(tmp_path))
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"n_estimators": 50, "learning_rate": 0.2, "max_depth": 2},
    )
    assert "完成" in res.summary
    assert res.estimates["cv_r2"] > 0.4


def test_degrade_too_few_rows(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"y": rng.normal(0, 1, 10), "x1": rng.normal(0, 1, 10), "x2": rng.normal(0, 1, 10)})
    csv = tmp_path / "tiny.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
    assert "cv_r2" not in res.estimates


def test_degrade_no_predictor(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"y": rng.normal(0, 1, 40)})
    csv = tmp_path / "onecol.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
    assert "cv_r2" not in res.estimates
