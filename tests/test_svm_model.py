"""Tests for the svm_model (SVC / SVR) executor branch.

Known structure:
  * classification — a well-separated 2-class problem (two Gaussian blobs) where the
    outcome is binary/categorical -> SVC should reach high cross-validated accuracy.
  * regression — a continuous outcome that is a smooth function of the predictors ->
    SVR should reach a positive cross-validated R-squared.
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
        id="svm_model", method="Support vector machine", domain="machine learning",
        family="ml", goal="predict", preconditions=Precondition(min_numeric_cols=2, min_rows=20),
    )


def _separable_two_class(tmp_path: Path, n: int = 120) -> Path:
    """Two well-separated Gaussian blobs labelled by a categorical 'group' column.
    'group' is the only non-continuous column -> classification target."""
    rng = np.random.default_rng(0)
    half = n // 2
    f1 = np.concatenate([rng.normal(-2.0, 0.6, half), rng.normal(2.0, 0.6, half)])
    f2 = np.concatenate([rng.normal(-2.0, 0.6, half), rng.normal(2.0, 0.6, half)])
    group = np.array(["A"] * half + ["B"] * half)
    df = pd.DataFrame({"group": group, "f1": f1.round(5), "f2": f2.round(5)})
    df = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
    csv = tmp_path / "blobs.csv"
    df.to_csv(csv, index=False)
    return csv


def _smooth_regression(tmp_path: Path, n: int = 120) -> Path:
    rng = np.random.default_rng(2)
    x1 = rng.uniform(-3, 3, n)
    x2 = rng.uniform(-3, 3, n)
    y = 2.0 * x1 + x2 + rng.normal(0, 0.3, n)  # y first -> continuous outcome
    df = pd.DataFrame({"y": y.round(5), "x1": x1.round(5), "x2": x2.round(5)})
    csv = tmp_path / "reg.csv"
    df.to_csv(csv, index=False)
    return csv


def test_classification_high_accuracy(tmp_path: Path) -> None:
    # engine convention prefers a continuous outcome (=regression); a categorical
    # label among continuous features must be named to trigger classification.
    fp = profile_dataset(_separable_two_class(tmp_path))
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"outcome": "group"})
    assert "完成" in res.summary
    # separable blobs -> SVC should classify almost perfectly out of sample
    assert res.estimates["cv_accuracy"] > 0.9
    assert res.estimates["cv_f1_macro"] > 0.9
    assert res.estimates["n_classes"] == 2
    assert res.estimates["n_support_vectors"] > 0
    cm = pd.read_csv(Path(res.output_dir) / "svm_confusion_matrix.csv", index_col=0)
    assert int(cm.to_numpy().sum()) == int(res.estimates["n"])
    assert "svm_confusion_matrix.png" in res.files


def test_regression_positive_r2(tmp_path: Path) -> None:
    fp = profile_dataset(_smooth_regression(tmp_path))
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "完成" in res.summary
    assert res.estimates["cv_r2"] > 0.5
    assert res.estimates["cv_rmse"] > 0
    assert res.estimates["n_support_vectors"] > 0
    assert "svm_pred_vs_actual.png" in res.files
    assert "cv_accuracy" not in res.estimates  # regression path, not classification


def test_config_kernel_linear(tmp_path: Path) -> None:
    fp = profile_dataset(_separable_two_class(tmp_path))
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "group", "kernel": "linear", "C": 0.5})
    assert "完成" in res.summary
    assert res.estimates["cv_accuracy"] > 0.9


def test_degrade_too_few_rows(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"group": ["A", "B"] * 5, "f1": rng.normal(0, 1, 10), "f2": rng.normal(0, 1, 10)})
    csv = tmp_path / "tiny.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
    assert "cv_accuracy" not in res.estimates and "cv_r2" not in res.estimates


def test_degrade_no_predictor(tmp_path: Path) -> None:
    df = pd.DataFrame({"group": ["A"] * 20 + ["B"] * 20})
    csv = tmp_path / "onecol.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
