"""Tests for xgboost executor branch (regression-first, classification).
Entry is constructed inline because xgboost is not in the catalog yaml."""

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog import AnalysisEntry
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _xgb_entry():
    return AnalysisEntry(
        id="xgboost",
        method="XGBoost (prediction)",
        domain="ml",
        family="ml",
        goal="predict",
        preconditions={"min_rows": 50},
    )


def _make_regression_csv(tmp_path: Path, n: int = 120) -> Path:
    rng = np.random.default_rng(42)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    y = 2.0 * x1 - 1.0 * x2 + 0.5 * x3 + rng.normal(0, 0.5, n)
    csv = tmp_path / "regression.csv"
    pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3}).to_csv(csv, index=False)
    return csv


def test_executor_xgboost_regression(tmp_path):
    fp = profile_dataset(_make_regression_csv(tmp_path))
    # No binary columns — regression path only
    assert not any(c.kind == "binary" for c in fp.columns)

    res = run_analysis(fp, _xgb_entry(), output_root=str(tmp_path / "outputs"))
    out = Path(res.output_dir)

    assert (out / "feature_importances.csv").exists()
    assert (out / "report.md").exists()
    assert "test_score" in res.estimates
    assert res.estimates["test_score"] > 0.5
    assert "回归" in res.summary


def _make_classification_csv(tmp_path: Path, n: int = 140) -> Path:
    rng = np.random.default_rng(7)
    c1 = rng.integers(0, 6, n)
    c2 = rng.integers(0, 6, n)
    p = 1 / (1 + np.exp(-((c1 - c2) / 2.0)))
    label = rng.binomial(1, p).astype(int)
    csv = tmp_path / "classification.csv"
    pd.DataFrame({"label": label, "c1": c1, "c2": c2}).to_csv(csv, index=False)
    return csv


def test_executor_xgboost_classification(tmp_path):
    fp = profile_dataset(_make_classification_csv(tmp_path))
    # binary outcome present and NO continuous column -> classification path
    assert any(c.kind == "binary" for c in fp.columns)
    assert not any(c.kind == "continuous" for c in fp.columns)

    res = run_analysis(fp, _xgb_entry(), output_root=str(tmp_path / "outputs"))
    out = Path(res.output_dir)

    assert (out / "feature_importances.csv").exists()
    assert "test_score" in res.estimates
    assert "分类" in res.summary
