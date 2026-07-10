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


def test_xgb_config_outcome_wins_over_continuous_tier(tmp_path):
    # Wave K F1: {y continuous, treated binary, x1 continuous} but config explicitly
    # asks for the binary "treated" as outcome. Before the fix the tier decision
    # (cont_cols present -> regress on y) ran BEFORE the config check, so the
    # config-specified outcome was silently ignored. Must now classify "treated".
    rng = np.random.default_rng(3)
    n = 120
    treated = rng.integers(0, 2, n)
    x1 = rng.normal(0, 1, n)
    y = 1.0 + 2.0 * x1 + 0.8 * treated + rng.normal(0, 0.5, n)
    csv = tmp_path / "mixed_config.csv"
    pd.DataFrame({"y": y, "treated": treated, "x1": x1}).to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _xgb_entry(), output_root=str(tmp_path / "outputs"), config={"outcome": "treated"}
    )

    assert "分类" in res.summary  # classification on the config-forced "treated"
    assert "treated" in res.summary
    feats = set(pd.read_csv(Path(res.output_dir) / "feature_importances.csv")["feature"])
    assert "treated" not in feats  # config-forced outcome excluded from features
    assert "y" in feats  # y is now just a feature


def test_xgb_no_config_behavior_unchanged(tmp_path):
    # Same mixed shape, no config override -> must still regress on the continuous
    # y (unchanged auto behavior), guarding against the fix over-firing.
    rng = np.random.default_rng(3)
    n = 120
    treated = rng.integers(0, 2, n)
    x1 = rng.normal(0, 1, n)
    y = 1.0 + 2.0 * x1 + 0.8 * treated + rng.normal(0, 0.5, n)
    csv = tmp_path / "mixed_noconfig.csv"
    pd.DataFrame({"y": y, "treated": treated, "x1": x1}).to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _xgb_entry(), output_root=str(tmp_path / "outputs"))

    assert "回归" in res.summary
    feats = set(pd.read_csv(Path(res.output_dir) / "feature_importances.csv")["feature"])
    assert "y" not in feats
    assert "treated" in feats
