"""Tests for random_forest — catalog + executor (regression-first, classification,
and the target-selection guard that the double-review surfaced)."""

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog import Catalog
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _rf_entry():
    return Catalog.load().by_id("random_forest")


def test_catalog_loads_random_forest():
    entry = _rf_entry()
    assert entry is not None
    assert entry.goal == "predict"
    assert entry.family == "ml"
    assert entry.preconditions.min_rows == 50


def _make_regression_csv(tmp_path: Path, n: int = 120) -> Path:
    rng = np.random.default_rng(42)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    y = 2.0 * x1 - 1.0 * x2 + 0.5 * x3 + rng.normal(0, 0.5, n)
    csv = tmp_path / "regression.csv"
    pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3}).to_csv(csv, index=False)
    return csv


def test_executor_random_forest_regression(tmp_path):
    fp = profile_dataset(_make_regression_csv(tmp_path))
    assert not any(c.kind == "binary" for c in fp.columns)

    res = run_analysis(fp, _rf_entry(), output_root=str(tmp_path / "outputs"))
    out = Path(res.output_dir)

    assert (out / "feature_importances.csv").exists()
    assert (out / "report.md").exists()
    assert "test_score" in res.estimates
    assert "回归" in res.summary
    assert res.estimates["test_score"] > 0.5  # the model actually learned, not just wrote a file


def _make_classification_csv(tmp_path: Path, n: int = 140) -> Path:
    rng = np.random.default_rng(7)
    c1 = rng.integers(0, 6, n)
    c2 = rng.integers(0, 6, n)
    p = 1 / (1 + np.exp(-((c1 - c2) / 2.0)))
    label = rng.binomial(1, p).astype(int)
    csv = tmp_path / "classification.csv"
    pd.DataFrame({"label": label, "c1": c1, "c2": c2}).to_csv(csv, index=False)
    return csv


def test_executor_random_forest_classification(tmp_path):
    fp = profile_dataset(_make_classification_csv(tmp_path))
    # binary outcome present and NO continuous column -> classification path
    assert any(c.kind == "binary" for c in fp.columns)
    assert not any(c.kind == "continuous" for c in fp.columns)

    res = run_analysis(fp, _rf_entry(), output_root=str(tmp_path / "outputs"))

    assert (Path(res.output_dir) / "feature_importances.csv").exists()
    assert "test_score" in res.estimates
    assert "分类" in res.summary


def test_rf_prefers_continuous_outcome_over_binary_feature(tmp_path):
    # {y continuous, treated binary, x1 continuous}: must REGRESS on y, not classify
    # the treatment flag. Guards the silent-misfire the double-review found.
    rng = np.random.default_rng(3)
    n = 120
    treated = rng.integers(0, 2, n)
    x1 = rng.normal(0, 1, n)
    y = 1.0 + 2.0 * x1 + 0.8 * treated + rng.normal(0, 0.5, n)
    csv = tmp_path / "mixed.csv"
    pd.DataFrame({"y": y, "treated": treated, "x1": x1}).to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _rf_entry(), output_root=str(tmp_path / "outputs"))

    assert "回归" in res.summary  # regression on y, not classification of treated
    feats = set(pd.read_csv(Path(res.output_dir) / "feature_importances.csv")["feature"])
    assert "y" not in feats  # outcome excluded from features
    assert "treated" in feats  # the binary is used as a feature
