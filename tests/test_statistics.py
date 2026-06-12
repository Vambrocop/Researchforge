"""Tests for the logistic_regression analysis — catalog, matcher, executor."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog import Catalog
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset
from researchforge.recommender import recommend


# ---------------------------------------------------------------------------
# 1. Catalog loads the entry
# ---------------------------------------------------------------------------

def test_catalog_loads_logistic_regression():
    entry = Catalog.load().by_id("logistic_regression")
    assert entry is not None
    assert entry.preconditions.requires_binary_outcome is True
    assert entry.preconditions.min_rows == 30
    assert entry.executor_ref == "empirical-analysis-python"


# ---------------------------------------------------------------------------
# 2. Recommender: feasible/infeasible based on binary column presence
# ---------------------------------------------------------------------------

def _make_binary_csv(tmp_path: Path, n: int = 40) -> Path:
    rng = np.random.default_rng(42)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    p = 1 / (1 + np.exp(-(0.3 + 0.8 * x1 - 0.5 * x2)))
    outcome = rng.binomial(1, p).astype(float)
    df = pd.DataFrame({"outcome": outcome, "x1": x1, "x2": x2})
    csv = tmp_path / "binary_data.csv"
    df.to_csv(csv, index=False)
    return csv


def _make_no_binary_csv(tmp_path: Path, n: int = 40) -> Path:
    rng = np.random.default_rng(7)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    csv = tmp_path / "no_binary.csv"
    df.to_csv(csv, index=False)
    return csv


def test_logistic_feasible_with_binary_column(tmp_path):
    csv = _make_binary_csv(tmp_path)
    fp = profile_dataset(csv)
    by_id = {r.entry.id: r for r in recommend(fp)}
    assert "logistic_regression" in by_id
    assert by_id["logistic_regression"].feasible


def test_logistic_infeasible_without_binary_column(tmp_path):
    csv = _make_no_binary_csv(tmp_path)
    fp = profile_dataset(csv)
    by_id = {r.entry.id: r for r in recommend(fp)}
    assert "logistic_regression" in by_id
    assert not by_id["logistic_regression"].feasible
    assert "需要二值结果变量" in by_id["logistic_regression"].rigor.unmet


# ---------------------------------------------------------------------------
# 3. Executor: run logistic_regression and check outputs
# ---------------------------------------------------------------------------

def _make_logistic_csv(tmp_path: Path, n: int = 80) -> Path:
    rng = np.random.default_rng(99)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    p = 1 / (1 + np.exp(-(0.5 + 1.2 * x1 - 0.8 * x2)))
    outcome = rng.binomial(1, p).astype(int)
    df = pd.DataFrame({"outcome": outcome, "x1": x1, "x2": x2})
    csv = tmp_path / "logistic_data.csv"
    df.to_csv(csv, index=False)
    return csv


def test_executor_logistic_regression_outputs(tmp_path):
    csv = _make_logistic_csv(tmp_path)
    fp = profile_dataset(csv)
    entry = Catalog.load().by_id("logistic_regression")
    assert entry is not None

    res = run_analysis(fp, entry, output_root=str(tmp_path / "outputs"))
    out = Path(res.output_dir)

    assert out.exists()
    assert (out / "report.md").exists()
    assert (out / "coefficients.csv").exists()
    assert res.summary
