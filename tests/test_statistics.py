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


# ---------------------------------------------------------------------------
# 4. group_comparison — catalog, matcher, executor
# ---------------------------------------------------------------------------

def test_catalog_loads_group_comparison():
    entry = Catalog.load().by_id("group_comparison")
    assert entry is not None
    assert entry.preconditions.requires_group is True
    assert entry.preconditions.min_continuous == 1
    assert entry.preconditions.min_rows == 10
    assert entry.executor_ref == "empirical-analysis-python"


def _make_group_csv(tmp_path: Path, n: int = 60, n_groups: int = 2) -> Path:
    rng = np.random.default_rng(123)
    labels = [chr(65 + i) for i in range(n_groups)]  # "A", "B", ...
    grp = [labels[i % n_groups] for i in range(n)]
    # make groups differ so test is significant
    value = [rng.normal(i * 2.0, 1.0) for i, g in enumerate(grp) for _ in [None]]
    value = [rng.normal((ord(g) - 65) * 2.0, 1.0) for g in grp]
    df = pd.DataFrame({"grp": grp, "value": value})
    csv = tmp_path / f"group_{n_groups}.csv"
    df.to_csv(csv, index=False)
    return csv


def _make_all_continuous_csv(tmp_path: Path, n: int = 40) -> Path:
    rng = np.random.default_rng(77)
    df = pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(0, 1, n),
        "x3": rng.normal(0, 1, n),
    })
    csv = tmp_path / "all_continuous.csv"
    df.to_csv(csv, index=False)
    return csv


def test_group_comparison_feasible_with_group_column(tmp_path):
    csv = _make_group_csv(tmp_path, n_groups=2)
    fp = profile_dataset(csv)
    by_id = {r.entry.id: r for r in recommend(fp)}
    assert "group_comparison" in by_id
    assert by_id["group_comparison"].feasible


def test_group_comparison_infeasible_without_group_column(tmp_path):
    csv = _make_all_continuous_csv(tmp_path)
    fp = profile_dataset(csv)
    by_id = {r.entry.id: r for r in recommend(fp)}
    assert "group_comparison" in by_id
    assert not by_id["group_comparison"].feasible
    assert "需要分组变量（分类/二值）" in by_id["group_comparison"].rigor.unmet


def test_executor_group_comparison_two_groups(tmp_path):
    csv = _make_group_csv(tmp_path, n_groups=2)
    fp = profile_dataset(csv)
    # 2-level string column -> kind "binary"
    assert any(c.kind == "binary" for c in fp.columns), \
        f"Expected binary kind for 2-level group; got {[(c.name, c.kind) for c in fp.columns]}"

    entry = Catalog.load().by_id("group_comparison")
    assert entry is not None

    res = run_analysis(fp, entry, output_root=str(tmp_path / "outputs"))
    out = Path(res.output_dir)

    assert out.exists()
    assert (out / "report.md").exists()
    assert (out / "group_means.csv").exists()
    assert "pvalue" in res.estimates
    assert res.summary
