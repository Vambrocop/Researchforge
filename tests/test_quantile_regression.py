"""Tests for quantile_regression: gate + quantile coefficient table across tau."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset
from researchforge.recommender.match import check_preconditions


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="quantile_regression",
        method="Quantile regression",
        domain="statistics",
        family="statistics",
        goal="explain",
        preconditions=Precondition(min_continuous=1, min_rows=30),
    )


def test_quantile_regression_executor(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(0, 1, n)
    # heteroskedastic: noise spread grows with x, so the slope on x should be
    # larger at tau=0.75 than at tau=0.25 — exactly what quantile reg reveals.
    y = 2.0 * x + rng.normal(0, 1, n) * (1.0 + 0.8 * (x - x.min()))
    # outcome must be the first continuous column (engine convention) -> y first
    df = pd.DataFrame({"y": y, "x": x})
    csv = tmp_path / "het.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    coefs = pd.read_csv(out / "coefficients.csv", index_col=0)
    assert list(coefs.columns) == ["tau=0.25", "tau=0.5", "tau=0.75"]
    assert (out / "summary.txt").exists()
    assert "x" in res.estimates
    # upper-tail slope on x exceeds lower-tail slope under this heteroskedasticity
    assert coefs.loc["Q('x')", "tau=0.75"] > coefs.loc["Q('x')", "tau=0.25"]


def test_quantile_regression_precondition_unmet(tmp_path: Path) -> None:
    df = pd.DataFrame({"g": ["a", "b"] * 5})  # no continuous column, too few rows
    csv = tmp_path / "cat.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)

    assert not ok
    assert any("连续" in u or "行" in u for u in unmet)
