"""Tests for heteroskedasticity_test: Breusch-Pagan + White.

Known-structure check: with error SD proportional to x (var grows with x), both
Breusch-Pagan and White should reject (small p). With homoskedastic errors they
should NOT reject. Plus config override and the no-predictor degrade.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="heteroskedasticity_test",
        method="Heteroskedasticity tests (Breusch-Pagan + White)",
        domain="statistics",
        family="regression",
        goal="explain",
        preconditions=Precondition(min_continuous=1, min_numeric_cols=2, min_rows=12),
    )


def test_het_detects_heteroskedasticity(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 400
    x = rng.uniform(1, 5, n)
    # error SD proportional to x -> non-constant variance
    err = rng.normal(0, 1, n) * x
    y = 2.0 * x + err  # y first -> outcome
    df = pd.DataFrame({"y": y, "x": x})
    csv = tmp_path / "het.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    tab = pd.read_csv(out / "heteroskedasticity_tests.csv")
    assert set(tab["test"]) == {"Breusch-Pagan", "White"}
    # both tests reject homoskedasticity
    assert res.estimates["bp_p"] < 0.05
    assert res.estimates["white_p"] < 0.05
    assert res.estimates["bp_lm_stat"] > 0
    assert res.estimates["n"] == float(n)
    assert (out / "residuals_vs_fitted.png").exists()


def test_het_homoskedastic_not_rejected(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n = 400
    x = rng.uniform(1, 5, n)
    y = 2.0 * x + rng.normal(0, 1.0, n)  # constant variance
    df = pd.DataFrame({"y": y, "x": x})
    csv = tmp_path / "homo.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    # homoskedastic data -> BP should not reject at 0.05
    assert res.estimates["bp_p"] > 0.05


def test_het_config_override(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    n = 300
    x = rng.uniform(1, 5, n)
    target = 2.0 * x + rng.normal(0, 1, n) * x  # heteroskedastic
    df = pd.DataFrame({"other": rng.normal(0, 1, n), "target": target, "x": x})
    csv = tmp_path / "ovr.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"outcome": "target", "predictors": ["x"]},
    )
    assert res.estimates["bp_p"] < 0.05
    assert res.estimates["n_predictors"] == 1.0


def test_het_degrade_no_predictor(tmp_path: Path) -> None:
    # single continuous column -> no predictor -> honest skip
    df = pd.DataFrame({"y": np.arange(30, dtype=float)})
    csv = tmp_path / "one.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
