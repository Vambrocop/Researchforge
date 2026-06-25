"""Tests for the MISSING-DATA family: mice_imputation, missingness_diagnosis.

Harness mirrors tests/test_techno_economic.py: write CSV -> profile_dataset ->
AnalysisEntry/Precondition -> run_analysis -> assert res.estimates / res.summary.
A fixed seed keeps the data (and the imputations) reproducible.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry(eid: str, method: str, goal: str = "explain") -> AnalysisEntry:
    return AnalysisEntry(
        id=eid,
        method=method,
        domain="statistics",
        family="missing_data",
        goal=goal,
        preconditions=Precondition(min_rows=1),
    )


def _csv(tmp_path: Path, name: str, df: pd.DataFrame) -> Path:
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


# --------------------------------------------------------------------------- #
# 1) mice_imputation — recovers the truth + FMI>0 + listwise drops rows
# --------------------------------------------------------------------------- #
def test_mice_recovers_truth_with_mar_missingness(tmp_path: Path) -> None:
    """y = 2.0 + 3.0*x1 - 1.5*x2 + noise. Inject MAR missingness into x1
    (missing more often when x2 is large). MICE-pooled coefficients should recover
    the TRUE slopes within tolerance; FMI on x1 should be > 0; listwise deletion
    should drop rows so n_complete_case < n_total."""
    rng = np.random.RandomState(42)
    n = 400
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = 2.0 + 3.0 * x1 - 1.5 * x2 + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    # MAR: probability x1 is missing increases with x2 (depends on OBSERVED x2)
    p_miss = 1.0 / (1.0 + np.exp(-(x2 - 0.3)))
    miss_mask = rng.uniform(size=n) < (0.45 * p_miss)
    df.loc[miss_mask, "x1"] = np.nan
    assert df["x1"].isna().sum() > 0  # sanity: we actually injected missingness

    csv = _csv(tmp_path, "mar.csv", df)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry("mice_imputation", "MICE"),
        output_root=str(tmp_path / "o"),
        config={"outcome": "y", "predictors": ["x1", "x2"],
                "m_imputations": 10, "seed": 7},
    )
    e = res.estimates
    # pooled coefficients recover the truth within tolerance
    assert math.isclose(e["coef__x1"], 3.0, abs_tol=0.4), e
    assert math.isclose(e["coef__x2"], -1.5, abs_tol=0.4), e
    # FMI on the column we made missing is strictly positive
    assert e["fmi__x1"] > 0.0, e
    # listwise deletion dropped rows
    assert e["n_complete_case"] < e["n_total"], e
    assert e["n_imputations"] >= 2
    # products exist
    out = Path(res.output_dir)
    assert (out / "mice_coefficients.csv").exists()
    coef = pd.read_csv(out / "mice_coefficients.csv")
    assert {"parameter", "pooled_coef", "pooled_se", "fmi",
            "complete_case_coef"}.issubset(coef.columns)
    assert "多重插补" in res.summary


def test_mice_degrade_on_complete_data(tmp_path: Path) -> None:
    """No missing values -> honest 跳过 (points the user to ols_regression)."""
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "y": rng.normal(0, 1, 50),
        "x1": rng.normal(0, 1, 50),
        "x2": rng.normal(0, 1, 50),
    })
    csv = _csv(tmp_path, "complete.csv", df)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry("mice_imputation", "MICE"),
        output_root=str(tmp_path / "o"),
        config={"outcome": "y", "predictors": ["x1", "x2"]},
    )
    assert "跳过" in res.summary
    assert "无缺失值" in res.summary
    assert "coef__x1" not in res.estimates


def test_mice_degrade_too_few_numeric(tmp_path: Path) -> None:
    """Only one numeric column -> cannot form outcome + predictor -> honest 跳过."""
    df = pd.DataFrame({"label": ["a", "b", "c", "d"], "y": [1.0, np.nan, 3.0, 4.0]})
    csv = _csv(tmp_path, "one.csv", df)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry("mice_imputation", "MICE"),
        output_root=str(tmp_path / "o"),
    )
    assert "跳过" in res.summary
    assert "coef__y" not in res.estimates


# --------------------------------------------------------------------------- #
# 2) missingness_diagnosis — patterns + correct rate + MCAR p in [0,1]
# --------------------------------------------------------------------------- #
def test_missingness_diagnosis_pattern_and_rate(tmp_path: Path) -> None:
    """Two distinct missingness patterns by construction; overall missing rate is
    computable exactly; MCAR screen p-value is a valid probability in [0, 1]."""
    rng = np.random.RandomState(123)
    n = 200
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    # pattern A: x2 missing where x1 is large (MAR-ish association -> MCAR screen has signal)
    mask2 = x1 > 0.6
    df.loc[mask2, "x2"] = np.nan
    # pattern B: x3 missing for a separate fixed block of rows
    df.loc[:9, "x3"] = np.nan

    n_missing_cells = int(df.isna().sum().sum())
    expected_rate = n_missing_cells / (n * 3)

    csv = _csv(tmp_path, "patterns.csv", df)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry("missingness_diagnosis", "Missingness diagnosis", goal="describe"),
        output_root=str(tmp_path / "o"),
    )
    e = res.estimates
    assert e["n_patterns"] >= 2, e
    assert math.isclose(e["overall_missing_rate"], expected_rate, abs_tol=1e-4), e
    assert e["n_cols_with_missing"] == 2.0, e
    # MCAR screen p-value is a valid probability (may be NaN only if infeasible; here it is feasible)
    assert 0.0 <= e["mcar_p"] <= 1.0, e
    # products exist
    out = Path(res.output_dir)
    assert (out / "missingness_by_column.csv").exists()
    assert (out / "missingness_patterns.csv").exists()
    pat = pd.read_csv(out / "missingness_patterns.csv")
    assert {"pattern_id", "n_rows", "frequency", "missing_columns"}.issubset(pat.columns)
    assert "缺失诊断" in res.summary


def test_missingness_diagnosis_no_missing(tmp_path: Path) -> None:
    """Complete data -> overall rate 0, single pattern, honest summary (no crash)."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [5.0, 4.0, 3.0, 2.0, 1.0]})
    csv = _csv(tmp_path, "full.csv", df)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry("missingness_diagnosis", "Missingness diagnosis", goal="describe"),
        output_root=str(tmp_path / "o"),
    )
    e = res.estimates
    assert e["overall_missing_rate"] == 0.0
    assert e["n_cols_with_missing"] == 0.0
    assert "无缺失" in res.summary
