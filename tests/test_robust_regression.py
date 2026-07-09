"""Tests for robust_regression: Huber M-estimator + Theil-Sen.

Known-structure check: with Y-outliers injected, the robust slope stays near the
clean (true) slope while OLS is pulled off -> |robust - truth| < |OLS - truth|.
Plus Theil-Sen for the 1-predictor case, multi-predictor, config override, and
degrade.
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
        id="robust_regression",
        method="Robust regression (Huber M + Theil-Sen)",
        domain="statistics",
        family="nonparametric",
        goal="explain",
        preconditions=Precondition(min_continuous=2, min_rows=12),
    )


def test_robust_beats_ols_with_outliers(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 200
    true_slope = 2.0
    x = rng.normal(0, 1, n)
    y = true_slope * x + rng.normal(0, 0.5, n)
    # inject Y-direction outliers: blow up the response for 8% of points
    k = int(0.08 * n)
    idx = rng.choice(n, size=k, replace=False)
    y[idx] += rng.normal(40, 5, k)  # large positive Y shifts
    df = pd.DataFrame({"y": y, "x": x})  # y first -> outcome
    csv = tmp_path / "out.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    comp = pd.read_csv(out / "robust_vs_ols.csv")
    row = comp[comp["term"] == "x"].iloc[0]
    robust_b = float(row["robust_coef"])
    ols_b = float(row["ols_coef"])

    # robust slope should stay closer to the truth than OLS
    assert abs(robust_b - true_slope) < abs(ols_b - true_slope)
    # robust should be close to truth; OLS clearly pulled off
    assert abs(robust_b - true_slope) < 0.4
    assert "theilsen_slope" in comp.columns
    assert abs(float(row["theilsen_slope"]) - true_slope) < 0.4
    assert "x" in res.estimates


def test_robust_theilsen_breakdown(tmp_path: Path) -> None:
    """Theil-Sen (high breakdown) survives a large fraction of Y-outliers."""
    rng = np.random.default_rng(1)
    n = 300
    true_slope = -1.5
    x = rng.normal(0, 1, n)
    y = true_slope * x + rng.normal(0, 0.4, n)
    k = int(0.20 * n)  # 20% outliers, below Theil-Sen's ~29% breakdown
    idx = rng.choice(n, size=k, replace=False)
    y[idx] += 50.0
    df = pd.DataFrame({"y": y, "x": x})
    csv = tmp_path / "ts.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)
    comp = pd.read_csv(out / "robust_vs_ols.csv")
    ts = float(res.estimates["theilsen_slope"])
    ols_b = float(comp[comp["term"] == "x"].iloc[0]["ols_coef"])
    # Theil-Sen recovers the sign+magnitude; OLS does not
    assert abs(ts - true_slope) < abs(ols_b - true_slope)
    assert abs(ts - true_slope) < 0.4
    assert (out / "robust_vs_ols.png").exists()


def test_robust_multi_predictor(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    n = 150
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = 1.0 * x1 - 2.0 * x2 + rng.normal(0, 0.5, n)
    # outliers in Y
    idx = rng.choice(n, size=12, replace=False)
    y[idx] += 30.0
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    csv = tmp_path / "multi.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)
    comp = pd.read_csv(out / "robust_vs_ols.csv")
    # both predictors present; robust closer to truth on x2 than OLS
    r2 = comp[comp["term"] == "x2"].iloc[0]
    assert abs(float(r2["robust_coef"]) - (-2.0)) < abs(float(r2["ols_coef"]) - (-2.0))
    # multi-predictor case has no theilsen column (only 1-predictor case)
    assert "theilsen_slope" not in res.estimates


def test_robust_config_override(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    n = 120
    x = rng.normal(0, 1, n)
    target = 3.0 * x + rng.normal(0, 0.5, n)
    other = rng.normal(0, 1, n)
    df = pd.DataFrame({"other": other, "target": target, "x": x})
    csv = tmp_path / "ovr.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "target", "predictors": ["x"]})
    out = Path(res.output_dir)
    comp = pd.read_csv(out / "robust_vs_ols.csv")
    row = comp[comp["term"] == "x"].iloc[0]
    assert abs(float(row["robust_coef"]) - 3.0) < 0.4
    assert "theilsen_slope" in comp.columns  # single chosen predictor


def test_robust_resolver_picks_high_confidence_outcome_not_first(tmp_path: Path) -> None:
    """A high-confidence-named outcome ('target') placed AFTER a decoy continuous
    column must still be resolved as the outcome (shared resolve_outcome, not raw
    cont_cols[0])."""
    rng = np.random.default_rng(9)
    n = 150
    decoy = rng.normal(0, 1, n)  # first continuous column, unrelated noise
    x = rng.normal(0, 1, n)
    target = 3.0 * x + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"decoy": decoy, "target": target, "x": x})
    csv = tmp_path / "resolver.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    assert fp.likely_outcome == "target" and fp.likely_outcome_confidence == "high"
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)
    comp = pd.read_csv(out / "robust_vs_ols.csv")
    row = comp[comp["term"] == "x"].iloc[0]
    # only true if 'target' (linear in x) was modeled, not 'decoy'
    assert abs(float(row["robust_coef"]) - 3.0) < 0.4


def test_robust_degrade_no_predictor(tmp_path: Path) -> None:
    # single continuous column -> no predictor -> honest failure
    df = pd.DataFrame({"y": np.arange(30, dtype=float)})
    csv = tmp_path / "one.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "失败" in res.summary
