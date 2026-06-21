"""Tests for zero_inflated_poisson: known excess-zero structure + degrade.

Synthetic data = a mixture of structural zeros (a fraction that can never count)
and a Poisson count process whose log-rate rises with x1. ZIP should fit,
recover a positive x1 count-coefficient, and beat plain Poisson by AIC.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _make_entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="zero_inflated_poisson",
        method="Zero-inflated Poisson (excess-zero counts)",
        domain="statistics",
        family="statistics",
        goal="explain",
        preconditions=Precondition(requires_count_outcome=True, min_rows=30),
    )


def _excess_zero_counts(seed: int = 0, n: int = 600) -> pd.DataFrame:
    """Structural-zero mixture + Poisson(exp(0.2 + 0.6 x1 - 0.3 x2))."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    lam = np.exp(0.2 + 0.6 * x1 - 0.3 * x2)
    counts = rng.poisson(lam)
    # 45% of observations are structural zeros (always 0 regardless of lam)
    structural = rng.random(n) < 0.45
    counts = np.where(structural, 0, counts).astype(int)
    return pd.DataFrame({"events": counts, "x1": x1, "x2": x2})


def test_zip_fits_and_beats_poisson(tmp_path: Path) -> None:
    df = _excess_zero_counts(seed=1)
    csv = tmp_path / "zip.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    assert any(c.kind == "count" for c in fp.columns), (
        f"'events' not detected as count: {[(c.name, c.kind) for c in fp.columns]}"
    )

    res = run_analysis(fp, _make_entry(), output_root=str(tmp_path / "out"))
    out = Path(res.output_dir)

    assert "失败" not in res.summary, f"ZIP reported failure: {res.summary}"
    assert (out / "summary.txt").exists()
    assert (out / "count_coefficients.csv").exists()
    assert (out / "inflation_coefficients.csv").exists()
    assert (out / "report.md").exists()

    # count-coefficient sign recovery (true x1 effect is +0.6)
    assert "x1" in res.estimates, f"x1 missing from estimates: {res.estimates}"
    assert res.estimates["x1"] > 0, f"x1 count-coef should be positive: {res.estimates['x1']}"

    # ZIP should beat plain Poisson by AIC on excess-zero data
    assert "aic_zip" in res.estimates and "aic_poisson" in res.estimates
    assert res.estimates["aic_zip"] < res.estimates["aic_poisson"], (
        f"ZIP AIC {res.estimates['aic_zip']:.1f} should beat Poisson "
        f"{res.estimates['aic_poisson']:.1f}"
    )

    # observed zeros should exceed the naive Poisson-expected zero fraction
    assert res.estimates["pct_zeros_observed"] > res.estimates["pct_zeros_poisson_expected"]


def test_zip_disclosure_present(tmp_path: Path) -> None:
    df = _excess_zero_counts(seed=2)
    csv = tmp_path / "zip2.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _make_entry(), output_root=str(tmp_path / "out"))
    assert "⚠" in res.summary
    assert "ZINB" in res.summary  # points to the overdispersion alternative


def test_zip_config_inflation_override(tmp_path: Path) -> None:
    """config['inflation'] adds a separate inflation covariate (still fits)."""
    df = _excess_zero_counts(seed=3)
    csv = tmp_path / "zip3.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp,
        _make_entry(),
        output_root=str(tmp_path / "out"),
        config={"outcome": "events", "predictors": ["x1"], "inflation": ["x2"]},
    )
    assert "失败" not in res.summary, res.summary
    assert "x1" in res.estimates


def test_zip_degrades_without_count(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    n = 80
    df = pd.DataFrame({"y": rng.normal(5, 1, n), "x1": rng.normal(0, 1, n)})
    csv = tmp_path / "nocount.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _make_entry(), output_root=str(tmp_path / "out"))
    assert "未找到计数型结果变量" in res.summary


def test_zip_degrades_too_few_rows(tmp_path: Path) -> None:
    rng = np.random.default_rng(10)
    n = 12
    df = pd.DataFrame(
        {"events": rng.poisson(2, n).astype(int), "x1": rng.normal(0, 1, n)}
    )
    csv = tmp_path / "fewrows.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _make_entry(), output_root=str(tmp_path / "out"))
    assert "失败" in res.summary
    assert "行数不足" in res.summary
