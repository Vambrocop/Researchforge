"""Tests for bootstrap_ci: BCa (bias-corrected & accelerated) bootstrap CI.

Known-structure checks: BCa CI on a Normal sample brackets the true mean; the
hand-rolled BCa (z0, acceleration, adjusted percentiles) is cross-checked against
scipy.stats.bootstrap(method='BCa') when available. Plus statistic/config
overrides (median/std/correlation), reproducibility, and degrade.
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
        id="bootstrap_ci",
        method="Bootstrap CI (BCa)",
        domain="statistics",
        family="nonparametric",
        goal="describe",
        preconditions=Precondition(min_continuous=1, min_rows=8),
    )


def test_bootstrap_mean_ci_brackets_truth(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(10.0, 2.0, 200)  # true mean = 10
    df = pd.DataFrame({"x": x})
    csv = tmp_path / "norm.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"statistic": "mean", "n_boot": 2000, "seed": 0})
    out = Path(res.output_dir)

    assert (out / "bootstrap_ci.csv").exists()
    lo, hi = res.estimates["ci_low"], res.estimates["ci_high"]
    assert lo < 10.0 < hi  # 95% CI brackets the true mean
    assert lo < res.estimates["estimate"] < hi
    assert res.estimates["ci_level"] == 0.95


def test_bootstrap_bca_matches_scipy(tmp_path: Path) -> None:
    """Cross-check the hand-rolled BCa against scipy.stats.bootstrap if available."""
    try:
        from scipy.stats import bootstrap as _scipy_bootstrap
    except Exception:  # pragma: no cover - scipy too old
        import pytest
        pytest.skip("scipy.stats.bootstrap unavailable")

    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 3.0, 120)
    df = pd.DataFrame({"x": x})
    csv = tmp_path / "x.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"statistic": "mean", "n_boot": 5000, "seed": 0})
    ours_lo, ours_hi = res.estimates["ci_low"], res.estimates["ci_high"]

    # scipy BCa on the SAME data; both use Monte-Carlo resampling with their own
    # RNGs, so the endpoints agree only up to resampling noise -> loose tolerance.
    sci = _scipy_bootstrap(
        (x,), np.mean, method="BCa", confidence_level=0.95,
        n_resamples=5000, random_state=np.random.default_rng(0),
    )
    sci_lo = float(sci.confidence_interval.low)
    sci_hi = float(sci.confidence_interval.high)

    # endpoints should match within Monte-Carlo tolerance (data sd ~3, n=120)
    assert abs(ours_lo - sci_lo) < 0.15
    assert abs(ours_hi - sci_hi) < 0.15


def test_bootstrap_bca_z0_acceleration_finite(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    # skewed data so acceleration is meaningfully non-zero
    x = rng.exponential(2.0, 150)
    df = pd.DataFrame({"x": x})
    csv = tmp_path / "exp.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"statistic": "mean", "n_boot": 3000, "seed": 0})
    assert np.isfinite(res.estimates["z0"])
    assert np.isfinite(res.estimates["acceleration"])
    # for right-skewed data the acceleration is nonzero
    assert abs(res.estimates["acceleration"]) > 1e-4


def test_bootstrap_median_override(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(0.0, 1.0, 100)
    df = pd.DataFrame({"x": x})
    csv = tmp_path / "med.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"statistic": "median", "n_boot": 1500, "seed": 0})
    lo, hi = res.estimates["ci_low"], res.estimates["ci_high"]
    assert lo < res.estimates["estimate"] < hi
    assert lo < 0.0 < hi  # true median is 0


def test_bootstrap_correlation_override(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    n = 200
    a = rng.normal(0, 1, n)
    b = 0.8 * a + rng.normal(0, 0.6, n)  # strong positive correlation
    df = pd.DataFrame({"a": a, "b": b})
    csv = tmp_path / "corr.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"statistic": "correlation", "column": "a",
                               "column2": "b", "n_boot": 2000, "seed": 0})
    lo, hi = res.estimates["ci_low"], res.estimates["ci_high"]
    assert 0.0 < lo < hi < 1.0          # positive correlation, CI within (0,1)
    assert lo < res.estimates["estimate"] < hi


def test_bootstrap_ci_level_override(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(0, 1, 150)
    df = pd.DataFrame({"x": x})
    csv = tmp_path / "lvl.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)

    r95 = run_analysis(fp, _entry(), output_root=str(tmp_path / "o95"),
                       config={"ci": 0.95, "n_boot": 2000, "seed": 0})
    r80 = run_analysis(fp, _entry(), output_root=str(tmp_path / "o80"),
                       config={"ci": 0.80, "n_boot": 2000, "seed": 0})
    w95 = r95.estimates["ci_high"] - r95.estimates["ci_low"]
    w80 = r80.estimates["ci_high"] - r80.estimates["ci_low"]
    assert r80.estimates["ci_level"] == 0.80
    assert w80 < w95  # narrower interval at lower confidence


def test_bootstrap_reproducible(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, 100)
    df = pd.DataFrame({"x": x})
    csv = tmp_path / "rep.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)

    r1 = run_analysis(fp, _entry(), output_root=str(tmp_path / "o1"),
                      config={"n_boot": 1000, "seed": 9})
    r2 = run_analysis(fp, _entry(), output_root=str(tmp_path / "o2"),
                      config={"n_boot": 1000, "seed": 9})
    assert r1.estimates["ci_low"] == r2.estimates["ci_low"]
    assert r1.estimates["ci_high"] == r2.estimates["ci_high"]


def test_bootstrap_degrade_no_continuous(tmp_path: Path) -> None:
    df = pd.DataFrame({"g": ["a", "b", "c", "d"] * 3})
    csv = tmp_path / "cat.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "失败" in res.summary
    assert "ci_low" not in res.estimates
