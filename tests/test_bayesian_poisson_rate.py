"""Tests for bayesian_poisson_rate — Gamma-Poisson conjugate rate estimation.

Known-value checks: posterior mean rate equals the exact (a0+sum_y)/(b0+sum_exp)
and approaches the empirical rate at large n; the overdispersion flag fires when the
count variance far exceeds the mean (and not for a clean Poisson); the rate-ratio
posterior detects a clearly-higher-rate group. Plus prior override, exposure offset,
and honest degrade.
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
        id="bayesian_poisson_rate",
        method="Bayesian Poisson rate",
        domain="statistics",
        family="bayesian",
        goal="describe",
        preconditions=Precondition(requires_count_outcome=True, min_rows=8),
    )


def test_posterior_mean_matches_formula_and_empirical(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    y = rng.poisson(3.0, 600)
    df = pd.DataFrame({"events": y.astype(int)})
    csv = tmp_path / "c.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "events"})
    out = Path(res.output_dir)

    assert (out / "bayesian_poisson_rate.csv").exists()
    e = res.estimates
    sum_y = float(y.sum())
    # default Gamma(0.001,0.001), exposure = 1 per row (n=600)
    expected = (0.001 + sum_y) / (0.001 + 600.0)
    assert abs(e["post_mean_rate"] - expected) < 1e-9
    # at large n the posterior mean ~ empirical rate; the weakly-informative
    # Gamma(0.001,0.001) prior shifts it by ~Σy·0.001/600² ≈ 3e-6, so compare loosely
    assert abs(e["post_mean_rate"] - e["empirical_rate"]) < 1e-3
    assert abs(e["post_mean_rate"] - 3.0) < 0.25
    assert e["ci_low"] < e["post_mean_rate"] < e["ci_high"]


def test_overdispersion_flag_fires(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    # negative-binomial-like: heavy overdispersion (var >> mean)
    lam = rng.gamma(0.5, 6.0, 400)          # mixing -> overdispersion
    y = rng.poisson(lam)
    df = pd.DataFrame({"events": y.astype(int)})
    csv = tmp_path / "od.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "events"})
    e = res.estimates
    assert e["dispersion"] > 1.5
    assert e["overdispersed"] == 1.0
    assert "过度离散" in res.summary


def test_no_overdispersion_flag_for_clean_poisson(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    y = rng.poisson(4.0, 800)
    df = pd.DataFrame({"events": y.astype(int)})
    csv = tmp_path / "cp.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "events"})
    assert res.estimates["overdispersed"] == 0.0


def test_exposure_offset(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    exposure = rng.integers(1, 5, 300).astype(float)
    y = rng.poisson(2.0 * exposure)         # true rate per unit exposure = 2.0
    df = pd.DataFrame({"events": y.astype(int), "exposure": exposure})
    csv = tmp_path / "ex.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "events", "exposure": "exposure"})
    e = res.estimates
    expected = (0.001 + float(y.sum())) / (0.001 + float(exposure.sum()))
    assert abs(e["post_mean_rate"] - expected) < 1e-9
    assert abs(e["post_mean_rate"] - 2.0) < 0.25   # recovers per-exposure rate


def test_rate_ratio_posterior(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    y_lo = rng.poisson(2.0, 300)
    y_hi = rng.poisson(5.0, 300)            # group 'hi' has clearly higher rate
    df = pd.DataFrame({
        "events": np.concatenate([y_lo, y_hi]).astype(int),
        "grp": (["lo"] * 300) + (["hi"] * 300),
    })
    csv = tmp_path / "rr.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "events", "group": "grp"})
    e = res.estimates
    # ratio = group2/group1 with levels sorted -> ('hi','lo') sorts to hi first,
    # so ratio = lo/hi < 1 (P(ratio>1) ~ 0), OR hi/lo > 1; either way it is decisive.
    assert "rate_ratio_mean" in e
    # decisive separation: P(ratio>1) is near 0 or near 1, not ~0.5
    assert e["prob_ratio_gt1"] < 0.05 or e["prob_ratio_gt1"] > 0.95


def test_prior_override(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    y = rng.poisson(3.0, 30)
    df = pd.DataFrame({"events": y.astype(int)})
    csv = tmp_path / "pr.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "events", "prior_a": 10.0, "prior_b": 2.0})
    e = res.estimates
    expected = (10.0 + float(y.sum())) / (2.0 + 30.0)
    assert abs(e["post_mean_rate"] - expected) < 1e-9


def test_degrade_no_count(tmp_path: Path) -> None:
    df = pd.DataFrame({"x": np.linspace(0.1, 5.0, 20)})  # continuous, not counts
    csv = tmp_path / "cont.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "失败" in res.summary
    assert "post_mean_rate" not in res.estimates
