"""Tests for bayesian_proportion — single-proportion Beta-Binomial estimation.

Known-value checks: posterior mean equals the exact (a+k)/(a+b+n); the 95% credible
interval brackets the true proportion; P(theta>0.5) is sensible (near 1 when the
true rate is high). Plus the Jeffreys prior override, HPD interval, ref override,
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
        id="bayesian_proportion",
        method="Bayesian proportion",
        domain="statistics",
        family="bayesian",
        goal="describe",
        preconditions=Precondition(requires_binary_outcome=True, min_rows=8),
    )


def _binary_frame(p: float, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"success": rng.binomial(1, p, n).astype(int)})


def test_posterior_mean_matches_conjugate_formula(tmp_path: Path) -> None:
    df = _binary_frame(p=0.7, n=200, seed=0)
    k = int(df["success"].sum())
    n = len(df)
    csv = tmp_path / "p.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "success"})
    out = Path(res.output_dir)

    assert (out / "bayesian_proportion.csv").exists()
    e = res.estimates
    # uniform prior Beta(1,1): posterior mean = (1+k)/(2+n)
    assert abs(e["post_mean"] - (1.0 + k) / (2.0 + n)) < 1e-12
    assert e["successes"] == k
    assert e["trials"] == n


def test_credible_interval_brackets_truth(tmp_path: Path) -> None:
    df = _binary_frame(p=0.7, n=400, seed=1)
    csv = tmp_path / "p.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "success"})
    e = res.estimates
    assert e["ci_low"] < 0.7 < e["ci_high"]       # 95% CrI brackets true 0.7
    assert e["ci_low"] < e["post_mean"] < e["ci_high"]
    assert e["prob_gt_ref"] > 0.99                # P(theta > 0.5) ~ 1 for true 0.7
    assert e["ref"] == 0.5


def test_jeffreys_prior_override(tmp_path: Path) -> None:
    df = _binary_frame(p=0.4, n=50, seed=2)
    k = int(df["success"].sum())
    n = len(df)
    csv = tmp_path / "p.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "success", "prior": "jeffreys"})
    e = res.estimates
    # Jeffreys Beta(0.5,0.5): posterior mean = (0.5+k)/(1+n)
    assert abs(e["post_mean"] - (0.5 + k) / (1.0 + n)) < 1e-12
    assert "Jeffreys" in res.summary


def test_hpd_interval_narrower_or_equal(tmp_path: Path) -> None:
    """For a skewed posterior the HPD interval is no wider than the equal-tail one."""
    df = _binary_frame(p=0.05, n=120, seed=3)  # skewed posterior near 0
    csv = tmp_path / "p.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    eq = run_analysis(fp, _entry(), output_root=str(tmp_path / "o1"),
                      config={"outcome": "success", "interval": "equal_tail"})
    hpd = run_analysis(fp, _entry(), output_root=str(tmp_path / "o2"),
                       config={"outcome": "success", "interval": "hpd"})
    w_eq = eq.estimates["ci_high"] - eq.estimates["ci_low"]
    w_hpd = hpd.estimates["ci_high"] - hpd.estimates["ci_low"]
    assert w_hpd <= w_eq + 1e-6


def test_ref_override(tmp_path: Path) -> None:
    # p=0.85, n=400 lands the proportion clearly above ref=0.75 across seeds (SE≈0.018,
    # so ~5σ margin) -> P(θ>0.75)≈1, robustly. (The earlier p=0.8/n=300/seed=4 unluckily
    # sampled 0.743 < 0.75, which correctly gives P<0.5 — a data artifact, not a bug:
    # the handler's prob_gt_ref = beta.sf(ref) direction is verified correct.)
    df = _binary_frame(p=0.85, n=400, seed=4)
    csv = tmp_path / "p.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "success", "ref": 0.75})
    e = res.estimates
    assert e["ref"] == 0.75
    assert 0.0 <= e["prob_gt_ref"] <= 1.0
    assert e["prob_gt_ref"] > 0.5   # proportion ~0.85 >> 0.75


def test_degrade_no_binary(tmp_path: Path) -> None:
    df = pd.DataFrame({"x": np.arange(20.0)})
    csv = tmp_path / "cont.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "失败" in res.summary
    assert "post_mean" not in res.estimates
