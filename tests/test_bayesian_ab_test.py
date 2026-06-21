"""Tests for bayesian_ab_test — Beta-Binomial conjugate A/B test.

Known-value checks: arm B clearly better -> P(B>A) near 1 and the lift credible
interval excludes 0; equal arms -> P(B>A) ~ 0.5; cross-check each Beta posterior
mean against the exact conjugate formula (a+k)/(a+b+n). Plus the counts-config input
mode, prior override, and honest degrade. The closed-form P(B>A) (Beta-inequality
sum) is also cross-checked against fixed-seed Monte-Carlo draws.
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
        id="bayesian_ab_test",
        method="Bayesian A/B test",
        domain="statistics",
        family="bayesian",
        goal="explain",
        preconditions=Precondition(requires_binary_outcome=True, requires_group=True, min_rows=8),
    )


def _ab_frame(p_a: float, p_b: float, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ya = rng.binomial(1, p_a, n)
    yb = rng.binomial(1, p_b, n)
    return pd.DataFrame(
        {
            "converted": np.concatenate([ya, yb]).astype(int),
            "variant": (["A"] * n) + (["B"] * n),
        }
    )


def test_b_clearly_better(tmp_path: Path) -> None:
    df = _ab_frame(p_a=0.20, p_b=0.50, n=400, seed=0)
    csv = tmp_path / "ab.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "converted", "group": "variant"})
    out = Path(res.output_dir)

    assert (out / "bayesian_ab_test.csv").exists()
    e = res.estimates
    assert e["post_mean_b"] > e["post_mean_a"]
    assert e["prob_b_gt_a"] > 0.99           # B almost surely better
    assert e["lift_ci_low"] > 0.0            # 95% lift CI excludes 0 (B - A)
    assert e["lift_mean"] > 0.0
    # decision-theoretic: choosing B has near-zero expected loss
    assert e["expected_loss_b"] < e["expected_loss_a"]


def test_equal_arms_prob_near_half(tmp_path: Path) -> None:
    # Deterministically SYMMETRIC arms (identical successes/trials) -> P(B>A) is exactly
    # 0.5 by symmetry, regardless of sampling luck. (Random equal-rate draws can differ
    # several pp at n=500 and then legitimately push P(B>A) far from 0.5 — e.g. seed 1
    # gave 144 vs 169 -> P=0.96, which is correct, just not "equal arms".)
    n, k = 500, 150
    df = pd.DataFrame({
        "converted": ([1] * k + [0] * (n - k)) * 2,
        "variant": ["A"] * n + ["B"] * n,
    })
    csv = tmp_path / "ab.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "converted", "group": "variant"})
    e = res.estimates
    assert abs(e["prob_b_gt_a"] - 0.5) < 0.02   # symmetric -> exact coin flip
    assert e["lift_ci_low"] < 0.0 < e["lift_ci_high"]  # lift CI brackets 0


def test_posterior_mean_matches_conjugate_formula(tmp_path: Path) -> None:
    """Cross-check Beta posterior mean against the exact (a+k)/(a+b+n)."""
    # counts-config mode -> exact, deterministic k and n
    res = run_analysis(
        _dummy_fp(tmp_path), _entry(), output_root=str(tmp_path / "o"),
        config={"successes_a": 30, "trials_a": 100,
                "successes_b": 55, "trials_b": 100,
                "prior_a": 1.0, "prior_b": 1.0},
    )
    e = res.estimates
    # arm A: Beta(1+30, 1+70) -> mean = 31/102 ; arm B: Beta(1+55, 1+45) -> 56/102
    assert abs(e["post_mean_a"] - 31.0 / 102.0) < 1e-12
    assert abs(e["post_mean_b"] - 56.0 / 102.0) < 1e-12
    # P(B>A) should be very high here and computed by closed form (integer alpha)
    assert e["prob_b_gt_a"] > 0.999


def test_prob_b_gt_a_closed_form_matches_draws(tmp_path: Path) -> None:
    """The closed-form Beta-inequality sum should match Monte-Carlo within noise."""
    res = run_analysis(
        _dummy_fp(tmp_path), _entry(), output_root=str(tmp_path / "o"),
        config={"successes_a": 40, "trials_a": 100,
                "successes_b": 52, "trials_b": 100},
    )
    closed = res.estimates["prob_b_gt_a"]

    # independent draws cross-check
    a1, b1 = 1 + 40, 1 + 60
    a2, b2 = 1 + 52, 1 + 48
    rng = np.random.default_rng(123)
    xa = rng.beta(a1, b1, 200_000)
    xb = rng.beta(a2, b2, 200_000)
    mc = float(np.mean(xb > xa))
    assert abs(closed - mc) < 0.01


def test_prior_override_shifts_posterior(tmp_path: Path) -> None:
    """A strong prior favouring A pulls its posterior mean up vs the uniform prior."""
    base = run_analysis(
        _dummy_fp(tmp_path), _entry(), output_root=str(tmp_path / "o1"),
        config={"successes_a": 2, "trials_a": 10, "successes_b": 5, "trials_b": 10},
    )
    strong = run_analysis(
        _dummy_fp(tmp_path), _entry(), output_root=str(tmp_path / "o2"),
        config={"successes_a": 2, "trials_a": 10, "successes_b": 5, "trials_b": 10,
                "prior_a": 50.0, "prior_b": 5.0},
    )
    # strong Beta(50,5) prior (mean ~0.91) pulls arm A's posterior mean up
    assert strong.estimates["post_mean_a"] > base.estimates["post_mean_a"]
    # exact: Beta(50+2, 5+8) mean = 52/65
    assert abs(strong.estimates["post_mean_a"] - 52.0 / 65.0) < 1e-12


def test_degrade_no_binary(tmp_path: Path) -> None:
    df = pd.DataFrame({"x": np.arange(20.0), "g": (["a", "b"] * 10)})
    csv = tmp_path / "cont.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "失败" in res.summary
    assert "post_mean_a" not in res.estimates


def _dummy_fp(tmp_path: Path):
    """A throwaway fingerprint — counts-config mode reads only cfg, not the frame."""
    df = pd.DataFrame({"converted": ([0, 1] * 6), "variant": (["A", "B"] * 6)})
    csv = tmp_path / "dummy.csv"
    df.to_csv(csv, index=False)
    return profile_dataset(csv)
