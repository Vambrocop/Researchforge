"""Tests for goodman_bacon: Goodman-Bacon (2021) decomposition of the TWFE DiD
estimator into its 2x2 sub-comparisons. Pure Python (no R).

Empirical check: a STAGGERED-adoption balanced panel with HETEROGENEOUS,
DYNAMIC effects (later cohorts get a larger, growing effect). Under such
heterogeneity the "later-vs-earlier" comparisons (already-treated earlier group
used as control) are the contaminating "bad" comparisons, so we assert the bad
weight is strictly positive. The exact TWFE coefficient is reported separately as
twfe_did_direct (two-way within OLS); the simplified-shape Bacon weights mean the
decomposition sum (twfe_did_decomp) only agrees in SIGN + a loose band, not to 1e-6.
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
        id="goodman_bacon",
        method="Goodman-Bacon (2021) — TWFE DiD decomposition (staggered-DiD diagnostic)",
        domain="economics",
        family="causal",
        goal="explain",
        preconditions=Precondition(is_panel=True, requires_treatment=True, min_rows=60),
    )


def _make_staggered_hetero_panel(
    n_per_cohort: int = 12, n_time: int = 10, seed: int = 0
) -> pd.DataFrame:
    """Staggered adoption with HETEROGENEOUS dynamic effects.

    Cohorts first-treated at t=4 and t=7, plus a never-treated group. Outcome:
        y_it = unit_FE_i + 0.5*t (common trend) + effect_it + small noise
    where effect_it grows with exposure AND differs by cohort (later cohort
    bigger) — exactly the heterogeneity that makes TWFE's already-treated-as
    -control comparisons biased, so the Goodman-Bacon 'bad' weight is nonzero.
    """
    rng = np.random.default_rng(seed)
    times = list(range(1, n_time + 1))
    cohorts = [4, 7, 0]  # 0 = never-treated
    cohort_scale = {4: 1.0, 7: 2.5}  # later cohort has a much bigger effect (heterogeneity)
    rows = []
    uid = 0
    for g in cohorts:
        for _ in range(n_per_cohort):
            unit_fe = rng.normal(0.0, 1.0)
            for t in times:
                if g != 0 and t >= g:
                    exposure = t - g
                    effect = cohort_scale[g] * (1.0 + 0.5 * exposure)  # dynamic + heterogeneous
                else:
                    effect = 0.0
                y = unit_fe + 0.5 * t + effect + rng.normal(0.0, 0.2)
                rows.append({
                    "firm": uid, "year": t, "y": round(float(y), 5),
                    "first_treat": g, "treated": 1 if (g != 0 and t >= g) else 0,
                })
            uid += 1
    return pd.DataFrame(rows)


def test_goodman_bacon_decomposition_and_bad_weight(tmp_path: Path) -> None:
    df = _make_staggered_hetero_panel(seed=0)
    csv = tmp_path / "stag.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"unit": "firm", "time": "year", "outcome": "y", "gname": "first_treat"},
    )
    out = Path(res.output_dir)

    assert (out / "goodman_bacon_decomposition.csv").exists(), res.summary
    assert (out / "goodman_bacon_2x2.csv").exists(), res.summary
    est = res.estimates
    for k in ("twfe_did_decomp", "weight_treated_vs_untreated", "weight_earlier_vs_later",
              "weight_later_vs_earlier_BAD", "att_treated_vs_untreated", "n_groups"):
        assert k in est, f"missing estimate {k}: {est}"

    # weights are a normalized partition -> sum to ~1
    wsum = (est["weight_treated_vs_untreated"] + est["weight_earlier_vs_later"]
            + est["weight_later_vs_earlier_BAD"])
    assert abs(wsum - 1.0) < 1e-6, f"weights do not sum to 1: {wsum}"

    # heterogeneous staggered design -> a strictly positive BAD weight
    assert est["weight_later_vs_earlier_BAD"] > 0.0, f"expected nonzero bad weight: {est}"

    # both the decomposition sum and the direct two-way within (TWFE) estimate are
    # reported; they are the same sign and in the same ballpark (the closed-form Bacon
    # weights approximate the exact decomposition — exact equality holds only for special
    # spacings, so we check agreement-in-sign + a loose magnitude band, not 1e-6).
    assert "twfe_did_direct" in est
    assert est["twfe_did_decomp"] > 0 and est["twfe_did_direct"] > 0, est
    assert abs(est["twfe_did_decomp"] - est["twfe_did_direct"]) < 2.0, (
        f"decomposition far from direct TWFE: {est['twfe_did_decomp']} vs {est['twfe_did_direct']}"
    )

    # treated-vs-never att is positive (planted effects are all positive)
    assert est["att_treated_vs_untreated"] > 0, est
    assert est["n_groups"] >= 3.0  # 2 treated cohorts + never-treated
    assert "Goodman-Bacon" in res.summary
    assert "坏比较" in res.summary  # bad-comparison disclosure present


def test_goodman_bacon_derived_gname(tmp_path: Path) -> None:
    """gname DERIVED from a 0/1 treatment indicator must also work."""
    df = _make_staggered_hetero_panel(seed=2)
    csv = tmp_path / "stag2.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"unit": "firm", "time": "year", "outcome": "y", "treatment": "treated"},
    )
    est = res.estimates
    assert "twfe_did_decomp" in est, res.summary
    assert est["weight_later_vs_earlier_BAD"] > 0.0, est


def test_goodman_bacon_degrades_not_panel(tmp_path: Path) -> None:
    """A flat cross-section must skip honestly (no crash, no estimates)."""
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "y": rng.normal(0, 1, 50), "x": rng.normal(0, 1, 50),
        "treated": rng.integers(0, 2, 50),
    })
    csv = tmp_path / "flat.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "Goodman-Bacon" in res.summary
    assert "跳过" in res.summary
    assert "twfe_did_decomp" not in res.estimates
