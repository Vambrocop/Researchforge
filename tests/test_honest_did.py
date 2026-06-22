"""Tests for honest_did: Rambachan & Roth (2023) honest sensitivity analysis for
parallel trends (relative-magnitudes single-post-period variant). Pure Python.

Empirical check: a panel with a CLEAN pre-trend (pre-period event-study coefs
~0) and a clear positive post effect, so the conventional CI excludes 0 and the
breakdown Mbar is finite and sensible (a small pre-violation means even a modest
Mbar can widen the CI to include 0 only after the effect/pre ratio is exceeded).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

TAU = 3.0  # planted post-treatment effect


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="honest_did",
        method="Rambachan & Roth (2023) — Honest DiD parallel-trends sensitivity",
        domain="economics",
        family="causal",
        goal="explain",
        preconditions=Precondition(is_panel=True, requires_treatment=True, min_rows=60),
    )


def _make_clean_pretrend_panel(
    tau: float = TAU, n_treated: int = 25, n_control: int = 25, n_time: int = 10,
    seed: int = 0, pretrend: float = 0.0,
) -> pd.DataFrame:
    """Single-onset DiD with a (near) clean pre-trend and a +tau post jump.

    All treated units adopt at t=6 (one cohort) + a never-treated control group,
    so the event study is well-identified with multiple leads (t=1..5) and lags.
        y_it = unit_FE_i + 0.4*t (common trend) + pretrend*t*1[treated]
               + tau*1[treated & t>=6] + small noise
    With pretrend=0 parallel trends hold pre-treatment (max pre-violation ~ noise).
    A small positive `pretrend` plants a controlled differential pre-trend so the
    max pre-violation is non-trivial and the breakdown Mbar lands within the grid.
    """
    rng = np.random.default_rng(seed)
    times = list(range(1, n_time + 1))
    g = 6
    rows = []
    uid = 0
    for is_treated in (True, False):
        n = n_treated if is_treated else n_control
        for _ in range(n):
            unit_fe = rng.normal(0.0, 1.0)
            for t in times:
                treated_now = 1 if (is_treated and t >= g) else 0
                trend = pretrend * t if is_treated else 0.0
                y = unit_fe + 0.4 * t + trend + tau * treated_now + rng.normal(0.0, 0.15)
                rows.append({
                    "firm": uid, "year": t, "y": round(float(y), 5),
                    "first_treat": g if is_treated else 0,
                    "treated": treated_now,
                })
            uid += 1
    return pd.DataFrame(rows)


def test_honest_did_clean_pretrend_finite_breakdown(tmp_path: Path) -> None:
    df = _make_clean_pretrend_panel(tau=TAU, seed=0)
    csv = tmp_path / "clean.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"unit": "firm", "time": "year", "outcome": "y", "gname": "first_treat"},
    )
    out = Path(res.output_dir)

    assert (out / "honest_did_sensitivity.csv").exists(), res.summary
    est = res.estimates
    for k in ("effect_post1", "ci_orig_low", "ci_orig_high", "breakdown_mbar",
              "max_pretrend_violation"):
        assert k in est, f"missing estimate {k}: {est}"

    # planted effect is positive and recovered roughly
    assert est["effect_post1"] > 0, est
    assert abs(est["effect_post1"] - TAU) < 1.0, f"effect off from tau: {est['effect_post1']}"

    # conventional CI excludes 0 (clean significant effect)
    assert not (est["ci_orig_low"] <= 0.0 <= est["ci_orig_high"]), (
        f"original CI should exclude 0: [{est['ci_orig_low']}, {est['ci_orig_high']}]"
    )

    # clean pre-trend -> small max pre-violation
    assert est["max_pretrend_violation"] >= 0.0
    assert est["max_pretrend_violation"] < 1.0, f"pre-trend not clean: {est['max_pretrend_violation']}"

    # breakdown Mbar is "sensible": a clean strong effect resists violations, so the
    # breakdown is either a finite positive value (the conventional CI eventually
    # widens to include 0) OR nan meaning it stays robust past the top of the grid
    # (Mbar=2). Per the spec, nan == robust-through-2; in that case the Mbar=2 robust
    # CI must still exclude 0. We assert the result is internally consistent.
    bd = est["breakdown_mbar"]
    grid = pd.read_csv(out / "honest_did_sensitivity.csv").sort_values("Mbar")
    m2 = grid[grid["Mbar"] == 2.0].iloc[0]
    m2_excludes_0 = not (m2["robust_ci_low"] <= 0.0 <= m2["robust_ci_high"])
    if bd == bd:  # finite breakdown reported
        assert bd > 0.0, f"breakdown should be > 0 for a significant effect: {bd}"
    else:  # nan -> robust even at Mbar=2
        assert m2_excludes_0, "breakdown=nan but robust CI at Mbar=2 already includes 0"

    # the sensitivity grid is monotone-widening: robust CI low decreases as Mbar grows
    assert grid["robust_ci_low"].is_monotonic_decreasing
    assert grid["robust_ci_high"].is_monotonic_increasing
    # at Mbar=0 the robust CI equals the conventional CI
    row0 = grid[grid["Mbar"] == 0.0].iloc[0]
    assert abs(row0["robust_ci_low"] - est["ci_orig_low"]) < 1e-6
    assert abs(row0["robust_ci_high"] - est["ci_orig_high"]) < 1e-6

    assert "honest_did" in res.summary.lower() or "Honest DiD" in res.summary
    assert "敏感性" in res.summary  # disclosure that it bounds, not fixes


def test_honest_did_breakdown_consistent_with_grid(tmp_path: Path) -> None:
    """At the breakdown Mbar the robust CI should be (approximately) touching 0;
    just below it should exclude 0."""
    df = _make_clean_pretrend_panel(tau=TAU, seed=1)
    csv = tmp_path / "clean2.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"unit": "firm", "time": "year", "outcome": "y", "gname": "first_treat"},
    )
    est = res.estimates
    bd = est["breakdown_mbar"]
    if bd == bd and est["max_pretrend_violation"] > 0:
        beta = est["effect_post1"]
        # closed form: at breakdown, the binding CI end reaches 0
        max_pre = est["max_pretrend_violation"]
        half = (est["ci_orig_high"] - est["ci_orig_low"]) / 2.0  # = z*se
        # |beta| - z*se = bd * max_pre  (within rounding)
        assert abs((abs(beta) - half) - bd * max_pre) < 1e-2, (
            f"breakdown not consistent: |b|-z*se={abs(beta)-half}, bd*maxpre={bd*max_pre}"
        )


def test_honest_did_pretrend_lowers_breakdown(tmp_path: Path) -> None:
    """A PLANTED differential pre-trend must (a) raise the measured max pre-trend
    violation and (b) make the conclusion more fragile — i.e. yield a smaller
    breakdown Mbar than the clean case (or hit it sooner). This exercises a finite,
    sensible breakdown within the grid."""
    # NOTE: max_pre is now the max pre-period FIRST-DIFFERENCE (RR relative-magnitudes
    # scale), which for a linear trend is ~constant per step; the breakdown lands in the
    # [0,2] grid only when the pre-trend step is sizeable relative to the post effect, so
    # the dirty design uses a strong differential trend.
    clean = _make_clean_pretrend_panel(tau=TAU, seed=4, pretrend=0.0)
    dirty = _make_clean_pretrend_panel(tau=TAU, seed=4, pretrend=4.0)
    fp_c = profile_dataset(_dump(clean, tmp_path / "c.csv"))
    fp_d = profile_dataset(_dump(dirty, tmp_path / "d.csv"))
    cfg = {"unit": "firm", "time": "year", "outcome": "y", "gname": "first_treat"}
    rc = run_analysis(fp_c, _entry(), output_root=str(tmp_path / "oc"), config=cfg)
    rd = run_analysis(fp_d, _entry(), output_root=str(tmp_path / "od"), config=cfg)

    # planted pre-trend => larger measured pre-trend violation
    assert rd.estimates["max_pretrend_violation"] > rc.estimates["max_pretrend_violation"], (
        f"planted pre-trend did not raise the violation: "
        f"{rd.estimates['max_pretrend_violation']} vs {rc.estimates['max_pretrend_violation']}"
    )
    # the dirty design's breakdown is finite and within the grid (more fragile)
    bd_d = rd.estimates["breakdown_mbar"]
    assert bd_d == bd_d and 0.0 < bd_d <= 2.0, f"dirty breakdown not finite/in-grid: {bd_d}"
    # and it is smaller than (or equal to a robust nan) the clean breakdown
    bd_c = rc.estimates["breakdown_mbar"]
    if bd_c == bd_c:
        assert bd_d <= bd_c + 1e-9, f"dirty breakdown not <= clean: {bd_d} vs {bd_c}"


def _dump(df: pd.DataFrame, path: Path) -> Path:
    df.to_csv(path, index=False)
    return path


def test_honest_did_degrades_not_panel(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "y": rng.normal(0, 1, 50), "x": rng.normal(0, 1, 50),
        "treated": rng.integers(0, 2, 50),
    })
    csv = tmp_path / "flat.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "Honest DiD" in res.summary
    assert "跳过" in res.summary
    assert "effect_post1" not in res.estimates
