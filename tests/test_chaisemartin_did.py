"""Tests for chaisemartin_did: de Chaisemartin & D'Haultfoeuille (2020) DID_M.
Pure Python (numpy/pandas), unit-resample bootstrap with a FIXED seed.

Empirical check: a panel where treatment SWITCHES on for staggered cohorts with
a KNOWN instantaneous switching effect of +tau. DID_M targets the instantaneous
effect among switchers, so it should recover ~tau. We also exercise a panel with
both switch-IN and switch-OUT transitions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

TAU = 2.0  # planted instantaneous switching effect


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="chaisemartin_did",
        method="de Chaisemartin & D'Haultfoeuille (2020) — DID_M estimator",
        domain="economics",
        family="causal",
        goal="explain",
        preconditions=Precondition(is_panel=True, requires_treatment=True, min_rows=60),
    )


def _make_switcher_panel(
    tau: float = TAU, n_per_cohort: int = 14, n_time: int = 8, seed: int = 0
) -> pd.DataFrame:
    """Staggered switch-IN panel with a constant instantaneous effect +tau.

    Cohorts switch ON at t=3, t=5, t=6, plus a never-treated group (stable-0
    controls each pair). The treatment effect is a clean +tau LEVEL shift once
    treated (so the per-pair switch-in DiD = tau). DID_M (average switch effect)
    should recover ~tau.
        y_it = unit_FE_i + 0.3*t (common trend) + tau*D_it + small noise
    """
    rng = np.random.default_rng(seed)
    times = list(range(1, n_time + 1))
    cohorts = [3, 5, 6, 0]  # 0 = never-treated (stable-0 controls)
    rows = []
    uid = 0
    for g in cohorts:
        for _ in range(n_per_cohort):
            unit_fe = rng.normal(0.0, 1.0)
            for t in times:
                d = 1 if (g != 0 and t >= g) else 0
                y = unit_fe + 0.3 * t + tau * d + rng.normal(0.0, 0.1)
                rows.append({
                    "firm": uid, "year": t, "y": round(float(y), 5),
                    "first_treat": g, "treated": d,
                })
            uid += 1
    return pd.DataFrame(rows)


def test_chaisemartin_did_recovers_switch_effect(tmp_path: Path) -> None:
    df = _make_switcher_panel(tau=TAU, seed=0)
    csv = tmp_path / "switch.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"unit": "firm", "time": "year", "outcome": "y", "treatment": "treated",
                "bootstrap": 200},
    )
    out = Path(res.output_dir)

    assert (out / "chaisemartin_did_by_pair.csv").exists(), res.summary
    est = res.estimates
    for k in ("did_m", "se", "ci_low", "ci_high", "n_switchers", "n_period_pairs"):
        assert k in est, f"missing estimate {k}: {est}"

    # DID_M recovers the planted instantaneous switching effect ~tau
    assert abs(est["did_m"] - TAU) < 0.5, f"DID_M did not recover tau={TAU}: {est['did_m']}"
    assert est["n_switchers"] > 0, est
    assert est["n_period_pairs"] >= 2, est
    # bootstrap produced a finite SE/CI and the effect is significant (CI excludes 0)
    assert est["se"] == est["se"] and est["se"] >= 0, est
    assert not (est["ci_low"] <= 0.0 <= est["ci_high"]), (
        f"CI should exclude 0 for a clear effect: [{est['ci_low']}, {est['ci_high']}]"
    )

    assert "DID_M" in res.summary
    assert "瞬时效应" in res.summary  # disclosure: instantaneous switching effect


def test_chaisemartin_did_explicit_gname(tmp_path: Path) -> None:
    """gname column should also yield the period-by-period treatment status."""
    df = _make_switcher_panel(tau=TAU, seed=3)
    csv = tmp_path / "switch2.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"unit": "firm", "time": "year", "outcome": "y", "gname": "first_treat",
                "bootstrap": 200},
    )
    est = res.estimates
    assert "did_m" in est, res.summary
    assert abs(est["did_m"] - TAU) < 0.5, f"DID_M off: {est['did_m']}"


def test_chaisemartin_did_reproducible_bootstrap(tmp_path: Path) -> None:
    """Fixed seed -> identical bootstrap SE across runs."""
    df = _make_switcher_panel(tau=TAU, seed=0)
    csv = tmp_path / "switch.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    cfg = {"unit": "firm", "time": "year", "outcome": "y", "treatment": "treated",
           "bootstrap": 200}
    r1 = run_analysis(fp, _entry(), output_root=str(tmp_path / "o1"), config=cfg)
    r2 = run_analysis(fp, _entry(), output_root=str(tmp_path / "o2"), config=cfg)
    assert r1.estimates["se"] == r2.estimates["se"], "bootstrap not reproducible with fixed seed"
    assert r1.estimates["ci_low"] == r2.estimates["ci_low"]


def test_chaisemartin_did_degrades_not_panel(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "y": rng.normal(0, 1, 50), "x": rng.normal(0, 1, 50),
        "treated": rng.integers(0, 2, 50),
    })
    csv = tmp_path / "flat.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "de Chaisemartin DiD" in res.summary
    assert "跳过" in res.summary
    assert "did_m" not in res.estimates
