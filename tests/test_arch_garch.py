"""ARCH-effect diagnostic precision + GARCH surfacing (priority ②).

Dogfood: on a volatility-clustering stock series, GARCH sat at #13 and never surfaced in the
study. Two problems, both fixed here:

1. The `volatility_clustering` diagnostic tested squared LEVEL deviations, so it fired on ANY
   trending / unit-root series (a plain random walk, a trend+noise line) — a false positive.
   It now runs Engle's ARCH LM test on the RETURNS (log-returns for a price series), which is
   zero for iid increments and only fires on genuine conditional heteroskedasticity.
2. GARCH was filed in the `time-series` family, so the study's ≤1-per-family pick shadowed it
   behind ARIMA even when ARCH was confirmed. GARCH models VOLATILITY (a finance/risk tool),
   not the mean — moved to the `finance` family so it competes with VaR/EVT for the finance
   slot, where the ARCH diagnostic lifts it above them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from researchforge.catalog import Catalog
from researchforge.profiler import profile_dataset
from researchforge.profiler.ingest import read_table
from researchforge.recommender import build_plan, select_top
from researchforge.recommender.diagnostics import diagnose_data


def _dates(n):
    return pd.bdate_range("2022-01-03", periods=n).strftime("%Y-%m-%d")


def _random_walk(n=400, seed=5, drift=0.0):
    # geometric random walk → iid, homoskedastic log-returns (NO volatility clustering),
    # with an optional drift so it also covers the "trending but non-ARCH" case.
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"date": _dates(n),
                         "close": (100 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))).round(3)})


def _garch(n=400, seed=5):
    rng = np.random.default_rng(seed)
    omega, a, b = 0.02, 0.08, 0.90
    eps = np.zeros(n)
    s2 = np.zeros(n)
    s2[0] = omega / (1 - a - b)
    for t in range(1, n):
        s2[t] = omega + a * eps[t - 1] ** 2 + b * s2[t - 1]
        eps[t] = np.sqrt(s2[t]) * rng.standard_normal()
    return pd.DataFrame({"date": _dates(n),
                         "close": (100 * np.exp(np.cumsum(eps / 100))).round(3)})


def _codes(df, tmp_path):
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    return {d.code for d in diagnose_data(read_table(fp.path), fp)}


# ── precision: fires only on genuine ARCH, not on trending / random-walk prices ────────
def test_arch_false_positive_rate_low_on_random_walks(tmp_path):
    # het_arch on iid returns has a ~5% Type-I rate by construction; the OLD squared-LEVEL
    # test fired on ~every trending series (≈100%). Assert the fire rate on random walks
    # (flat and drifting) stays low — a systematic false positive would blow past this.
    fires = 0
    trials = 0
    for seed in range(8):
        for drift in (0.0, 0.0006):
            trials += 1
            fires += "volatility_clustering" in _codes(_random_walk(seed=seed, drift=drift), tmp_path)
    assert fires <= trials * 0.25, f"random walks should rarely fire ARCH, got {fires}/{trials}"


def test_arch_fired_on_genuine_garch(tmp_path):
    # a real GARCH process fires decisively (LM p ≈ 1e-9), so this is robust across seeds.
    assert "volatility_clustering" in _codes(_garch(), tmp_path)


# ── surfacing: GARCH appears in the study pick on ARCH data, not on non-financial ─────
def _study_picks(df, tmp_path):
    from researchforge.study import _PICK_SKIP, _diversity_pick

    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    cat = Catalog.load()
    plan = build_plan(fp, read_table(fp.path), cat)
    pool = select_top(fp, top=12, catalog=cat, plan=plan, diagnostic_aware=True)
    sub = [r for r in pool if r.entry.id not in _PICK_SKIP]
    return {r.entry.id for r in _diversity_pick(sub, 3)}


def test_garch_surfaces_on_volatility_clustering(tmp_path):
    assert "garch" in _study_picks(_garch(), tmp_path)


def test_garch_absent_on_nonfinancial_series(tmp_path):
    # a plain sales/temperature series (no finance signal, no ARCH) must NOT headline GARCH.
    rng = np.random.default_rng(3)
    n = 60
    t = np.arange(n)
    df = pd.DataFrame({"month": pd.date_range("2020-01-01", periods=n, freq="MS").strftime("%Y-%m"),
                       "sales": (200 + 2.5 * t + rng.normal(0, 6, n)).round(1)})
    assert "garch" not in _study_picks(df, tmp_path)


def test_garch_is_finance_family():
    assert Catalog.load().by_id("garch").family == "finance"
