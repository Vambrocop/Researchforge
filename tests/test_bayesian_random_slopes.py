"""Tests for bayesian_random_slopes (PyMC NUTS) + the PPC wiring on the bayesian family.

A varying-intercept AND varying-slope multilevel model: each group j gets its own
slope b_j = mu_b + N(0, tau) around a clear nonzero population slope mu_b, plus its own
intercept. The test simulates this KNOWN structure and asserts parameter recovery
(population slope HDI excludes 0 / recovers sign, real slope variation slope_sd > 0,
intercept-slope correlation reported), convergence (max R-hat < 1.1), the per-group CSV
artifact, and that the posterior predictive check populated ppc_bayes_p_mean. A degrade
test monkeypatches PyMC away to confirm the honest skip. It also confirms the NEW PPC
capability now fires for bayesian_regression too.

Slow (real MCMC fits) — conftest SLOW_MODULES should add "test_bayesian_random_slopes".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

pytest.importorskip("pymc")
pytest.importorskip("arviz")

# small sampler for speed; generous tolerances downstream
_FAST = {"draws": 300, "tune": 300, "chains": 2, "seed": 0}


def _entry(eid: str, method: str) -> AnalysisEntry:
    return AnalysisEntry(id=eid, method=method, domain="statistics", family="bayesian",
                         goal="explain", preconditions=Precondition(min_rows=10))


def _run(tmp_path, name, df, eid, method, config):
    p = tmp_path / name
    df.to_csv(p, index=False)
    fp = profile_dataset(p)
    return run_analysis(fp, _entry(eid, method), output_root=str(tmp_path / "o"),
                        config={**_FAST, **config})


def _make_random_slopes_df(seed: int = 7) -> pd.DataFrame:
    """Multilevel data with KNOWN varying slopes: b_j = mu_b + N(0, tau), mu_b = 4.0,
    tau = 1.5 (real slope variation), plus varying intercepts a_j = 50 + N(0, 5)."""
    rng = np.random.default_rng(seed)
    n_groups = 10
    mu_b, tau = 4.0, 1.5
    a = 50.0 + rng.normal(0, 5.0, n_groups)      # varying intercepts
    b = mu_b + rng.normal(0, tau, n_groups)      # varying slopes around mu_b
    rows = []
    for g in range(n_groups):
        for _ in range(rng.integers(12, 20)):
            x = rng.normal(0, 1.0)
            y = a[g] + b[g] * x + rng.normal(0, 2.0)
            rows.append({"grp": f"g{g}", "x": x, "y": y})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 1) random slopes — population slope + slope variation + correlation + PPC
# --------------------------------------------------------------------------- #
def test_random_slopes_recovers_population_slope(tmp_path: Path):
    df = _make_random_slopes_df()
    res = _run(tmp_path, "rs.csv", df, "bayesian_random_slopes",
               "Bayesian random-slopes multilevel model",
               {"group": "grp", "outcome": "y", "predictor": "x"})
    e = res.estimates

    # headline: population slope (std-x). True mu_b = 4.0; x has SD ~1, so std-x slope ~4.
    assert "population_slope_std" in e
    # HDI excludes 0 OR at least recovers the positive sign within tolerance
    excludes0 = e["population_slope_hdi_low"] > 0
    assert excludes0 or e["population_slope_std"] > 1.0
    assert e["population_slope_hdi_low"] <= e["population_slope_hdi_high"]   # HDI ordering

    # real between-group slope variation was injected (tau = 1.5) -> slope_sd > 0
    assert e["slope_sd"] > 0.0
    assert e["intercept_sd"] > 0.0
    assert "intercept_slope_corr" in e
    assert -1.0 <= e["intercept_slope_corr"] <= 1.0

    # raw-x back-transform present and finite
    assert "population_slope_raw" in e and e["population_slope_raw"] == e["population_slope_raw"]

    # convergence honest + numeric
    assert isinstance(e["max_rhat"], float) and e["max_rhat"] == e["max_rhat"]
    assert e["max_rhat"] < 1.1
    assert e["n_groups"] == 10.0
    assert 0.0 < e["icc"] < 1.0

    # per-group artifact with both intercept and slope columns
    grp_csv = Path(res.output_dir) / "bayesian_random_slopes_groups.csv"
    assert grp_csv.exists()
    tbl = pd.read_csv(grp_csv)
    assert {"group", "intercept", "slope_std", "slope_raw"}.issubset(tbl.columns)
    assert len(tbl) == 10

    # posterior predictive check fired
    assert "ppc_bayes_p_mean" in e
    assert 0.0 <= e["ppc_bayes_p_mean"] <= 1.0
    assert "ppc_bayes_p_sd" in e

    assert "随机斜率" in res.summary
    assert "HDI" in res.summary


def test_random_slopes_resolver_picks_named_outcome_not_first(tmp_path: Path):
    """A decoy continuous column ('noise_col', no group/slope structure) is placed
    BEFORE 'y' — the shared resolver must still pick 'y' as outcome. 'predictor' is
    pinned via config to isolate the (untouched) predictor pick from this outcome
    resolution check."""
    rng = np.random.default_rng(17)
    n_groups = 10
    mu_b, tau = 4.0, 1.5
    a = 50.0 + rng.normal(0, 5.0, n_groups)
    b = mu_b + rng.normal(0, tau, n_groups)
    rows = []
    for g in range(n_groups):
        for _ in range(rng.integers(12, 20)):
            x = rng.normal(0, 1.0)
            rows.append({
                "noise_col": rng.normal(0, 1.0),
                "grp": f"g{g}", "x": x,
                "y": a[g] + b[g] * x + rng.normal(0, 2.0),
            })
    df = pd.DataFrame(rows)
    res = _run(tmp_path, "rs_resolver.csv", df, "bayesian_random_slopes",
               "Bayesian random-slopes multilevel model",
               {"group": "grp", "predictor": "x"})  # no "outcome" in config
    e = res.estimates
    # real slope structure (mu_b=4.0, tau=1.5) only lives in y~x; a wrong (positional)
    # pick of noise_col as outcome would show ~0 slope and an HDI including 0.
    assert e["slope_sd"] > 0.0
    excludes0 = e["population_slope_hdi_low"] > 0
    assert excludes0 or e["population_slope_std"] > 1.0


# --------------------------------------------------------------------------- #
# 2) honest skip when only ONE continuous column (no slope covariate)
# --------------------------------------------------------------------------- #
def test_random_slopes_skips_without_predictor(tmp_path: Path):
    rng = np.random.default_rng(3)
    rows = []
    for g in range(5):
        for _ in range(10):
            rows.append({"grp": f"g{g}", "y": 10.0 + g + rng.normal(0, 1.0)})
    df = pd.DataFrame(rows)
    res = _run(tmp_path, "noslope.csv", df, "bayesian_random_slopes",
               "Bayesian random-slopes multilevel model",
               {"group": "grp", "outcome": "y"})
    assert "跳过" in res.summary
    assert "bayesian_hierarchical" in res.summary
    assert not res.estimates  # nothing fabricated


# --------------------------------------------------------------------------- #
# 3) honest degrade when PyMC is unavailable
# --------------------------------------------------------------------------- #
def test_random_slopes_degrade_without_pymc(tmp_path: Path, monkeypatch):
    import researchforge.executor.branches.bayesian_mcmc as bm

    monkeypatch.setattr(bm, "_have_pymc", lambda: False)
    df = _make_random_slopes_df()
    res = _run(tmp_path, "deg.csv", df, "bayesian_random_slopes",
               "Bayesian random-slopes multilevel model",
               {"group": "grp", "outcome": "y", "predictor": "x"})
    assert "跳过" in res.summary
    assert "pymc" in res.summary.lower()
    assert not res.estimates  # nothing fabricated


# --------------------------------------------------------------------------- #
# 4) the NEW PPC capability now fires for bayesian_regression too
# --------------------------------------------------------------------------- #
def test_bayesian_regression_emits_ppc(tmp_path: Path):
    rng = np.random.default_rng(11)
    n = 120
    x1 = rng.normal(size=n)
    y = 2.0 + 3.0 * x1 + rng.normal(0, 1.0, n)
    df = pd.DataFrame({"x1": x1, "y": y})
    res = _run(tmp_path, "reg.csv", df, "bayesian_regression",
               "Bayesian linear regression", {"outcome": "y", "predictors": ["x1"]})
    e = res.estimates
    assert "ppc_bayes_p_mean" in e
    assert 0.0 <= e["ppc_bayes_p_mean"] <= 1.0
    assert "ppc_bayes_p_sd" in e
