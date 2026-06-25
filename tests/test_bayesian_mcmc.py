"""Full Bayesian regression family (PyMC NUTS) — bayesian_mcmc.py.

Modern PyMC needs no Stan/JAGS compiler, so these run on a bare Python. Tests use
small data, few draws, and a fixed seed: they assert PARAMETER RECOVERY (posterior
mean near the known truth / HDI brackets it) and CONVERGENCE (max R-hat < 1.1), not
exact values. A degrade test monkeypatches PyMC away to confirm the honest skip.
These are slow (MCMC fits) — tagged in conftest SLOW_MODULES.
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

_FAST = {"draws": 500, "tune": 500, "chains": 2, "seed": 0}


def _entry(eid: str, method: str) -> AnalysisEntry:
    return AnalysisEntry(id=eid, method=method, domain="statistics", family="bayesian",
                         goal="explain", preconditions=Precondition(min_rows=10))


def _run(tmp_path, name, df, eid, method, config):
    p = tmp_path / name
    df.to_csv(p, index=False)
    fp = profile_dataset(p)
    return run_analysis(fp, _entry(eid, method), output_root=str(tmp_path / "o"),
                        config={**_FAST, **config})


# --------------------------------------------------------------------------- #
# 1) bayesian_regression — coefficient recovery + convergence
# --------------------------------------------------------------------------- #
def test_bayesian_regression_recovers_coefficients(tmp_path: Path):
    rng = np.random.default_rng(1)
    n = 120
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 2.0 + 3.0 * x1 - 1.5 * x2 + rng.normal(0, 1.0, n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    res = _run(tmp_path, "reg.csv", df, "bayesian_regression", "Bayesian linear regression",
               {"outcome": "y", "predictors": ["x1", "x2"]})
    e = res.estimates
    assert abs(e["beta__x1"] - 3.0) < 0.5
    assert abs(e["beta__x2"] - (-1.5)) < 0.5
    # the strong x1 effect's HDI should exclude 0
    assert e["beta__x1__hdi_low"] > 0
    assert e["beta__x1__hdi_low"] <= e["beta__x1__hdi_high"]   # HDI ordering (back-transform sane)
    assert isinstance(e["max_rhat"], float) and e["max_rhat"] == e["max_rhat"]  # numeric, not NaN/string
    assert e["max_rhat"] < 1.1
    assert 0.7 < e["bayes_r2"] <= 1.0
    assert "贝叶斯" in res.summary and "HDI" in res.summary


# --------------------------------------------------------------------------- #
# 2) bayesian_logistic_regression — odds-ratio direction + convergence
# --------------------------------------------------------------------------- #
def test_bayesian_logistic_recovers_direction(tmp_path: Path):
    rng = np.random.default_rng(2)
    n = 200
    x = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-(0.3 + 1.5 * x)))
    y = (rng.random(n) < p).astype(int)
    df = pd.DataFrame({"x": x, "y": y})
    res = _run(tmp_path, "logit.csv", df, "bayesian_logistic_regression",
               "Bayesian logistic regression", {"outcome": "y", "predictors": ["x"]})
    e = res.estimates
    assert e["or__x"] > 1.0            # positive effect -> OR > 1
    assert e["or__x__hdi_low"] > 1.0   # clearly positive: HDI excludes 1
    assert e["or__x__hdi_low"] <= e["or__x__hdi_high"]  # HDI ordering
    assert e["max_rhat"] < 1.1
    assert "OR" in res.summary or "优势比" in res.summary


# --------------------------------------------------------------------------- #
# 3) bayesian_hierarchical — variance components + partial pooling
# --------------------------------------------------------------------------- #
def test_bayesian_hierarchical_partial_pooling(tmp_path: Path):
    rng = np.random.default_rng(3)
    n_groups = 6
    offsets = rng.normal(0, 5.0, n_groups)   # real between-group variation
    rows = []
    for g in range(n_groups):
        for _ in range(rng.integers(8, 16)):
            rows.append({"grp": f"g{g}", "y": 50.0 + offsets[g] + rng.normal(0, 2.0)})
    df = pd.DataFrame(rows)
    res = _run(tmp_path, "hier.csv", df, "bayesian_hierarchical",
               "Bayesian hierarchical model", {"group": "grp", "outcome": "y"})
    e = res.estimates
    assert e["n_groups"] == float(n_groups)
    assert e["between_group_sd"] > 0 and e["within_group_sd"] > 0
    assert 0.0 < e["icc"] < 1.0
    assert e["max_rhat"] < 1.15        # hierarchical geometry is harder; allow a touch more
    # partial pooling: a group's pooled intercept sits between its observed mean and the
    # population mean (shrinkage), so it never lands further from the pop mean than the raw mean
    grp_csv = Path(res.output_dir) / "bayesian_hierarchical_groups.csv"
    tbl = pd.read_csv(grp_csv)
    pop = e["population_mean"]
    shrunk = (np.abs(tbl["partial_pooled_intercept"] - pop)
              <= np.abs(tbl["observed_mean"] - pop) + 1e-6)
    assert shrunk.mean() >= 0.8        # shrinkage holds for the large majority of groups
    assert "部分汇集" in res.summary or "ICC" in res.summary


# --------------------------------------------------------------------------- #
# 4) honest degrade when PyMC is unavailable
# --------------------------------------------------------------------------- #
def test_degrade_without_pymc(tmp_path: Path, monkeypatch):
    import researchforge.executor.branches.bayesian_mcmc as bm

    monkeypatch.setattr(bm, "_have_pymc", lambda: False)
    df = pd.DataFrame({"x1": range(30), "y": [i * 1.1 for i in range(30)]})
    res = _run(tmp_path, "deg.csv", df, "bayesian_regression", "Bayesian linear regression",
               {"outcome": "y", "predictors": ["x1"]})
    assert "跳过" in res.summary
    assert "pymc" in res.summary.lower()
    assert not res.estimates  # nothing fabricated
