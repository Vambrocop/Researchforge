"""Tests for bayesian_survival — Bayesian Weibull AFT with right-censoring (PyMC).

Simulates Weibull survival times with a KNOWN covariate AFT effect plus administrative
right-censoring, then asserts the posterior recovers the effect's SIGN and rough
magnitude, the Weibull shape is positive, the event/censored counts are correct, and the
chains converge (max R-hat < 1.15). A degrade test monkeypatches PyMC away to confirm the
honest skip (nothing fabricated). These are slow (MCMC fits) — tagged in conftest
SLOW_MODULES.

The censored-likelihood is the thing under test: we generate true Weibull times, apply an
administrative censoring cap so ~30% of rows are right-censored (event=0 at the cap), and
check the recovered AFT effect points the right way — a POSITIVE AFT coefficient lengthens
survival (acceleration factor AF>1).
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

_FAST = {"draws": 400, "tune": 400, "chains": 2, "seed": 0}


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="bayesian_survival",
        method="Bayesian parametric survival (Weibull AFT)",
        domain="statistics",
        family="survival",
        goal="explain",
        preconditions=Precondition(min_continuous=1, min_rows=15),
    )


def _run(tmp_path, name, df, config):
    p = tmp_path / name
    df.to_csv(p, index=False)
    fp = profile_dataset(p)
    return run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                        config={**_FAST, **config})


def _simulate_weibull(rng, n, shape, beta_x, log_scale0):
    """True Weibull AFT: log(scale_i) = log_scale0 + beta_x * x_i; T_i ~ Weibull(shape, scale_i).
    A POSITIVE beta_x lengthens survival (larger scale -> larger times)."""
    x = rng.normal(size=n)
    scale = np.exp(log_scale0 + beta_x * x)
    # inverse-CDF Weibull draw: T = scale * (-ln U)^(1/shape)
    u = rng.uniform(size=n)
    t = scale * (-np.log(u)) ** (1.0 / shape)
    return x, t


# --------------------------------------------------------------------------- #
# 1) covariate AFT effect recovery + censoring counts + shape + convergence
# --------------------------------------------------------------------------- #
def test_bayesian_survival_recovers_aft_effect(tmp_path: Path):
    rng = np.random.default_rng(11)
    n = 220
    shape_true = 1.5
    beta_true = 0.6          # positive -> lengthens survival (AF > 1)
    x, t = _simulate_weibull(rng, n, shape_true, beta_true, log_scale0=2.0)

    # administrative right-censoring at a cap: event=0 and duration=cap if T > cap
    cap = float(np.quantile(t, 0.7))     # ~30% censored
    event = (t <= cap).astype(int)
    duration = np.minimum(t, cap)
    df = pd.DataFrame({"duration": duration, "event": event, "x": x})

    res = _run(tmp_path, "surv.csv", df, {"duration": "duration", "event": "event",
                                          "predictors": ["x"]})
    e = res.estimates

    # event / censored bookkeeping is exact
    assert e["n_events"] == float(int(event.sum()))
    assert e["n_censored"] == float(int((event == 0).sum()))
    assert e["n_events"] + e["n_censored"] == float(n)
    assert int(e["n_censored"]) > 0          # censoring actually happened (likelihood exercised)

    # Weibull shape is positive and roughly recovered
    assert e["weibull_shape"] > 0
    assert abs(e["weibull_shape"] - shape_true) < 0.7

    # AFT effect points the RIGHT way: positive std-scale coef, acceleration factor > 1
    assert e["aft_x"] > 0
    assert e["af_x"] > 1.0
    assert e["aft_x__hdi_low"] <= e["aft_x__hdi_high"]   # HDI ordering
    assert e["aft_x__hdi_low"] > 0                        # clearly positive -> HDI excludes 0

    # median survival is a sane positive time
    assert e["median_survival"] > 0

    # honest convergence + Chinese disclosures
    assert isinstance(e["max_rhat"], float) and e["max_rhat"] == e["max_rhat"]
    assert e["max_rhat"] < 1.15
    assert "Weibull" in res.summary and "AFT" in res.summary
    assert "⚠" in res.summary

    # acceleration-factor CSV produced
    assert (Path(res.output_dir) / "bayesian_survival_aft.csv").exists()


# --------------------------------------------------------------------------- #
# 2) no event column -> treat all as observed (no censoring) + disclose
# --------------------------------------------------------------------------- #
def test_bayesian_survival_no_event_column(tmp_path: Path):
    rng = np.random.default_rng(12)
    n = 80
    x, t = _simulate_weibull(rng, n, shape=1.2, beta_x=0.0, log_scale0=1.5)
    # NO event column at all; only a duration + a covariate
    df = pd.DataFrame({"duration": t, "x": x})

    res = _run(tmp_path, "noevent.csv", df, {"duration": "duration", "predictors": ["x"]})
    e = res.estimates
    assert e["n_censored"] == 0.0          # all treated as observed
    assert e["n_events"] == float(n)
    assert e["weibull_shape"] > 0
    assert "无删失" in res.summary or "无事件指示列" in res.summary or "完整事件" in res.summary


# --------------------------------------------------------------------------- #
# 3) honest degrade when PyMC is unavailable
# --------------------------------------------------------------------------- #
def test_bayesian_survival_degrade_without_pymc(tmp_path: Path, monkeypatch):
    import researchforge.executor.branches.bayesian_survival as bs

    monkeypatch.setattr(bs, "_have_pymc", lambda: False)
    df = pd.DataFrame({
        "duration": [float(i) + 1.0 for i in range(30)],
        "event": [i % 2 for i in range(30)],
        "x": list(range(30)),
    })
    res = _run(tmp_path, "deg.csv", df, {"duration": "duration", "event": "event",
                                         "predictors": ["x"]})
    assert "跳过" in res.summary
    assert "pymc" in res.summary.lower()
    assert not res.estimates       # nothing fabricated


# --------------------------------------------------------------------------- #
# 4) too few rows -> honest skip (no crash, no fabricated estimates)
# --------------------------------------------------------------------------- #
def test_bayesian_survival_too_few_rows(tmp_path: Path):
    df = pd.DataFrame({
        "duration": [1.0, 2.0, 3.0, 4.0, 5.0],
        "event": [1, 0, 1, 1, 0],
        "x": [0.1, 0.2, 0.3, 0.4, 0.5],
    })
    res = _run(tmp_path, "few.csv", df, {"duration": "duration", "event": "event",
                                         "predictors": ["x"]})
    assert "跳过" in res.summary
    assert not res.estimates
