"""Auto-diagnose → recommend-with-a-plan engine (smarter auto-selection, v1.5).

These exercise ``recommender.diagnostics``: each value-level diagnostic fires on a
synthetic dataset where the condition is engineered in (and stays silent on a clean
control), the plan reuses the slice-1 outcome role hint, every suggested method id
is real (catalog-filtered), and the whole thing degrades without crashing on junk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.registry import Catalog
from researchforge.profiler import profile_dataset
from researchforge.recommender.diagnostics import build_plan, diagnose_data


def _plan(tmp_path: Path, name: str, df: pd.DataFrame):
    p = tmp_path / name
    df.to_csv(p, index=False)
    fp = profile_dataset(p)
    return fp, build_plan(fp)


def _codes(plan) -> set[str]:
    return {d.code for d in plan.diagnostics}


def _by_code(plan, code: str):
    return next(d for d in plan.diagnostics if d.code == code)


# --------------------------------------------------------------------------- #
# overdispersion / zero-inflation (count outcome)
# --------------------------------------------------------------------------- #
def test_overdispersion_detected(tmp_path: Path):
    # negative-binomial counts: var >> mean -> overdispersion, prefer negbin over poisson
    rng = np.random.default_rng(0)
    events = rng.negative_binomial(2, 0.2, 300)  # mean ~8, var ~ mean*5
    df = pd.DataFrame({"x": rng.normal(size=300), "events": events})
    fp, plan = _plan(tmp_path, "od.csv", df)
    assert fp.column("events").kind == "count"
    assert "overdispersion" in _codes(plan)
    d = _by_code(plan, "overdispersion")
    assert "negative_binomial_regression" in d.prefer
    assert "poisson_regression" in d.over


def test_poisson_no_overdispersion(tmp_path: Path):
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(size=400), "events": rng.poisson(4, 400)})
    _, plan = _plan(tmp_path, "pois.csv", df)
    assert "overdispersion" not in _codes(plan)


def test_zero_inflation_detected(tmp_path: Path):
    rng = np.random.default_rng(2)
    base = rng.poisson(5, 400)
    mask = rng.random(400) < 0.45          # ~45% structural zeros
    events = np.where(mask, 0, base)
    df = pd.DataFrame({"x": rng.normal(size=400), "events": events})
    _, plan = _plan(tmp_path, "zi.csv", df)
    assert "zero_inflation" in _codes(plan)
    assert "zero_inflated_poisson" in _by_code(plan, "zero_inflation").prefer


# --------------------------------------------------------------------------- #
# non-normal continuous outcome
# --------------------------------------------------------------------------- #
def test_non_normal_outcome_detected(tmp_path: Path):
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "g": rng.integers(0, 2, 200),
        "outcome": rng.exponential(2.0, 200),  # right-skewed -> non-normal
    })
    _, plan = _plan(tmp_path, "skew.csv", df)
    assert "non_normal_outcome" in _codes(plan)
    pref = _by_code(plan, "non_normal_outcome").prefer
    assert "mann_whitney" in pref and "bootstrap_ci" in pref


def test_normal_outcome_not_flagged(tmp_path: Path):
    rng = np.random.default_rng(4)
    df = pd.DataFrame({
        "g": rng.integers(0, 2, 300),
        "outcome": rng.normal(10, 2, 300),
    })
    _, plan = _plan(tmp_path, "norm.csv", df)
    assert "non_normal_outcome" not in _codes(plan)


# --------------------------------------------------------------------------- #
# heteroskedasticity (Koenker BP)
# --------------------------------------------------------------------------- #
def test_heteroskedasticity_detected(tmp_path: Path):
    rng = np.random.default_rng(5)
    x = rng.uniform(1, 10, 300)
    outcome = 2 * x + rng.normal(0, 1, 300) * x   # error scales with x
    df = pd.DataFrame({"outcome": outcome, "x": x})
    fp, plan = _plan(tmp_path, "het.csv", df)
    assert fp.likely_outcome == "outcome"
    assert "heteroskedasticity" in _codes(plan)
    assert "robust_regression" in _by_code(plan, "heteroskedasticity").prefer


def test_homoskedastic_not_flagged(tmp_path: Path):
    rng = np.random.default_rng(6)
    x = rng.uniform(1, 10, 300)
    outcome = 2 * x + rng.normal(0, 1, 300)       # constant-variance error
    df = pd.DataFrame({"outcome": outcome, "x": x})
    _, plan = _plan(tmp_path, "homo.csv", df)
    assert "heteroskedasticity" not in _codes(plan)


# --------------------------------------------------------------------------- #
# multicollinearity (VIF)
# --------------------------------------------------------------------------- #
def test_multicollinearity_detected(tmp_path: Path):
    rng = np.random.default_rng(7)
    x1 = rng.normal(size=200)
    x2 = x1 + rng.normal(0, 0.01, 200)            # near-duplicate predictor
    outcome = x1 + rng.normal(0, 1, 200)
    df = pd.DataFrame({"outcome": outcome, "x1": x1, "x2": x2})
    _, plan = _plan(tmp_path, "vif.csv", df)
    assert "multicollinearity" in _codes(plan)
    assert "vif_multicollinearity" in _by_code(plan, "multicollinearity").prefer


def test_independent_predictors_not_flagged(tmp_path: Path):
    rng = np.random.default_rng(8)
    df = pd.DataFrame({
        "outcome": rng.normal(size=200),
        "x1": rng.normal(size=200),
        "x2": rng.normal(size=200),
    })
    _, plan = _plan(tmp_path, "indep.csv", df)
    assert "multicollinearity" not in _codes(plan)


# --------------------------------------------------------------------------- #
# small sample
# --------------------------------------------------------------------------- #
def test_small_sample_detected(tmp_path: Path):
    rng = np.random.default_rng(9)
    df = pd.DataFrame({"outcome": rng.normal(size=18), "x": rng.normal(size=18)})
    _, plan = _plan(tmp_path, "small.csv", df)
    assert "small_sample" in _codes(plan)
    assert "bootstrap_ci" in _by_code(plan, "small_sample").prefer


# --------------------------------------------------------------------------- #
# plan-level: outcome reuses the role hint, ids are real, junk degrades
# --------------------------------------------------------------------------- #
def test_plan_outcome_uses_role_hint(tmp_path: Path):
    rng = np.random.default_rng(10)
    df = pd.DataFrame({"a": rng.normal(size=60), "b": rng.normal(size=60),
                       "target": rng.normal(size=60)})
    p = tmp_path / "roles.csv"
    df.to_csv(p, index=False)
    fp = profile_dataset(p)
    assert fp.likely_outcome == "target"
    plan = build_plan(fp)
    assert plan.outcome == "target"


def test_all_suggested_ids_are_real(tmp_path: Path):
    # catalog-filtered build must keep every diagnostic (no dangling/renamed ids)
    rng = np.random.default_rng(11)
    df = pd.DataFrame({"x": rng.normal(size=300),
                       "events": rng.negative_binomial(2, 0.2, 300)})
    p = tmp_path / "real.csv"
    df.to_csv(p, index=False)
    fp = profile_dataset(p)
    catalog = Catalog.load()
    ids = {e.id for e in catalog.all()}
    plan = build_plan(fp, catalog=catalog)
    assert plan.diagnostics  # at least overdispersion survives catalog filtering
    for d in plan.diagnostics:
        assert d.prefer and all(m in ids for m in d.prefer)
        assert all(m in ids for m in d.over)


def test_junk_data_never_crashes(tmp_path: Path):
    df = pd.DataFrame({"name": ["a", "b", "c"] * 5, "tag": list("xyzwv") * 3})
    fp, plan = _plan(tmp_path, "junk.csv", df)
    assert isinstance(plan.diagnostics, list)  # no crash; may be empty/small-sample only


def test_diagnose_data_direct_smoke():
    # diagnose_data works directly on an in-memory frame + fingerprint
    rng = np.random.default_rng(12)
    df = pd.DataFrame({"outcome": rng.exponential(2, 100), "g": rng.integers(0, 2, 100)})
    fp = profile_dataset_from_df(df)
    diags = diagnose_data(df, fp)
    assert any(d.code == "non_normal_outcome" for d in diags)


def profile_dataset_from_df(df: pd.DataFrame):
    # helper: profile an in-memory frame via a tmp file (profile reads from disk)
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.csv"
        df.to_csv(p, index=False)
        return profile_dataset(p)
