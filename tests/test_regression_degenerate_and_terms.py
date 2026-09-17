"""dogfood：回归族报「完成」却零估计，而模型其实是秩亏的。

Measured on a textbook repeated-measures RCT `[subject, week, arm, pain_score]`:

    did → estimates={}, summary = "…完成：因变量 pain_score"   （一个系数都没有）

…and `did` was ranked #2 by the recommender, so that empty result took a headline slot.
Two independent defects, both general to the `_REGRESSION` family (ols_regression /
panel_fixed_effects / did):

(a) COEFFICIENT LOOKUP — estimates were recorded only under the exact key `Q('<v>')`. A
    STRING-coded binary/categorical predictor is treatment-coded by patsy, so its real name
    is `Q('arm')[T.placebo]`; the lookup missed and the number was dropped silently, leaving
    a naked "完成". coefficients.csv had the value all along.

(b) RANK DEFICIENCY — `arm` is constant within subject, hence perfectly collinear with the
    subject fixed effects. The design matrix had rank 64 of 65 columns and statsmodels
    returned `Q('arm')[T.placebo] = -1.398` with **SE = 1.09e-13**. The branch reported 完成.
    Note the first version of this guard used `max(bse)` and did NOT fire: rank deficiency
    kills one DIRECTION, so the other 64 SEs were normal (max |bse| = 0.80). The rank of the
    design matrix is the honest test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog import Catalog
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_CAT = Catalog.load()


def _fp(df, tmp_path, name="d.csv"):
    csv = tmp_path / name
    df.to_csv(csv, index=False)
    return profile_dataset(csv)


def _rm_rct(n_subj=60, n_time=5, seed=3):
    """`arm` is constant per subject → collinear with the subject fixed effects."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subj):
        grp = "drug" if s % 2 == 0 else "placebo"
        re_s = rng.normal(0, 4.0)
        for t in range(n_time):
            rows.append({"subject": f"p{s:03d}", "week": t, "arm": grp,
                         "pain_score": round(float(50 + re_s - 1.5 * t
                                                   - 2.5 * t * (grp == "drug")
                                                   + rng.normal(0, 2.0)), 2)})
    return pd.DataFrame(rows)


def _switching_panel(n_unit=30, n_t=14, seed=11):
    """A REAL staggered-adoption panel, true ATT +3.0 — the guard must not touch it."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_unit):
        fe = rng.normal(0, 1)
        g = 6 if u % 3 == 0 else (10 if u % 3 == 1 else 10 ** 9)
        for t in range(n_t):
            d = 1 if t >= g else 0
            rows.append({"unit": f"u{u:02d}", "year": 2005 + t,
                         "y": round(float(10 + fe + 3.0 * d + 0.1 * t + rng.normal(0, 0.6)), 3),
                         "adopted": d})
    return pd.DataFrame(rows)


# ── (b) rank deficiency ──────────────────────────────────────────────────────
def test_a_rank_deficient_design_fails_honestly(tmp_path):
    fp = _fp(_rm_rct(), tmp_path)
    res = run_analysis(fp, _CAT.by_id("did"), output_root=str(tmp_path / "o"))
    assert "失败" in res.summary, res.summary[:200]
    assert "秩亏" in res.summary
    assert not res.estimates, "a rank-deficient fit must not report coefficients"


def test_the_failure_points_at_the_right_method(tmp_path):
    """An honest degrade names the alternative — this frame IS analysable, just not by DiD."""
    fp = _fp(_rm_rct(), tmp_path)
    res = run_analysis(fp, _CAT.by_id("did"), output_root=str(tmp_path / "p"))
    assert "repeated_measures_anova" in res.summary or "mixed_effects" in res.summary


def test_the_guard_does_not_fire_on_a_real_switching_panel(tmp_path):
    fp = _fp(_switching_panel(), tmp_path, name="panel.csv")
    res = run_analysis(fp, _CAT.by_id("did"), output_root=str(tmp_path / "q"))
    assert "秩亏" not in res.summary
    assert 2.0 < res.estimates["adopted"] < 4.0, res.estimates


def test_the_guard_does_not_fire_on_plain_ols(tmp_path):
    rng = np.random.default_rng(4)
    n = 300
    x = rng.normal(0, 1, n)
    z = rng.normal(0, 1, n)
    df = pd.DataFrame({"y": (10 + 2 * x - z + rng.normal(0, 1, n)).round(3),
                       "x": x.round(3), "z": z.round(3)})
    fp = _fp(df, tmp_path, name="ols.csv")
    res = run_analysis(fp, _CAT.by_id("ols_regression"), output_root=str(tmp_path / "r"))
    assert "秩亏" not in res.summary
    assert res.estimates


# ── (a) categorical coefficient names ────────────────────────────────────────
def test_a_string_coded_predictor_is_reported(tmp_path):
    """`Q('arm')[T.placebo]`, not `Q('arm')` — the estimate used to vanish silently."""
    rng = np.random.default_rng(4)
    n = 300
    arm = rng.choice(["drug", "placebo"], n)
    x = rng.normal(0, 1, n)
    df = pd.DataFrame({"y": (10 - 3 * (arm == "drug") + 2 * x + rng.normal(0, 1, n)).round(3),
                       "arm": arm, "x": x.round(3)})
    fp = _fp(df, tmp_path, name="str.csv")
    res = run_analysis(fp, _CAT.by_id("ols_regression"), output_root=str(tmp_path / "s"))
    assert res.estimates, "a fitted model must report its coefficients"
    key = next((k for k in res.estimates if k.startswith("arm")), None)
    assert key is not None, res.estimates
    assert 2.0 < res.estimates[key] < 4.0, "drug lowers y by 3, so placebo is +3"
    assert "关键系数" in res.summary and "arm" in res.summary


def test_a_dropped_categorical_predictor_is_disclosed(tmp_path):
    """My first version of this test asserted the factor's levels appear as coefficients —
    wrong premise: this family's predictor set is continuous/count/binary, so a multi-level
    categorical never enters the model at all. The measurement that matters: on `[y, grp, x]`
    where `grp` (a=0, b=+4, c=+8, noise sd 1) explains ~95% of the variance, the report read
    "关键系数 x = -0.0571 (p=0.724)" and said nothing about `grp` — a reader concludes nothing
    predicts y. mixed_effects fixed the same silent drop in Wave K-B3 by dummy-coding; here we
    at least say it out loud (dummy-coding the regression family is a spec change, deferred)."""
    rng = np.random.default_rng(6)
    n = 450
    grp = rng.choice(["a", "b", "c"], n)
    shift = pd.Series(grp).map({"a": 0.0, "b": 4.0, "c": 8.0}).to_numpy()
    df = pd.DataFrame({"y": (10 + shift + rng.normal(0, 1, n)).round(3),
                       "grp": grp, "x": rng.normal(0, 1, n).round(3)})
    fp = _fp(df, tmp_path, name="multi.csv")
    res = run_analysis(fp, _CAT.by_id("ols_regression"), output_root=str(tmp_path / "t"))
    assert "grp" not in res.estimates                      # it really is not in the model
    assert "分类预测变量" in res.summary and "grp" in res.summary, res.summary[:200]


def test_no_disclosure_when_nothing_was_dropped(tmp_path):
    """The note must not fire on a frame with no categorical predictors."""
    rng = np.random.default_rng(12)
    n = 300
    df = pd.DataFrame({"y": rng.normal(10, 2, n).round(3),
                       "x": rng.normal(0, 1, n).round(3)})
    fp = _fp(df, tmp_path, name="plain.csv")
    res = run_analysis(fp, _CAT.by_id("ols_regression"), output_root=str(tmp_path / "u"))
    assert "分类预测变量" not in res.summary


# ── cold review B1: the guard's first version killed the flagship panel shape ─
def _panel_with_time_invariant_covariate(n_unit=40, n_t=6, seed=11):
    """A firm x year panel carrying ONE time-invariant covariate — a baseline value, a
    sex code, a region code. Under two-way fixed effects that column is absorbed by the
    unit dummies BY CONSTRUCTION, so this is not an exotic shape: it is the ordinary one.
    True ATT = 3.0."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_unit):
        baseline = rng.normal(50, 10)
        a = rng.normal(0, 2)
        for t in range(n_t):
            treat = 1 if (i % 2 == 0 and t >= 3) else 0
            rows.append({"firm": f"F{i:02d}", "year": 2010 + t, "baseline": round(baseline, 2),
                         "capex": round(rng.normal(20, 5), 2), "treat": treat,
                         "profit": round(10 + a + 0.5 * t + 3.0 * treat
                                         + 0.2 * rng.normal(), 3)})
    return pd.DataFrame(rows)


def test_an_absorbed_covariate_does_not_void_the_whole_panel_analysis(tmp_path):
    """v1 of the guard returned 失败 with estimates={} here. But the effect IS estimable:
    rank deficiency kills the absorbed DIRECTION, not the fit."""
    fp = _fp(_panel_with_time_invariant_covariate(), tmp_path, name="tinv.csv")
    res = run_analysis(fp, _CAT.by_id("panel_fixed_effects"), output_root=str(tmp_path / "o"))
    assert "失败" not in res.summary, res.summary[:260]
    assert res.estimates, "a panel whose treatment effect is identified must report it"
    assert 2.5 < res.estimates["treat"] < 3.5, res.estimates


def test_the_absorbed_column_is_named_not_silently_dropped(tmp_path):
    fp = _fp(_panel_with_time_invariant_covariate(), tmp_path, name="tinv2.csv")
    res = run_analysis(fp, _CAT.by_id("panel_fixed_effects"), output_root=str(tmp_path / "p"))
    assert "完全共线" in res.summary and "baseline" in res.summary
    assert "baseline" not in res.estimates, "an aliased column must not be reported"
    # the headline moved off the dropped column — say so rather than let it shift silently
    assert "关键系数" in res.summary and "已从 baseline" in res.summary


def test_the_surviving_estimate_equals_the_manual_drop_and_refit(tmp_path):
    """The claim the fix rests on: dropping the aliased column changes nothing else."""
    import statsmodels.formula.api as smf

    df = _panel_with_time_invariant_covariate()
    fp = _fp(df, tmp_path, name="tinv3.csv")
    res = run_analysis(fp, _CAT.by_id("panel_fixed_effects"), output_root=str(tmp_path / "q"))
    manual = smf.ols(
        "Q('profit') ~ Q('capex') + Q('treat') + C(Q('firm')) + C(Q('year'))", data=df
    ).fit(cov_type="cluster", cov_kwds={"groups": df["firm"]})
    assert res.estimates["treat"] == pytest.approx(
        float(manual.params["Q('treat')"]), rel=1e-9
    )


def test_nothing_estimable_still_fails_honestly(tmp_path):
    """When every predictor is absorbed there is nothing to salvage — keep failing."""
    fp = _fp(_rm_rct(), tmp_path, name="none.csv")
    res = run_analysis(fp, _CAT.by_id("did"), output_root=str(tmp_path / "r"))
    assert "失败" in res.summary and "秩亏" in res.summary
    assert not res.estimates
