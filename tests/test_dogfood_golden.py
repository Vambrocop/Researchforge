"""Wave K · batch 0 — dogfooding golden-selection regression lock.

Six Wave-J dogfooding personas (``docs/dogfood-findings.md``) caught the "automatic
transmission" (recommend/pick/study auto-selection + role detection) systematically
mis-routing common data shapes — questionnaires, cohorts, panels, RCBD trials, leaky ML
tables. This file pins the CORRECT post-Wave-K behavior as a regression lock, mirroring
``test_golden_selection.py``'s pattern (build data -> profile -> recommend/run -> assert).

Most assertions here are currently RED (the underlying Wave K batches haven't landed yet)
and are marked ``xfail(strict=False)`` with the batch that should turn them green — flip
each to a hard assertion once its batch lands (that XPASS is the proof the fix worked).
A few assertions are ALREADY correct today and are hard assertions (regression guards).

Fixtures live in ``tests/fixtures/dogfood.py`` (structural replicas of the 6 personas,
NOT copies of the scratch files in ``e:/tmp/dogfood/`` which will disappear).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from researchforge.catalog import Catalog
from researchforge.executor import run_analysis
from researchforge.profiler import diagnose, profile_dataset
from researchforge.recommender import recommend, select_top

from fixtures.dogfood import (
    build_p1_likert,
    build_p2_cohort,
    build_p3_panel,
    build_p4_rcbd,
    build_p5_churn,
    build_p6_messy,
)

_CAT = Catalog.load()
_TOP_K = 6


def _profile(df, tmp_path: Path):
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    return profile_dataset(csv)


def _feasible_ids(fp) -> set[str]:
    return {r.entry.id for r in recommend(fp) if r.feasible}


def _top_ids(fp, k: int = _TOP_K) -> list[str]:
    return [r.entry.id for r in select_top(fp, top=k)]


# ─────────────────────────────────────────────────────────────────────────────
# P1 — 医学生 Likert 问卷 (发现 1、2；Wave K-A1/A2/E7)
# ─────────────────────────────────────────────────────────────────────────────
# Wave K-A1+A2 已落地 → 硬断言：count 模型退出可行集后 cronbach/factor 升进 top-6（附带红利）。
def test_p1_likert_surfaces_psychometrics(tmp_path: Path) -> None:
    fp = _profile(build_p1_likert(), tmp_path)
    top = set(_top_ids(fp))
    assert {"cronbach_alpha", "factor_analysis"} & top, (
        f"expected cronbach_alpha/factor_analysis in top-{_TOP_K}, got {sorted(top)}"
    )


# Wave K-A1(排 ordinal_like)+A2(排非结果 count 协变量 + 单一真源 has_count_outcome) 已落地 → 硬断言。
def test_p1_likert_ecology_and_count_models_infeasible(tmp_path: Path) -> None:
    fp = _profile(build_p1_likert(), tmp_path)
    feasible = _feasible_ids(fp)
    bad = feasible & {
        "permanova", "rda", "indicator_species",
        "poisson_regression", "zero_inflated_poisson", "negative_binomial_regression",
    }
    assert not bad, f"ecology/count models should not be feasible on bounded Likert data: {sorted(bad)}"


# ─────────────────────────────────────────────────────────────────────────────
# P2 — 流行病 cohort (发现 1、3；Wave K-A1/A2)
# ─────────────────────────────────────────────────────────────────────────────
# Wave M1 已落地 → 硬断言:①has_treatment 语义化(处理名列才发 requires_treatment bonus,
# 因果族在无处理名的 cohort 上退出 top)②disease/diagnos 进二值结局词表(disease 高置信结局,
# logistic/epi_risk/diagnostic_test/calibration 上位)。26 条 dogfood 发现至此全部收口。
def test_p2_cohort_surfaces_logistic_and_epi(tmp_path: Path) -> None:
    fp = _profile(build_p2_cohort(), tmp_path)
    top = set(_top_ids(fp))
    assert {"logistic_regression", "epi_risk_measures"} & top, (
        f"expected logistic_regression/epi_risk_measures in top-{_TOP_K}, got {sorted(top)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# P3 — 公司-年面板 (面板结构检测已是强项 -> 硬断言，非 xfail)
# ─────────────────────────────────────────────────────────────────────────────
def test_p3_panel_family_feasible(tmp_path: Path) -> None:
    fp = _profile(build_p3_panel(), tmp_path)
    assert fp.is_panel and fp.unit_col == "firm_id" and fp.time_col == "year"
    feasible = _feasible_ids(fp)
    hit = feasible & {"panel_fixed_effects", "random_effects", "first_difference"}
    assert hit, f"expected a panel estimator feasible, got feasible={sorted(feasible)}"


# Wave K 轨F D1/D2 已落地 → 硬断言：pooled OLS / panel FE 在面板数据上默认 HC1 稳健 SE 忽略
# 同单位内序列相关，把 p 值压到虚假显著（dogfood 观测：panel_fixed_effects 上 HC1 给
# cashflow p≈6.7e-39）；按 firm_id 聚类后 SE 明显变大、p 不再离谱（仍显著——cashflow 对
# investment 确有真实效应，聚类只是把过度自信的 SE 修正回诚实水平，不是把信号抹掉）。
def test_p3_panel_ols_and_fe_use_clustered_se(tmp_path: Path) -> None:
    import statsmodels.formula.api as smf

    from researchforge.executor.run import _regression

    df = build_p3_panel()
    csv = tmp_path / "p3.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    assert fp.is_panel and fp.unit_col == "firm_id"

    for eid in ("ols_regression", "panel_fixed_effects"):
        entry = _CAT.by_id(eid)
        _, rhs_vars, formula, model = _regression(df, fp, entry, {})
        key = "Q('cashflow')"
        assert key in model.bse.index, f"{eid}: cashflow dropped from spec, rhs={rhs_vars}"
        clustered_bse, clustered_p = float(model.bse[key]), float(model.pvalues[key])
        # same spec, unclustered HC1 — the pre-fix behavior — as the baseline to beat.
        unclustered = smf.ols(formula, data=df).fit(cov_type="HC1")
        hc1_bse, hc1_p = float(unclustered.bse[key]), float(unclustered.pvalues[key])
        assert clustered_bse > hc1_bse * 1.05, (
            f"{eid}: clustered SE ({clustered_bse}) not visibly larger than unclustered HC1 "
            f"({hc1_bse}) — clustering may not be engaged"
        )
        assert clustered_p > hc1_p * 100, (
            f"{eid}: clustered p ({clustered_p}) still as extreme as unclustered HC1 p "
            f"({hc1_p}) — SE fix not reflected in significance"
        )

    entry = _CAT.by_id("ols_regression")
    res = run_analysis(fp, entry, output_root=str(tmp_path / "o"), config=None)
    assert "聚类" in res.summary, "summary 应披露按 unit 聚类的稳健 SE"
    assert "panel_fixed_effects" in res.summary, "pooled OLS 在面板数据上应提示改用 panel_fixed_effects"


# ─────────────────────────────────────────────────────────────────────────────
# P4 — 农学 RCBD (发现 11；Wave K-B1 已落地 → 硬断言)
# B1 给 _TRT_HINTS/_BLOCK_HINTS 加中文子串(处理/区组…)后，角色不再反转：处理绑 treatment、区组绑 block。
# ─────────────────────────────────────────────────────────────────────────────
def test_p4_rcbd_roles_not_reversed(tmp_path: Path) -> None:
    df = build_p4_rcbd()
    csv = tmp_path / "p4.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _CAT.by_id("rcbd"), output_root=str(tmp_path / "o"), config=None)
    # 处理 has 5 levels (对照/处理A/B/C/D); 区组 has 4 levels (block1..4).
    assert res.estimates.get("n_treatments") == 5.0, (
        f"treatment role should bind to 处理(5 levels), got n_treatments={res.estimates.get('n_treatments')}"
    )
    assert res.estimates.get("n_blocks") == 4.0, (
        f"block role should bind to 区组(4 levels), got n_blocks={res.estimates.get('n_blocks')}"
    )


def test_p4_rcbd_anova_roles_not_reversed(tmp_path: Path) -> None:
    df = build_p4_rcbd()
    csv = tmp_path / "p4.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _CAT.by_id("rcbd_anova"), output_root=str(tmp_path / "o"), config=None)
    assert res.estimates.get("n_treatments") == 5.0, (
        f"treatment role should bind to 处理(5 levels), got n_treatments={res.estimates.get('n_treatments')}"
    )
    assert res.estimates.get("n_blocks") == 4.0, (
        f"block role should bind to 区组(4 levels), got n_blocks={res.estimates.get('n_blocks')}"
    )


def test_p4_rcbd_at_least_feasible(tmp_path: Path) -> None:
    """Already true today: rcbd/rcbd_anova ARE feasible on this shape (just not top-ranked
    and, per the xfail tests above, currently bind roles backwards when they DO run)."""
    fp = _profile(build_p4_rcbd(), tmp_path)
    feasible = _feasible_ids(fp)
    hit = feasible & {"rcbd", "rcbd_anova"}
    assert hit, f"expected rcbd/rcbd_anova feasible on a textbook RCBD shape, got {sorted(feasible)}"


# ─────────────────────────────────────────────────────────────────────────────
# P5 — churn 预测/泄漏 (发现 22, 真 bug；Wave K-F1 已落地 → 硬断言/回归护栏)
# ─────────────────────────────────────────────────────────────────────────────
def test_p5_config_outcome_binds_random_forest(tmp_path: Path) -> None:
    fp = _profile(build_p5_churn(), tmp_path)
    res = run_analysis(fp, _CAT.by_id("random_forest"), output_root=str(tmp_path / "o"),
                       config={"outcome": "churn"})
    assert "预测 churn" in res.summary, f"config outcome=churn not honored: {res.summary!r}"


def test_p5_config_outcome_binds_xgboost(tmp_path: Path) -> None:
    fp = _profile(build_p5_churn(), tmp_path)
    res = run_analysis(fp, _CAT.by_id("xgboost"), output_root=str(tmp_path / "o"),
                       config={"outcome": "churn"})
    assert "预测 churn" in res.summary, f"config outcome=churn not honored: {res.summary!r}"


def test_p5_config_outcome_binds_logistic(tmp_path: Path) -> None:
    """Already true today (Wave H4 outcome-resolve sweep wired logistic_regression's
    _regression() helper through resolve_outcome correctly) — a regression guard."""
    fp = _profile(build_p5_churn(), tmp_path)
    res = run_analysis(fp, _CAT.by_id("logistic_regression"), output_root=str(tmp_path / "o"),
                       config={"outcome": "churn"})
    assert "结果变量 churn" in res.summary, f"config outcome=churn not honored: {res.summary!r}"


# Wave L ③: WITHOUT config, rf/xgboost previously regressed a continuous FEATURE (tenure)
# because the cont-first tier never considered churn (binary). A high-confidence detected
# outcome (churn matches the event-name pattern) now binds across kinds → classification on
# churn, not regression on tenure. churn is event-named (a target), not treatment-named
# (a design factor), so binding it is safe.
def test_p5_no_config_rf_binds_high_conf_churn(tmp_path: Path) -> None:
    fp = _profile(build_p5_churn(), tmp_path)
    assert fp.likely_outcome == "churn" and fp.likely_outcome_confidence == "high", (
        f"churn should be a high-confidence detected outcome; got "
        f"{fp.likely_outcome!r}/{fp.likely_outcome_confidence!r}"
    )
    for eid in ("random_forest", "xgboost"):
        res = run_analysis(fp, _CAT.by_id(eid), output_root=str(tmp_path / eid), config=None)
        assert "预测 churn" in res.summary, (
            f"{eid} without config should predict high-conf churn, not a continuous feature: {res.summary!r}"
        )


# Wave L 收尾: ml_supervised._resolve_xy (gradient_boosting/svm/regularized_regression) had the
# SAME cont-first-before-high-confidence gap as ml.py's rf/xgboost above, pre-③. Fixed by
# reusing ml.py's _resolve_ml_outcome for the no-config fallback (one ladder, zero drift) —
# without config, gradient_boosting must now CLASSIFY churn, not regress a continuous feature
# (tenure).
def test_p5_no_config_gbm_binds_high_conf_churn(tmp_path: Path) -> None:
    fp = _profile(build_p5_churn(), tmp_path)
    assert fp.likely_outcome == "churn" and fp.likely_outcome_confidence == "high", (
        f"churn should be a high-confidence detected outcome; got "
        f"{fp.likely_outcome!r}/{fp.likely_outcome_confidence!r}"
    )
    res = run_analysis(
        fp, _CAT.by_id("gradient_boosting"), output_root=str(tmp_path / "gbm"), config=None
    )
    assert "分类 churn" in res.summary, (
        f"gradient_boosting without config should classify high-conf churn, not regress a "
        f"continuous feature: {res.summary!r}"
    )


# Wave L D: gamm's requires_group gate accepted ANY binary/categorical column (region has only
# 4 levels, churn — the outcome itself — has 2), so gamm was ranking rank-1 on this cross-sectional
# churn data under goal=predict then failing at runtime with "需要分组变量...(≥5 组)". min_group_levels
# tightens the gate to a genuine grouping structure (>=5 levels) — hard assertion, regression guard.
def test_p5_gamm_infeasible_no_real_group(tmp_path: Path) -> None:
    fp = _profile(build_p5_churn(), tmp_path)
    feasible = _feasible_ids(fp)
    assert "gamm" not in feasible, (
        "gamm should be infeasible on cross-sectional churn data with no >=5-level "
        f"grouping column (region has 4, churn has 2), got feasible={sorted(feasible)}"
    )
    top = set(r.entry.id for r in select_top(fp, goal="predict", top=_TOP_K))
    assert "gamm" not in top, f"gamm (doomed at runtime) should not rank in top-{_TOP_K}, got {sorted(top)}"


# ─────────────────────────────────────────────────────────────────────────────
# P6 — 脏 admin 表 (clean 层是强项 -> 硬断言，非 xfail)
# ─────────────────────────────────────────────────────────────────────────────
def test_p6_messy_profiler_survives_and_flags_issues(tmp_path: Path) -> None:
    df = build_p6_messy()
    csv = tmp_path / "p6.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)  # must not raise on comma-text numbers / non-padded dates
    assert fp.n_rows == len(df)
    issues = diagnose(df)
    kinds = {i.kind for i in issues}
    assert {"duplicate_rows", "constant", "missing"} <= kinds, (
        f"expected duplicate/constant/missing issues flagged, got kinds={sorted(kinds)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# C1 — goal=compare 排序 (发现16；Wave K-C1 已落地 → 硬断言)
# 观测数据(无 treatment/block 列)在 --goal compare 下，DoE 怪兽(factorial/split-plot/RCBD/
# Latin/AMMI)不得压过朴素 group_comparison；有设计信号(P4 处理/区组)时 DoE 排位不动。
# ─────────────────────────────────────────────────────────────────────────────
def _compare_top_ids(fp, k: int = _TOP_K) -> list[str]:
    return [r.entry.id for r in select_top(fp, goal="compare", top=k)]


def test_p6_compare_goal_surfaces_naive_comparison(tmp_path: Path) -> None:
    from researchforge.recommender.goals import has_design_signal

    fp = _profile(build_p6_messy(), tmp_path)
    assert not has_design_signal(fp), "P6 is observational — no treatment/block column expected"
    top = _compare_top_ids(fp)
    assert "group_comparison" in top, (
        f"naive group_comparison should surface in top-{_TOP_K} on observational compare data, got {top}"
    )
    doe = {"factorial_anova", "split_plot", "rcbd", "latin_square", "ammi", "gge_biplot"}
    assert not (doe & set(top)), (
        f"designed-experiment methods must not crowd top-{_TOP_K} on observational data, got {sorted(doe & set(top))}"
    )


def test_p4_compare_goal_keeps_doe_when_design_signal(tmp_path: Path) -> None:
    """Gate check: P4 HAS a design signal (处理/区组), so the C1 demotion must NOT fire — a real
    RCBD trial keeps its designed-experiment methods eligible while naive comparison also shows."""
    from researchforge.recommender.goals import has_design_signal

    fp = _profile(build_p4_rcbd(), tmp_path)
    assert has_design_signal(fp), "P4 is a designed RCBD — 处理/区组 should signal design"
    top = _compare_top_ids(fp)
    assert "group_comparison" in top, f"group_comparison should still surface on P4, got {top}"
    assert {"factorial_anova", "rcbd", "split_plot"} & set(top), (
        f"designed-experiment methods must stay eligible when a design signal is present, got {top}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# C2 — 面板误检硬化根治 (发现16 根因；Wave L-C2 → 硬断言)
# P6 的 备注 列 89% 空、日期 高基数：旧 unit_col 检测在 dropna 子集上判唯一 → 误当面板单位 →
# is_panel=True → system_gmm 劫持 compare。加覆盖率/重复守卫后 P6 不再是面板；真面板(P3)不受影响。
# ─────────────────────────────────────────────────────────────────────────────
def test_p6_not_misdetected_as_panel(tmp_path: Path) -> None:
    fp = _profile(build_p6_messy(), tmp_path)
    assert not fp.is_panel, (
        f"observational admin data must not be a panel; got is_panel with unit_col={fp.unit_col!r}"
    )
    assert fp.unit_col is None, f"an 89%-empty notes column must not be a unit; got {fp.unit_col!r}"
    # the panel misdetection was what floated system_gmm into the compare pool (发现16)
    feasible = _feasible_ids(fp)
    assert "system_gmm" not in feasible, (
        "panel-only estimators must not be feasible on non-panel data"
    )


def test_p3_still_detected_as_panel(tmp_path: Path) -> None:
    """Regression guard: the C2 unit_col guards must NOT break a genuine balanced panel —
    firm_id (0 missing, repeats across years) is still a valid unit."""
    fp = _profile(build_p3_panel(), tmp_path)
    assert fp.is_panel and fp.unit_col == "firm_id" and fp.time_col == "year", (
        f"real panel must stay detected; got is_panel={fp.is_panel} unit={fp.unit_col} time={fp.time_col}"
    )
