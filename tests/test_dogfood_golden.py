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
@pytest.mark.xfail(
    reason="Wave K-A1/K-E7 未落地：8 个 ordinal_like 满意度列仍在可行性/排名上跟 ecology/agreement 抢位，"
    "cronbach_alpha 现排名第 7（top-6 之外）、factor_analysis 连 top-10 都进不去（发现1）",
    strict=False,
)
def test_p1_likert_surfaces_psychometrics(tmp_path: Path) -> None:
    fp = _profile(build_p1_likert(), tmp_path)
    top = set(_top_ids(fp))
    assert {"cronbach_alpha", "factor_analysis"} & top, (
        f"expected cronbach_alpha/factor_analysis in top-{_TOP_K}, got {sorted(top)}"
    )


@pytest.mark.xfail(
    reason="Wave K-A1/K-A2 未落地：match.py 的 requires_count_outcome/min_count_cols 门只看 c.kind=='count'、"
    "不排除 ordinal_like，8 个 Likert 列仍判定为可行的计数结果 -> permanova/indicator_species/"
    "poisson/zip/nb 全部 feasible=True（发现1，应对有界评分不可行）",
    strict=False,
)
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
@pytest.mark.xfail(
    reason="Wave K-A2 未落地：age(20-80 整数, count kind, 非 likely_outcome、无 count/n_/events 命名) "
    "仍被判 has_count_outcome=True，把 NB/ZIP/Poisson(fit=81) 顶到 logistic_regression/"
    "epi_risk_measures(fit=67) 之上，二者未进 top-6（发现1、3）",
    strict=False,
)
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


# ─────────────────────────────────────────────────────────────────────────────
# P4 — 农学 RCBD (发现 11；Wave K-B1 — role hints 英文 only 反转角色)
# ─────────────────────────────────────────────────────────────────────────────
_P4_ROLE_XFAIL = pytest.mark.xfail(
    reason="Wave K-B1 未落地：_TRT_HINTS/_BLOCK_HINTS 只认英文子串，中文'处理'/'区组'两个都不命中，"
    "兜底按列声明顺序分配角色 -> 处理(5 水平)被当 block、区组(4 水平)被当 treatment，"
    "角色反转、无披露（发现11，阻断级）",
    strict=False,
)


@_P4_ROLE_XFAIL
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


@_P4_ROLE_XFAIL
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
