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
@pytest.mark.xfail(
    reason="Wave K-A2 后已重归属(Fable A2 验)：count 模型确已退场，但 P2 top-6 现被因果族"
    "(disease/smoking 被读成 treatment-outcome 的 double_ml/causal_forest/psm/ipw)+tweedie 压住，"
    "logistic/epi 仍未进 top-6。真凶=二值结局 cohort 上因果族过度排名，待后续排名/affinity 批降权(非 A2)",
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
