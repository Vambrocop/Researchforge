"""Tests for the logistic_regression analysis — catalog, matcher, executor."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from researchforge.catalog import Catalog
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset
from researchforge.recommender import recommend


# ---------------------------------------------------------------------------
# 1. Catalog loads the entry
# ---------------------------------------------------------------------------

def test_catalog_loads_logistic_regression():
    entry = Catalog.load().by_id("logistic_regression")
    assert entry is not None
    assert entry.preconditions.requires_binary_outcome is True
    assert entry.preconditions.min_rows == 30
    assert entry.executor_ref == "empirical-analysis-python"


# ---------------------------------------------------------------------------
# 2. Recommender: feasible/infeasible based on binary column presence
# ---------------------------------------------------------------------------

def _make_binary_csv(tmp_path: Path, n: int = 40) -> Path:
    rng = np.random.default_rng(42)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    p = 1 / (1 + np.exp(-(0.3 + 0.8 * x1 - 0.5 * x2)))
    outcome = rng.binomial(1, p).astype(float)
    df = pd.DataFrame({"outcome": outcome, "x1": x1, "x2": x2})
    csv = tmp_path / "binary_data.csv"
    df.to_csv(csv, index=False)
    return csv


def _make_no_binary_csv(tmp_path: Path, n: int = 40) -> Path:
    rng = np.random.default_rng(7)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    csv = tmp_path / "no_binary.csv"
    df.to_csv(csv, index=False)
    return csv


def test_logistic_feasible_with_binary_column(tmp_path):
    csv = _make_binary_csv(tmp_path)
    fp = profile_dataset(csv)
    by_id = {r.entry.id: r for r in recommend(fp)}
    assert "logistic_regression" in by_id
    assert by_id["logistic_regression"].feasible


def test_logistic_infeasible_without_binary_column(tmp_path):
    csv = _make_no_binary_csv(tmp_path)
    fp = profile_dataset(csv)
    by_id = {r.entry.id: r for r in recommend(fp)}
    assert "logistic_regression" in by_id
    assert not by_id["logistic_regression"].feasible
    assert "需要二值结果变量" in by_id["logistic_regression"].rigor.unmet


# ---------------------------------------------------------------------------
# 3. Executor: run logistic_regression and check outputs
# ---------------------------------------------------------------------------

def _make_logistic_csv(tmp_path: Path, n: int = 80) -> Path:
    rng = np.random.default_rng(99)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    p = 1 / (1 + np.exp(-(0.5 + 1.2 * x1 - 0.8 * x2)))
    outcome = rng.binomial(1, p).astype(int)
    df = pd.DataFrame({"outcome": outcome, "x1": x1, "x2": x2})
    csv = tmp_path / "logistic_data.csv"
    df.to_csv(csv, index=False)
    return csv


def test_executor_logistic_regression_outputs(tmp_path):
    csv = _make_logistic_csv(tmp_path)
    fp = profile_dataset(csv)
    entry = Catalog.load().by_id("logistic_regression")
    assert entry is not None

    res = run_analysis(fp, entry, output_root=str(tmp_path / "outputs"))
    out = Path(res.output_dir)

    assert out.exists()
    assert (out / "report.md").exists()
    assert (out / "coefficients.csv").exists()
    assert res.summary


# ---------------------------------------------------------------------------
# 4. group_comparison — catalog, matcher, executor
# ---------------------------------------------------------------------------

def test_catalog_loads_group_comparison():
    entry = Catalog.load().by_id("group_comparison")
    assert entry is not None
    assert entry.preconditions.requires_group is True
    assert entry.preconditions.min_continuous == 1
    assert entry.preconditions.min_rows == 10
    assert entry.executor_ref == "empirical-analysis-python"


def _make_group_csv(tmp_path: Path, n: int = 60, n_groups: int = 2) -> Path:
    rng = np.random.default_rng(123)
    labels = [chr(65 + i) for i in range(n_groups)]  # "A", "B", ...
    grp = [labels[i % n_groups] for i in range(n)]
    # make groups differ so test is significant
    value = [rng.normal(i * 2.0, 1.0) for i, g in enumerate(grp) for _ in [None]]
    value = [rng.normal((ord(g) - 65) * 2.0, 1.0) for g in grp]
    df = pd.DataFrame({"grp": grp, "value": value})
    csv = tmp_path / f"group_{n_groups}.csv"
    df.to_csv(csv, index=False)
    return csv


def _make_all_continuous_csv(tmp_path: Path, n: int = 40) -> Path:
    rng = np.random.default_rng(77)
    df = pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(0, 1, n),
        "x3": rng.normal(0, 1, n),
    })
    csv = tmp_path / "all_continuous.csv"
    df.to_csv(csv, index=False)
    return csv


def test_group_comparison_feasible_with_group_column(tmp_path):
    csv = _make_group_csv(tmp_path, n_groups=2)
    fp = profile_dataset(csv)
    by_id = {r.entry.id: r for r in recommend(fp)}
    assert "group_comparison" in by_id
    assert by_id["group_comparison"].feasible


def test_group_comparison_infeasible_without_group_column(tmp_path):
    csv = _make_all_continuous_csv(tmp_path)
    fp = profile_dataset(csv)
    by_id = {r.entry.id: r for r in recommend(fp)}
    assert "group_comparison" in by_id
    assert not by_id["group_comparison"].feasible
    assert "需要分组变量（分类/二值）" in by_id["group_comparison"].rigor.unmet


def test_executor_group_comparison_two_groups(tmp_path):
    csv = _make_group_csv(tmp_path, n_groups=2)
    fp = profile_dataset(csv)
    # 2-level string column -> kind "binary"
    assert any(c.kind == "binary" for c in fp.columns), \
        f"Expected binary kind for 2-level group; got {[(c.name, c.kind) for c in fp.columns]}"

    entry = Catalog.load().by_id("group_comparison")
    assert entry is not None

    res = run_analysis(fp, entry, output_root=str(tmp_path / "outputs"))
    out = Path(res.output_dir)

    assert out.exists()
    assert (out / "report.md").exists()
    assert (out / "group_means.csv").exists()
    assert "pvalue" in res.estimates
    assert res.summary


# ---------------------------------------------------------------------------
# 5. group_comparison — P3-4: min-per-group guard + variance-assumption
#    disclosure for the k>=3 (ANOVA) path.
# ---------------------------------------------------------------------------

def _make_group_csv_singleton(tmp_path: Path) -> Path:
    """3 groups, one of which has only n=1 — must not yield a NaN statistic."""
    rng = np.random.default_rng(11)
    grp = ["A"] * 10 + ["B"] * 10 + ["C"] * 1
    value = (
        list(rng.normal(0, 1, 10))
        + list(rng.normal(2, 1, 10))
        + list(rng.normal(4, 1, 1))
    )
    df = pd.DataFrame({"grp": grp, "value": value})
    csv = tmp_path / "group_singleton.csv"
    df.to_csv(csv, index=False)
    return csv


def _make_group_csv_unequal_variance(tmp_path: Path) -> Path:
    """3 groups, clearly unequal variances (std 1, 1, 20) -> Levene should fire."""
    rng = np.random.default_rng(55)
    n_per = 30
    grp = ["A"] * n_per + ["B"] * n_per + ["C"] * n_per
    value = (
        list(rng.normal(0, 1, n_per))
        + list(rng.normal(0, 1, n_per))
        + list(rng.normal(0, 20, n_per))
    )
    df = pd.DataFrame({"grp": grp, "value": value})
    csv = tmp_path / "group_unequal_var.csv"
    df.to_csv(csv, index=False)
    return csv


def test_executor_group_comparison_singleton_group_no_nan(tmp_path):
    csv = _make_group_csv_singleton(tmp_path)
    fp = profile_dataset(csv)
    entry = Catalog.load().by_id("group_comparison")
    assert entry is not None

    res = run_analysis(fp, entry, output_root=str(tmp_path / "outputs"))

    # a group with n=1 must not produce a NaN "result" — honest skip instead.
    assert "statistic" not in res.estimates
    assert "pvalue" not in res.estimates
    assert "失败" in res.summary
    assert "样本量" in res.summary


def test_executor_group_comparison_unequal_variance_disclosure(tmp_path):
    csv = _make_group_csv_unequal_variance(tmp_path)
    fp = profile_dataset(csv)
    entry = Catalog.load().by_id("group_comparison")
    assert entry is not None

    res = run_analysis(fp, entry, output_root=str(tmp_path / "outputs"))

    assert "pvalue" in res.estimates
    assert not np.isnan(res.estimates["pvalue"])
    assert "Levene" in res.summary
    assert "⚠" in res.summary
    assert "不齐" in res.summary


def test_executor_group_comparison_three_groups_balanced_regression(tmp_path):
    """Regression: a normal balanced >=3-group case still yields a valid ANOVA
    F/p (the guard + disclosure must not affect valid inputs)."""
    csv = _make_group_csv(tmp_path, n_groups=3)
    fp = profile_dataset(csv)
    entry = Catalog.load().by_id("group_comparison")
    assert entry is not None

    res = run_analysis(fp, entry, output_root=str(tmp_path / "outputs"))

    assert "statistic" in res.estimates
    assert "pvalue" in res.estimates
    assert not np.isnan(res.estimates["statistic"])
    assert not np.isnan(res.estimates["pvalue"])
    # variance-assumption disclosure is present regardless of Levene outcome
    assert "Levene" in res.summary


# ---------------------------------------------------------------------------
# 6. group_comparison — P3-4b: hand-rolled Welch's ANOVA (used UNCONDITIONALLY
#    on the k>=3 path, replacing the classic equal-variance f_oneway).
# ---------------------------------------------------------------------------

def _hand_welch_anova(groups: list) -> tuple[float, float, float, float]:
    """Test-local, independently-written implementation of Welch's F-test
    (Welch 1951, Satterthwaite-corrected df) — does NOT import or call the
    executor's `_welch_anova` helper, so it serves as a genuine correctness
    cross-check rather than testing the formula against itself."""
    k = len(groups)
    ns = np.array([len(g) for g in groups], dtype=float)
    means = np.array([np.mean(g) for g in groups], dtype=float)
    variances = np.array([np.var(g, ddof=1) for g in groups], dtype=float)
    w = ns / variances
    W = w.sum()
    m_bar = (w * means).sum() / W
    numer = (w * (means - m_bar) ** 2).sum() / (k - 1)
    A = ((1 - w / W) ** 2 / (ns - 1)).sum()
    denom = 1 + (2 * (k - 2) / (k**2 - 1)) * A
    F = numer / denom
    df1 = float(k - 1)
    df2 = (k**2 - 1) / (3 * A)
    p = float(stats.f.sf(F, df1, df2))
    return float(F), p, df1, float(df2)


def _make_group_csv_known(tmp_path: Path) -> Path:
    """3 groups with hand-picked, KNOWN unequal n and variance:
        A: n=4  [10.5,12.5,14.5,16.5]   mean=13.5   var=6.666667
        B: n=6  [1.5,2.5,3.5,4.5,5.5,6.5]  mean=4.0  var=3.5
        C: n=3  [100.5,106.5,112.5]  mean=106.5  var=36
    (values use a +0.5 offset vs. the "obvious" integer picks purely to dodge
    the profiler's all-unique-integers -> "id" kind misclassification — see
    CLAUDE.md's "profiler id 陷阱"; Welch's F/p are translation-invariant so
    this does not change the statistic.)
    Independently verified against statsmodels.stats.oneway.anova_oneway
    (use_var='unequal', welch_correction=True): Welch F=361.95828989622004,
    p=3.128006484127184e-05, df1=2, df2=3.983257770316186 — vs. classic
    f_oneway F=1036.4120126449238 (the low-variance group B dominates the
    classic pooled-variance F but is properly down-weighted by Welch).
    """
    grp = ["A"] * 4 + ["B"] * 6 + ["C"] * 3
    value = [10.5, 12.5, 14.5, 16.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 100.5, 106.5, 112.5]
    df = pd.DataFrame({"grp": grp, "value": value})
    csv = tmp_path / "group_known.csv"
    df.to_csv(csv, index=False)
    return csv


def test_executor_group_comparison_welch_matches_hand_formula(tmp_path):
    """Correctness cross-check: the handler's Welch F/p/df must match an
    independently-written implementation of the same formula to ~1e-6."""
    csv = _make_group_csv_known(tmp_path)
    fp = profile_dataset(csv)
    entry = Catalog.load().by_id("group_comparison")
    assert entry is not None

    res = run_analysis(fp, entry, output_root=str(tmp_path / "outputs"))

    groups = [
        np.array([10.5, 12.5, 14.5, 16.5]),
        np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5]),
        np.array([100.5, 106.5, 112.5]),
    ]
    F, p, df1, df2 = _hand_welch_anova(groups)

    assert res.estimates["statistic"] == pytest.approx(F, abs=1e-6)
    assert res.estimates["pvalue"] == pytest.approx(p, abs=1e-9, rel=1e-6)
    assert res.estimates["welch_df1"] == pytest.approx(df1)
    assert res.estimates["welch_df2"] == pytest.approx(df2, abs=1e-6)

    # sanity: on this heavily unbalanced/heteroscedastic fixture Welch's F is
    # sharply different from the classic equal-variance ANOVA's F.
    classic_stat, _classic_p = stats.f_oneway(*groups)
    assert abs(res.estimates["statistic"] - classic_stat) > 100


def test_executor_group_comparison_welch_diverges_under_heteroscedasticity(tmp_path):
    """On clearly heteroscedastic data (std 1/1/20, unequal n), Welch's F/p
    must differ from the classic equal-variance f_oneway result."""
    csv = _make_group_csv_unequal_variance(tmp_path)
    fp = profile_dataset(csv)
    entry = Catalog.load().by_id("group_comparison")
    assert entry is not None

    res = run_analysis(fp, entry, output_root=str(tmp_path / "outputs"))

    raw = pd.read_csv(csv)
    levels = raw["grp"].unique().tolist()
    groups = [raw.loc[raw["grp"] == lv, "value"].values for lv in levels]
    classic_stat, classic_p = stats.f_oneway(*groups)

    assert not np.isclose(res.estimates["statistic"], classic_stat, rtol=1e-3)
    assert not np.isclose(res.estimates["pvalue"], classic_p, rtol=1e-3)


def test_executor_group_comparison_welch_converges_under_homogeneity(tmp_path):
    """On homoscedastic, balanced data, Welch's F should closely track the
    classic f_oneway F (the two coincide as variances/n become equal). Uses a
    larger balanced sample (n=300, 100/group) so the per-group sample
    variances are close enough for a tight-ish comparison; with only ~20/group
    sampling noise in the per-group variances alone can push the two apart by
    more than a few percent even though both are asymptotically the same."""
    csv = _make_group_csv(tmp_path, n=300, n_groups=3)
    fp = profile_dataset(csv)
    entry = Catalog.load().by_id("group_comparison")
    assert entry is not None

    res = run_analysis(fp, entry, output_root=str(tmp_path / "outputs"))

    raw = pd.read_csv(csv)
    levels = raw["grp"].unique().tolist()
    groups = [raw.loc[raw["grp"] == lv, "value"].values for lv in levels]
    classic_stat, _classic_p = stats.f_oneway(*groups)

    assert res.estimates["statistic"] == pytest.approx(classic_stat, rel=0.1)


def test_executor_group_comparison_welch_disclosure_is_unconditional(tmp_path):
    """The Levene note is now a diagnostic explaining the (unconditional)
    Welch default, not a pre-test switch — check the new framing and that
    the Satterthwaite df are disclosed in estimates."""
    csv = _make_group_csv_unequal_variance(tmp_path)
    fp = profile_dataset(csv)
    entry = Catalog.load().by_id("group_comparison")
    assert entry is not None

    res = run_analysis(fp, entry, output_root=str(tmp_path / "outputs"))

    assert "已默认用 Welch 稳健单因素方差分析" in res.summary
    assert "Levene" in res.summary
    assert "welch_df1" in res.estimates
    assert "welch_df2" in res.estimates

    # same wording/behavior must also hold on homoscedastic data (p>=0.05
    # case) — the default is NOT gated on the Levene result.
    csv2 = _make_group_csv(tmp_path, n_groups=3)
    fp2 = profile_dataset(csv2)
    res2 = run_analysis(fp2, entry, output_root=str(tmp_path / "outputs2"))
    assert "已默认用 Welch 稳健单因素方差分析" in res2.summary


def _make_group_csv_degenerate(tmp_path: Path) -> Path:
    """3 groups, one of which is exactly constant (var=0, n=5) so it passes
    the min-per-group>=2 guard but makes Welch's weight w_i = n_i/v_i
    infinite for that group."""
    rng = np.random.default_rng(3)
    grp = ["A"] * 8 + ["B"] * 8 + ["C"] * 5
    value = (
        list(rng.normal(0, 1, 8))
        + list(rng.normal(3, 1, 8))
        + [7.0] * 5  # constant within group -> var(ddof=1) == 0
    )
    df = pd.DataFrame({"grp": grp, "value": value})
    csv = tmp_path / "group_degenerate.csv"
    df.to_csv(csv, index=False)
    return csv


def test_executor_group_comparison_degenerate_variance_skips_honestly(tmp_path):
    """A constant group (var=0, n>=2) must not yield an inf/NaN Welch
    statistic — the handler should skip with an honest failure message."""
    csv = _make_group_csv_degenerate(tmp_path)
    fp = profile_dataset(csv)
    entry = Catalog.load().by_id("group_comparison")
    assert entry is not None

    res = run_analysis(fp, entry, output_root=str(tmp_path / "outputs"))

    assert "statistic" not in res.estimates
    assert "pvalue" not in res.estimates
    assert "失败" in res.summary
    assert "方差为 0" in res.summary


# ---------------------------------------------------------------------------
# 7. group_comparison — Wave K-B2: block-hint layering must not let a
#    lower-cardinality *block* (RCBD nuisance factor) shadow the real
#    treatment/grouping factor just because it has fewer levels.
# ---------------------------------------------------------------------------

def _make_rcbd_csv(tmp_path: Path) -> Path:
    """Textbook RCBD shape: 4 blocks x 5 treatments x 3 reps. Block has FEWER
    levels (4) than treatment (5) — the old `sort by nunique ascending` picked
    block first, which is backwards (docs/dogfood-findings.md #12)."""
    rng = np.random.default_rng(9)
    blocks = [f"B{i}" for i in range(4)]
    trts = [f"T{i}" for i in range(5)]
    rows = []
    for b in blocks:
        for i, t in enumerate(trts):
            for _ in range(3):
                rows.append({
                    "block": b,
                    "trt": t,
                    "yield": 10.0 + i * 2.0 + rng.normal(0, 0.3),
                })
    df = pd.DataFrame(rows)
    csv = tmp_path / "rcbd.csv"
    df.to_csv(csv, index=False)
    return csv


def test_group_comparison_prefers_treatment_over_lower_card_block(tmp_path):
    csv = _make_rcbd_csv(tmp_path)
    fp = profile_dataset(csv)
    entry = Catalog.load().by_id("group_comparison")
    assert entry is not None

    res = run_analysis(fp, entry, output_root=str(tmp_path / "outputs"))
    gm = pd.read_csv(Path(res.output_dir) / "group_means.csv")

    # must group by 'trt' (5 levels), NOT 'block' (4 levels, fewer -> would have
    # won under the old nunique-ascending sort).
    assert gm.columns[0] == "trt", f"expected grouped by trt, got columns {list(gm.columns)}"
    assert len(gm) == 5
    assert "⚠ 自动选分组=trt" in res.summary


def test_group_comparison_config_group_override(tmp_path):
    """config["group"] must win over the auto heuristic even when it points at
    a block-hint-named column."""
    csv = _make_rcbd_csv(tmp_path)
    fp = profile_dataset(csv)
    entry = Catalog.load().by_id("group_comparison")
    assert entry is not None

    res = run_analysis(
        fp, entry, output_root=str(tmp_path / "outputs2"), config={"group": "block"}
    )
    gm = pd.read_csv(Path(res.output_dir) / "group_means.csv")

    assert gm.columns[0] == "block"
    assert len(gm) == 4
    # explicit override -> no "auto-picked" disclosure
    assert "⚠ 自动选分组" not in res.summary


# ---------------------------------------------------------------------------
# 8. group_comparison — Wave K-E5: on a significant difference, the summary
#    must spell out a plain-language conclusion (highest group, % vs lowest,
#    p-value), not just raw stat/p numbers.
# ---------------------------------------------------------------------------

def test_group_comparison_significant_synthesizes_human_conclusion(tmp_path):
    csv = _make_group_csv(tmp_path, n_groups=3)  # A mean~0, B mean~2, C mean~4
    fp = profile_dataset(csv)
    entry = Catalog.load().by_id("group_comparison")
    assert entry is not None

    res = run_analysis(fp, entry, output_root=str(tmp_path / "outputs"))

    assert "最高" in res.summary
    assert "显著" in res.summary
    assert "C" in res.summary  # C has the highest mean in this fixture


def test_group_comparison_not_significant_skips_human_conclusion(tmp_path):
    """No synthesized "X 组均值最高...显著" sentence when the difference is not
    significant — the sentence must be conditional on p<0.05, not unconditional."""
    rng = np.random.default_rng(321)
    n = 30
    grp = ["A"] * n + ["B"] * n
    value = list(rng.normal(0, 1, n)) + list(rng.normal(0, 1, n))  # same mean -> not significant
    df = pd.DataFrame({"grp": grp, "value": value})
    csv = tmp_path / "group_ns.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    entry = Catalog.load().by_id("group_comparison")
    assert entry is not None

    res = run_analysis(fp, entry, output_root=str(tmp_path / "outputs"))

    assert res.estimates.get("pvalue", 0) >= 0.05, (
        f"fixture unexpectedly significant, cannot test the negative case: {res.estimates}"
    )
    assert "均值最高" not in res.summary


# ---------------------------------------------------------------------------
# 9. logistic_regression — Wave K-B4b: Chinese (non-formula-identifier-safe)
#    column names must not crash the formula, and coefficients.csv/summary
#    must show the ORIGINAL Chinese names, not the internal v1/v2/... aliases
#    that safe_formula_terms() uses to keep patsy happy.
# ---------------------------------------------------------------------------

def _make_chinese_logistic_csv(tmp_path: Path, n: int = 100) -> Path:
    rng = np.random.default_rng(2024)
    pred1 = rng.normal(0, 1, n)
    pred2 = rng.normal(0, 1, n)
    p = 1 / (1 + np.exp(-(0.4 + 1.0 * pred1 - 0.6 * pred2)))
    outcome = rng.binomial(1, p).astype(int)
    df = pd.DataFrame({"结果": outcome, "预测1": pred1, "预测2": pred2})
    csv = tmp_path / "chinese_logistic.csv"
    df.to_csv(csv, index=False)
    return csv


def test_logistic_regression_chinese_columns_do_not_crash(tmp_path):
    csv = _make_chinese_logistic_csv(tmp_path)
    fp = profile_dataset(csv)
    entry = Catalog.load().by_id("logistic_regression")
    assert entry is not None

    res = run_analysis(fp, entry, output_root=str(tmp_path / "outputs"))

    assert "未收敛/失败" not in res.summary, f"crashed on Chinese columns: {res.summary!r}"
    coefs = pd.read_csv(Path(res.output_dir) / "coefficients.csv", index_col=0)
    # original Chinese names must appear in the coefficients table, NOT the
    # internal v1/v2/... formula-safety aliases.
    assert "预测1" in coefs.index
    assert "预测2" in coefs.index
    assert not any(str(i).startswith("v") and str(i)[1:].isdigit() for i in coefs.index)
    assert "结果变量 结果" in res.summary
    assert "预测1" in res.estimates and "预测2" in res.estimates


# ---------------------------------------------------------------------------
# 10. logistic_regression — Wave K-E1 (odds ratios) + E3 (binary predictors
#     included by default), on the P2 epidemiology cohort fixture
#     (disease ~ smoking + age + sex + bmi, true smoking OR = 2.0).
# ---------------------------------------------------------------------------

def test_logistic_regression_p2_cohort_or_and_binary_predictors(tmp_path):
    from fixtures.dogfood import build_p2_cohort

    df = build_p2_cohort()
    csv = tmp_path / "p2_cohort.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    entry = Catalog.load().by_id("logistic_regression")
    assert entry is not None

    res = run_analysis(fp, entry, output_root=str(tmp_path / "outputs"))

    assert "结果变量 disease" in res.summary, f"unexpected outcome pick: {res.summary!r}"
    # E3: smoking (binary) and sex (binary) must be included as predictors —
    # previously only continuous columns (age, bmi) were auto-selected.
    assert "smoking" in res.estimates, f"smoking missing from predictors: {sorted(res.estimates)}"
    assert "sex" in res.estimates, f"sex missing from predictors: {sorted(res.estimates)}"

    # E1: OR = exp(coef); CI endpoints = exp(conf_int) (never exp(point ± se)).
    assert "smoking_OR" in res.estimates
    assert res.estimates["smoking_OR"] == pytest.approx(np.exp(res.estimates["smoking"]))

    coefs = pd.read_csv(Path(res.output_dir) / "coefficients.csv", index_col=0)
    assert {"OR", "OR_CI_low", "OR_CI_high"} <= set(coefs.columns)
    row = coefs.loc["smoking"]
    assert row["OR"] == pytest.approx(np.exp(row["Coef."]))
    assert row["OR_CI_low"] == pytest.approx(np.exp(row["[0.025"]))
    assert row["OR_CI_high"] == pytest.approx(np.exp(row["0.975]"]))
