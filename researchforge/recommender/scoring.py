"""Methodology score card — a multi-dimensional read on a recommended method.

Offline + deterministic: dimensions are derived from a rule/metadata rubric
(by family, with per-id overrides) plus the data-specific rigor verdict. No
network. The self-evolution trend engine (later phase) can refine the popularity
and novelty dimensions from live CRAN/PyPI/GitHub/literature signals; until then
these are static editorial priors, surfaced honestly as such.

Dimensions (0-100, higher = more of that attribute):
  popularity     流行   how widely used the method is
  publishability 可发表 how much it supports high-impact publication
  aesthetics     美观   strength of its signature figures/visual output
  difficulty     难度   interpretation / assumption burden (a COST, not a good)
  fit            契合   how well it fits THIS dataset (= the rigor score)
  novelty        新颖   how fresh / trendy the method currently is
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import BaseModel

from researchforge.catalog.schema import AnalysisEntry
from researchforge.profiler.fingerprint import DataFingerprint
from researchforge.recommender.affinity import data_signals, get_affinity, match_score
from researchforge.recommender.rigor import RigorVerdict

# base scores by family: (popularity, publishability, aesthetics, difficulty, novelty).
# Editorial priors (subjective, disclosed as such) — keys MUST match catalog family
# strings exactly (test_scoring guards full coverage, so a family never silently falls
# back to _DEFAULT). Populated for all 45 catalog families.
_FAMILY: dict[str, tuple[int, int, int, int, int]] = {
    "statistics": (85, 55, 45, 35, 25),
    "causal": (70, 88, 68, 70, 70),
    "sem": (65, 82, 80, 75, 55),
    "meta": (60, 80, 76, 50, 50),
    "ml": (80, 65, 70, 60, 65),
    "time-series": (70, 62, 66, 62, 42),
    "spatial": (55, 74, 88, 66, 62),
    "ecology": (60, 66, 80, 55, 46),
    "mcda": (60, 46, 60, 40, 46),
    "efficiency": (52, 72, 60, 72, 56),
    "econometrics": (66, 82, 55, 78, 56),
    "configurational": (42, 70, 62, 70, 76),
    "soil": (46, 42, 70, 32, 36),
    # — filled in so the scorecard is meaningful for every family (was _DEFAULT before) —
    "bayesian": (62, 80, 72, 78, 66),
    "survival": (72, 80, 80, 62, 48),
    "regression": (60, 45, 42, 32, 22),
    "conditional_process": (66, 74, 66, 68, 58),
    "irt": (55, 76, 66, 74, 52),
    "psychometrics": (58, 66, 60, 55, 40),
    "latent_class": (52, 74, 70, 76, 62),
    "mixture": (55, 66, 72, 68, 56),
    "nonparametric": (72, 48, 44, 32, 28),
    "categorical": (70, 50, 48, 36, 26),
    "categorical_tests": (72, 48, 45, 32, 24),
    "distribution": (58, 46, 56, 42, 32),
    "distribution_extra": (52, 48, 58, 46, 42),
    "effect_sizes": (66, 62, 50, 36, 34),
    "epidemiology": (64, 78, 68, 55, 50),
    "finance": (60, 60, 66, 62, 52),
    "hydrology": (44, 60, 66, 58, 46),
    "marketing": (58, 50, 62, 45, 50),
    "actuarial": (46, 58, 55, 66, 44),
    "operations_research": (56, 52, 56, 55, 40),
    "game_theory": (48, 62, 58, 70, 58),
    "reliability": (50, 62, 60, 62, 46),
    "spc": (58, 55, 66, 48, 36),
    "survey_methods": (60, 66, 50, 60, 44),
    "nlp": (74, 68, 66, 62, 68),
    "policy": (56, 66, 78, 60, 72),
    "choice": (58, 74, 60, 72, 52),
    "missing_data": (58, 66, 52, 64, 54),
    "resource": (50, 60, 66, 52, 56),
    "techno_economic": (52, 58, 62, 50, 50),
    "experimental_design": (60, 66, 58, 48, 40),
    "experimental_stats": (64, 62, 55, 50, 38),
    "agreement": (58, 60, 55, 42, 34),
}
_DEFAULT = (50, 56, 56, 55, 50)


@lru_cache(maxsize=1)
def _trend_snapshot() -> Optional[dict]:
    """Process-cached momentum snapshot (written by `cli discover --live`). Read once
    per process — a refreshed snapshot is picked up on the next run. Hot-path safe:
    pure file read, never network. Returns None when no fresh snapshot exists."""
    try:
        from researchforge.catalog.trends import load_snapshot

        return load_snapshot()
    except Exception:
        return None


def _live_momentum(entry_id: str, family: str) -> Optional[int]:
    """Real PyPI/GitHub/CRAN momentum for this method from the cached snapshot:
    per-id if known, else the per-family mean. None when no live signal exists."""
    snap = _trend_snapshot()
    if not snap:
        return None
    by_id = snap.get("by_id", {})
    if entry_id in by_id:
        return int(by_id[entry_id])
    fam = snap.get("by_family", {})
    if family in fam:
        return int(fam[family])
    return None

# per-id overrides (only the dimensions worth nudging from the family base)
_ID: dict[str, dict[str, int]] = {
    "synthetic_control": {"novelty": 88, "publishability": 88, "aesthetics": 80},
    "did": {"publishability": 90, "aesthetics": 78, "popularity": 75},
    "gam": {"novelty": 70, "aesthetics": 82, "publishability": 76},
    "glmm": {"publishability": 80, "difficulty": 78},
    "meta_analysis": {"aesthetics": 80, "publishability": 82},
    "dynamic_panel_gmm": {"difficulty": 90, "publishability": 84},
    "fsqca": {"novelty": 80}, "qca_necessity": {"novelty": 80}, "csqca": {"novelty": 76},
    "nca": {"novelty": 82},
    "spatial_regression": {"aesthetics": 88, "publishability": 78},
    "survival_analysis": {"aesthetics": 82, "publishability": 78},
    "sfa": {"difficulty": 82, "publishability": 76},
    "descriptive_stats": {"publishability": 30, "novelty": 12, "aesthetics": 35},
    "correlation": {"publishability": 35, "novelty": 15},
}


# specific (non-generic) precondition flag -> (data signal that satisfies it, weight). A
# method whose SPECIFIC precondition matches this data's structure is tailored to it, so
# it earns a fit bonus over generic methods in the same family (this is what lifts logistic
# on binary data, network methods on an edge list, spatial on geo, … out of the
# ols/random_forest/descriptive soup). requires_edgelist is weighted highest because there
# is no "network" family to grant a structure bonus, so this stands in for it.
_SPECIFIC_PRECOND = {
    "requires_edgelist": ("has_edgelist", 14.0),
    "is_panel": ("is_panel", 16.0),
    "is_timeseries": ("is_timeseries", 16.0),
    "requires_count_outcome": ("has_count_outcome", 14.0),
    # ordinal outcome (bounded 1..k rating): a rating scale profiles as `count`, so ordinal
    # regression (proportional-odds / ordered-probit) would otherwise tie count models and get
    # buried. Weighted just above count so on a genuine rating it edges out Poisson/NB — which
    # stay feasible (not wrong, just less ideal). Gated on has_ordinal_OUTCOME (1–2 rating cols),
    # so a ≥3-rater block goes to agreement methods, not ordinal regression.
    "requires_ordinal": ("has_ordinal_outcome", 15.0),
    "requires_geo": ("has_geo", 12.0),
    # gated on outcome_is_binary (not raw has_binary): a binary DESIGN FACTOR alongside a
    # continuous outcome must not lift logistic/epi over the ANOVA/regression modeling the
    # response. Fires when the binary is the role-detected outcome, or the table is pure-binary.
    "requires_binary_outcome": ("outcome_is_binary", 12.0),
    "requires_treatment": ("has_treatment", 12.0),
}

# Structure-defining preconditions: when the data has this structure, a method that
# requires it is clearly appropriate REGARDLESS of its (possibly heterogeneous) family —
# e.g. network methods live in family "ml" alongside random forests but don't need ml's
# outcome/predictors, so they must not eat the ml family penalty on an edge list. When
# one of these is met we floor the family base at a structure-match level.
_STRUCTURE_PRECOND = {
    "requires_edgelist": "has_edgelist",
    "requires_geo": "has_geo",
    "is_panel": "is_panel",
    "is_timeseries": "is_timeseries",
}
_STRUCTURE_FLOOR = 72.0

# A ≥3-parallel-ordinal-column block is a data STRUCTURE owned by the reliability families —
# just as an edge list is owned by network methods — so their members are floored above the
# generic multi-numeric methods the ratings would otherwise attract. The block SPLITS (see
# affinity.data_signals): PEOPLE raters (has_rater_block) → agreement (κ/ICC) AND
# psychometrics; SCALE ITEMS (has_scale_items) → psychometrics only (α/ω/EFA), NOT κ. Applied
# in _affinity_fit.
_RATER_FLOOR = 85.0


# ── small-data model-tier tilt (Wave S) ──────────────────────────────────────────────
# The "由简到繁 / start simple" philosophy for small data: with few rows (or few rows per
# predictor), low-capacity, regularized and prior-regularized methods GENERALIZE better,
# while data-hungry flexible learners OVERFIT. So on a small-data regime we tilt the
# data-fit score toward the simple end of the ladder and disclose the overfit risk — this
# is auto-selection encoding the model-capacity ladder, not just a bias-count.
#
# Data-hungry / high-variance learners: demoted on small data (+ an overfit ⚠).
_HIGH_CAPACITY = {
    "random_forest", "gradient_boosting", "xgboost", "bart",
    "gaussian_process_regression", "surrogate_model", "explainable_boosting",
    "svm_model",  # kernel SVM overfits small n without enough support-vector coverage
}
# Small-data-friendly: regularized / low-variance / constraint- or prior-regularized → boosted.
_SMALL_DATA_FRIENDLY = {
    "regularized_regression", "naive_bayes", "robust_regression",
    "ols_regression", "logistic_regression", "bootstrap_ci", "monotonic_constraints",
}
# Families whose members regularize via priors (small-data friendly by construction).
_FRIENDLY_FAMILIES = {"bayesian"}


def _small_data_tilt(entry: AnalysisEntry, signals: dict) -> tuple[float, str]:
    """(data-fit delta, disclosure note) for the small-data model-tier. Fires on a
    small-data regime — few rows (n < ~100) OR few rows per predictor (n/p < ~10, the real
    overfit measure). Demotes data-hungry learners (with an overfit ⚠), boosts
    regularized / Bayesian ones. Returns (0.0, "") outside the small-data regime."""
    n = int(signals.get("n_rows", 0) or 0)
    if n <= 0:
        return 0.0, ""
    p = max(1, int(signals.get("n_numeric", 0) or 0))
    ratio = n / p
    sev_n = min(1.0, (100 - n) / 70.0) if n < 100 else 0.0
    sev_r = min(1.0, (10 - ratio) / 8.0) if ratio < 10 else 0.0
    sev = max(0.0, sev_n, sev_r)
    if sev <= 0.0:
        return 0.0, ""
    if entry.id in _HIGH_CAPACITY:
        return -14.0 * sev, (
            f"⚠ 小数据（n={n}，约 {ratio:.0f} 行/预测变量）：{entry.method} 容量高、易过拟合——"
            "小数据首选正则线性/贝叶斯；若用树/集成，限深(max_depth≤3)并 bootstrap 验稳。"
        )
    if entry.id in _SMALL_DATA_FRIENDLY or entry.family in _FRIENDLY_FAMILIES:
        return 8.0 * sev, ""
    return 0.0, ""


# ── classification-target-first tilt (Wave M5) ───────────────────────────────────────
# When the data's natural outcome is a MULTICLASS categorical TARGET and there is no
# trustworthy continuous outcome (signals["has_class_target"]), the continuous columns are
# FEATURES. The analyses that answer the data's real question MODEL / DISCRIMINATE that
# target; a regression modeling an arbitrary feature is semantically weak (dogfood: on wine
# — target=cultivar, 13 continuous features — mixed_effects/robust_regression modeled
# `alcohol`, and the study's ≤1-per-family filter buried discriminant/manova behind them).
# So on such data we boost the target-modeling methods and mildly demote feature-only
# regressions. Tightly gated on has_class_target (class-label name + no high/med-confidence
# continuous outcome) so it cannot misfire on ordinary regression data that merely carries a
# category column. Binary targets are handled elsewhere (outcome_is_binary) — this is the
# multiclass gap.
#
# Model / discriminate the categorical target directly (features → target): boosted.
_CLASSIFY_TARGET = {
    "discriminant_analysis", "linear_discriminant", "naive_bayes",
    "multinomial_logistic", "manova", "hotelling_t2",
}
# Model a continuous FEATURE as outcome, ignoring / nuisance-ing the categorical target:
# demoted (on a classification table this analysis answers a question nobody asked).
_FEATURE_MODEL = {
    "mixed_effects", "glmm", "quantile_regression", "robust_regression",
    "influence_diagnostics",
}
_FEATURE_MODEL_FAMILIES = {"regression"}


def _class_target_tilt(entry: AnalysisEntry, signals: dict) -> tuple[float, str]:
    """(data-fit delta, disclosure note) for the classification-target-first tilt. Fires
    only when the data has a genuine multiclass categorical target and no trustworthy
    continuous outcome (signals["has_class_target"]). Boosts methods that model/discriminate
    the target, mildly demotes feature-only regressions, and gives a small lift to
    across-group comparisons (which at least use the target as a grouping). Returns (0.0, "")
    off a class-target regime."""
    if not signals.get("has_class_target"):
        return 0.0, ""
    if entry.id in _CLASSIFY_TARGET:
        return 12.0, ""
    if entry.id in _FEATURE_MODEL or entry.family in _FEATURE_MODEL_FAMILIES:
        return -8.0, (
            "⚠ 本数据的天然结果是多类别目标（分类），本法却建模某连续特征、未以目标为核心——"
            "语义偏弱；分类/判别（discriminant/naive_bayes/MANOVA）更契合该数据。"
        )
    # compares a continuous feature ACROSS the target's groups (ANOVA/Kruskal-style) — not a
    # classifier, but it does center the target, so a small lift over feature-only regressions.
    if entry.preconditions.requires_group:
        return 5.0, ""
    return 0.0, ""


# ── finance-relevance tilt (Wave M7) ─────────────────────────────────────────────────
# The finance family (value_at_risk / extreme_value / risk_adjusted_return) has the same
# timeseries affinity as the generic time-series family, so on ANY continuous series it ties
# ARIMA/ETS at the ceiling — surfacing "Value at Risk" as a headline analysis on a plain
# monthly SALES series (dogfood: sales_ts). These methods presuppose a security's return
# series; without a financial signal (a return/close/stock/portfolio-named column) they are
# off-target. So absent that signal we demote the finance family below generic forecasting.
# They stay FEASIBLE (still run if the user picks them, or sets the branch's is_returns) — the
# tilt only reorders, and the disclosure says why.
_FINANCE_DEMOTE = -18.0


def _finance_relevance_tilt(entry: AnalysisEntry, signals: dict) -> tuple[float, str]:
    """(data-fit delta, disclosure note) for the finance-relevance tilt: demote finance-family
    methods on a series with no financial-asset signal, so a plain sales/temperature series
    gets generic forecasting (ARIMA/ETS) as its headline, not VaR/EVT/Sharpe. Returns (0.0,"")
    for non-finance families or when a finance signal is present."""
    if entry.family != "finance" or signals.get("has_finance_signal"):
        return 0.0, ""
    return _FINANCE_DEMOTE, (
        "⚠ 未检测到金融资产信号（收益/收盘价/股票/组合类列名）——本法（VaR/极值/风险调整收益）"
        "假定的是证券收益序列，用在一般时序（如销量/温度）上语义偏弱；已降到通用预测法之下。"
        "若这确是金融序列，可直接选它并按需设 config['is_returns']。"
    )


def small_data_advisory(fp: DataFingerprint) -> str:
    """A once-per-dataset 由简到繁 (start-simple) guidance card when the data is small
    (Wave S). Empty string on ample data. Surfaced by the CLI recommend/pick output so the
    small-data model-tier philosophy is VISIBLE, not just baked into the ranking."""
    signals = data_signals(fp)
    n = int(signals.get("n_rows", 0) or 0)
    if n <= 0:
        return ""
    p = max(1, int(signals.get("n_numeric", 0) or 0))
    ratio = n / p
    if n >= 100 and ratio >= 10:
        return ""
    return (
        f"📐 小数据（n={n}，约 {ratio:.0f} 行/预测变量）——建议由简到繁选模：\n"
        "   ① 线性/逻辑回归（配强正则）→ ② 朴素贝叶斯 → ③ SVM → ④ 受限/单调约束树 "
        "→ ⑤ 贝叶斯方法（注入先验、量化不确定性）。\n"
        "   引擎已把数据饥渴的集成（RF/GBM/xgboost）在小数据上相应降权；"
        "若坚持用会自动限深，并请用 bootstrap 验稳。"
    )


def _precond_bonus(signals: dict, pre) -> float:
    """Per-method tailoring bonus (0–30): reward a method whose specific precondition
    matches this data's special structure."""
    pm = pre.model_dump()
    bonus = sum(w for flag, (sig, w) in _SPECIFIC_PRECOND.items()
                if pm.get(flag) and signals.get(sig))
    # requires_group fires on a GENUINE group (a binary/categorical that isn't the
    # role-detected outcome). With binary-outcome role detection, this no longer mis-fires
    # for binary-OUTCOME regression data (where logistic, not a 2-arm test, is right), so
    # it gets its full weight again.
    if pm.get("requires_group") and signals.get("has_group"):
        bonus += 10.0
    if pm.get("min_count_cols") and signals["has_count_outcome"]:
        bonus += 8.0
    if pm.get("min_categorical_cols") and signals["has_categorical"]:
        bonus += 6.0
    return min(bonus, 30.0)


def _affinity_fit(
    fp: DataFingerprint, entry: AnalysisEntry, rigor: RigorVerdict, signals: dict | None = None
) -> int:
    """Real data-fit (0–100): how well this method suits THIS dataset = family
    structure/outcome affinity (affinity.match_score) + per-method precondition
    tailoring + the small-data model-tier tilt. Replaces the old fit = rigor.score (which
    was just bias-count). An infeasible (red) method can't be a good fit no matter its
    affinity, so it stays capped at its (low) rigor score; feasible methods are ranked by
    affinity. `signals` may be passed in to avoid recomputing data_signals per call."""
    if signals is None:
        signals = data_signals(fp)
    base = match_score(signals, get_affinity(entry.family))
    pm = entry.preconditions.model_dump()
    if any(pm.get(flag) and signals.get(sig) for flag, sig in _STRUCTURE_PRECOND.items()):
        base = max(base, _STRUCTURE_FLOOR)
    # People raters → agreement (κ/ICC) AND psychometrics (α as inter-rater consistency);
    # scale items → psychometrics only (α/ω/EFA), NOT agreement (there is no rater to agree).
    if entry.family == "agreement" and signals.get("has_rater_block"):
        base = max(base, _RATER_FLOOR)
    if entry.family == "psychometrics" and (
        signals.get("has_rater_block") or signals.get("has_scale_items")
    ):
        base = max(base, _RATER_FLOOR)
    raw = min(100.0, base + _precond_bonus(signals, entry.preconditions))
    raw = max(0.0, min(100.0, raw + _small_data_tilt(entry, signals)[0]))
    raw = max(0.0, min(100.0, raw + _class_target_tilt(entry, signals)[0]))
    raw = max(0.0, min(100.0, raw + _finance_relevance_tilt(entry, signals)[0]))
    if rigor.light == "red":
        return max(0, min(int(round(rigor.score)), int(round(raw))))
    return int(round(raw))


class MethodologyScore(BaseModel):
    popularity: int
    publishability: int
    aesthetics: int
    difficulty: int
    fit: int
    novelty: int
    overall: int  # display blend (excludes difficulty, which is a cost)
    note: str = ""

    def as_dict(self) -> dict[str, int]:
        return {
            "popularity": self.popularity,
            "publishability": self.publishability,
            "aesthetics": self.aesthetics,
            "difficulty": self.difficulty,
            "fit": self.fit,
            "novelty": self.novelty,
            "overall": self.overall,
        }


def score_method(
    fp: DataFingerprint, entry: AnalysisEntry, rigor: RigorVerdict
) -> MethodologyScore:
    """Compute the methodology score card for a data/method pairing. `fit` comes
    from the data-specific rigor score; the rest from the offline rubric."""
    pop, pub, aes, diff, nov = _FAMILY.get(entry.family, _DEFAULT)
    for k, v in _ID.get(entry.id, {}).items():
        if k == "popularity":
            pop = v
        elif k == "publishability":
            pub = v
        elif k == "aesthetics":
            aes = v
        elif k == "difficulty":
            diff = v
        elif k == "novelty":
            nov = v

    # Live trend feed (phase 2): blend real PyPI/GitHub/CRAN momentum from the cached
    # snapshot into popularity when available. Hot-path safe (file read, no network).
    mom = _live_momentum(entry.id, entry.family)
    if mom is not None:
        pop = int(round(0.5 * pop + 0.5 * mom))
        trend_note = "流行含实时趋势（PyPI/GitHub/CRAN 动量，快照缓存）"
    else:
        trend_note = "流行·新颖为离线编辑先验，趋势引擎接入后将动态更新"

    signals = data_signals(fp)
    fit = _affinity_fit(fp, entry, rigor, signals)
    # overall display blend — fit and publishability weighted most; difficulty is a
    # cost and deliberately excluded (shown separately).
    overall = round(0.35 * fit + 0.25 * pub + 0.15 * pop + 0.15 * nov + 0.10 * aes)
    sd_note = _small_data_tilt(entry, signals)[1]  # overfit disclosure on small data
    ct_note = _class_target_tilt(entry, signals)[1]  # class-target semantic disclosure
    fin_note = _finance_relevance_tilt(entry, signals)[1]  # finance-relevance disclosure
    extra = " ".join(n for n in (sd_note, ct_note, fin_note) if n)
    note = (
        f"契合 {fit}（本数据）/ 流行 {pop} / 可发表 {pub} / 美观 {aes} / 新颖 {nov} / "
        f"难度 {diff}（越高越难）。{trend_note}。{(' ' + extra) if extra else ''}"
    )
    return MethodologyScore(
        popularity=pop, publishability=pub, aesthetics=aes, difficulty=diff,
        fit=fit, novelty=nov, overall=int(overall), note=note,
    )
