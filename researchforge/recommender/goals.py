"""Goal taxonomy for the fast method selector.

With 75+ methods, listing every feasible one is overwhelming ("看了还是不知道选哪个").
This maps a user's research GOAL to the catalog families / ids / keywords it implies,
so `recommend --goal X` surfaces the right handful. Tuned for the focus domains
(ecology / agronomy / economics / environmental / social science).
"""

from __future__ import annotations

from researchforge.catalog.schema import AnalysisEntry

# goal key -> label, matching families, explicit ids, and keyword hints (method/description/domain)
GOALS: dict[str, dict] = {
    "compare": {"label": "比较组间差异 (ANOVA/t/设计)", "families": {"experimental_design"},
                "ids": {"group_comparison"}, "kw": ("anova", "compare")},
    "relate": {"label": "找关系 / 回归", "families": {"econometrics"},
               "ids": {"correlation", "ols_regression", "logistic_regression", "poisson_regression",
                       "negative_binomial_regression", "quantile_regression", "gam", "gamm", "glmm",
                       "mixed_effects", "multinomial_logit", "ordered_logit", "sem", "pls_sem", "efa"},
               "kw": ("regression", "correlation")},
    "causal": {"label": "因果效应", "families": {"causal"}, "ids": set(),
               "kw": ("causal", "treatment effect", "difference-in-differences", "instrument")},
    "predict": {"label": "预测", "families": set(),
                "ids": {"random_forest", "xgboost", "bart", "conformal_prediction"},
                "kw": ("predict", "forecast")},
    "reduce": {"label": "降维 / 聚类", "families": set(),
               "ids": {"pca", "efa", "kmeans_clustering", "hierarchical_clustering", "nmds"},
               "kw": ("cluster", "dimension", "ordination")},
    "design": {"label": "设计实验 / 样本量", "families": {"experimental_design"}, "ids": set(),
               "kw": ("design", "power", "sample size")},
    "spatial": {"label": "空间分析", "families": {"spatial"}, "ids": set(),
                "kw": ("spatial", "kriging", "moran")},
    "survival": {"label": "生存 / 事件史", "families": {"survival"}, "ids": set(),
                 "kw": ("survival", "hazard")},
    "timeseries": {"label": "时间序列", "families": {"time-series"}, "ids": set(),
                   "kw": ("time series", "arima", "autoregress")},
    "qca": {"label": "定性比较 / 配置", "families": {"configurational"}, "ids": set(),
            "kw": ("qca", "configuration", "necessary condition")},
    "efficiency": {"label": "效率 / 生产率", "families": {"efficiency"}, "ids": set(),
                   "kw": ("efficiency", "frontier", "envelopment")},
    "evaluate": {"label": "综合评价 / 排序", "families": {"mcda"}, "ids": set(),
                 "kw": ("mcda", "topsis", "ranking", "criteria")},
    "meta": {"label": "元分析", "families": {"meta"}, "ids": set(),
             "kw": ("meta-analysis", "meta analysis", "meta-regression")},
    "diversity": {"label": "生态多样性 / 丰度", "families": {"ecology"}, "ids": set(),
                  "kw": ("diversity", "abundance", "richness")},
}


def resolve_goal(text: str | None) -> str | None:
    """Map a key or free phrase to a goal key (exact key, else label/keyword match)."""
    if not text:
        return None
    t = text.strip().lower()
    if t in GOALS:
        return t
    for key, g in GOALS.items():
        if t in g["label"].lower() or any(t in kw or kw in t for kw in g["kw"]):
            return key
    return None


def entry_matches_goal(entry: AnalysisEntry, goal_key: str) -> bool:
    g = GOALS.get(goal_key)
    if not g:
        return True
    if entry.family in g["families"] or entry.id in g["ids"]:
        return True
    # Trust the catalog's own goal field where the value domains actually align: entry.goal
    # is describe/explain/predict (catalog/schema.py), which only overlaps the user-facing
    # goal keys at "predict" — so this only widens the predict route (dogfood finding P5/F2).
    if goal_key == "predict" and getattr(entry, "goal", None) == "predict":
        return True
    hay = f"{entry.method} {entry.description} {entry.domain}".lower()
    return any(kw in hay for kw in g["kw"])


# Treatment/block vocabulary that marks a DESIGNED experiment (vs. observational data).
# A coarse bilingual subset of experimental_design/_shared {_BLOCK_HINTS, _TRT_HINTS}, kept
# recommender-local on purpose: importing the executor.branches package here would drag in
# ~130 branch modules (walk_packages registration) on the recommendation hot path. Wave L
# (ColumnSemantics) should relocate this vocabulary to a neutral column-semantics home so the
# recommender and the executor share one definition. Keep in sync with _shared until then.
_DESIGN_SIGNAL_HINTS = (
    "treat", "trt", "block", "rep", "replicate", "plot", "variety", "cultivar",
    "genotype", "hybrid", "dose",
    "处理", "区组", "重复", "组块", "品种", "剂量", "水平", "施肥", "组别",
)


def has_design_signal(fp) -> bool:
    """True when a column name signals a designed experiment (a treatment or block factor).
    Distinguishes a real DoE (RCBD/factorial/split-plot…) from observational data that merely
    has categorical groups — used to stop designed-experiment methods from crowding out the
    naive group comparison under ``--goal compare`` on observational data (Wave K-C1, 发现16)."""
    names = [str(c.name).lower() for c in fp.columns]
    return any(h in nm for nm in names for h in _DESIGN_SIGNAL_HINTS)
