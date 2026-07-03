"""Family-level data-affinity profiles — Stage 2 of smarter auto-selection.

The recommender's old `fit` was just the rigor score (≈ how few biases the catalog
author wrote), so within a rigor tier the *appropriate* method for the data's
structure was often buried under generic ones (ols / random_forest / descriptive).

This module defines, for each method FAMILY, an "ideal data profile": the data
structure it wants (panel / time-series / geo / cross-section), the outcome kinds it
models, whether it needs predictors, and a sensible minimum sample size. `data_signals`
extracts those structural signals from a DataFingerprint, and `match_score` grades how
well a family suits THIS data (0–100).

Family granularity is deliberate (45 families, not 294 methods): structure/outcome
affinity is a family property. Per-method tailoring (e.g. an edge-list method vs a
random forest, both family "ml") is layered on in Stage 3 via the method's own
preconditions (precondition-specificity bonus), not here.

Pure, offline, deterministic; no network, no new deps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from researchforge.profiler.fingerprint import DataFingerprint

# outcome-kind vocabulary a family may target
_OUTCOMES = {"continuous", "count", "binary", "categorical", "survival", "multi_numeric", "none"}

# a categorical column needs at least this many distinct values to be read as an edge-list
# node column (vs a low-cardinality demographic categorical like region/sector).
_NODE_MIN_UNIQUE = 12


@dataclass(frozen=True)
class FamilyAffinity:
    """A family's ideal data shape. `structure` ∈ {panel,timeseries,geo,edgelist,
    cross_section,any}; `outcomes` ⊆ _OUTCOMES; `needs_predictors` = wants ≥2 numeric
    columns; `min_rows` = below this the family is a poor fit on this data."""

    structure: str = "any"
    outcomes: frozenset = field(default_factory=lambda: frozenset({"continuous", "none"}))
    needs_predictors: bool = False
    min_rows: int = 0


def _a(structure="any", outcomes=("continuous", "none"), needs_predictors=False, min_rows=0):
    return FamilyAffinity(structure, frozenset(outcomes), needs_predictors, min_rows)


# One profile per catalog family (all 45 are covered; a missing family falls back to
# _DEFAULT). Kept compact and editable — these are structural priors, not fitted values.
FAMILY_AFFINITY: dict[str, FamilyAffinity] = {
    "statistics": _a("cross_section", ("continuous", "binary", "categorical", "multi_numeric", "none"), min_rows=8),
    "regression": _a("cross_section", ("continuous", "none"), needs_predictors=True, min_rows=20),
    "ml": _a("any", ("continuous", "binary", "multi_numeric"), needs_predictors=True, min_rows=50),
    "causal": _a("cross_section", ("continuous", "binary"), needs_predictors=True, min_rows=30),
    "time-series": _a("timeseries", ("continuous",), min_rows=20),
    # "any" not "panel": this family also holds ols_regression (a cross-section method),
    # so a panel penalty would wrongly demote ols on ordinary data. The panel-specific
    # members (FE/RE/GMM/Mundlak) carry an is_panel precondition, so they still get the
    # precondition-specificity bonus on panel data and stay red/capped off-panel.
    "econometrics": _a("any", ("continuous",), needs_predictors=True, min_rows=30),
    "spatial": _a("geo", ("continuous",), min_rows=20),
    "ecology": _a("any", ("count", "multi_numeric", "none"), min_rows=10),
    "bayesian": _a("any", ("continuous", "binary", "count"), needs_predictors=True, min_rows=20),
    "survival": _a("cross_section", ("survival",), min_rows=20),
    "experimental_design": _a("any", ("none",), min_rows=0),
    "experimental_stats": _a("cross_section", ("continuous",), min_rows=8),
    "mcda": _a("cross_section", ("multi_numeric", "none"), min_rows=2),
    "resource": _a("any", ("continuous", "none"), min_rows=1),
    "techno_economic": _a("any", ("continuous", "none"), min_rows=1),
    "nonparametric": _a("cross_section", ("continuous", "binary"), min_rows=6),
    "configurational": _a("cross_section", ("binary",), min_rows=8),
    "conditional_process": _a("cross_section", ("continuous",), needs_predictors=True, min_rows=30),
    "irt": _a("cross_section", ("multi_numeric", "binary"), min_rows=50),
    "actuarial": _a("any", ("count", "continuous", "none"), min_rows=1),
    "agreement": _a("cross_section", ("continuous", "categorical", "count"), min_rows=10),
    "categorical": _a("cross_section", ("categorical", "binary"), min_rows=10),
    "categorical_tests": _a("cross_section", ("categorical", "binary"), min_rows=10),
    "distribution": _a("cross_section", ("continuous",), min_rows=20),
    "distribution_extra": _a("cross_section", ("continuous",), min_rows=20),
    "effect_sizes": _a("cross_section", ("continuous",), min_rows=6),
    "efficiency": _a("cross_section", ("multi_numeric",), needs_predictors=True, min_rows=5),
    "epidemiology": _a("cross_section", ("binary", "continuous"), min_rows=20),
    "finance": _a("timeseries", ("continuous",), min_rows=20),
    "hydrology": _a("timeseries", ("continuous",), min_rows=10),
    "marketing": _a("any", ("continuous", "none"), min_rows=10),
    "mixture": _a("cross_section", ("multi_numeric", "continuous"), min_rows=30),
    "operations_research": _a("any", ("none", "continuous"), min_rows=1),
    "psychometrics": _a("cross_section", ("multi_numeric",), min_rows=30),
    "reliability": _a("cross_section", ("survival", "count", "continuous"), min_rows=10),
    "spc": _a("cross_section", ("continuous",), min_rows=20),
    "sem": _a("cross_section", ("multi_numeric",), needs_predictors=True, min_rows=100),
    "survey_methods": _a("cross_section", ("continuous", "binary", "none"), min_rows=30),
    "nlp": _a("cross_section", ("none",), min_rows=10),
    "choice": _a("cross_section", ("categorical", "binary"), min_rows=50),
    "game_theory": _a("any", ("none",), min_rows=1),
    "latent_class": _a("cross_section", ("categorical", "binary", "multi_numeric"), min_rows=50),
    "missing_data": _a("any", ("continuous", "none"), min_rows=20),
    "meta": _a("cross_section", ("none",), min_rows=3),
    "soil": _a("cross_section", ("continuous",), min_rows=5),
}

_DEFAULT = _a("any", ("continuous", "none"), min_rows=10)


def get_affinity(family: str) -> FamilyAffinity:
    return FAMILY_AFFINITY.get(family, _DEFAULT)


# name hints for the (otherwise structure-invisible) survival signal
_SURV_DUR = ("dur", "time", "tenure", "surv", "lifetime", "follow", "age_at", "tte", "los")
_SURV_EVT = ("event", "status", "censor", "death", "dead", "fail", "relapse", "recur")


def data_signals(fp: DataFingerprint) -> dict:
    """Extract structural selection signals from a fingerprint (analysis columns only —
    unit/time excluded). These are what `match_score` (and Stage-3 fit) compare against
    the family affinity profiles. Deterministic; never raises."""
    excl = {fp.unit_col, fp.time_col}
    cols = [c for c in fp.columns if c.name not in excl]
    # an edge list's two node-identifier columns are STRUCTURE, not analysis variables —
    # exclude them from the categorical count so they don't masquerade as a categorical
    # outcome (which would wrongly favour agreement/contingency methods over network ones).
    # A node column has MANY distinct values (many nodes); a plain demographic categorical
    # (region/sector, few levels) is NOT an edge endpoint — requiring high cardinality
    # stops ordinary 2-categorical data from being read as an edge list (which would
    # otherwise float network methods up via the structure floor).
    id_cols = [
        c.name for c in fp.columns
        if c.name != fp.time_col
        and (c.kind == "id" or (c.kind == "categorical" and getattr(c, "n_unique", 0) >= _NODE_MIN_UNIQUE))
    ]
    edge_cols = set(id_cols[:2]) if len(id_cols) >= 2 else set()
    n_cont = sum(1 for c in cols if c.kind == "continuous")
    n_count = sum(1 for c in cols if c.kind == "count")
    n_bin = sum(1 for c in cols if c.kind == "binary")
    n_cat = sum(1 for c in cols if c.kind == "categorical" and c.name not in edge_cols)
    n_id = len(id_cols)
    # ordinal_like = a bounded 1..k rating scale (profiles as `count`; see types.is_ordinal_like).
    # Structure splits the two things a rating scale can mean: ≥3 parallel rating columns are
    # RATERS (inter-rater agreement); 1–2 are an ORDINAL OUTCOME (ordinal regression). Mutually
    # exclusive so ordinal-regression and agreement methods never both float up on the same data.
    n_ordinal = sum(1 for c in cols if getattr(c, "ordinal_like", False))
    has_rater_block = n_ordinal >= 3
    has_ordinal_outcome = n_ordinal >= 1 and not has_rater_block
    names = [str(c.name).lower() for c in fp.columns]

    def _hint(words):
        return any(any(w in nm for w in words) for nm in names)

    has_survival = (
        n_bin >= 1 and (n_cont + n_count) >= 1 and _hint(_SURV_DUR) and _hint(_SURV_EVT)
    )
    # a genuine GROUP/arm = a binary/categorical column that is NOT the (role-detected)
    # outcome, and not an edge endpoint. Lets grouping methods (group_comparison, A/B)
    # fire when there's a real grouping variable, but NOT when the only binary is the
    # outcome itself (then a regression like logistic is the right call, not a 2-arm test).
    lo = getattr(fp, "likely_outcome", None)
    has_group = any(
        c.kind in {"binary", "categorical"} and c.name != lo and c.name not in excl and c.name not in edge_cols
        for c in fp.columns
    )
    return {
        "n_rows": fp.n_rows,
        "is_panel": bool(fp.is_panel),
        "is_timeseries": bool(fp.is_timeseries),
        "has_geo": bool(getattr(fp, "has_geo", False)),
        "has_edgelist": n_id >= 2,
        "n_continuous": n_cont,
        "n_count": n_count,
        "n_binary": n_bin,
        "n_categorical": n_cat,
        "n_numeric": n_cont + n_count,
        "has_binary": n_bin >= 1,
        "has_categorical": n_cat >= 1,
        "has_count": n_count >= 1,
        "has_ordinal": n_ordinal >= 1,
        "has_ordinal_outcome": has_ordinal_outcome,
        "has_rater_block": has_rater_block,
        "has_survival": has_survival,
        "has_group": has_group,
        "has_treatment": bool(getattr(fp, "treatment_candidates", None)),
    }


def _available_outcomes(signals: dict) -> set:
    """Outcome kinds the data could support (for the outcome-match term)."""
    avail = {"none"}  # descriptive is always applicable
    if signals["n_continuous"] > 0:
        avail.add("continuous")
    if signals["has_count"]:
        avail.add("count")
    if signals["has_binary"]:
        avail.add("binary")
    if signals["has_categorical"]:
        avail.add("categorical")
    if signals["has_survival"]:
        avail.add("survival")
    if signals["n_numeric"] >= 4:
        avail.add("multi_numeric")
    return avail


def match_score(signals: dict, fam: FamilyAffinity) -> float:
    """Grade how well a family's ideal profile matches the data signals (0–100).
    Structure match dominates; outcome match and a survival/multivariate bonus refine;
    too-small samples and missing predictors are penalised. 50 = neutral."""
    score = 50.0

    s = fam.structure
    if s == "panel":
        score += 25 if signals["is_panel"] else -22
    elif s == "timeseries":
        score += 25 if signals["is_timeseries"] else -22
    elif s == "geo":
        score += 25 if signals["has_geo"] else -22
    elif s == "edgelist":
        score += 25 if signals["has_edgelist"] else -22
    elif s == "cross_section":
        # cross-sectional methods are mildly disfavoured when the data is clearly
        # panel/time-series (those structures have dedicated families)
        score += -8 if (signals["is_panel"] or signals["is_timeseries"]) else 5

    avail = _available_outcomes(signals)
    score += 12 if (fam.outcomes & avail) else -12
    if "survival" in fam.outcomes and signals["has_survival"]:
        score += 15
    if "multi_numeric" in fam.outcomes and signals["n_numeric"] >= 4:
        score += 8

    if fam.needs_predictors and signals["n_numeric"] < 2:
        score -= 10
    if signals["n_rows"] < fam.min_rows:
        score -= 12

    return max(0.0, min(100.0, score))
