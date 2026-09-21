"""Structured description of a dataset — the input to the Recommender."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

ColumnKind = Literal[
    "continuous",
    "categorical",
    "count",
    "binary",
    "datetime",
    "id",
    "geo",
    "unknown",
]


class ColumnInfo(BaseModel):
    name: str
    kind: ColumnKind
    dtype: str
    n_missing: int
    n_unique: int
    # True for a rating-scale-like column: a small run of consecutive positive
    # integers (e.g. a 1–5 Likert), which profiles as `kind="count"` but is really
    # ORDINAL. Distinguishes a bounded rating from an unbounded count (which starts at
    # 0 / has many levels), so ordinal-regression and rater-agreement methods can be
    # surfaced without changing the coarse `count` type. Defaults False (additive).
    ordinal_like: bool = False
    # True for a FREE-TEXT column (prose: reviews / open-ends / documents), which profiles as
    # `kind="categorical"`/`"id"` but should route to the text-mining family (TF-IDF / sentiment
    # / topics), not to contingency/agreement methods. See types.is_text_like. Defaults False.
    is_text: bool = False


class Issue(BaseModel):
    """A data-quality finding produced by the Profiler, consumed by Cleaning."""

    kind: str  # missing | duplicate_rows | constant | outliers | high_cardinality
    severity: str  # low | medium | high
    detail: str
    column: Optional[str] = None
    count: int = 0


class DataFingerprint(BaseModel):
    path: str
    n_rows: int
    n_cols: int
    columns: list[ColumnInfo]
    is_panel: bool = False
    unit_col: Optional[str] = None
    time_col: Optional[str] = None
    is_timeseries: bool = False
    has_geo: bool = False
    # EVERY binary column, in file order — a shape fact, not a role claim. It was called
    # `treatment_candidates` until 2026-09-08, which asserted a treatment signal the value
    # never carried; three measured bugs came out of code trusting that name (PSM matching
    # on a survival event indicator, a study-site `group` outranking the arm, and
    # synthetic_control failing outright on a decoy-first panel). Only
    # `executor._helpers.core.resolve_treatment` is entitled to say which column is the
    # treatment; this list is just its candidate pool.
    binary_columns: list[str] = Field(default_factory=list)
    # Multi-label SHAPE facts, computed only when there are >=3 binary columns. They exist
    # because the recommender sees a fingerprint and never the data, and "3 co-occurring
    # labels" cannot be told apart from "3 unrelated binary covariates" by marginals alone.
    #   label_cardinality  — mean number of binary columns that are 1 in a row
    #   multi_label_row_frac — fraction of rows carrying MORE THAN ONE of them
    # One-hot multiclass gives exactly (1.0, 0.0), which is how it is ruled out. Neither is
    # a role claim: they say what the table looks like, not what anything means.
    label_cardinality: Optional[float] = None
    multi_label_row_frac: Optional[float] = None
    # CLOSED (compositional) components: the largest set of >=3 non-negative numeric columns
    # whose ROW SUM is constant — parts of a whole (percentages, proportions, mineral or
    # budget shares). Closure is a shape fact with a hard statistical consequence: the parts
    # cannot vary freely, so ordinary correlations among them are forced negative and mean
    # nothing (Pearson 1897). Empty when the frame is not compositional.
    closed_components: list[str] = Field(default_factory=list)
    # DISCRETE-CHOICE design: a binary flag that is 1 exactly ONCE inside every stratum, with
    # >=2 rows per stratum. That is the conditional-logit / McFadden setting — a conjoint or
    # choice-based survey (respondent x task), and equally a matched case-control study
    # (one case per matched set). Unmistakable: ordinary binary outcomes, staggered-adoption
    # panels and one-hot encodings all fail it. Empty/None when the frame is not one.
    choice_flag: Optional[str] = None
    choice_set_cols: list[str] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)
    # Non-binding semantic role hints (see profiler/roles.py). They do NOT change
    # run-time defaults — only suggest a `config` to the user.
    likely_outcome: Optional[str] = None
    # Confidence in likely_outcome: "high" = an unambiguous dependent-variable name
    # (outcome/target/y/…) or a clear binary event → safe to BIND as the run-time outcome;
    # "medium" = a domain word that is often but not always the DV (price/sales/score…) →
    # surfaced as a hint only, never binds; "low" = position convention; "" = none detected.
    likely_outcome_confidence: str = ""
    likely_treatment: Optional[str] = None
    # Evidence behind likely_treatment: "high" = a strong treatment word (treated/arm/trt/
    # dose…), "medium" = a weak whole-name or compound match (a bare `group`, `study_group`),
    # "low" = no name signal at all — the first non-outcome binary, i.e. column order.
    # The hint is shown to users AND binds (resolve_treatment tier 3), so the distinction
    # has to travel with it.
    likely_treatment_confidence: str = ""
    likely_time: Optional[str] = None
    role_hint_reason: str = ""

    def column(self, name: str) -> Optional[ColumnInfo]:
        for c in self.columns:
            if c.name == name:
                return c
        return None
