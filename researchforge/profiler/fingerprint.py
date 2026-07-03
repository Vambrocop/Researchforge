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
    treatment_candidates: list[str] = Field(default_factory=list)
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
    likely_time: Optional[str] = None
    role_hint_reason: str = ""

    def column(self, name: str) -> Optional[ColumnInfo]:
        for c in self.columns:
            if c.name == name:
                return c
        return None
