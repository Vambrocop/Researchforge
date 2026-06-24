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
    likely_treatment: Optional[str] = None
    likely_time: Optional[str] = None
    role_hint_reason: str = ""

    def column(self, name: str) -> Optional[ColumnInfo]:
        for c in self.columns:
            if c.name == name:
                return c
        return None
