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

    def column(self, name: str) -> Optional[ColumnInfo]:
        for c in self.columns:
            if c.name == name:
                return c
        return None
