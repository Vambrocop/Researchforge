"""Catalog data model — each entry is one analysis the engine can recommend.

`preconditions` are matched against a DataFingerprint by the Recommender;
`biases` feed the rigor review.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Precondition(BaseModel):
    is_panel: Optional[bool] = None
    is_timeseries: Optional[bool] = None
    requires_treatment: Optional[bool] = None
    requires_time: Optional[bool] = None
    min_rows: Optional[int] = None
    min_continuous: Optional[int] = None
    requires_binary_outcome: Optional[bool] = None
    requires_group: Optional[bool] = None
    requires_count_outcome: Optional[bool] = None
    min_count_cols: Optional[int] = None
    requires_ordinal: Optional[bool] = None


class AnalysisEntry(BaseModel):
    id: str
    method: str
    domain: str
    family: str  # statistics | econometrics | causal | ml | ...
    goal: str  # describe | explain | predict
    description: str = ""
    preconditions: Precondition = Field(default_factory=Precondition)
    produces: list[str] = Field(default_factory=list)
    executor_ref: str = ""
    biases: list[str] = Field(default_factory=list)
