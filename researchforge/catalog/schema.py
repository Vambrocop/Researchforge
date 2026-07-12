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
    min_numeric_cols: Optional[int] = None  # continuous OR count (Likert items profile as count)
    min_categorical_cols: Optional[int] = None  # categorical/binary/count (rater codes profile as count)
    requires_binary_outcome: Optional[bool] = None
    requires_group: Optional[bool] = None
    # stricter than requires_group: needs a categorical/binary column whose level
    # count is >= N — a genuine grouping structure for a random-intercept/random-slope
    # term, not just any 2-level flag (e.g. a binary OUTCOME column). Wave L D: gamm's
    # random intercept needs >=5 groups for a stable variance component (mgcv/lme4).
    min_group_levels: Optional[int] = None
    requires_count_outcome: Optional[bool] = None
    min_count_cols: Optional[int] = None
    requires_ordinal: Optional[bool] = None
    requires_geo: Optional[bool] = None
    requires_soil_texture: Optional[bool] = None
    requires_effect_sizes: Optional[bool] = None
    requires_edgelist: Optional[bool] = None


class ParamSpec(BaseModel):
    """Machine-readable spec for one user-configurable parameter (a key the
    analysis reads from ``config``). The single source of truth consumed by the
    Web UI form, the recommend explanation, and run-time config validation — so
    nobody has to guess an analysis's parameters. Optional: an entry with no
    declared ``params`` simply isn't config-validated (it still runs on its auto
    defaults).

    ``type``: column (one column name) | columns (list of names) | str | int |
    float | bool | choice. ``default`` is a human note on the auto-default the
    engine uses when the key is omitted (the default must still run)."""

    name: str
    type: str = "column"
    required: bool = False
    description: str = ""
    choices: list[str] = Field(default_factory=list)
    default: str = ""


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
    params: list[ParamSpec] = Field(default_factory=list)
