"""Heuristic column-type inference. Type only — the column's *role* (unit /
time / treatment) is decided later in `profile.py`.

Note: pandas >= 3.0 gives text columns the `str` dtype (not `object`), so we
branch on numeric-vs-not rather than testing for `object`.
"""

from __future__ import annotations

import warnings

import pandas as pd

from researchforge.profiler.fingerprint import ColumnKind

_GEO_NAMES = {"lat", "latitude", "lon", "lng", "long", "longitude"}


def infer_kind(s: pd.Series) -> ColumnKind:
    nn = s.dropna()
    if len(nn) == 0:
        return "unknown"
    name = str(s.name).strip().lower()

    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"

    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        uniq = set(nn.unique().tolist())
        if uniq <= {0, 1} and len(uniq) == 2:
            return "binary"
        if name in _GEO_NAMES:
            return "geo"
        is_int = pd.api.types.is_integer_dtype(nn) or bool((nn % 1 == 0).all())
        if is_int:
            if nn.is_unique:
                return "id"
            if bool((nn >= 0).all()):
                return "count"
        return "continuous"

    if pd.api.types.is_bool_dtype(s):
        return "binary"

    # textual (object or pandas>=3.0 `str` dtype)
    sample = nn.astype(str)
    if sample.str.contains(r"\d").mean() > 0.5:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().mean() > 0.9:
            return "datetime"
    if nn.is_unique:
        return "id"
    if nn.nunique() == 2:
        return "binary"
    return "categorical"
