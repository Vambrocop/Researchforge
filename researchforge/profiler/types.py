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
        is_int_dtype = pd.api.types.is_integer_dtype(nn)
        is_whole = is_int_dtype or bool((nn % 1 == 0).all())
        if is_whole:
            if nn.is_unique:
                return "id"
            if len(uniq) == 2:
                return "binary"
            if bool((nn >= 0).all()):
                # A count is an event/abundance tally (Poisson/NB). But a whole-valued
                # FLOAT column with many distinct values is a continuous measurement
                # recorded without decimals (e.g. a clinical progression score 25–346),
                # not a count — a genuine count is stored as int. Calling it count would
                # wrongly surface Poisson/NB over OLS (real-data dogfood: diabetes target
                # float64, 214 distinct, 25–346). Keep int-typed and low-cardinality-float
                # whole numbers as count.
                if (not is_int_dtype) and len(uniq) > 15:
                    return "continuous"
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


def is_ordinal_like(s: pd.Series) -> bool:
    """A rating-scale hint: does this column look like an ORDINAL Likert scale rather
    than an unbounded count? True when the values are a short run of CONSECUTIVE
    POSITIVE integers (min ≥ 1, no zero, ≤ 7 distinct levels, e.g. {1,2,3,4,5}).

    This is intentionally strict — a count that starts at 0, has many levels, or skips
    values (a genuine Poisson/abundance) is NOT ordinal_like — so it separates a bounded
    rating (surface ordinal regression / rater agreement) from an unbounded count
    (surface Poisson/NB) even though both profile as `kind="count"`.
    """
    nn = s.dropna()
    if len(nn) == 0 or not pd.api.types.is_numeric_dtype(nn) or pd.api.types.is_bool_dtype(nn):
        return False
    if not (pd.api.types.is_integer_dtype(nn) or bool((nn % 1 == 0).all())):
        return False
    if nn.is_unique:  # an id-like column, not a rating
        return False
    uniq = sorted(int(v) for v in nn.unique())
    if len(uniq) < 3 or len(uniq) > 7:
        return False
    if uniq[0] < 1:  # a 0-based / negative scale reads as a count, not a rating
        return False
    return uniq == list(range(uniq[0], uniq[-1] + 1))  # consecutive levels
