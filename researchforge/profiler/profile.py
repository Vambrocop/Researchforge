"""Read a tabular dataset and produce a DataFingerprint."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from researchforge.profiler.fingerprint import ColumnInfo, DataFingerprint
from researchforge.profiler.ingest import read_table as _robust_read_table
from researchforge.profiler.quality import diagnose
from researchforge.profiler.semantics import looks_like_survival
from researchforge.profiler.types import infer_kind, is_ordinal_like, is_text_like

_TIME_NAMES = {"year", "yr", "date", "time", "month", "quarter", "period", "day", "week"}


def read_table(path: Path) -> pd.DataFrame:
    """Robust read: encoding fallback, delimiter sniff, and conservative numeric
    coercion of text columns that are really numbers (see profiler.ingest).
    Drop-in replacement; coercions/encoding are recorded in ``df.attrs``."""
    return _robust_read_table(path)


def profile_dataset(path: str | Path) -> DataFingerprint:
    path = Path(path)
    df = read_table(path)

    columns = [
        ColumnInfo(
            name=str(c),
            kind=infer_kind(df[c]),
            dtype=str(df[c].dtype),
            n_missing=int(df[c].isna().sum()),
            n_unique=int(df[c].nunique(dropna=True)),
            ordinal_like=is_ordinal_like(df[c]),
            is_text=is_text_like(df[c]),
        )
        for c in df.columns
    ]
    fp = DataFingerprint(
        path=str(path), n_rows=int(len(df)), n_cols=int(df.shape[1]), columns=columns
    )
    _detect_structure(df, fp)
    fp.issues = diagnose(df)
    # non-binding semantic role hints (do not change run-time defaults)
    from researchforge.profiler.roles import detect_roles

    roles = detect_roles(columns, df)
    fp.likely_outcome = roles["likely_outcome"]
    fp.likely_outcome_confidence = roles.get("likely_outcome_confidence", "")
    fp.likely_treatment = roles["likely_treatment"]
    fp.likely_treatment_confidence = roles["likely_treatment_confidence"]
    fp.likely_time = roles["likely_time"]
    fp.role_hint_reason = roles["reason"]
    return fp


def _find_time_col(df: pd.DataFrame, fp: DataFingerprint) -> str | None:
    for c in fp.columns:
        if c.kind == "datetime":
            return c.name
    for c in fp.columns:
        if c.name.lower() in _TIME_NAMES:
            return c.name
    # Fallback: a bare integer column whose values sit in a plausible calendar
    # year range (1900-2100) with at least one duplicate (repeat years across
    # units, e.g. a panel). This is ONLY trustworthy when the column name also
    # carries a temporal anchor (e.g. "obs_year", "survey_date", "fiscal_yr")
    # -- otherwise a bounded, duplicated integer column that is really an Elo
    # rating, a score, or an index gets misread as the panel/time axis (P3-1).
    for c in fp.columns:
        name_lower = c.name.lower()
        if not any(token in name_lower for token in _TIME_NAMES):
            continue
        s = df[c.name].dropna()
        if not s.empty and pd.api.types.is_integer_dtype(s):
            if int(s.min()) >= 1900 and int(s.max()) <= 2100 and s.nunique() < len(s):
                return c.name
    return None


def _label_shape(df: pd.DataFrame, fp: DataFingerprint) -> None:
    """How much do the binary columns CO-OCCUR? (see DataFingerprint.label_cardinality)

    The recommender only ever sees a fingerprint, so a data-derived fact has to be computed
    here or not at all. Measured on five frames while designing the multi-label tilt:

        真多标签       k=5  cardinality 2.35  multi-label rows 84%
        医学二值协变量  k=4              1.27                   37%
        独热多分类     k=3              1.00                    0%
        RCT 标志列    k=3              1.21                   37%
        独立四标志     k=4              1.70                   58%

    Mean |phi| among the columns was the obvious candidate and is USELESS on its own: one-hot
    multiclass scores highest of all (0.500 — mutual exclusion is strong NEGATIVE correlation),
    so it would have promoted exactly the shape that must be refused.
    """
    cols = [c for c in fp.binary_columns if c in df.columns]
    if len(cols) < 3:
        return
    try:
        sub = df[cols]
        ones = sub.apply(lambda s: s == s.dropna().max() if s.dropna().nunique() == 2 else s)
        per_row = ones.sum(axis=1)
        if not len(per_row):
            return
        fp.label_cardinality = round(float(per_row.mean()), 4)
        fp.multi_label_row_frac = round(float((per_row > 1).mean()), 4)
    except Exception:  # noqa: BLE001 — a shape fact must never break profiling
        fp.label_cardinality = fp.multi_label_row_frac = None


def _detect_structure(df: pd.DataFrame, fp: DataFingerprint) -> None:
    fp.binary_columns = [c.name for c in fp.columns if c.kind == "binary"]
    fp.has_geo = any(c.kind == "geo" for c in fp.columns)
    _label_shape(df, fp)

    time_col = _find_time_col(df, fp)
    fp.time_col = time_col
    if time_col is None:
        return

    # Survival data: the `time`-named column is a follow-up DURATION, not a panel/series time
    # axis. Keep it as time_col (survival branches read it as the duration — see CLAUDE.md) but
    # infer NO panel/timeseries structure from it: otherwise a repeating covariate (age) is
    # picked as a panel UNIT and the near-unique durations as a series, surfacing DID /
    # forecasting methods that model the durations — nonsense (dogfood: clinical survival.csv).
    if looks_like_survival(fp):
        return

    n = len(df)
    unit_candidates = [
        c.name
        for c in fp.columns
        if c.name != time_col and c.kind in {"categorical", "id", "count", "unknown"}
    ]
    unit_col = None
    for u in unit_candidates:
        # A real panel UNIT identifies (almost) every row's entity AND repeats across periods.
        # Guards computed on the FULL table, not a dropna subset (the subset hides missingness):
        #   (a) coverage — a mostly-empty column (e.g. an 89%-blank free-text 备注/notes field)
        #       pairs uniquely with a high-cardinality date over its few non-null rows and
        #       masquerades as a unit; a genuine unit is present in most rows (dogfooding #16);
        #   (b) repetition — a near-unique id never repeats across time, so it is not a panel
        #       unit either. Require each unit to appear ≥2× on average.
        nunq = int(df[u].nunique(dropna=True))
        if df[u].notna().mean() < 0.5:
            continue
        if nunq <= 1 or n / nunq < 2:
            continue
        pair = df[[u, time_col]].dropna()
        if pair.duplicated().sum() == 0 and nunq < n:
            unit_col = u
            break

    if unit_col is not None and n > df[time_col].nunique():
        fp.is_panel = True
        fp.unit_col = unit_col
    elif df[time_col].nunique() > 1 and n / df[time_col].nunique() <= 1.5:
        # A univariate time series is indexed by time with ~one observation per period.
        # Many rows sharing each timestamp is a date-stamped cross-section / batch (which
        # value do you forecast at t?), not a series — don't flag it timeseries just
        # because a date column exists (dogfooding P6 structural over-detection).
        fp.is_timeseries = True
