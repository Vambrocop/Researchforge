"""Read a tabular dataset and produce a DataFingerprint."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from researchforge.profiler.fingerprint import ColumnInfo, DataFingerprint
from researchforge.profiler.ingest import read_table as _robust_read_table
from researchforge.profiler.quality import diagnose
from researchforge.profiler.types import infer_kind

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
        )
        for c in df.columns
    ]
    fp = DataFingerprint(
        path=str(path), n_rows=int(len(df)), n_cols=int(df.shape[1]), columns=columns
    )
    _detect_structure(df, fp)
    fp.issues = diagnose(df)
    return fp


def _find_time_col(df: pd.DataFrame, fp: DataFingerprint) -> str | None:
    for c in fp.columns:
        if c.kind == "datetime":
            return c.name
    for c in fp.columns:
        if c.name.lower() in _TIME_NAMES:
            return c.name
    for c in fp.columns:
        s = df[c.name].dropna()
        if not s.empty and pd.api.types.is_integer_dtype(s):
            if int(s.min()) >= 1900 and int(s.max()) <= 2100 and s.nunique() < len(s):
                return c.name
    return None


def _detect_structure(df: pd.DataFrame, fp: DataFingerprint) -> None:
    fp.treatment_candidates = [c.name for c in fp.columns if c.kind == "binary"]
    fp.has_geo = any(c.kind == "geo" for c in fp.columns)

    time_col = _find_time_col(df, fp)
    fp.time_col = time_col
    if time_col is None:
        return

    unit_candidates = [
        c.name
        for c in fp.columns
        if c.name != time_col and c.kind in {"categorical", "id", "count", "unknown"}
    ]
    unit_col = None
    for u in unit_candidates:
        pair = df[[u, time_col]].dropna()
        if pair.duplicated().sum() == 0 and 1 < df[u].nunique() < len(df):
            unit_col = u
            break

    if unit_col is not None and len(df) > df[time_col].nunique():
        fp.is_panel = True
        fp.unit_col = unit_col
    elif df[time_col].nunique() > 1:
        fp.is_timeseries = True
