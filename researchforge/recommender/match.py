"""Match a DataFingerprint against an analysis entry's hard preconditions."""

from __future__ import annotations

from researchforge.catalog.schema import Precondition
from researchforge.profiler.fingerprint import DataFingerprint


def check_preconditions(fp: DataFingerprint, pre: Precondition) -> tuple[bool, list[str]]:
    """Return (all_met, unmet_reasons). Reasons are human-readable (Chinese)."""
    unmet: list[str] = []

    if pre.is_panel and not fp.is_panel:
        unmet.append("需要面板数据")
    if pre.is_timeseries and not fp.is_timeseries:
        unmet.append("需要时间序列")
    if pre.requires_treatment and not fp.treatment_candidates:
        unmet.append("需要处理组指示变量")
    if pre.requires_time and fp.time_col is None:
        unmet.append("需要时间变量")
    if pre.min_rows is not None and fp.n_rows < pre.min_rows:
        unmet.append(f"需要 ≥ {pre.min_rows} 行（现有 {fp.n_rows}）")
    if pre.min_continuous is not None:
        n_cont = sum(1 for c in fp.columns if c.kind == "continuous")
        if n_cont < pre.min_continuous:
            unmet.append(f"需要 ≥ {pre.min_continuous} 个连续变量（现有 {n_cont}）")
    if pre.requires_binary_outcome and not any(c.kind == "binary" for c in fp.columns):
        unmet.append("需要二值结果变量")
    if pre.requires_group and not any(c.kind in {"binary", "categorical"} for c in fp.columns):
        unmet.append("需要分组变量（分类/二值）")
    if pre.requires_count_outcome and not any(c.kind == "count" for c in fp.columns):
        unmet.append("需要计数型结果变量")
    if pre.min_count_cols is not None:
        n_count = sum(
            1 for c in fp.columns if c.kind == "count" and c.name not in {fp.unit_col, fp.time_col}
        )
        if n_count < pre.min_count_cols:
            unmet.append(f"需要 ≥ {pre.min_count_cols} 个计数列（物种丰度，现有 {n_count}）")
    if pre.requires_ordinal and not any(
        c.kind in {"count", "categorical"}
        and 3 <= c.n_unique <= 10
        and c.name not in {fp.unit_col, fp.time_col}
        for c in fp.columns
    ):
        unmet.append("需要有序结果变量（3–10 个有序等级，如 Likert 量表）")
    if pre.requires_geo:
        n_geo = sum(1 for c in fp.columns if c.kind == "geo")
        if n_geo < 2:
            unmet.append(f"需要经纬度坐标（≥2 个地理列，现有 {n_geo}）")

    return (len(unmet) == 0, unmet)
