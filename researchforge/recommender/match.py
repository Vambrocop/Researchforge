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

    return (len(unmet) == 0, unmet)
