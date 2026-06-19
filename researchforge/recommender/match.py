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
    if pre.min_numeric_cols is not None:
        # continuous OR count — for item-scale methods (Likert items profile as count, not
        # continuous), so gating on min_continuous alone would hide them from ordinal data.
        n_num = sum(
            1 for c in fp.columns
            if c.kind in {"continuous", "count"} and c.name not in {fp.unit_col, fp.time_col}
        )
        if n_num < pre.min_numeric_cols:
            unmet.append(f"需要 ≥ {pre.min_numeric_cols} 个数值列（连续/计数，现有 {n_num}）")
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
    if pre.requires_soil_texture:
        names = [c.name.lower() for c in fp.columns]
        if not (
            any("sand" in n for n in names)
            and any("silt" in n for n in names)
            and any("clay" in n for n in names)
        ):
            unmet.append("需要 sand / silt / clay（砂/粉/黏粒）百分比列")
    if pre.requires_effect_sizes:
        names = {c.name.lower() for c in fp.columns}

        def _has(*opts):
            return any(o in names for o in opts)

        gen = _has("yi", "effect", "es", "effect_size", "smd", "logor", "d", "g", "lnrr") and _has(
            "vi", "var", "variance", "v", "sei", "se", "std_err", "se_effect"
        )
        smd = all(
            _has(*o)
            for o in (("m1", "m1i", "mean1", "mean_t"), ("sd1", "sd1i", "sd_t"), ("n1", "n1i", "nt"),
                      ("m2", "m2i", "mean2", "mean_c"), ("sd2", "sd2i", "sd_c"), ("n2", "n2i", "nc"))
        )
        counts = all(_has(o) for o in ("ai", "bi", "ci", "di"))
        if not (gen or smd or counts):
            unmet.append(
                "需要研究层效应量数据（yi+vi/sei，或两组 m/sd/n，或 2×2 ai/bi/ci/di）"
            )
    if pre.requires_edgelist:
        n_id = sum(1 for c in fp.columns if c.kind in {"id", "categorical"} and c.name != fp.time_col)
        if n_id < 2:
            unmet.append("需要 ≥2 个节点标识列（边的 source / target）")

    return (len(unmet) == 0, unmet)
