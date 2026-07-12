"""ONE bilingual role-vocabulary registry (Wave L · ColumnSemantics C0).

Collapses the 5 scattered treatment/block vocab definitions (executor _shared /
field_trials / statistics.group_compare, recommender.goals, profiler.roles) into a
single source. Import ``role_hint`` from here; never re-declare these tuples.
Neutral layer: no imports from executor/recommender (profiler is the low dependency).
"""
from __future__ import annotations

ROLE_HINTS: dict[str, tuple[str, ...]] = {
    "block": ("block", "blk", "rep", "replicate", "replication", "plot", "site",
              "field", "batch", "区组", "重复", "组块", "块", "批次", "场地", "地块"),
    "treatment": ("treat", "treatment", "treated", "trt", "arm", "intervention",
                  "exposed", "exposure", "dose", "fert", "variety", "cultivar",
                  "genotype", "hybrid", "factor", "level", "处理", "品种", "剂量",
                  "水平", "施肥", "组别"),
    "time": ("year", "yr", "date", "time", "month", "quarter", "period",
             "day", "week", "wave", "日期", "年份", "月份", "季度"),
}


def role_hint(name: str, role: str) -> bool:
    """True if a column NAME carries the vocabulary signal for `role` (bilingual, 子串匹配)."""
    low = str(name).strip().lower()
    return any(h in low for h in ROLE_HINTS.get(role, ()))
