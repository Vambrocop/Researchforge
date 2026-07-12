"""ONE bilingual role-vocabulary registry (Wave L · ColumnSemantics C0/C1).

Collapses the scattered treatment/block vocab definitions (executor _shared /
field_trials / statistics.group_compare, recommender.goals, profiler.roles) into a
single source. Import ``role_hint`` / ``is_treatment_named`` / ``has_design_signal`` from
here; never re-declare these. Neutral layer: no imports from executor/recommender
(profiler is the low dependency), so both may depend on it without a cycle.
"""
from __future__ import annotations

import re

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

# Word-boundary treatment/arm/exposure regex — a DIFFERENT concept from ROLE_HINTS["treatment"]
# and deliberately NOT merged (the reframe): this is high-precision role detection for the
# executor's outcome resolver (`is_treatment_named` skips a treatment column when falling back
# to "first candidate"). Boundary semantics matter — a bare substring like "group" would match
# age_group/blood_group and mis-skip real columns. Its vocabulary is clinical/policy (arm/
# intervention/policy/assigned); ROLE_HINTS["treatment"] is agronomic (variety/dose/水平). They
# co-exist here as one module (single home) without collapsing into one bag. Migrated from roles.py.
_TREATMENT_RE = re.compile(
    r"(?:^|_|\b)(treat|treatment|treated|intervention|arm|group|condition|"
    r"exposed|exposure|policy|program|assigned|dose)(?:$|_|\b)",
    re.I,
)


def role_hint(name: str, role: str) -> bool:
    """True if a column NAME carries the vocabulary signal for `role` (bilingual, 子串匹配)."""
    low = str(name).strip().lower()
    return any(h in low for h in ROLE_HINTS.get(role, ()))


def is_treatment_named(name: str) -> bool:
    """True when a column NAME carries a treatment/arm signal (treat/arm/exposed/dose…).

    Word-boundary match (not the substring ``role_hint``): a treatment INDICATOR is almost
    never the dependent variable, so the executor's outcome resolver skips such a column when
    falling back to the "first candidate". Consumed by executor _helpers.core / epidemiology;
    re-exported by profiler.roles for backward compatibility."""
    return bool(_TREATMENT_RE.search(str(name)))


def has_design_signal(fp) -> bool:
    """True when the data looks like a DESIGNED experiment (RCBD/factorial/split-plot…) rather
    than observational data that merely has categorical groups. Used to stop designed-experiment
    methods from crowding out the naive group comparison under ``--goal compare`` (发现16).

    Double gate (Wave L-C1) —防单歧义词误伤: a TREATMENT word (处理/dose/variety — low ambiguity,
    strongly experimental) is sufficient on its own; a BLOCK word alone (site/batch/区组… — also
    common in observational data) is NOT — it requires a LAYOUT of ≥2 design-vocab columns (e.g.
    block + replicate, or two blocking factors). So a single ambiguous block-named column can no
    longer flip observational data to "designed"."""
    names = [str(c.name) for c in fp.columns]
    if any(role_hint(nm, "treatment") for nm in names):
        return True
    design_cols = sum(
        1 for nm in names if role_hint(nm, "treatment") or role_hint(nm, "block")
    )
    return design_cols >= 2
