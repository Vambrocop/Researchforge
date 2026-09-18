"""Shared covariate hygiene for the causal family (cold review A#5/A#6)."""
from __future__ import annotations

def _drop_post_treatment(fp, treatment, cols, summary):
    """Shared with psm/ipw/aipw: the survival event indicator is a descendant of the
    treatment, and a control/negated flag is a deterministic function of it. Neither is a
    confounder. AUTO set only — an explicit config list stays the user's call."""
    from researchforge.profiler.semantics import survival_event_column, treatment_never_named

    drop = {c for c in cols if c != treatment and treatment_never_named(c)}
    ev = survival_event_column(fp)
    if ev and ev in cols and ev != treatment:
        drop.add(ev)
    if drop:
        summary.append(
            f"⚠ 已把 {'、'.join(sorted(drop))} 排除在自动协变量之外："
            "生存事件指示列是时长、进而是处理变量的后代；对照/否定标记是处理变量的"
            "确定性函数。两者都不是混杂因子，放进模型属于处理后调整。"
            '若确需纳入，用 config 显式指定协变量。'
        )
    return [c for c in cols if c not in drop]
