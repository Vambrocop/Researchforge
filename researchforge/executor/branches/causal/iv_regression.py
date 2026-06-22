"""Causal family branch handler: iv_regression (split from causal.py)."""
from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


@register("iv_regression")
def _branch_iv_regression(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    summary.append(
        "工具变量回归（2SLS）需要你指定外生工具变量（instrument），引擎无法自动识别。"
        "请在指定工具变量后手动运行；或先用 panel_fixed_effects / did 作为可自动执行的替代。"
    )
