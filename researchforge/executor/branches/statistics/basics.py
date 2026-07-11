"""Branch handlers: correlation, descriptive_stats (statistics family).

Each handler unpacks ctx into the same local names run_analysis used and runs the
original branch body verbatim. See executor/_branch_api.py.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.run import _heatmap, _plotly_corr_heatmap


@register("correlation")
def _branch_correlation(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    num = df.select_dtypes(include="number")
    corr = num.corr()
    corr.to_csv(d / "correlation.csv", encoding="utf-8")
    files.append("correlation.csv")
    _heatmap(corr, d / "correlation_heatmap.png")
    files.append("correlation_heatmap.png")
    _plotly_corr_heatmap(corr, d / "correlation_heatmap.html")
    if (d / "correlation_heatmap.html").exists():
        files.append("correlation_heatmap.html")
    summary.append(f"相关分析完成：{num.shape[1]} 个数值变量")
    code += ["num = df.select_dtypes(include='number')", "num.corr().to_csv('correlation.csv')"]



@register("descriptive_stats")
def _branch_descriptive_stats(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    df.describe(include="all").transpose().to_csv(d / "table_describe.csv", encoding="utf-8")
    files.append("table_describe.csv")
    summary.append(f"描述统计完成：{df.shape[0]} 行 × {df.shape[1]} 列")
    high_card = [c.name for c in fp.columns if c.kind in {"id", "categorical"} and c.n_unique > 50]
    if high_card:
        summary.append(f"注意：{len(high_card)} 个高基数列（如 {high_card[0]}）描述统计意义有限。")
    code.append("df.describe(include='all').transpose().to_csv('table_describe.csv')")
