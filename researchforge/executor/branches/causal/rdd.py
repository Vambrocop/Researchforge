"""Causal family branch handler: rdd (split from causal.py)."""
from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.run import _rdd_via_rdrobust


@register("rdd")
def _branch_rdd(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import importlib.util

    import pandas as pd

    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in {fp.unit_col, fp.time_col}]
    running = cfg.get("running") or cfg.get("running_var")
    # guard: never accept the running variable as the outcome (regressing it on itself
    # around the cutoff is meaningless) — same guard as fuzzy_rdd.
    outcome = (
        cfg["outcome"] if (cfg.get("outcome") in cont and cfg.get("outcome") != running)
        else next((c for c in cont if c != running), None)
    )
    try:
        cutoff = float(cfg.get("cutoff", 0.0))
    except (TypeError, ValueError):
        cutoff = None
    if importlib.util.find_spec("rdrobust") is None:
        summary.append("RDD 需要 rdrobust 包（未检测到）。安装：pip install rdrobust；或用 did（面板双重差分）。")
    elif not running or running not in df.columns:
        summary.append(
            "RDD 失败：需用 config 指定 running（驱动/分配变量），例 "
            "config={\"running\":\"score\",\"cutoff\":60}。RDD 是设计驱动的，引擎无法自动猜驱动变量。"
        )
    elif outcome is None:
        summary.append("RDD 失败：需要一个连续结果变量（≠ running）。可用 config['outcome'] 指定。")
    elif cutoff is None:
        summary.append("RDD 失败：cutoff 需为数值。")
    else:
        rv = pd.to_numeric(df[running], errors="coerce").dropna()
        if rv.empty or not (rv.min() < cutoff < rv.max()):
            summary.append(
                f"RDD 失败：cutoff={cutoff:g} 不在 {running} 取值范围内"
                f"（{rv.min():.3g}–{rv.max():.3g}）；用 config['cutoff'] 指定正确断点。"
            )
        else:
            try:
                res = _rdd_via_rdrobust(df, outcome, running, cutoff, d / "rdd_plot.png")
                if (d / "rdd_plot.png").exists():
                    files.append("rdd_plot.png")
                rd, lb, ub = res["rd_robust"], res["ci_lb"], res["ci_ub"]
                p, h = res["p_robust"], res["bw_h"]
                nl, nr = res["n_left"], res["n_right"]
                pd.DataFrame(
                    {
                        "estimator": ["Conventional", "Robust (bias-corrected)"],
                        "rd_effect": [round(res["rd_conventional"], 4), round(rd, 4)],
                        "ci_lower": ["", round(lb, 4)],
                        "ci_upper": ["", round(ub, 4)],
                        "p_value": ["", round(p, 4)],
                    }
                ).to_csv(d / "rdd_estimate.csv", index=False, encoding="utf-8")
                files.append("rdd_estimate.csv")
                estimates["rd_effect"] = round(rd, 4)
                estimates["ci_lb"] = round(lb, 4)
                estimates["ci_ub"] = round(ub, 4)
                estimates["p_value"] = round(p, 4)
                estimates["bandwidth"] = round(h, 4)
                estimates["n_effective"] = float(nl + nr)
                sig = "显著" if p < 0.05 else "不显著"
                (d / "rdd_summary.txt").write_text(
                    f"断点回归 RDD（rdrobust，sharp，{res['bwselect']} 最优带宽）\n"
                    f"驱动变量 {running} @ 断点 {cutoff:g}，结果 {outcome}\n"
                    f"断点处跳跃（局部处理效应 LATE）= {rd:.4f}，稳健 95% CI [{lb:.4f}, {ub:.4f}]，p={p:.4g}\n"
                    f"MSE 最优带宽 h={h:.4f}；带内有效样本 左 {nl} / 右 {nr}\n"
                    "注：RDD 只识别断点局部的处理效应（外推性有限）；这是 sharp 设计，"
                    "假定断点处处理概率从 0 跳到 1（完全合规）——若实际为模糊断点（take-up<100%）"
                    "本估计有偏，需 fuzzy RDD（未启用）；并依赖潜在结果均值在断点连续、"
                    "无其他政策/混杂同址跳变、个体无法精确操纵驱动变量（应做 McCrary 密度检验）。\n",
                    encoding="utf-8",
                )
                files.append("rdd_summary.txt")
                summary.append(
                    f"{entry.method} 完成（rdrobust，sharp）：{running}@{cutoff:g} 处 {outcome} "
                    f"跳跃 = {rd:.4f}（稳健 95% CI [{lb:.4f}, {ub:.4f}]，{sig}，p={p:.3g}）；"
                    f"最优带宽 h={h:.3f}，带内有效样本 {nl}+{nr}。"
                    "⚠ 仅识别断点局部效应（外推有限）；sharp 设计（假定断点处处理概率 0→1、"
                    "完全合规，模糊断点需 fuzzy RDD，未启用）；假定潜在结果均值在断点连续、"
                    "无其他政策/混杂同址跳变、驱动变量不可精确操纵（建议 McCrary 密度检验）。"
                )
                code += [
                    "from rdrobust import rdrobust  # 断点回归(局部线性, 稳健偏差校正)",
                    f"# rdrobust(y={outcome!r}, x={running!r}, c={cutoff!r}); MSE 最优带宽 + robust CI",
                ]
            except Exception as err:
                summary.append(f"RDD 拟合失败：{err}")
