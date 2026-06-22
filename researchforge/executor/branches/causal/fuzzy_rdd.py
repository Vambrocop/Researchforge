"""Causal family branch handler: fuzzy_rdd (split from causal.py)."""
from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.run import _rdd_via_rdrobust


@register("fuzzy_rdd")
def _branch_fuzzy_rdd(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import importlib.util

    import pandas as pd

    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in {fp.unit_col, fp.time_col}]
    running = cfg.get("running") or cfg.get("running_var")
    treatment = cfg.get("treatment")
    outcome = cfg["outcome"] if (cfg.get("outcome") in cont and cfg.get("outcome") not in {running, treatment}) else next(
        (c for c in cont if c not in {running, treatment}), None
    )
    try:
        cutoff = float(cfg.get("cutoff", 0.0))
    except (TypeError, ValueError):
        cutoff = None
    if importlib.util.find_spec("rdrobust") is None:
        summary.append("模糊断点回归需要 rdrobust 包（未检测到）。安装：pip install rdrobust。")
    elif not running or running not in df.columns:
        summary.append(
            "模糊 RDD 失败：需用 config 指定 running（驱动/分配变量），例 "
            "config={\"running\":\"score\",\"cutoff\":60,\"treatment\":\"got_program\"}。"
        )
    elif not treatment or treatment not in df.columns:
        summary.append(
            "模糊 RDD 失败：需用 config['treatment'] 指定『实际接受处理』列（0/1，断点处 take-up 不完全）；"
            "这正是模糊 RDD 区别于 sharp 的地方（sharp 假定断点处处理概率 0→1 完全合规）。"
        )
    elif outcome is None:
        summary.append("模糊 RDD 失败：需要一个连续结果变量（≠ running/treatment）。可用 config['outcome'] 指定。")
    elif cutoff is None:
        summary.append("模糊 RDD 失败：cutoff 需为数值。")
    else:
        rv = pd.to_numeric(df[running], errors="coerce").dropna()
        tv = pd.to_numeric(df[treatment], errors="coerce").dropna()
        if rv.empty or not (rv.min() < cutoff < rv.max()):
            summary.append(
                f"模糊 RDD 失败：cutoff={cutoff:g} 不在 {running} 取值范围内"
                f"（{rv.min():.3g}–{rv.max():.3g}）；用 config['cutoff'] 指定正确断点。"
            )
        elif tv.nunique() < 2:
            summary.append(f"模糊 RDD 失败：处理列 {treatment} 无变异（需 0/1 两种 take-up）。")
        else:
            try:
                res = _rdd_via_rdrobust(df, outcome, running, cutoff, d / "fuzzy_rdd_plot.png", fuzzy=treatment)
                if (d / "fuzzy_rdd_plot.png").exists():
                    files.append("fuzzy_rdd_plot.png")
                rd, lb, ub = res["rd_robust"], res["ci_lb"], res["ci_ub"]
                p, h = res["p_robust"], res["bw_h"]
                nl, nr = res["n_left"], res["n_right"]
                fsj, fsp = res.get("first_stage_jump", float("nan")), res.get("first_stage_p", float("nan"))
                weak = (not (abs(fsj) >= 0.1)) or (not (fsp < 0.05))  # NaN-safe: weak unless clearly strong
                pd.DataFrame(
                    {
                        "quantity": ["first_stage_jump(处理概率跳跃)", "LATE(模糊, 稳健偏差校正)"],
                        "estimate": [round(fsj, 4), round(rd, 4)],
                        "ci_lower": ["", round(lb, 4)],
                        "ci_upper": ["", round(ub, 4)],
                        "p_value": [round(fsp, 4), round(p, 4)],
                    }
                ).to_csv(d / "fuzzy_rdd_estimate.csv", index=False, encoding="utf-8")
                files.append("fuzzy_rdd_estimate.csv")
                estimates["late"] = round(rd, 4)
                estimates["ci_lb"] = round(lb, 4)
                estimates["ci_ub"] = round(ub, 4)
                estimates["p_value"] = round(p, 4)
                estimates["bandwidth"] = round(h, 4)
                estimates["first_stage_jump"] = round(fsj, 4)
                estimates["n_effective"] = float(nl + nr)
                sig = "显著" if p < 0.05 else "不显著"
                weak_note = (
                    f"⚠ 第一阶段弱（处理概率跳跃≈{fsj:.3f}，p={fsp:.3g}，不足以可信识别）——LATE 可能严重不稳，慎用。 "
                    if weak else ""
                )
                (d / "fuzzy_rdd_summary.txt").write_text(
                    f"模糊断点回归 fuzzy RDD（rdrobust，局部 IV，{res['bwselect']} 最优带宽）\n"
                    f"驱动变量 {running} @ 断点 {cutoff:g}，实际处理 {treatment}，结果 {outcome}\n"
                    f"第一阶段：断点处处理概率跳跃 ≈ {fsj:.4f}（p={fsp:.4g}）\n"
                    f"LATE（= 结果跳跃 / 处理概率跳跃）= {rd:.4f}，稳健 95% CI [{lb:.4f}, {ub:.4f}]，p={p:.4g}\n"
                    f"MSE 最优带宽 h={h:.4f}；带内有效样本 左 {nl} / 右 {nr}\n"
                    "注：模糊 RDD 仅识别断点附近『合规者(compliers)』的局部处理效应（外推性有限）；"
                    "依赖第一阶段在断点有真实跳跃（弱则 LATE 不可信）、潜在结果均值与第一阶段在断点连续、"
                    "无其他政策/混杂同址跳变、个体无法精确操纵驱动变量（应做 McCrary 密度检验）。\n",
                    encoding="utf-8",
                )
                files.append("fuzzy_rdd_summary.txt")
                summary.append(
                    f"{entry.method} 完成（rdrobust，fuzzy）：{running}@{cutoff:g} 处 {outcome} 的局部处理效应 "
                    f"LATE = {rd:.4f}（稳健 95% CI [{lb:.4f}, {ub:.4f}]，{sig}，p={p:.3g}）；"
                    f"第一阶段处理概率跳跃 ≈ {fsj:.3f}（p={fsp:.3g}）；最优带宽 h={h:.3f}，带内 {nl}+{nr}。"
                    + weak_note
                    + "⚠ 模糊 RDD 的 LATE = (结果跳跃)/(处理概率跳跃)，仅对断点附近合规者成立、外推有限；"
                    "假定潜在结果与第一阶段在断点连续、无其他政策同址跳变、驱动变量不可精确操纵（建议 McCrary 密度检验）。"
                )
                code += [
                    "from rdrobust import rdrobust  # 模糊断点回归(fuzzy RDD, 局部 IV)",
                    f"# rdrobust(y={outcome!r}, x={running!r}, c={cutoff!r}, fuzzy={treatment!r}); LATE=结果跳/处理跳",
                ]
            except Exception as err:
                summary.append(f"模糊 RDD 拟合失败：{err}")
