"""Causal family branch handler: gsynth (split from causal.py)."""
from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.run import _gsynth_via_r


@register("gsynth")
def _branch_gsynth(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import re

    from researchforge.executor import rbridge

    unit, time = fp.unit_col, fp.time_col
    _excl = {unit, time, *fp.treatment_candidates}
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
    outcome = cfg["outcome"] if cfg.get("outcome") in cont else (cont[0] if cont else None)
    treat = cfg.get("treatment") or next((c for c in fp.treatment_candidates if c in df.columns), None)
    try:
        n_boots = max(100, int(cfg.get("nboots", 200)))
    except (TypeError, ValueError):
        n_boots = 200
    try:
        gs_seed = int(cfg.get("seed", 2024))
    except (TypeError, ValueError):
        gs_seed = 2024  # malformed cfg["seed"] -> same reproducible default (audit: was 20260616)
    names_safe = all(
        x and re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", str(x)) for x in [outcome, treat, unit, time]
    )
    if not (unit and time):
        summary.append("广义合成控制失败：需要面板数据（单位列 + 时间列）。")
    elif not (rbridge.r_available() and rbridge.r_package_available("gsynth")):
        summary.append("广义合成控制需要 R 的 gsynth 包（未检测到）。安装：install.packages('gsynth')；或用 synthetic_control（单处理单位）/ did。")
    elif outcome is None or treat is None:
        summary.append("广义合成控制失败：需要连续结果变量 + 处理指示列(0/1, 随时间变)。用 config={\"treatment\":\"<列>\"} 指定。")
    elif not names_safe:
        summary.append("广义合成控制失败：列名需为标识符式（字母/数字/. _），R 公式要求。")
    else:
        import pandas as pd

        sub = df[[unit, time, outcome, treat]].dropna()
        tvals = set(pd.to_numeric(sub[treat], errors="coerce").dropna().unique())
        n_treated_units = sub[sub[treat] == 1][unit].nunique() if tvals <= {0, 1} else 0
        if not tvals <= {0, 1}:
            summary.append("广义合成控制失败：处理指示列需编码为 0/1（1=该单位该期受处理）。")
        elif n_treated_units < 1 or sub[sub[treat] == 0][unit].nunique() < 5:
            summary.append(
                f"广义合成控制失败：处理单位 {n_treated_units} 个或对照单位 "
                f"{sub[sub[treat] == 0][unit].nunique()} 个太少（对照需 ≥5 以建反事实）。"
            )
        else:
            import pandas as pd

            csv = d / "_gsynth_input.csv"
            sub.to_csv(csv, index=False)
            try:
                meta, att_t = _gsynth_via_r(csv, outcome, treat, unit, time, n_boots, d / "gsynth_att.png", seed=gs_seed)
                att_t.to_csv(d / "gsynth_att_by_time.csv", index=False, encoding="utf-8")
                files.append("gsynth_att_by_time.csv")
                if (d / "gsynth_att.png").exists():
                    files.append("gsynth_att.png")
                att, lb, ub = meta["att"], meta["ci_lb"], meta["ci_ub"]
                p, rcv = meta["pval"], int(meta["r_cv"])
                ntr, nco = int(meta["n_treated"]), int(meta["n_control"])
                estimates["att"] = round(att, 4)
                estimates["att_lb"] = round(lb, 4)
                estimates["att_ub"] = round(ub, 4)
                estimates["p_value"] = round(p, 4)
                estimates["n_factors"] = float(rcv)
                estimates["n_treated_units"] = float(ntr)
                sig = "显著" if (lb > 0 or ub < 0) else "不显著"
                # gsynth silently drops treated units with <5 pre-treatment periods
                # (needs them to estimate the factors); disclose if it happened (Opus catch).
                drop_txt = (
                    f"；⚠ {n_treated_units - ntr} 个处理单位因干预前期不足(需≥5)被 gsynth 剔除"
                    if n_treated_units > ntr else ""
                )
                (d / "gsynth_summary.txt").write_text(
                    f"广义合成控制 GSC（R gsynth，交互固定效应，two-way）\n"
                    f"结果 {outcome}，处理 {treat}；处理单位 {ntr} 个 / 对照 {nco} 个，潜在因子数 r={rcv}（CV 选）\n"
                    f"平均处理效应 ATT = {att:.4f}，95% CI [{lb:.4f}, {ub:.4f}]（{sig}，p={p:.4g}）\n"
                    "GSC = 合成控制 + 交互固定效应（潜在因子×载荷），可处理多处理单位/交错采纳，"
                    "比经典合成控制更灵活；反事实由对照单位 + 估计的因子结构外推。\n"
                    "假定：无未观测的时变混杂超出低维因子结构、对照未受溢出影响、平行的因子结构。\n\n"
                    "动态 ATT（按时间）：\n" + att_t.to_string(index=False),
                    encoding="utf-8",
                )
                files.append("gsynth_summary.txt")
                summary.append(
                    f"{entry.method} 完成（R/gsynth，交互固定效应）：结果 {outcome}，处理 {treat}；"
                    f"ATT={att:.4f}（95% CI [{lb:.4f}, {ub:.4f}]，{sig}，p={p:.3g}）；"
                    f"{ntr} 处理 / {nco} 对照单位，CV 选潜在因子 r={rcv}（seed={gs_seed} 可复现）。"
                    "⚠ 假定时变混杂被低维因子结构吸收、对照无溢出、处理单位需足够干预前期；多处理单位/交错采纳可用。"
                    + drop_txt
                )
                code += [
                    "library(gsynth)  # 广义合成控制(交互固定效应)",
                    f"# gsynth({outcome}~{treat}, index=c('{unit}','{time}'), force='two-way', CV=TRUE, r=c(0,5), se=TRUE)",
                ]
            except Exception as err:
                summary.append(f"广义合成控制拟合失败：{err}")
            finally:
                try:
                    csv.unlink()
                except OSError:
                    pass
