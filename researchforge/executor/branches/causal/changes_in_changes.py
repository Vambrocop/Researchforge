"""Causal family branch handler: changes_in_changes (split from causal.py)."""
from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.run import _cic_via_r


@register("changes_in_changes")
def _branch_changes_in_changes(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import re

    import pandas as pd

    from researchforge.executor import rbridge

    time = cfg.get("time") or fp.time_col
    _excl = {fp.unit_col, time}
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
    outcome = cfg["outcome"] if cfg.get("outcome") in cont else (cont[0] if cont else None)
    treat = cfg.get("treatment")
    if treat is None:
        treat = next(
            (c.name for c in fp.columns if c.kind == "binary" and c.name not in {outcome, time, fp.unit_col}),
            None,
        )
    # periods: config [pre,post], else the last two distinct time values; must be numeric
    periods = sorted(pd.to_numeric(df[time], errors="coerce").dropna().unique()) if (time and time in df.columns) else []
    want = cfg.get("periods")
    if isinstance(want, (list, tuple)) and len(want) == 2:
        try:
            t_pre, t_post = float(want[0]), float(want[1])
        except (TypeError, ValueError):
            t_pre = t_post = None
    elif len(periods) >= 2:
        t_pre, t_post = float(periods[-2]), float(periods[-1])
    else:
        t_pre = t_post = None
    probs = cfg.get("probs") or [round(0.1 * i, 2) for i in range(1, 10)]
    names_safe = all(
        outcome and treat and time and re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", str(c))
        for c in [outcome, treat, time]
    )
    if not (rbridge.r_available() and rbridge.r_package_available("qte")):
        summary.append("Changes-in-changes 需要 R 的 qte 包（未检测到）。安装：install.packages('qte')；或用 did / quantile_regression。")
    elif outcome is None or treat is None or not time:
        summary.append(
            "Changes-in-changes 失败：需要 结果变量 + 处理组指示(二值) + 时间变量(两期)。"
            "用 config={\"outcome\":..,\"treatment\":..,\"time\":..} 指定。"
        )
    elif t_pre is None or t_post is None:
        summary.append("Changes-in-changes 失败：时间列需为数值且 ≥2 期（或 config periods=[前,后]）。")
    elif not names_safe:
        summary.append("Changes-in-changes 失败：列名需为标识符式（字母/数字/. _），R 公式要求。")
    else:
        sub = df[[outcome, treat, time]].copy()
        sub[time] = pd.to_numeric(sub[time], errors="coerce")
        sub = sub[sub[time].isin([t_pre, t_post])].dropna()
        # Normalize the group indicator to {0,1}: CiC imputes the COUNTERFACTUAL
        # for group==1, so which value maps to 1 (treated) determines the SIGN and
        # WHICH group's effect is estimated — a lexicographic guess can silently
        # invert it (Opus catch). Prefer config['treated_group']; else use the {0,1}
        # convention; else map sorted but disclose the direction prominently.
        tvals = sorted(sub[treat].dropna().unique().tolist(), key=lambda v: str(v))
        treated_group = cfg.get("treated_group")
        treat_map, treated_label, dir_explicit = None, None, False
        if len(tvals) == 2:
            if treated_group is not None and treated_group in tvals:
                other = [v for v in tvals if v != treated_group][0]
                treat_map = {other: 0, treated_group: 1}
                sub[treat] = sub[treat].map(treat_map)
                treated_label, dir_explicit = treated_group, True
            elif set(tvals) == {0, 1}:
                treated_label = 1  # standard 1=treated convention; no remap
            else:
                treat_map = {tvals[0]: 0, tvals[1]: 1}
                sub[treat] = sub[treat].map(treat_map)
                treated_label = tvals[1]
        if sub[treat].dropna().nunique() != 2:
            summary.append("Changes-in-changes 失败：处理组指示需恰好 2 类（处理 vs 对照）。")
        elif any(sub[(sub[time] == p)][treat].nunique() < 2 for p in (t_pre, t_post)):
            summary.append("Changes-in-changes 失败：每期都需同时有处理组与对照组样本。")
        else:
            csv = d / "_cic_input.csv"
            sub.to_csv(csv, index=False)
            try:
                meta, qte = _cic_via_r(csv, outcome, treat, time, t_post, t_pre, probs, d / "cic_qte.png")
                qte.to_csv(d / "cic_qte.csv", index=False, encoding="utf-8")
                files.append("cic_qte.csv")
                if (d / "cic_qte.png").exists():
                    files.append("cic_qte.png")
                att, alb, aub = meta["ate"], meta["ate_lb"], meta["ate_ub"]
                estimates["att"] = round(att, 4)
                estimates["att_lb"] = round(alb, 4)
                estimates["att_ub"] = round(aub, 4)
                estimates["qte_min"] = round(float(qte["qte"].min()), 4)
                estimates["qte_max"] = round(float(qte["qte"].max()), 4)
                sig = "显著" if (alb > 0 or aub < 0) else "不显著"
                spread = float(qte["qte"].max() - qte["qte"].min())
                het = "效应随分位明显变化（分布性异质）" if spread > 0.5 * (abs(att) + 1e-9) else "效应在各分位较一致"
                # ALWAYS disclose which value is treated(=1) — direction sets the sign
                enc_txt = (
                    f"；处理组(=1) 取 [{treated_label}]"
                    + ("（config 指定）" if dir_explicit else "，方向若反请用 config treated_group 指定")
                )
                if len(periods) > 2 and not (isinstance(want, (list, tuple)) and len(want) == 2):
                    enc_txt += f"；⚠ 检测到 >2 期，仅用最后两期 {t_pre:g}→{t_post:g}（其余忽略，可用 config periods 指定）"
                (d / "cic_summary.txt").write_text(
                    f"Changes-in-changes（Athey-Imbens 2006，R qte::CiC，{t_pre:g}→{t_post:g}）\n"
                    f"结果 {outcome}，处理组 {treat}，时间 {time}{enc_txt}\n"
                    f"总体 ATT = {att:.4f}，95% CI [{alb:.4f}, {aub:.4f}]（{sig}）\n"
                    f"分位处理效应 QTE 范围 [{qte['qte'].min():.4f}, {qte['qte'].max():.4f}]；{het}\n"
                    "CiC 是 DID 的分布推广：放松「平行趋势」为单调/秩不变假定，识别整条反事实分布；"
                    "QTE 看处理对结果分布不同位置的异质影响。仍需无预期/无组成变化等 DID 类假定。\n\n"
                    + qte.to_string(index=False),
                    encoding="utf-8",
                )
                files.append("cic_summary.txt")
                summary.append(
                    f"{entry.method} 完成（R/qte，{t_pre:g}→{t_post:g}）：结果 {outcome}，处理组 {treat}；"
                    f"ATT={att:.4f}（95% CI [{alb:.4f}, {aub:.4f}]，{sig}）；"
                    f"QTE 范围 [{qte['qte'].min():.3f}, {qte['qte'].max():.3f}]，{het}。"
                    "⚠ DID 的分布版：放松平行趋势为单调/秩不变；仍依赖无预期/无组成变化等假定。" + enc_txt
                )
                code += [
                    "library(qte)  # changes-in-changes (Athey-Imbens 分布 DID)",
                    f"# CiC({outcome} ~ {treat}, t={t_post:g}, tmin1={t_pre:g}, tname='{time}', panel=FALSE, se=TRUE)",
                ]
            except Exception as err:
                summary.append(f"Changes-in-changes 拟合失败：{err}")
            finally:
                try:
                    csv.unlink()
                except OSError:
                    pass
