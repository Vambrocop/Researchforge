"""Branch handlers for the causal family (migrated from the run.py monolith).

Each handler unpacks ctx into the same local names run_analysis used and runs the
original branch body verbatim. See executor/_branch_api.py.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.run import (
    _causal_forest_via_econml,
    _cic_via_r,
    _dml_via_doubleml,
    _gsynth_via_r,
    _rdd_via_rdrobust,
    _synthetic_control,
)


@register("causal_forest")
def _branch_causal_forest(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import importlib.util

    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in {fp.unit_col, fp.time_col}]
    treatment = cfg.get("treatment")
    if treatment is None:
        treatment = next(
            (c.name for c in fp.columns if c.kind == "binary" and c.name not in {fp.unit_col, fp.time_col}),
            None,
        )
    outcome = cfg["outcome"] if cfg.get("outcome") in cont else next((c for c in cont if c != treatment), None)
    forced_mod = [c for c in (cfg.get("effect_modifiers") or cfg.get("controls") or cfg.get("predictors") or []) if c in df.columns and c not in {outcome, treatment}]
    if forced_mod:
        modifiers = forced_mod[:15]
    else:
        modifiers = [
            c.name for c in fp.columns
            if c.kind in {"continuous", "count", "binary"}
            and c.name not in {outcome, treatment, fp.unit_col, fp.time_col}
        ][:12]
    try:
        n_folds = max(2, int(cfg.get("n_folds", 4)))
    except (TypeError, ValueError):
        n_folds = 4
    try:
        seed = int(cfg.get("seed", 0))
    except (TypeError, ValueError):
        seed = 0
    if importlib.util.find_spec("econml") is None:
        summary.append("因果森林需要 econml 包（未检测到）。安装：pip install econml；或用 double_ml（平均效应 ATE）。")
    elif treatment is None:
        summary.append("因果森林失败：需要一个处理变量。用 config={\"treatment\":\"<列>\"} 指定。")
    elif outcome is None:
        summary.append("因果森林失败：需要一个连续结果变量（≠ treatment）。")
    elif not modifiers:
        summary.append("因果森林失败：需要 ≥1 个协变量/效应修饰因子（用于估计异质效应 CATE）。")
    else:
        try:
            import pandas as pd

            disc = df[treatment].dropna().nunique() == 2
            fdr_method = cfg.get("fdr_method") if cfg.get("fdr_method") in {"fdr_bh", "fdr_by"} else "fdr_bh"
            res = _causal_forest_via_econml(
                df, outcome, treatment, modifiers, n_folds, disc, seed,
                d / "cate_hist.png", d / "cate_vs_modifier.png", fdr_method=fdr_method,
            )
            for png in ("cate_hist.png", "cate_vs_modifier.png"):
                if (d / png).exists():
                    files.append(png)
            ate, alb, aub = res["ate"], res["ate_lb"], res["ate_ub"]
            csd, p10, p90 = res["cate_sd"], res["cate_p10"], res["cate_p90"]
            frac = res["frac_significant"]
            frac_bh = res.get("frac_significant_bh", float("nan"))
            fdr_lab = "BY-FDR(任意相关下有效)" if res.get("fdr_method") == "fdr_by" else "BH-FDR"
            drivers = res["drivers"]
            pd.DataFrame(drivers, columns=["modifier", "importance"]).to_csv(
                d / "heterogeneity_drivers.csv", index=False, encoding="utf-8"
            )
            files.append("heterogeneity_drivers.csv")
            estimates["ate"] = round(ate, 4)
            estimates["ate_lb"] = round(alb, 4)
            estimates["ate_ub"] = round(aub, 4)
            estimates["cate_sd"] = round(csd, 4)
            estimates["frac_significant"] = round(frac, 3)
            estimates["frac_significant_bh"] = round(frac_bh, 3) if frac_bh == frac_bh else float("nan")
            top = drivers[0]
            # heterogeneity heuristic: CATE spread large relative to |ATE|
            het = "明显" if csd > 0.25 * (abs(ate) + 1e-9) else "较弱"
            enc_txt = ""
            if res.get("treat_map"):
                k0 = [k for k, v in res["treat_map"].items() if v == 0][0]
                k1 = [k for k, v in res["treat_map"].items() if v == 1][0]
                enc_txt = f"；处理编码 {k0}→0、{k1}→1"
            (d / "causal_forest.txt").write_text(
                f"因果森林 CATE（econml CausalForestDML，RandomForest，{n_folds} 折，seed={seed}）\n"
                f"处理 {treatment} → 结果 {outcome}，效应修饰因子 {len(modifiers)} 个{enc_txt}\n"
                f"总体 ATE = {ate:.4f}，95% CI [{alb:.4f}, {aub:.4f}]\n"
                f"CATE 分布：均值 {res['cate_mean']:.4f}，SD {csd:.4f}，[P10,P90]=[{p10:.4f},{p90:.4f}]\n"
                f"个体效应显著（逐行 95% CI 不含 0）占比 {frac:.0%}（未校正，零效应下基线约 5%）；"
                f"{fdr_lab} 多重比较校正后占比 {frac_bh:.0%}（FDR≈5%；BH 在独立/正相关下成立，"
                "森林 CATE 近似满足、未严格证；如需任意相关下的保守保证用 config['fdr_method']='fdr_by'）；"
                f"异质性{het}\n"
                f"异质性主要驱动（特征重要性）：{drivers[:3]}\n"
                "CATE 用 ML 估计异质效应；因果解释同样依赖无未观测混杂 + 重叠假定；"
                "CATE 偏探索性，子组结论需预注册/外部验证（防多重比较假阳性）；"
                "若处理编码方向反了，请用 config['treatment'] 传入预编码 {0,1}。\n",
                encoding="utf-8",
            )
            files.append("causal_forest.txt")
            summary.append(
                f"{entry.method} 完成（econml，RF×{n_folds}折，seed={seed}）：处理 {treatment} → {outcome}；"
                f"ATE={ate:.4f}（95% CI [{alb:.4f}, {aub:.4f}]）；CATE 异质性{het}"
                f"（SD={csd:.3f}，P10–P90=[{p10:.3f},{p90:.3f}]，个体效应显著 {frac:.0%} 未校正"
                f"/ {frac_bh:.0%} {fdr_lab} 校正）；"
                f"主要驱动 {top[0]}（重要性 {top[1]}）。"
                "⚠ 因果解释依赖无未观测混杂 + 重叠；CATE 偏探索、子组结论需外部验证。" + enc_txt
            )
            code += [
                "from econml.dml import CausalForestDML  # 因果森林(异质处理效应 CATE)",
                f"# CausalForestDML(discrete_treatment={disc}, random_state={seed}).fit(Y,T,X=modifiers); effect(X)",
            ]
        except Exception as err:
            summary.append(f"因果森林拟合失败：{err}")



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



@register("double_ml")
def _branch_double_ml(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import importlib.util

    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in {fp.unit_col, fp.time_col}]
    # treatment: config, else a binary column (treatment candidate)
    treatment = cfg.get("treatment")
    if treatment is None:
        treatment = next(
            (c.name for c in fp.columns if c.kind == "binary" and c.name not in {fp.unit_col, fp.time_col}),
            None,
        )
    outcome = cfg["outcome"] if cfg.get("outcome") in cont else next((c for c in cont if c != treatment), None)
    forced_ctrl = [c for c in (cfg.get("controls") or cfg.get("predictors") or []) if c in df.columns and c not in {outcome, treatment}]
    if forced_ctrl:
        controls = forced_ctrl[:20]
    else:
        controls = [
            c.name for c in fp.columns
            if c.kind in {"continuous", "count", "binary"}
            and c.name not in {outcome, treatment, fp.unit_col, fp.time_col}
        ][:15]
    try:
        n_folds = max(2, int(cfg.get("n_folds", 5)))
    except (TypeError, ValueError):
        n_folds = 5
    if importlib.util.find_spec("doubleml") is None:
        summary.append("双重机器学习需要 doubleml 包（未检测到）。安装：pip install doubleml；或用 ols_regression（线性控制）。")
    elif treatment is None:
        summary.append(
            "双重机器学习失败：需要一个处理变量。用 config={\"treatment\":\"<列>\"} 指定"
            "（二值→ATE/IRM，连续→偏线性/PLR）。"
        )
    elif outcome is None:
        summary.append("双重机器学习失败：需要一个连续结果变量（≠ treatment）。")
    elif not controls:
        summary.append("双重机器学习失败：需要 ≥1 个混杂/控制变量（DML 靠它们去混杂）。")
    else:
        try:
            import pandas as pd

            disc = df[treatment].dropna().nunique() == 2
            try:
                seed = int(cfg.get("seed", 0))
            except (TypeError, ValueError):
                seed = 0
            res = _dml_via_doubleml(df, outcome, treatment, controls, n_folds, disc, d / "dml_ate.png", seed=seed)
            if (d / "dml_ate.png").exists():
                files.append("dml_ate.png")
            ate, lb, ub = res["ate"], res["ci_lb"], res["ci_ub"]
            p, kind = res["p_value"], res["model"]
            label = res["estimand"]  # "ATE" (IRM) or "PLR_coefficient" (continuous treatment)
            pd.DataFrame(
                {"quantity": [label], "estimate": [round(ate, 4)], "se": [round(res["se"], 4)],
                 "ci_lower": [round(lb, 4)], "ci_upper": [round(ub, 4)], "p_value": [round(p, 4)]}
            ).to_csv(d / "dml_estimate.csv", index=False, encoding="utf-8")
            files.append("dml_estimate.csv")
            estimates["ate"] = round(ate, 4)
            estimates["ci_lb"] = round(lb, 4)
            estimates["ci_ub"] = round(ub, 4)
            estimates["p_value"] = round(p, 4)
            estimates["n_controls"] = float(len(controls))
            sig = "显著" if p < 0.05 else "不显著"
            overlap_txt = (
                "；⚠ 倾向得分接近 0/1（重叠/正值性可疑，ATE 不稳，检查处理组与对照的协变量重叠）"
                if res["overlap_warn"] else ""
            )
            mlabel = "IRM（二值处理 ATE）" if kind == "IRM" else "PLR（偏线性）"
            est_word = "平均处理效应 ATE" if kind == "IRM" else "PLR 偏线性系数（假定处理的边际效应恒定）"
            enc_txt = ""
            if res.get("treat_map"):
                k0 = [k for k, v in res["treat_map"].items() if v == 0][0]
                k1 = [k for k, v in res["treat_map"].items() if v == 1][0]
                enc_txt = f"；处理编码 {k0}→0、{k1}→1（ATE 为 1 相对 0 的效应）"
            r2y, r2d = res.get("nuisance_r2_y", float("nan")), res.get("nuisance_r2_d", float("nan"))
            if r2y == r2y:
                estimates["nuisance_r2_y"] = round(r2y, 4)
            if r2d == r2d:
                estimates["nuisance_r2_d"] = round(r2d, 4)
            if kind == "PLR":
                nui_txt = f"干扰项交叉拟合 R²（样本外）：结果 E[Y|X]={r2y:.3f}，处理 E[D|X]={r2d:.3f}"
            else:
                nui_txt = f"干扰项交叉拟合 R²（样本外）：结果 E[Y|X,D]={r2y:.3f}（倾向模型为分类器，R² 不适用）"
            nui_note = "；⚠ 结果干扰项 R²≈0（控制变量几乎不解释 Y，混杂调整很弱）" if (r2y == r2y and r2y < 0.05) else ""
            (d / "dml_summary.txt").write_text(
                f"双重/去偏机器学习 DML（doubleml {kind}，RandomForest 学习器，{n_folds} 折交叉拟合，seed={seed}）\n"
                f"处理 {treatment} → 结果 {outcome}，控制 {len(controls)} 个协变量{enc_txt}\n"
                f"{est_word} = {ate:.4f}（SE={res['se']:.4f}），95% CI [{lb:.4f}, {ub:.4f}]，p={p:.4g}\n"
                f"{nui_txt}\n"
                "DML 用 ML 学习干扰项(结果模型+处理/倾向模型)再正交化+交叉拟合，去偏后做 √n 推断。\n"
                "假定：无未观测混杂(条件可忽略)、重叠(正值性)；因果解释依赖这些假定，非自动成立。\n"
                "注：干扰项 R² 是诊断（看控制变量解释力）——DML 正交化对中等干扰误差一阶稳健，"
                "R² 高低非有效性门槛；但 R²≈0 表示控制变量近乎无信息、混杂调整弱。\n"
                "注：交叉拟合样本切分已按 seed 固定（可复现）；可用 config seed 改。\n",
                encoding="utf-8",
            )
            files.append("dml_summary.txt")
            summary.append(
                f"{entry.method} 完成（doubleml {mlabel}，RF×{n_folds}折，seed={seed}）：处理 {treatment} → {outcome}，"
                f"控制 {len(controls)} 协变量；{est_word}={ate:.4f}（95% CI [{lb:.4f}, {ub:.4f}]，{sig}，p={p:.3g}）；"
                f"{nui_txt}。"
                + enc_txt + "⚠ 因果解释依赖无未观测混杂 + 重叠假定；≠净相关；干扰项 R² 为诊断(非有效性门槛)。"
                + nui_note + overlap_txt
            )
            code += [
                "import doubleml as dml  # 双重/去偏机器学习",
                f"# DoubleML{kind}(DoubleMLData(y={outcome!r}, d={treatment!r}, x=controls), RF, n_folds={n_folds}).fit()",
            ]
        except Exception as err:
            summary.append(f"双重机器学习拟合失败：{err}")



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
        gs_seed = 20260616
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



@register("iv_regression")
def _branch_iv_regression(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    summary.append(
        "工具变量回归（2SLS）需要你指定外生工具变量（instrument），引擎无法自动识别。"
        "请在指定工具变量后手动运行；或先用 panel_fixed_effects / did 作为可自动执行的替代。"
    )



@register("mediation")
def _branch_mediation(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    _excl = {fp.unit_col, fp.time_col}
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
    y_col = cont[0] if cont else None
    cand = [
        c.name
        for c in fp.columns
        if c.kind in {"continuous", "binary"} and c.name not in _excl | {y_col}
    ]
    if y_col is None or len(cand) < 2:
        summary.append("中介分析失败：需要连续结果变量 Y + ≥2 个变量（自变量 X、中介 M）。")
    else:
        x_col, m_col = cand[0], cand[1]  # default by column order; X→M→Y assumption
        try:
            import statsmodels.api as sm
            from statsmodels.stats.mediation import Mediation

            sub = df[[y_col, x_col, m_col]].dropna().rename(
                columns={y_col: "_y", x_col: "_x", m_col: "_m"}
            )
            om = sm.OLS.from_formula("_y ~ _x + _m", sub)  # outcome: Y ~ X + M
            mm = sm.OLS.from_formula("_m ~ _x", sub)  # mediator: M ~ X
            med = Mediation(om, mm, "_x", "_m").fit(n_rep=1000)
            s = med.summary()
            s.to_csv(d / "mediation_summary.csv", encoding="utf-8")
            files.append("mediation_summary.csv")

            def _row(label):
                return s.loc[label] if label in s.index else None

            acme = _row("ACME (average)")
            ade = _row("ADE (average)")
            tot = _row("Total effect")
            pm = _row("Prop. mediated (average)")
            indirect = float(acme["Estimate"])
            direct = float(ade["Estimate"])
            total = float(tot["Estimate"])
            prop = float(pm["Estimate"]) if pm is not None else float("nan")
            acme_p = float(acme["P-value"])
            estimates["indirect_effect_ACME"] = round(indirect, 4)
            estimates["direct_effect_ADE"] = round(direct, 4)
            estimates["total_effect"] = round(total, 4)
            estimates["prop_mediated"] = round(prop, 4)
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                labels = ["indirect (ACME)", "direct (ADE)", "total"]
                est = [indirect, direct, total]
                lo = [float(acme["Lower CI bound"]), float(ade["Lower CI bound"]), float(tot["Lower CI bound"])]
                hi = [float(acme["Upper CI bound"]), float(ade["Upper CI bound"]), float(tot["Upper CI bound"])]
                err = [[e - l for e, l in zip(est, lo)], [h - e for e, h in zip(est, hi)]]
                fig, ax = plt.subplots(figsize=(5.5, 3.2))
                ax.errorbar(est, range(3), xerr=err, fmt="o", capsize=4)
                ax.axvline(0, color="grey", ls="--")
                ax.set_yticks(range(3))
                ax.set_yticklabels(labels)
                ax.set_xlabel("effect (95% CI)")
                ax.set_title(f"Mediation {x_col} → {m_col} → {y_col}")
                fig.tight_layout()
                fig.savefig(d / "mediation_effects.png", dpi=150)
                plt.close(fig)
                files.append("mediation_effects.png")
            except Exception:
                pass
            verdict = "存在显著中介" if acme_p < 0.05 else "中介效应不显著"
            # prop. mediated is meaningless under suppression (opposite signs) or
            # near-zero total effect — flag rather than print a misleading % (Opus catch).
            suppression = abs(total) < 0.05 or (direct * indirect < 0)
            prop_txt = "不稳定（抑制效应/总效应近零，比例无意义）" if suppression else f"{prop:.1%}"
            summary.append(
                f"{entry.method} 完成：路径 {x_col} → {m_col} → {y_col}；"
                f"间接效应 ACME={indirect:.4f}（p={acme_p:.3g}，{verdict}），"
                f"直接效应 ADE={direct:.4f}，总效应={total:.4f}，中介比例={prop_txt}（Monte-Carlo CI）。"
                "⚠ X/M/Y 按列序默认（首连续=Y，其后=X、M），且 **X↔M 不对称**——交换二者是不同模型、"
                "列序只是选了一个假设而非事实，请核对你的理论路径；中介推断需 no-unmeasured-confounding 假定（非纯相关即因果）。"
            )
            code += [
                "from statsmodels.stats.mediation import Mediation",
                f"# OLS('{y_col}~{x_col}+{m_col}') + OLS('{m_col}~{x_col}'); Mediation(...).fit(n_rep=1000)",
            ]
        except Exception as err:
            summary.append(f"中介分析失败：{err}")



@register("rdd")
def _branch_rdd(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import importlib.util

    import pandas as pd

    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in {fp.unit_col, fp.time_col}]
    running = cfg.get("running") or cfg.get("running_var")
    outcome = cfg["outcome"] if cfg.get("outcome") in cont else next((c for c in cont if c != running), None)
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



@register("synthetic_control")
def _branch_synthetic_control(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    unit, time = fp.unit_col, fp.time_col
    _excl = {unit, time, *fp.treatment_candidates}
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
    outcome = cfg["outcome"] if cfg.get("outcome") in cont else (cont[0] if cont else None)
    # treated unit + treatment time: config, else derive from a treatment 0/1 column
    treated = cfg.get("treated_unit")
    treat_time = cfg.get("treatment_time")
    ever_treated: set = set()  # ALL units ever treated → excluded from donor pool
    if unit and time and fp.treatment_candidates:
        tcol = next((c for c in fp.treatment_candidates if c in df.columns), None)
        if tcol is not None:
            trows = df[df[tcol] == 1]
            if len(trows):
                ever_treated = set(trows[unit].dropna().unique())
                if treated is None:
                    treated = trows[unit].dropna().unique()[0]
                if treat_time is None:
                    treat_time = df[(df[unit] == treated) & (df[tcol] == 1)][time].min()
    # coerce a JSON-supplied treatment_time to the time column's dtype
    if treat_time is not None and time and time in df.columns:
        try:
            treat_time = type(df[time].dropna().iloc[0])(treat_time)
        except Exception:
            pass
    forced = [c for c in (cfg.get("predictors") or []) if c in df.columns and c not in {outcome, unit, time}]
    preds = forced or [c for c in cont if c != outcome][:5]
    if not (unit and time):
        summary.append("合成控制失败：需要面板数据（单位列 + 时间列）。")
    elif outcome is None:
        summary.append("合成控制失败：需要一个连续结果变量。")
    elif treated is None or treat_time is None:
        summary.append(
            "合成控制失败：需指定干预单位与干预时点。"
            "用 config={\"treated_unit\": \"<单位>\", \"treatment_time\": <时点>}，"
            "或在数据中提供处理指示列（0/1）。"
        )
    else:
        try:
            import pysyncon  # noqa: F401
        except ImportError:
            summary.append("合成控制需要 pysyncon 包（未检测到）。安装：pip install pysyncon；或用 did。")
        else:
            try:
                import pandas as pd

                # exclude OTHER ever-treated units from donors (contamination bias).
                other_treated = {u for u in ever_treated if u != treated}
                weights, att, pre_rmspe, n_don, post = _synthetic_control(
                    df, unit, time, outcome, treated, treat_time, preds, d / "gaps.png",
                    exclude=other_treated,
                )
                if len(ever_treated) > 1:
                    contam_note = (
                        f"；⚠ 检测到 {len(ever_treated)} 个被处理单位（疑交错采纳），"
                        f"仅对 {treated} 建模、其余已剔出供体池；交错处理建议改用 did"
                    )
                elif not ever_treated:
                    contam_note = "；⚠ 由 config 指定处理单位、未据处理列校验供体，假定供体未受干预"
                else:
                    contam_note = ""
                wdf = weights.reset_index()
                wdf.columns = ["donor", "weight"]
                wdf = wdf[wdf["weight"] > 1e-4].reset_index(drop=True)
                wdf.to_csv(d / "donor_weights.csv", index=False, encoding="utf-8")
                files.append("donor_weights.csv")
                if (d / "gaps.png").exists():
                    files.append("gaps.png")
                att_val = float(att.get("att", float("nan")))
                att_se = float(att.get("se", float("nan")))
                estimates["att"] = round(att_val, 4)
                estimates["att_se"] = round(att_se, 4)
                estimates["pre_rmspe"] = round(pre_rmspe, 4)
                estimates["n_donors_used"] = float(len(wdf))
                top = wdf.head(5)
                top_txt = ", ".join(f"{r.donor}={r.weight:.2f}" for r in top.itertuples())
                (d / "synth_summary.txt").write_text(
                    f"合成控制法（Abadie）：干预单位 {treated}，干预时点 {treat_time}，"
                    f"结果变量 {outcome}\n"
                    f"ATT（干预后平均处理效应）= {att_val:.4f}（SE≈{att_se:.4f}）\n"
                    f"干预前拟合 RMSPE = {pre_rmspe:.4f}（越小=合成体越贴合真实前期路径）\n"
                    f"合成体权重（前 5）：{top_txt}\n"
                    f"对照供体池 {n_don} 个，实际赋权 {len(wdf)} 个。\n"
                    "注：ATT 可信度取决于干预前拟合好坏（RMSPE 小）；推断需做安慰剂检验。\n\n"
                    + wdf.to_string(index=False),
                    encoding="utf-8",
                )
                files.append("synth_summary.txt")
                summary.append(
                    f"{entry.method} 完成（pysyncon）：干预单位 {treated} @ {treat_time}，"
                    f"结果 {outcome}；ATT={att_val:.4f}（SE≈{att_se:.4f}）；"
                    f"干预前 RMSPE={pre_rmspe:.4f}；合成体由 {len(wdf)} 个供体加权（{top_txt}）。"
                    "⚠ ATT 可信度依赖干预前拟合；正式推断需安慰剂检验（in-space/in-time placebo）。"
                    + contam_note
                )
                code += [
                    "from pysyncon import Dataprep, Synth  # 合成控制法",
                    f"# 干预单位={treated}, 干预时点={treat_time}; 干预前拟合权重 -> 后期 gap=ATT",
                ]
            except Exception as err:
                summary.append(f"合成控制拟合失败：{err}")


@register("psm")
def _branch_psm(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    _excl = {fp.unit_col, fp.time_col}
    bins = [c.name for c in fp.columns if c.kind == "binary" and c.name not in _excl]
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
    treatment = cfg.get("treatment") if cfg.get("treatment") in df.columns else (
        fp.treatment_candidates[0] if fp.treatment_candidates else (bins[0] if bins else None))
    outcome = cfg.get("outcome") if cfg.get("outcome") in df.columns else (
        next((c for c in cont if c != treatment), None))
    if cfg.get("covariates"):
        covs = [c for c in cfg["covariates"] if c in df.columns and c not in {outcome, treatment}]
    else:
        covs = [c.name for c in fp.columns if c.kind in {"continuous", "binary", "count"}
                and c.name not in (_excl | {outcome, treatment})]
    if treatment is None or outcome is None or not covs:
        summary.append('倾向得分匹配失败：需要 二值处理 + 连续结果 + ≥1 协变量。'
                       'config={"treatment":..,"outcome":..,"covariates":[..]}。')
        return

    sub = df[[outcome, treatment, *covs]].dropna().copy()
    tvals = set(pd.unique(sub[treatment].dropna()))
    if not (tvals <= {0, 1}):
        if len(tvals) == 2:  # map two arbitrary values to 0/1 (higher = treated)
            hi = sorted(tvals)[1]
            sub[treatment] = (sub[treatment] == hi).astype(int)
        else:
            summary.append("倾向得分匹配失败：处理变量必须是二值（0/1 或恰两类）。")
            return
    sub[treatment] = sub[treatment].astype(int)
    n_t, n_c = int((sub[treatment] == 1).sum()), int((sub[treatment] == 0).sum())
    if n_t < 5 or n_c < 5:
        summary.append(f"倾向得分匹配失败：处理组 {n_t}、对照组 {n_c}，样本太少。")
        return

    try:
        import statsmodels.formula.api as smf
        from scipy import stats as _st

        rhs = " + ".join(f'Q("{c}")' for c in covs)
        ps_model = smf.logit(f'Q("{treatment}") ~ {rhs}', data=sub).fit(disp=0)
        ps = ps_model.predict(sub).clip(1e-6, 1 - 1e-6)
        sub["_lp"] = np.log(ps / (1 - ps))  # match on the logit (linear predictor), per Austin
        caliper = 0.2 * float(sub["_lp"].std(ddof=1))
        treated = sub[sub[treatment] == 1]
        controls = sub[sub[treatment] == 0]
        ctrl_lp = controls["_lp"].to_dict()

        used: set = set()
        pairs: list[tuple] = []  # greedy 1:1 NN on _lp, no replacement, within caliper
        for ti, trow in treated.sort_values("_lp", ascending=False).iterrows():
            best, bestd = None, None  # nearest unused control WITHIN the caliper
            for ci, lp in ctrl_lp.items():
                if ci in used:
                    continue
                dlp = abs(trow["_lp"] - lp)
                if dlp <= caliper and (bestd is None or dlp < bestd):
                    bestd, best = dlp, ci
            if best is not None:
                used.add(best)
                pairs.append((ti, best))
        if len(pairs) < 3:
            summary.append(f"倾向得分匹配失败：卡尺内仅匹配到 {len(pairs)} 对（共同支撑不足）。")
            return

        t_idx = [p[0] for p in pairs]
        c_idx = [p[1] for p in pairs]
        diffs = sub.loc[t_idx, outcome].to_numpy(dtype=float) - sub.loc[c_idx, outcome].to_numpy(dtype=float)
        att = float(diffs.mean())
        se = float(diffs.std(ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else float("nan")
        tstat = att / se if se and se > 0 else float("nan")
        pval = float(2 * _st.norm.sf(abs(tstat))) if tstat == tstat else float("nan")

        def _smd(a, b):
            a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
            sp = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
            return float((a.mean() - b.mean()) / sp) if sp > 1e-12 else 0.0

        bal = pd.DataFrame([
            {"covariate": c,
             "smd_before": round(_smd(treated[c], controls[c]), 3),
             "smd_after": round(_smd(sub.loc[t_idx, c], sub.loc[c_idx, c]), 3)}
            for c in covs
        ])
        bal.to_csv(d / "balance.csv", index=False, encoding="utf-8")
        files.append("balance.csv")
        max_smd_after = float(bal["smd_after"].abs().max())

        estimates.update({"att": att, "se": se, "pvalue": pval, "n_treated": float(n_t),
                          "n_matched_pairs": float(len(pairs)), "max_abs_smd_after": max_smd_after})

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(treated["_lp"], bins=20, alpha=0.5, label="treated", color="#C44E52")
            ax.hist(controls["_lp"], bins=20, alpha=0.5, label="control", color="#4C72B0")
            ax.set_xlabel("propensity (logit)")
            ax.set_ylabel("count")
            ax.set_title("Propensity overlap (common support)")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "propensity_overlap.png", dpi=150)
            plt.close(fig)
            files.append("propensity_overlap.png")
        except Exception:
            pass

        bal_ok = "达标(|SMD|<0.1)" if max_smd_after < 0.1 else f"⚠ 残留不平衡(最大|SMD|={max_smd_after:.2f})"
        sig = "显著" if (pval == pval and pval < 0.05) else "不显著"
        summary.append(
            f"{entry.method} 完成：ATT={att:.4f}（SE={se:.4f}, p={pval:.3g}，{sig}）；"
            f"匹配 {len(pairs)} 对（处理组 {n_t}/对照 {n_c}）；匹配后协变量平衡 {bal_ok}。"
            " ⚠ PSM 假定**可忽略性/选择仅基于可观测**（无未观测混杂）——不可检验的强假设，"
            "PSM 不能修正未观测混杂；估计的是 **ATT**（对处理组）非 ATE；需共同支撑 + 匹配后平衡。"
            " SE 为配对差简化估计（未计倾向得分估计不确定性、亦未计无放回匹配的依赖；Abadie-Imbens 方差更严）。"
        )
        code += [
            "import statsmodels.formula.api as smf  # 倾向得分匹配 (PSM)",
            f"# logit({treatment} ~ 协变量) -> 倾向得分 -> 线性预测子上 1:1 最近邻(卡尺 0.2σ) -> ATT",
        ]
    except Exception as err:
        summary.append(f"倾向得分匹配失败：{err}")


@register("ipw")
def _branch_ipw(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    _excl = {fp.unit_col, fp.time_col}
    bins = [c.name for c in fp.columns if c.kind == "binary" and c.name not in _excl]
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
    treatment = cfg.get("treatment") if cfg.get("treatment") in df.columns else (
        fp.treatment_candidates[0] if fp.treatment_candidates else (bins[0] if bins else None))
    outcome = cfg.get("outcome") if cfg.get("outcome") in df.columns else (
        next((c for c in cont if c != treatment), None))
    if cfg.get("covariates"):
        covs = [c for c in cfg["covariates"] if c in df.columns and c not in {outcome, treatment}]
    else:
        covs = [c.name for c in fp.columns if c.kind in {"continuous", "binary", "count"}
                and c.name not in (_excl | {outcome, treatment})]
    if treatment is None or outcome is None or not covs:
        summary.append('逆概率加权失败：需要 二值处理 + 连续结果 + ≥1 协变量。'
                       'config={"treatment":..,"outcome":..,"covariates":[..]}。')
        return

    sub = df[[outcome, treatment, *covs]].dropna().copy()
    tvals = set(pd.unique(sub[treatment].dropna()))
    treated_level = None
    if not (tvals <= {0, 1}):
        if len(tvals) == 2:
            treated_level = sorted(tvals)[1]  # higher value = treated; disclose so the ATE sign is unambiguous
            sub[treatment] = (sub[treatment] == treated_level).astype(int)
        else:
            summary.append("逆概率加权失败：处理变量必须是二值（0/1 或恰两类）。")
            return
    sub[treatment] = sub[treatment].astype(int)
    n_t, n_c = int((sub[treatment] == 1).sum()), int((sub[treatment] == 0).sum())
    if n_t < 5 or n_c < 5:
        summary.append(f"逆概率加权失败：处理组 {n_t}、对照组 {n_c}，样本太少。")
        return

    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf

        rhs = " + ".join(f'Q("{c}")' for c in covs)
        ps = smf.logit(f'Q("{treatment}") ~ {rhs}', data=sub).fit(disp=0).predict(sub).to_numpy()
        ps = np.clip(ps, 1e-3, 1 - 1e-3)  # bound to avoid exploding weights
        t = sub[treatment].to_numpy(dtype=float)
        y = sub[outcome].to_numpy(dtype=float)
        p_treat = float(t.mean())
        sw = np.where(t == 1, p_treat / ps, (1 - p_treat) / (1 - ps))  # stabilized weights

        # ATE via the marginal structural model: WLS of y ~ T weighted by stabilized weights
        wls = sm.WLS(y, sm.add_constant(t), weights=sw).fit(cov_type="HC1")
        ate, se, pval = float(wls.params[1]), float(wls.bse[1]), float(wls.pvalues[1])
        ess = float(sw.sum() ** 2 / (sw ** 2).sum())          # effective sample size
        extreme = float((((ps < 0.05) | (ps > 0.95)).mean()))  # poor-overlap fraction
        estimates.update({"ate": ate, "se": se, "pvalue": pval, "ess": ess,
                          "max_weight": float(sw.max()), "extreme_ps_frac": extreme,
                          "n": float(len(sub))})

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(ps[t == 1], bins=20, alpha=0.5, label="treated", color="#C44E52")
            ax.hist(ps[t == 0], bins=20, alpha=0.5, label="control", color="#4C72B0")
            ax.set_xlabel("propensity score")
            ax.set_ylabel("count")
            ax.set_title("Propensity overlap (IPW positivity)")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "propensity_overlap.png", dpi=150)
            plt.close(fig)
            files.append("propensity_overlap.png")
        except Exception:
            pass

        overlap = "良好" if extreme < 0.05 and ess > 0.5 * len(sub) else f"⚠ 重叠/正性存疑(极端倾向 {extreme:.0%}, ESS={ess:.0f}/{len(sub)})"
        sig = "显著" if (pval == pval and pval < 0.05) else "不显著"
        trt_note = f"（处理组 = {treatment}='{treated_level}'）" if treated_level is not None else ""
        summary.append(
            f"{entry.method} 完成：ATE={ate:.4f}（HC1 SE={se:.4f}, p={pval:.3g}，{sig}）{trt_note}；"
            f"稳定化权重，ESS={ess:.0f}/{len(sub)}，最大权重 {sw.max():.2f}；重叠 {overlap}。"
            " ⚠ IPW 估 **ATE**，因果有效仅在**可忽略性 + 正性/重叠**(无未观测混杂、处理概率不近 0/1)下成立——"
            "极端权重会放大方差/偏差(已报 ESS 与极端倾向占比);HC1 SE 未计倾向得分估计不确定性(自助/三明治更严)。"
        )
        code += [
            "import statsmodels.api as sm, statsmodels.formula.api as smf  # 逆概率加权(IPW/MSM)",
            f"# 稳定化权重 sw = T·P(T)/e + (1-T)·P(0)/(1-e); WLS({outcome}~T, weights=sw).T 系数 = ATE",
        ]
    except Exception as err:
        summary.append(f"逆概率加权失败：{err}")


@register("event_study")
def _branch_event_study(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import re

    import numpy as np  # noqa: F401
    import pandas as pd

    unit = cfg.get("unit") if cfg.get("unit") in df.columns else fp.unit_col
    time = cfg.get("time") if cfg.get("time") in df.columns else fp.time_col
    if not unit or not time:
        summary.append('事件研究失败：需要面板数据（单位列 + 时间列）。config={"unit":..,"time":..}。')
        return
    _excl = {unit, time}
    bins_ = [c.name for c in fp.columns if c.kind == "binary" and c.name not in _excl]
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
    treatment = cfg.get("treatment") if cfg.get("treatment") in df.columns else (
        fp.treatment_candidates[0] if fp.treatment_candidates else (bins_[0] if bins_ else None))
    outcome = cfg.get("outcome") if cfg.get("outcome") in df.columns else (
        next((c for c in cont if c != treatment), None))
    if treatment is None or outcome is None:
        summary.append('事件研究失败：需要 二值处理(随时间开启) + 连续结果。config={"treatment":..,"outcome":..}。')
        return

    sub = df[[unit, time, treatment, outcome]].dropna().copy()
    tvals = set(pd.unique(sub[treatment].dropna()))
    if not (tvals <= {0, 1}):
        if len(tvals) == 2:
            sub[treatment] = (sub[treatment] == sorted(tvals)[1]).astype(int)
        else:
            summary.append("事件研究失败：处理变量必须二值。")
            return
    sub[treatment] = sub[treatment].astype(int)
    sub[time] = pd.to_numeric(sub[time], errors="coerce")
    sub = sub.dropna(subset=[time])

    onset = sub[sub[treatment] == 1].groupby(unit)[time].min()
    if onset.empty:
        summary.append("事件研究失败：没有任何单位被处理（处理从未开启）。")
        return
    sub["_onset"] = sub[unit].map(onset)
    sub["_evt"] = sub[time] - sub["_onset"]  # NaN for never-treated (kept as comparison group)

    try:
        L = max(2, min(12, int(cfg.get("window", 5))))
    except (TypeError, ValueError):
        L = 5

    def _bin(e):
        return "never" if pd.isna(e) else str(int(max(-L, min(L, e))))

    sub["_evtb"] = sub["_evt"].apply(_bin)
    has_never = bool(sub["_evt"].isna().any())
    treated_bins = sorted({b for b in sub["_evtb"] if b != "never"}, key=int)
    if len(treated_bins) < 2:
        summary.append("事件研究失败：处理单位的事件时间变化不足（需要前后多期）。")
        return
    # fully-staggered with NO never-treated comparison -> TWFE event study is under-identified
    # (needs a 2nd normalization beyond k=-1) and biased under heterogeneity; statsmodels pinv
    # would silently spread the estimate -> fail honestly instead (inference-reviewer must-fix).
    if not has_never and onset.nunique() > 1:
        summary.append("事件研究失败：所有单位最终都被处理且为交错采纳——纯 TWFE 事件研究此情形需第二个"
                       "归一化(再固定一个远端 lead)、且异质效应下有偏；请改用 Callaway-Sant'Anna / de Chaisemartin。")
        return
    # reference MUST be a pre-treatment lead — a lag reference would invert the event-study reading
    leads_avail = [b for b in treated_bins if int(b) < 0]
    if "-1" in treated_bins:
        ref = "-1"
    elif leads_avail:
        ref = max(leads_avail, key=int)  # closest available pre-period
    else:
        summary.append("事件研究失败：没有任何处理前(lead)期可作参照——无法识别动态效应基线"
                       "（单位是否在 onset 当期才进入面板？）。")
        return

    try:
        import statsmodels.formula.api as smf

        formula = (f'Q("{outcome}") ~ C(_evtb, Treatment("{ref}")) '
                   f'+ C(Q("{unit}")) + C(Q("{time}"))')
        model = smf.ols(formula, data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub[unit]})

        ci = model.conf_int()
        rx = re.compile(r"C\(_evtb.*?\)\[T\.(-?\d+)\]")
        rows = [{"event_time": int(ref), "coef": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p": float("nan")}]
        for name in model.params.index:
            m = rx.match(name)
            if not m:
                continue
            rows.append({"event_time": int(m.group(1)), "coef": float(model.params[name]),
                         "ci_low": float(ci.loc[name, 0]), "ci_high": float(ci.loc[name, 1]),
                         "p": float(model.pvalues[name])})
        es = pd.DataFrame(rows).sort_values("event_time").reset_index(drop=True)
        es.to_csv(d / "event_study.csv", index=False, encoding="utf-8")
        files.append("event_study.csv")

        leads = es[es["event_time"] < -1]
        pretrend_bad = bool((leads["p"] < 0.05).any()) if len(leads) else False
        post = es[es["event_time"] >= 0]
        att_post = float(post["coef"].mean()) if len(post) else float("nan")
        staggered = bool(onset.nunique() > 1)  # >1 distinct onset TIME = staggered
        estimates.update({"att_post_mean": att_post, "n_event_coefs": float(len(es) - 1),
                          "pretrend_violation": 1.0 if pretrend_bad else 0.0,
                          "n_treated_units": float(len(onset))})  # count of treated units

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.axhline(0, color="gray", lw=0.8)
            ax.axvline(-0.5, color="red", ls="--", lw=0.8)
            ax.errorbar(es["event_time"], es["coef"],
                        yerr=[es["coef"] - es["ci_low"], es["ci_high"] - es["coef"]],
                        fmt="o", capsize=3, color="#4C72B0")
            ax.set_xlabel(f"event time (relative to onset; ref={ref})")
            ax.set_ylabel(f"effect on {outcome}")
            ax.set_title("Event study (dynamic treatment effects)")
            fig.tight_layout()
            fig.savefig(d / "event_study.png", dpi=150)
            plt.close(fig)
            files.append("event_study.png")
        except Exception:
            pass

        pt = "⚠ 检出预趋势(平行趋势存疑)" if pretrend_bad else "前置期系数未见显著(支持平行趋势)"
        stag_note = (" ⚠ 处理时点**交错**——双向固定效应事件研究在异质处理效应下可能有偏(负权重)，"
                     "稳健做法见 Callaway-Sant'Anna / de Chaisemartin。" if staggered else "")
        summary.append(
            f"{entry.method} 完成：{outcome} 围绕处理开启的动态效应（参照期 k={ref}，窗口 ±{L}）；"
            f"处理后(k≥0)平均效应 {att_post:.4f}；{len(es)-1} 个事件期系数（图 event_study.png）。{pt}。"
            f" ⚠ DiD 识别依赖**平行趋势**（前置期系数应≈0，已检）；单位+时间双向固定效应、按单位聚类 SE；"
            f"处理后平均为描述性汇总(非加权 ATT)；端点(±{L})为合并累计期、非单期效应。{stag_note}"
        )
        code += [
            "import statsmodels.formula.api as smf  # 事件研究(动态 DiD)",
            f'# evt=time-onset; ols(Q("{outcome}") ~ C(evt,Treatment("{ref}"))+C(unit)+C(time)).fit(cluster=unit)',
        ]
    except Exception as err:
        summary.append(f"事件研究失败：{err}")


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


@register("staggered_did")
def _branch_staggered_did(ctx: Ctx) -> None:
    # Sun & Abraham (2021) interaction-weighted estimator: heterogeneity-robust event-study /
    # ATT under STAGGERED adoption. Fits cohort-specific CATT(g,e) (cohort x relative-time dummies,
    # e=-1 omitted, never-treated as the clean control), then aggregates ATT(e) = Σ_g (N_g/ΣN) CATT(g,e)
    # weighted by cohort sample shares — avoiding the negative-weight ("bad comparison") bias of pooled
    # TWFE event studies. SE via delta method on the cluster-robust covariance. Pure Python (no R did).
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    unit = cfg.get("unit") if cfg.get("unit") in df.columns else fp.unit_col
    time = cfg.get("time") if cfg.get("time") in df.columns else fp.time_col
    if not unit or not time:
        summary.append('交错DiD失败：需要面板数据(单位列+时间列)。config={"unit":..,"time":..}。')
        return
    _excl = {unit, time}
    bins_ = [c.name for c in fp.columns if c.kind == "binary" and c.name not in _excl]
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
    treatment = cfg.get("treatment") if cfg.get("treatment") in df.columns else (
        fp.treatment_candidates[0] if fp.treatment_candidates else (bins_[0] if bins_ else None))
    outcome = cfg.get("outcome") if cfg.get("outcome") in df.columns else (
        next((c for c in cont if c != treatment), None))
    if treatment is None or outcome is None:
        summary.append('交错DiD失败：需要 二值处理(随时间开启) + 连续结果。config={"treatment":..,"outcome":..}。')
        return

    sub = df[[unit, time, treatment, outcome]].dropna().copy()
    tvals = set(pd.unique(sub[treatment].dropna()))
    if not (tvals <= {0, 1}):
        if len(tvals) == 2:
            sub[treatment] = (sub[treatment] == sorted(tvals)[1]).astype(int)
        else:
            summary.append("交错DiD失败：处理变量必须二值。")
            return
    sub[treatment] = sub[treatment].astype(int)
    sub[time] = pd.to_numeric(sub[time], errors="coerce")
    sub = sub.dropna(subset=[time])

    onset = sub[sub[treatment] == 1].groupby(unit)[time].min()
    if onset.empty:
        summary.append("交错DiD失败：没有任何单位被处理。")
        return
    sub["_cohort"] = sub[unit].map(onset)        # NaN = never-treated
    has_never = bool(sub["_cohort"].isna().any())
    # Sun-Abraham needs a CLEAN control group. We require never-treated units (the most robust
    # control): with none, the saturated cohort x period spec is collinear with unit+time FE (would
    # need last-cohort-as-control + period trimming) -> degrade honestly toward Callaway-Sant'Anna.
    if not has_never:
        summary.append("交错DiD失败：Sun-Abraham 交互加权估计需要『从未处理』对照组(本数据所有单位最终都被处理)；"
                       "请改用 R did(Callaway-Sant'Anna，可用『尚未处理』作对照)或提供含从未处理单位的面板。")
        return
    cohorts = sorted(float(c) for c in onset.unique())

    try:
        L = max(2, min(12, int(cfg.get("window", 5))))
    except (TypeError, ValueError):
        L = 5

    sub["_rel"] = sub[time] - sub["_cohort"]   # NaN for never-treated

    def _relbin(r):
        return np.nan if pd.isna(r) else int(max(-L, min(L, r)))

    sub["_relb"] = sub["_rel"].apply(_relbin)

    # cohort-specific relative-time dummies (e=-1 reference); never-treated & e=-1 are all-zero.
    # Column names are guaranteed-safe identifiers so patsy param names == column names (no quoting).
    dummy_map = {}   # (cohort, e) -> column name
    for gi, g in enumerate(cohorts):
        for e in range(-L, L + 1):
            if e == -1:
                continue
            mask = (sub["_cohort"] == g) & (sub["_relb"] == e)
            if int(mask.sum()) == 0:
                continue
            col = f"saD_{gi}_{e + L}"
            sub[col] = mask.astype(float)
            dummy_map[(g, e)] = col
    if not dummy_map:
        summary.append("交错DiD失败：处理单位的事件时间变化不足(需要前后多期)。")
        return

    try:
        import statsmodels.formula.api as smf
        from scipy.stats import norm

        rhs = " + ".join(dummy_map.values())
        formula = f'Q("{outcome}") ~ {rhs} + C(Q("{unit}")) + C(Q("{time}"))'
        model = smf.ols(formula, data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub[unit]})
        params, Vc = model.params, model.cov_params()
        # bare safe identifier names match patsy param names verbatim; fail LOUD if any coefficient is
        # missing (e.g. dropped for collinearity) rather than silently mislabeling a CATT.
        missing = [col for col in dummy_map.values() if col not in params.index]
        if missing:
            summary.append(f"交错DiD失败：{len(missing)} 个队列×事件期系数未进入模型(可能共线)，无法可靠聚合。")
            return
        pname = {ge: col for ge, col in dummy_map.items()}
        # Sun-Abraham weight for CATT(g,e): share of cohort g among units OBSERVED at relative time e
        # (per-(g,e) unit counts, not a fixed cohort size -> correct under UNBALANCED panels too;
        # for a balanced panel this reduces to the cohort-size share).
        treated_obs = sub[sub["_cohort"].notna()]
        ng_e = treated_obs.groupby(["_cohort", "_relb"])[unit].nunique()

        def _agg_vector(e):
            present = [g for g in cohorts if (g, e) in dummy_map]
            counts = {g: float(ng_e.get((g, e), 0.0)) for g in present}
            tot = float(sum(counts.values()))
            if tot <= 0:
                return None, present
            vec = pd.Series(0.0, index=params.index)
            for g in present:
                vec[pname[(g, e)]] = counts[g] / tot
            return vec, present

        def _att_se(vec):
            att = float(vec.values @ params.values)
            var = float(vec.values @ Vc.values @ vec.values)
            se = float(np.sqrt(max(var, 0.0)))
            z = att / se if se > 0 else float("nan")
            p = float(2 * (1 - norm.cdf(abs(z)))) if se > 0 else float("nan")
            return att, se, p

        event_times = sorted({e for (_, e) in dummy_map})
        rows = [{"event_time": -1, "att": 0.0, "se": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                 "p": float("nan"), "n_cohorts": 0}]
        post_vecs = []
        for e in event_times:
            vec, present = _agg_vector(e)
            if vec is None:
                continue
            att, se, p = _att_se(vec)
            rows.append({"event_time": e, "att": att, "se": se, "ci_low": att - 1.96 * se,
                         "ci_high": att + 1.96 * se, "p": p, "n_cohorts": len(present)})
            if e >= 0:
                post_vecs.append(vec.values)
        es = pd.DataFrame(rows).sort_values("event_time").reset_index(drop=True)
        es.to_csv(d / "staggered_did.csv", index=False, encoding="utf-8")
        files.append("staggered_did.csv")

        # overall post ATT = simple average of post-period IW event-study ATTs (linear combo -> delta-method SE)
        if post_vecs:
            lpost = pd.Series(sum(post_vecs) / len(post_vecs), index=params.index)
            att_overall, se_overall, p_overall = _att_se(lpost)
        else:
            att_overall = se_overall = p_overall = float("nan")

        leads = es[es["event_time"] < -1]
        pretrend_bad = bool((leads["p"] < 0.05).any()) if len(leads) else False
        n_never = int(sub[sub["_cohort"].isna()][unit].nunique())
        estimates.update({
            "att_overall": round(att_overall, 4), "att_overall_se": round(se_overall, 4),
            "att_overall_p": round(p_overall, 4) if p_overall == p_overall else float("nan"),
            "n_cohorts": float(len(cohorts)), "n_treated_units": float(len(onset)),
            "n_never_treated": float(n_never), "pretrend_violation": 1.0 if pretrend_bad else 0.0,
        })

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.axhline(0, color="gray", lw=0.8)
            ax.axvline(-0.5, color="red", ls="--", lw=0.8)
            ax.errorbar(es["event_time"], es["att"],
                        yerr=[es["att"] - es["ci_low"], es["ci_high"] - es["att"]],
                        fmt="o", capsize=3, color="#55A868")
            ax.set_xlabel("event time (relative to onset; ref = -1)")
            ax.set_ylabel(f"IW ATT on {outcome}")
            ax.set_title("Staggered DiD — Sun-Abraham interaction-weighted ATT")
            fig.tight_layout()
            fig.savefig(d / "staggered_did.png", dpi=150)
            plt.close(fig)
            files.append("staggered_did.png")
        except Exception:
            pass

        pt = "⚠ 检出预趋势(平行趋势存疑)" if pretrend_bad else "前置期 ATT 未见显著(支持平行趋势)"
        sig = "显著" if (p_overall == p_overall and p_overall < 0.05) else "不显著"
        summary.append(
            f"{entry.method} 完成(Sun-Abraham 交互加权)：{len(cohorts)} 个处理队列 / {len(onset)} 个处理单位 / "
            f"{n_never} 个从未处理对照；总体处理后 ATT = {att_overall:.4f}"
            f"(SE {se_overall:.4f}，{sig}，p={p_overall:.3g})；事件期 IW-ATT 见 staggered_did.png。{pt}。"
            " ⚠ 交错采纳下纯 TWFE 事件研究会受『坏对照(已处理单位被当对照)』负权重污染——本估计先拟合队列特定 "
            "CATT(g,e)、再按队列样本份额加权聚合(Sun & Abraham 2021)，对异质处理效应稳健；识别仍依赖**平行趋势**"
            "(前置期 ATT 应≈0，已检；队列数少时该检验功效有限，无显著≠平行趋势成立)与从未处理对照干净；总体 ATT "
            "为处理后各事件期 IW-ATT 的**简单平均**(非按格元样本量加权，与 Callaway-Sant'Anna simple 聚合略异)；"
            "端点(±窗口)为合并累计期。"
        )
        code += [
            "import statsmodels.formula.api as smf  # 交错DiD(Sun-Abraham 交互加权)",
            '# y ~ Σ 1[cohort=g & (t-g)=e, e≠-1] + C(unit) + C(time); cluster by unit',
            "# ATT(e)=Σ_g (N_g/ΣN)·CATT(g,e); SE via L·cov·Lᵀ (delta method)",
        ]
    except Exception as err:
        summary.append(f"交错DiD失败：{err}")

