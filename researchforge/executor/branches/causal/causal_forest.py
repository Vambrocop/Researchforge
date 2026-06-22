"""Causal family branch handler: causal_forest (split from causal.py)."""
from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.run import _causal_forest_via_econml


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
