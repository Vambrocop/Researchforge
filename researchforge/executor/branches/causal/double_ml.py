"""Causal family branch handler: double_ml (split from causal.py)."""
from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.run import _dml_via_doubleml


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
