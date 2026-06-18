"""Branch handlers for the experimental-design family — design-aware analyses that
force explicit block/treatment/plot roles instead of treating a field trial as a flat
table. First member: RCBD. See executor/_branch_api.py.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register

# column-name hints for auto-detecting roles when config doesn't specify them
_BLOCK_HINTS = ("block", "rep", "replicate", "site", "field", "batch")
_TRT_HINTS = ("treat", "trt", "variety", "cultivar", "genotype", "hybrid", "factor", "dose", "level")


@register("rcbd")
def _branch_rcbd(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    _excl = {fp.unit_col, fp.time_col}
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
    # treatment/block are often integer-coded → accept count/id too (profiler "id" trap)
    role_cols = [c.name for c in fp.columns
                 if c.kind in {"categorical", "binary", "count", "id"} and c.name not in _excl]

    y = cfg["outcome"] if cfg.get("outcome") in cont else (cont[0] if cont else None)
    # config overrides accept ANY column (a role factor may profile as count/id, not categorical)
    trt_cfg = cfg.get("treatment") if cfg.get("treatment") in df.columns else None
    blk_cfg = cfg.get("block") if cfg.get("block") in df.columns else None
    treatment, block = trt_cfg, blk_cfg

    def _pick(hints: tuple[str, ...], taken: set) -> str | None:
        for c in role_cols:
            if c == y or c in taken:
                continue
            if any(h in c.lower() for h in hints):
                return c
        return None

    if treatment is None:
        treatment = _pick(_TRT_HINTS, {block})
    if block is None:
        block = _pick(_BLOCK_HINTS, {treatment})
    # last resort: fill remaining roles from leftover role columns
    leftover = [c for c in role_cols if c not in {treatment, block} and c != y]
    if treatment is None and leftover:
        treatment = leftover.pop(0)
    if block is None and leftover:
        block = leftover.pop(0)
    # nudge to set config unless BOTH roles were explicitly configured (name-hint/leftover picks too)
    guessed = not (trt_cfg is not None and blk_cfg is not None)

    if y is None or treatment is None or block is None or treatment == block:
        summary.append(
            "RCBD 失败：需要 1 个连续结果 + 2 个分类因子（处理 treatment + 区组 block）。"
            '用 config={"outcome":..,"treatment":..,"block":..} 指定角色。'
        )
        return

    sub = df[[y, treatment, block]].dropna()
    n_trt = int(sub[treatment].nunique())
    n_blk = int(sub[block].nunique())
    if n_trt < 2 or n_blk < 2 or len(sub) < n_trt + n_blk:
        summary.append(f"RCBD 失败：处理水平={n_trt}、区组={n_blk}、有效行={len(sub)}，不足以拟合。")
        return

    try:
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm

        formula = f'Q("{y}") ~ C(Q("{treatment}")) + C(Q("{block}"))'
        model = smf.ols(formula, data=sub).fit()
        # guard degenerate/near-saturated designs: tiny residual df or ~0 residual MS makes the
        # F-ratio explode into a spurious "highly significant" result (inference-reviewer must-fix).
        resid_df = float(model.df_resid)
        mse_resid = float(model.mse_resid)
        y_var = float(np.var(sub[y].to_numpy(dtype=float)))
        if resid_df < 1 or not np.isfinite(mse_resid) or mse_resid < 1e-9 * max(y_var, 1e-12):
            summary.append(
                f"RCBD 失败：残差自由度={resid_df:.0f}、残差均方≈0 —— 设计近饱和/不完整"
                "（单次重复无足够误差项），F 检验不可靠（分母≈0）。需每区组每处理≥1 次且有可估残差。"
            )
            return
        aov = anova_lm(model, typ=2)
        aov.to_csv(d / "anova_table.csv", encoding="utf-8")
        files.append("anova_table.csv")

        # exact term first (removes substring ambiguity when one name ⊂ the other), substring fallback
        trt_key = f'C(Q("{treatment}"))'
        trt_term = trt_key if trt_key in aov.index else next(
            (t for t in aov.index if t != "Residual" and treatment in t), None)
        f_trt = float(aov.loc[trt_term, "F"]) if trt_term else float("nan")
        p_trt = float(aov.loc[trt_term, "PR(>F)"]) if trt_term else float("nan")
        estimates["treatment_F"] = f_trt
        estimates["treatment_p"] = p_trt
        estimates["n_treatments"] = float(n_trt)
        estimates["n_blocks"] = float(n_blk)
        estimates["r_squared"] = float(model.rsquared)

        means = sub.groupby(treatment, observed=True)[y].agg(["mean", "std", "count"])
        means.to_csv(d / "treatment_means.csv", encoding="utf-8")
        files.append("treatment_means.csv")

        # complete RCBD = each treatment exactly once per block
        cell = sub.groupby([block, treatment], observed=True).size()
        balanced = bool((cell == 1).all()) and len(cell) == n_trt * n_blk
        bal_note = "完全平衡(每区组每处理 1 次)" if balanced else "⚠ 非完全平衡(有缺失/重复)——RCBD 假定每区组每处理恰 1 次"

        tukey_done = False
        try:
            from statsmodels.stats.multicomp import pairwise_tukeyhsd

            tuk = pairwise_tukeyhsd(sub[y].to_numpy(dtype=float), sub[treatment].astype(str).to_numpy())
            pd.DataFrame(tuk.summary().data[1:], columns=tuk.summary().data[0]).to_csv(
                d / "tukey_hsd.csv", index=False, encoding="utf-8")
            files.append("tukey_hsd.csv")
            tukey_done = True
        except Exception:
            pass

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            m = means["mean"]
            se = means["std"] / np.sqrt(means["count"].clip(lower=1))
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.errorbar(range(len(m)), m.to_numpy(), yerr=(1.96 * se).to_numpy(),
                        fmt="o", capsize=4, color="#4C72B0")
            ax.set_xticks(range(len(m)))
            ax.set_xticklabels([str(i) for i in m.index], rotation=30, ha="right")
            ax.set_xlabel(f"treatment ({treatment})")
            ax.set_ylabel(f"mean {y} (95% CI, block-unadjusted)")
            ax.set_title(f"RCBD treatment means — {y}")
            fig.tight_layout()
            fig.savefig(d / "treatment_means.png", dpi=150)
            plt.close(fig)
            files.append("treatment_means.png")
        except Exception:
            pass

        role_note = "（角色自动猜测，建议 config 明确 treatment/block）" if guessed else "（角色由 config/列名确定）"
        sig = "显著" if (p_trt == p_trt and p_trt < 0.05) else "不显著"
        summary.append(
            f"{entry.method} 完成{role_note}：{y} ~ 处理 {treatment}（{n_trt} 水平）+ 区组 {block}（{n_blk} 区组）；"
            f"处理效应 F={f_trt:.3f}, p={p_trt:.3g}（{sig}）；R²={model.rsquared:.3f}。{bal_note}。"
            + ("已出 Tukey HSD 两两比较。" if tukey_done else "")
            + " ⚠ 区组为控制田间/批次梯度的干扰因子（非研究兴趣）；假定处理+区组可加（单次重复无法估"
            "处理×区组交互）；残差正态/等方差假定；区组亦可作随机效应（混合模型）。"
            " Tukey HSD 与均值±CI **未扣除区组**（用一元误差项），比 RCBD 的 F 检验保守、口径不同。"
        )
        code += [
            "import statsmodels.formula.api as smf",
            "from statsmodels.stats.anova import anova_lm",
            f'model = smf.ols(\'Q("{y}") ~ C(Q("{treatment}")) + C(Q("{block}"))\', data=df).fit()',
            "print(anova_lm(model, typ=2))  # RCBD：处理 F 检验，控制区组",
        ]
    except Exception as err:
        summary.append(f"RCBD 失败：{err}")
