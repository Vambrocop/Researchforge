"""Experimental-design family branch handler: latin_square (split from experimental_design.py)."""
from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register

from ._shared import _COL_HINTS, _ROW_HINTS, _TRT_HINTS, _degenerate_fit


@register("latin_square")
def _branch_latin_square(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    _excl = {fp.unit_col, fp.time_col}
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
    role_cols = [c.name for c in fp.columns
                 if c.kind in {"categorical", "binary", "count", "id"} and c.name not in _excl]

    y = cfg["outcome"] if cfg.get("outcome") in cont else (cont[0] if cont else None)
    treatment = cfg.get("treatment") if cfg.get("treatment") in df.columns else None
    row = cfg.get("row") if cfg.get("row") in df.columns else None
    col = cfg.get("col") if cfg.get("col") in df.columns else None

    def _pick(hints, taken):
        for c in role_cols:
            if c == y or c in taken:
                continue
            if any(h in c.lower() for h in hints):
                return c
        return None

    if row is None:
        row = _pick(_ROW_HINTS, {treatment, col})
    if col is None:
        col = _pick(_COL_HINTS, {treatment, row})
    if treatment is None:
        treatment = _pick(_TRT_HINTS, {row, col})
    leftover = [c for c in role_cols if c not in {treatment, row, col} and c != y]
    if treatment is None and leftover:
        treatment = leftover.pop(0)
    if row is None and leftover:
        row = leftover.pop(0)
    if col is None and leftover:
        col = leftover.pop(0)
    guessed = not (cfg.get("treatment") in df.columns and cfg.get("row") in df.columns
                   and cfg.get("col") in df.columns)

    roles = [treatment, row, col]
    if y is None or any(r is None for r in roles) or len(set(roles)) < 3:
        summary.append(
            "拉丁方设计失败：需要 1 个连续结果 + 处理 treatment + 行 row + 列 col（两个区组方向）。"
            '用 config={"outcome":..,"treatment":..,"row":..,"col":..} 指定。'
        )
        return

    sub = df[[y, treatment, row, col]].dropna()
    t, nr, nc = int(sub[treatment].nunique()), int(sub[row].nunique()), int(sub[col].nunique())
    if not (t == nr == nc) or t < 3:
        summary.append(f"拉丁方设计失败：处理={t}、行={nr}、列={nc} —— 拉丁方需 处理数=行数=列数 ≥3。")
        return
    # valid Latin square: one obs per (row,col); each treatment once per row and once per col
    rc = sub.groupby([row, col], observed=True).size()
    rt = sub.groupby([row, treatment], observed=True).size()
    ct = sub.groupby([col, treatment], observed=True).size()
    if not (bool((rc == 1).all()) and len(rc) == t * t
            and bool((rt == 1).all()) and bool((ct == 1).all())):
        summary.append("拉丁方设计失败：非有效拉丁方（每处理须在每行、每列各恰现 1 次，且每行列格 1 obs）。")
        return

    try:
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm

        formula = f'Q("{y}") ~ C(Q("{treatment}")) + C(Q("{row}")) + C(Q("{col}"))'
        model = smf.ols(formula, data=sub).fit()
        if _degenerate_fit(model, sub[y]):
            summary.append("拉丁方设计失败：残差自由度不足/残差均方≈0 —— 方阵太小或近饱和，F 检验不可靠。")
            return
        aov = anova_lm(model, typ=2)
        aov.to_csv(d / "anova_table.csv", encoding="utf-8")
        files.append("anova_table.csv")

        trt_key = f'C(Q("{treatment}"))'
        trt_term = trt_key if trt_key in aov.index else next(
            (tt for tt in aov.index if tt != "Residual" and treatment in tt), None)
        f_trt = float(aov.loc[trt_term, "F"]) if trt_term else float("nan")
        p_trt = float(aov.loc[trt_term, "PR(>F)"]) if trt_term else float("nan")
        estimates.update({"treatment_F": f_trt, "treatment_p": p_trt,
                          "n_treatments": float(t), "r_squared": float(model.rsquared)})

        means = sub.groupby(treatment, observed=True)[y].agg(["mean", "std", "count"])
        means.to_csv(d / "treatment_means.csv", encoding="utf-8")
        files.append("treatment_means.csv")

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
            ax.errorbar(range(len(m)), m.to_numpy(), yerr=(1.96 * se).to_numpy(), fmt="o", capsize=4, color="#4C72B0")
            ax.set_xticks(range(len(m)))
            ax.set_xticklabels([str(i) for i in m.index], rotation=30, ha="right")
            ax.set_xlabel(f"treatment ({treatment})")
            ax.set_ylabel(f"mean {y} (95% CI, unadjusted)")
            ax.set_title(f"Latin square treatment means — {y}")
            fig.tight_layout()
            fig.savefig(d / "treatment_means.png", dpi=150)
            plt.close(fig)
            files.append("treatment_means.png")
        except Exception:
            pass

        role_note = "（行/列/处理角色自动猜测，建议 config 明确）" if guessed else ""
        sig = "显著" if (p_trt == p_trt and p_trt < 0.05) else "不显著"
        summary.append(
            f"{entry.method} 完成{role_note}：{y} ~ 处理 {treatment}（{t}）+ 行 {row} + 列 {col}（{t}×{t} 方阵）；"
            f"处理效应 F={f_trt:.3f}, p={p_trt:.3g}（{sig}）；R²={model.rsquared:.3f}。"
            + ("已出 Tukey HSD。" if tukey_done else "")
            + " ⚠ 拉丁方同时控制行、列两个梯度（干扰因子）；残差自由度仅 (t-1)(t-2)，小方阵功效低；"
            "假定无 处理×行/列 交互（可加模型）；Tukey/CI 未扣行列。残差正态/等方差假定。"
        )
        code += [
            "import statsmodels.formula.api as smf",
            "from statsmodels.stats.anova import anova_lm",
            f'model = smf.ols(\'Q("{y}") ~ C(Q("{treatment}")) + C(Q("{row}")) + C(Q("{col}"))\', data=df).fit()',
            "print(anova_lm(model, typ=2))  # 拉丁方：处理 F，控制行 + 列",
        ]
    except Exception as err:
        summary.append(f"拉丁方设计失败：{err}")
