"""Experimental-design family branch handler: factorial_anova (split from experimental_design.py)."""
from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register

from ._shared import _degenerate_fit


@register("factorial_anova")
def _branch_factorial_anova(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np  # noqa: F401  (kept for parity / future use)
    import pandas as pd  # noqa: F401

    _excl = {fp.unit_col, fp.time_col}
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
    role_cols = [c.name for c in fp.columns
                 if c.kind in {"categorical", "binary", "count", "id"} and c.name not in _excl]

    y = cfg["outcome"] if cfg.get("outcome") in cont else (cont[0] if cont else None)
    fa = cfg.get("factor_a") if cfg.get("factor_a") in df.columns else None
    fb = cfg.get("factor_b") if cfg.get("factor_b") in df.columns else None
    cands = [c for c in role_cols if c != y and c not in {fa, fb}]
    if fa is None and cands:
        fa = cands.pop(0)
    if fb is None and cands:
        fb = cands.pop(0)
    guessed = not (cfg.get("factor_a") in df.columns and cfg.get("factor_b") in df.columns)

    if y is None or fa is None or fb is None or fa == fb:
        summary.append(
            "双因素方差分析失败：需要 1 个连续结果 + 2 个分类因子。"
            '用 config={"outcome":..,"factor_a":..,"factor_b":..} 指定。'
        )
        return

    sub = df[[y, fa, fb]].dropna()
    na, nb = int(sub[fa].nunique()), int(sub[fb].nunique())
    n_cells = sub.groupby([fa, fb], observed=True).ngroups
    if na < 2 or nb < 2 or len(sub) <= na * nb:
        summary.append(
            f"双因素方差分析失败：A={na} 水平、B={nb} 水平、有效行={len(sub)}；"
            "含交互的因子设计需每格 >1 次重复（否则无残差估交互）。"
        )
        return

    try:
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm

        formula = f'Q("{y}") ~ C(Q("{fa}")) * C(Q("{fb}"))'
        model = smf.ols(formula, data=sub).fit()
        if _degenerate_fit(model, sub[y]):
            summary.append("双因素方差分析失败：残差自由度不足/残差均方≈0 —— 设计近饱和，F 检验不可靠。")
            return
        aov = anova_lm(model, typ=2)
        aov.to_csv(d / "anova_table.csv", encoding="utf-8")
        files.append("anova_table.csv")

        # exact term keys (robust to one factor name being a substring of the other)
        ka, kb = f'C(Q("{fa}"))', f'C(Q("{fb}"))'
        t_a = ka if ka in aov.index else None
        t_b = kb if kb in aov.index else None
        t_int = next((t for t in (f"{ka}:{kb}", f"{kb}:{ka}") if t in aov.index), None)

        def _fp(term):
            if term and term in aov.index:
                return float(aov.loc[term, "F"]), float(aov.loc[term, "PR(>F)"])
            return float("nan"), float("nan")

        fA, pA = _fp(t_a)
        fB, pB = _fp(t_b)
        fI, pI = _fp(t_int)
        estimates.update({"A_F": fA, "A_p": pA, "B_F": fB, "B_p": pB,
                          "interaction_F": fI, "interaction_p": pI, "r_squared": float(model.rsquared)})

        cell = sub.groupby([fa, fb], observed=True)[y].mean().unstack()
        cell.to_csv(d / "cell_means.csv", encoding="utf-8")
        files.append("cell_means.csv")

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 4.5))
            for b_lvl in cell.columns:
                ax.plot([str(a) for a in cell.index], cell[b_lvl].to_numpy(), marker="o", label=str(b_lvl))
            ax.set_xlabel(f"factor A ({fa})")
            ax.set_ylabel(f"mean {y}")
            ax.set_title(f"Interaction plot — {y}")
            ax.legend(fontsize=8, title=str(fb))
            fig.tight_layout()
            fig.savefig(d / "interaction_plot.png", dpi=150)
            plt.close(fig)
            files.append("interaction_plot.png")
        except Exception:
            pass

        sizes = sub.groupby([fa, fb], observed=True).size()
        balanced = bool((sizes == sizes.iloc[0]).all()) and n_cells == na * nb
        bal_note = "（平衡设计）" if balanced else "（⚠ 不平衡/有空格 —— Type II SS，主效应解释依赖）"
        role_note = "（因子角色自动猜测，建议 config 明确）" if guessed else ""
        if pI != pI:  # NaN — interaction term not extracted/estimable (don't silently say "not sig")
            emph = "⚠ 交互项无法提取/估计，主效应解读须谨慎。"
        elif pI < 0.05:
            emph = "⚠ 交互显著 —— 主效应须在交互背景下解读（不可单独看）。"
        else:
            emph = "交互不显著 —— 主效应可独立解读。"
        summary.append(
            f"{entry.method} 完成{role_note}{bal_note}：{y} ~ {fa}（{na}）× {fb}（{nb}）；"
            f"A: F={fA:.3f},p={pA:.3g} ｜ B: F={fB:.3f},p={pB:.3g} ｜ A×B: F={fI:.3f},p={pI:.3g}；"
            f"R²={model.rsquared:.3f}。{emph}"
            " ⚠ 残差正态/等方差假定；Type II SS（主效应行不随交互校正；不平衡且含交互时 Type III + sum 对照是另一口径）。"
        )
        code += [
            "import statsmodels.formula.api as smf",
            "from statsmodels.stats.anova import anova_lm",
            f'model = smf.ols(\'Q("{y}") ~ C(Q("{fa}")) * C(Q("{fb}"))\', data=df).fit()',
            "print(anova_lm(model, typ=2))  # 主效应 A、B + 交互 A×B",
        ]
    except Exception as err:
        summary.append(f"双因素方差分析失败：{err}")
