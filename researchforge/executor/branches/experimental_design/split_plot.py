"""Experimental-design family branch handler: split_plot (split from experimental_design.py)."""
from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register

from ._shared import _BLOCK_HINTS


@register("split_plot")
def _branch_split_plot(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    _excl = {fp.unit_col, fp.time_col}
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
    role_cols = [c.name for c in fp.columns
                 if c.kind in {"categorical", "binary", "count", "id"} and c.name not in _excl]

    y = cfg["outcome"] if cfg.get("outcome") in cont else (cont[0] if cont else None)
    block = cfg.get("block") if cfg.get("block") in df.columns else None
    wp = cfg.get("whole_plot") if cfg.get("whole_plot") in df.columns else None
    sp = cfg.get("sub_plot") if cfg.get("sub_plot") in df.columns else None
    if block is None:  # block is name-guessable; whole/sub plot really need config
        for c in role_cols:
            if c != y and any(h in c.lower() for h in _BLOCK_HINTS):
                block = c
                break
    leftover = [c for c in role_cols if c != y and c not in {block, wp, sp}]
    if wp is None and leftover:
        wp = leftover.pop(0)
    if sp is None and leftover:
        sp = leftover.pop(0)
    guessed = not (cfg.get("whole_plot") in df.columns and cfg.get("sub_plot") in df.columns
                   and cfg.get("block") in df.columns)

    roles = [block, wp, sp]
    if y is None or any(r is None for r in roles) or len(set(roles)) < 3:
        summary.append(
            "裂区设计失败：需要 1 个连续结果 + 区组 block + 主区因子 whole_plot + 裂区因子 sub_plot。"
            '用 config={"outcome":..,"block":..,"whole_plot":..,"sub_plot":..} 指定（主区/裂区角色无法可靠自动判定）。'
        )
        return

    sub = df[[y, block, wp, sp]].dropna()
    r, a, b = int(sub[block].nunique()), int(sub[wp].nunique()), int(sub[sp].nunique())
    if r < 2 or a < 2 or b < 2:
        summary.append(f"裂区设计失败：区组={r}、主区水平={a}、裂区水平={b}，均需 ≥2。")
        return
    # split-plot ANOVA assumes a complete balanced design (one obs per block×wholeplot×subplot)
    cells = sub.groupby([block, wp, sp], observed=True).size()
    if not (bool((cells == 1).all()) and len(cells) == r * a * b):
        summary.append(
            f"裂区设计失败：需完全平衡（每 区组×主区×裂区 恰 1 次，应 {r*a*b} 行得 {len(sub)}）。"
            "不平衡/有重复请改用混合模型（lmer/MixedLM）。"
        )
        return

    try:
        import statsmodels.formula.api as smf
        from scipy import stats
        from statsmodels.stats.anova import anova_lm

        # Full model; whole-plot error stratum = block:whole_plot, sub-plot error = residual.
        formula = (f'Q("{y}") ~ C(Q("{block}")) + C(Q("{wp}")) + C(Q("{block}")):C(Q("{wp}")) '
                   f'+ C(Q("{sp}")) + C(Q("{wp}")):C(Q("{sp}"))')
        model = smf.ols(formula, data=sub).fit()
        aov = anova_lm(model, typ=2)  # balanced design -> Type I == II (verified); order-independent

        kb, kw, ks = f'C(Q("{block}"))', f'C(Q("{wp}"))', f'C(Q("{sp}"))'
        k_wpe = f"{kb}:{kw}"   # whole-plot error
        k_int = f"{kw}:{ks}"   # whole×sub interaction (sub-plot stratum)

        def _ss_df(key):
            if key in aov.index:
                return float(aov.loc[key, "sum_sq"]), float(aov.loc[key, "df"])
            return float("nan"), float("nan")

        ss_a, df_a = _ss_df(kw)
        ss_wpe, df_wpe = _ss_df(k_wpe)
        ss_b, df_b = _ss_df(ks)
        ss_ab, df_ab = _ss_df(k_int)
        ss_res, df_res = _ss_df("Residual")

        # guard degenerate strata (no error df / ~0 error MS -> spurious F)
        y_var = float(np.var(sub[y].to_numpy(dtype=float)))
        ms_wpe = ss_wpe / df_wpe if df_wpe and df_wpe >= 1 else float("nan")
        ms_res = ss_res / df_res if df_res and df_res >= 1 else float("nan")
        tiny = 1e-9 * max(y_var, 1e-12)
        if not (np.isfinite(ms_wpe) and np.isfinite(ms_res)) or ms_wpe < tiny or ms_res < tiny:
            summary.append("裂区设计失败：误差层自由度不足/误差均方≈0——设计过小或近饱和，F 检验不可靠。")
            return

        f_a = (ss_a / df_a) / ms_wpe
        p_a = float(stats.f.sf(f_a, df_a, df_wpe))
        f_b = (ss_b / df_b) / ms_res
        p_b = float(stats.f.sf(f_b, df_b, df_res))
        f_ab = (ss_ab / df_ab) / ms_res
        p_ab = float(stats.f.sf(f_ab, df_ab, df_res))

        estimates.update({
            "wholeplot_F": float(f_a), "wholeplot_p": p_a,
            "subplot_F": float(f_b), "subplot_p": p_b,
            "interaction_F": float(f_ab), "interaction_p": p_ab,
            "n_blocks": float(r), "n_wholeplots": float(a), "n_subplots": float(b),
        })

        # explicit split-plot ANOVA table with the two error strata
        tbl = pd.DataFrame(
            [
                ["whole-plot: " + wp, ss_a, df_a, f_a, p_a, "whole-plot error"],
                ["  whole-plot error (block×" + wp + ")", ss_wpe, df_wpe, np.nan, np.nan, ""],
                ["sub-plot: " + sp, ss_b, df_b, f_b, p_b, "residual"],
                [wp + " × " + sp, ss_ab, df_ab, f_ab, p_ab, "residual"],
                ["  sub-plot error (residual)", ss_res, df_res, np.nan, np.nan, ""],
            ],
            columns=["term", "sum_sq", "df", "F", "p", "tested_against"],
        )
        tbl.to_csv(d / "split_plot_anova.csv", index=False, encoding="utf-8")
        files.append("split_plot_anova.csv")

        cell = sub.groupby([wp, sp], observed=True)[y].mean().unstack()
        cell.to_csv(d / "cell_means.csv", encoding="utf-8")
        files.append("cell_means.csv")
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 4.5))
            for s_lvl in cell.columns:
                ax.plot([str(w) for w in cell.index], cell[s_lvl].to_numpy(), marker="o", label=str(s_lvl))
            ax.set_xlabel(f"whole-plot ({wp})")
            ax.set_ylabel(f"mean {y}")
            ax.set_title(f"Split-plot means — {y}")
            ax.legend(fontsize=8, title=str(sp))
            fig.tight_layout()
            fig.savefig(d / "split_plot_means.png", dpi=150)
            plt.close(fig)
            files.append("split_plot_means.png")
        except Exception:
            pass

        role_note = "（主区/裂区角色自动猜测，强烈建议 config 明确）" if guessed else ""

        def _sig(p):
            return "显著" if (p == p and p < 0.05) else "不显著"

        summary.append(
            f"{entry.method} 完成{role_note}：{y}；区组 {block}（{r}）、主区 {wp}（{a}）、裂区 {sp}（{b}）。"
            f"主区 {wp}：F={f_a:.3f}, p={p_a:.3g}（{_sig(p_a)}，对主区误差 df={df_wpe:.0f}）｜"
            f"裂区 {sp}：F={f_b:.3f}, p={p_b:.3g}（{_sig(p_b)}）｜{wp}×{sp}：F={f_ab:.3f}, p={p_ab:.3g}（{_sig(p_ab)}）。"
            " ⚠ 裂区设计有**两个误差层**：主区因子对主区误差(block×主区)检验、功效较低；裂区因子与交互对"
            "裂区(残差)误差检验、功效较高。需完全平衡；不平衡请用混合模型。残差正态/等方差假定。"
        )
        code += [
            "import statsmodels.formula.api as smf",
            "from scipy import stats; from statsmodels.stats.anova import anova_lm",
            f'm = smf.ols(\'Q("{y}") ~ C(Q("{block}"))+C(Q("{wp}"))+C(Q("{block}")):C(Q("{wp}"))'
            f'+C(Q("{sp}"))+C(Q("{wp}")):C(Q("{sp}"))\', data=df).fit()',
            "aov = anova_lm(m, typ=2)  # 主区 F=MS_主区/MS_(block:主区); 裂区&交互 F=MS/MS_残差",
        ]
    except Exception as err:
        summary.append(f"裂区设计失败：{err}")
