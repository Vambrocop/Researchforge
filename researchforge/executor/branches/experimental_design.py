"""Branch handlers for the experimental-design family — design-aware analyses that
force explicit block/treatment/plot roles instead of treating a field trial as a flat
table. First member: RCBD. See executor/_branch_api.py.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register

# column-name hints for auto-detecting roles when config doesn't specify them
_BLOCK_HINTS = ("block", "rep", "replicate", "site", "field", "batch")
_TRT_HINTS = ("treat", "trt", "variety", "cultivar", "genotype", "hybrid", "factor", "dose", "level")
_ROW_HINTS = ("row", "lane")
_COL_HINTS = ("col", "column", "position")


def _degenerate_fit(model, y_vals) -> bool:
    """True if an OLS fit has too little residual error (near-saturated design) for a
    trustworthy F-test — residual MS ~ 0 or residual df < 1 makes F explode spuriously."""
    import numpy as np

    rdf = float(model.df_resid)
    mse = float(model.mse_resid)
    yv = float(np.var(np.asarray(y_vals, dtype=float)))
    return rdf < 1 or not np.isfinite(mse) or mse < 1e-9 * max(yv, 1e-12)


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
        if _degenerate_fit(model, sub[y]):
            summary.append(
                "RCBD 失败：残差自由度不足/残差均方≈0 —— 设计近饱和/不完整"
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
