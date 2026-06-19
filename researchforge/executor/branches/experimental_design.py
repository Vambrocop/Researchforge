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
_GENO_HINTS = ("genotype", "geno", "variety", "cultivar", "hybrid", "line", "entry", "accession")
_ENV_HINTS = ("environment", "env", "site", "location", "loc", "year", "season", "trial")


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


@register("power_analysis")
def _branch_power_analysis(ctx: Ctx) -> None:
    """DoE advisory: required replications / sample size for a one-way comparison.
    Reports required n per group for CONVENTIONAL effect sizes (small/medium/large
    Cohen's f) at 80% & 90% power — the statistically sound planning output — plus the
    pilot data's observed effect as context (NOT 'observed/post-hoc power', which is a
    deterministic function of the p-value and not a planning tool)."""
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import math

    import numpy as np
    import pandas as pd

    _excl = {fp.unit_col, fp.time_col}
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
    cats = [c.name for c in fp.columns if c.kind in {"categorical", "binary"} and c.name not in _excl]
    cats.sort(key=lambda name: int(df[name].nunique()))  # prefer low-cardinality group

    y = cfg["outcome"] if cfg.get("outcome") in cont else (cont[0] if cont else None)
    group = cfg.get("group") if cfg.get("group") in df.columns else (cats[0] if cats else None)
    if y is None or group is None:
        summary.append(
            "功效/样本量失败：需要 1 个连续结果 + 1 个分组变量。"
            'config={"outcome":..,"group":..,"alpha":0.05} 指定。'
        )
        return
    try:
        alpha = min(0.5, max(1e-4, float(cfg.get("alpha", 0.05))))
    except (TypeError, ValueError):
        alpha = 0.05

    sub = df[[y, group]].dropna()
    k = int(sub[group].nunique())
    if k < 2 or len(sub) <= k:
        summary.append(f"功效/样本量失败：分组数={k}、有效行={len(sub)}，不足以估效应。")
        return

    try:
        from statsmodels.stats.power import FTestAnovaPower

        grp = sub.groupby(group, observed=True)[y]
        means, ns = grp.mean(), grp.count()
        grand = float(sub[y].to_numpy(dtype=float).mean())
        n_total = int(len(sub))
        ss_within = float(((sub[y] - sub[group].map(means)) ** 2).sum())
        sd_within = math.sqrt(ss_within / (n_total - k)) if n_total > k else float("nan")
        sigma_m = math.sqrt(float(((ns / n_total) * (means - grand) ** 2).sum()))
        f_obs = sigma_m / sd_within if sd_within and sd_within > 1e-12 else float("nan")

        analyzer = FTestAnovaPower()
        levels = [("small", 0.10), ("medium", 0.25), ("large", 0.40)]
        rows = []
        for label, f in levels:
            for pw in (0.80, 0.90):
                ntot = float(analyzer.solve_power(effect_size=f, alpha=alpha, power=pw, k_groups=k))
                per = int(math.ceil(ntot / k))
                rows.append({"effect": f"{label} (f={f})", "power": pw,
                             "n_per_group": per, "n_total": per * k})
                estimates[f"n_per_group_{label}_p{int(pw*100)}"] = float(per)
        tbl = pd.DataFrame(rows)
        tbl.to_csv(d / "required_sample_size.csv", index=False, encoding="utf-8")
        files.append("required_sample_size.csv")

        estimates["observed_f"] = float(f_obs) if f_obs == f_obs else float("nan")
        estimates["k_groups"] = float(k)
        estimates["n_current"] = float(n_total)
        estimates["alpha"] = float(alpha)

        # plot: required n/group vs effect size at 80% power
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            xs = [f for _, f in levels]
            ys = [int(math.ceil(float(analyzer.solve_power(effect_size=f, alpha=alpha, power=0.8, k_groups=k)) / k))
                  for _, f in levels]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(xs, ys, marker="o", color="#4C72B0")
            ax.set_xlabel("effect size (Cohen's f)")
            ax.set_ylabel("required n per group (80% power)")
            ax.set_title(f"Sample size vs effect — {k} groups, α={alpha}")
            fig.tight_layout()
            fig.savefig(d / "sample_size_curve.png", dpi=150)
            plt.close(fig)
            files.append("sample_size_curve.png")
        except Exception:
            pass

        med80 = estimates.get("n_per_group_medium_p80")
        fobs_txt = f"{f_obs:.3f}" if f_obs == f_obs else "不可估"
        summary.append(
            f"{entry.method} 完成：{k} 组比较（结果 {y}，分组 {group}，α={alpha}）。"
            f"**所需每组样本量**（单因素 ANOVA）：中等效应 f=0.25 → 每组 {int(med80) if med80 else '—'}（80% 功效）。"
            f"小/中/大(f=0.1/0.25/0.4) × 80%/90% 全表见 required_sample_size.csv。"
            f" 试点数据观测效应 f={fobs_txt}（当前每组≈{n_total//k}）。"
            " ⚠ 规划请用**有意义的目标效应**（小/中/大），别用观测效应当目标；"
            "「观测/事后功效」是 p 值的函数、非规划工具,故不作主输出。假定单因素均衡设计、正态/等方差。"
        )
        code += [
            "from statsmodels.stats.power import FTestAnovaPower",
            f"n_total = FTestAnovaPower().solve_power(effect_size=0.25, alpha={alpha}, power=0.8, k_groups={k})",
            f"print('每组样本量(中等效应,80%功效):', -(-int(n_total) // {k}))",
        ]
    except Exception as err:
        summary.append(f"功效/样本量失败：{err}")


def _pick_geno_env(fp, df, cfg, y):
    """Resolve genotype + environment role columns for G×E methods (AMMI / GGE).

    Config overrides accept ANY column (a genotype/env factor may profile as count/id,
    not categorical). Auto-default: name-hint match first, else leftover categorical/id
    columns in declaration order. Returns (genotype, environment, guessed)."""
    _excl = {fp.unit_col, fp.time_col}
    role_cols = [c.name for c in fp.columns
                 if c.kind in {"categorical", "binary", "count", "id"} and c.name not in _excl]
    g_cfg = cfg.get("genotype") if cfg.get("genotype") in df.columns else None
    e_cfg = cfg.get("environment") if cfg.get("environment") in df.columns else None
    genotype, environment = g_cfg, e_cfg

    def _pick(hints, taken):
        for c in role_cols:
            if c == y or c in taken:
                continue
            if any(h in c.lower() for h in hints):
                return c
        return None

    if genotype is None:
        genotype = _pick(_GENO_HINTS, {environment})
    if environment is None:
        environment = _pick(_ENV_HINTS, {genotype})
    leftover = [c for c in role_cols if c not in {genotype, environment} and c != y]
    if genotype is None and leftover:
        genotype = leftover.pop(0)
    if environment is None and leftover:
        environment = leftover.pop(0)
    guessed = not (g_cfg is not None and e_cfg is not None)
    return genotype, environment, guessed


def _ge_means_matrix(sub, genotype, environment, y):
    """Genotype×environment table of cell means (genotypes=rows, environments=cols),
    dropping any row/col with missing cells so SVD has a complete matrix."""
    import pandas as pd  # noqa: F401

    mat = sub.groupby([genotype, environment], observed=True)[y].mean().unstack()
    mat = mat.dropna(axis=0, how="any").dropna(axis=1, how="any")
    return mat


@register("ammi")
def _branch_ammi(ctx: Ctx) -> None:
    """AMMI = Additive Main effects + Multiplicative Interaction. Two-way additive model
    (grand + genotype + environment main effects), then SVD of the G×E interaction
    residual matrix → IPCA axes ranked by % interaction explained, an AMMI-2 biplot, and
    a per-genotype stability readout (IPCA1 magnitude / interaction-residual norm)."""
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    _excl = {fp.unit_col, fp.time_col}
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
    y = cfg["outcome"] if cfg.get("outcome") in cont else (cont[0] if cont else None)
    genotype, environment, guessed = _pick_geno_env(fp, df, cfg, y)

    if y is None or genotype is None or environment is None or genotype == environment:
        summary.append(
            "AMMI 失败：需要 1 个连续结果 + 基因型 genotype + 环境 environment 两个因子。"
            '用 config={"outcome":..,"genotype":..,"environment":..} 指定角色。'
        )
        return

    sub = df[[y, genotype, environment]].dropna()
    mat = _ge_means_matrix(sub, genotype, environment, y)
    g, e = mat.shape
    if g < 3 or e < 3:
        summary.append(
            f"AMMI 失败：完整 G×E 均值表为 {g} 基因型 × {e} 环境（需各 ≥3 才有可分解的交互结构）。"
            "缺格已删行/列；请提供更完整的多环境试验。"
        )
        return

    try:
        M = mat.to_numpy(dtype=float)
        grand = float(M.mean())
        g_main = M.mean(axis=1) - grand          # genotype main effects
        e_main = M.mean(axis=0) - grand          # environment main effects
        # additive expectation; interaction residual = observed - additive (double-centered)
        additive = grand + g_main[:, None] + e_main[None, :]
        inter = M - additive                     # G×E interaction matrix (row & col centered)

        # SVD of the interaction matrix → IPCA axes
        U, S, Vt = np.linalg.svd(inter, full_matrices=False)
        # number of non-trivial interaction axes = min(g-1, e-1)
        rank = min(g - 1, e - 1)
        S = S[:rank]
        U = U[:, :rank]
        Vt = Vt[:rank, :]
        ss = S ** 2
        total_inter_ss = float(ss.sum())
        pct = (ss / total_inter_ss * 100.0) if total_inter_ss > 1e-12 else np.zeros_like(ss)

        # genotype / environment IPCA scores (symmetric scaling: sqrt(singular value))
        sqrtS = np.sqrt(S)
        g_scores = U * sqrtS[None, :]            # g × rank
        e_scores = (Vt.T) * sqrtS[None, :]       # e × rank

        n_axes = int(rank)
        ipca_df = pd.DataFrame(
            {f"IPCA{i+1}_pct": [pct[i]] for i in range(n_axes)}
        )
        ipca_df.insert(0, "axis_ss", [total_inter_ss])
        ipca_df.to_csv(d / "ammi_ipca_variance.csv", index=False, encoding="utf-8")
        files.append("ammi_ipca_variance.csv")

        # genotype scores + stability: AMMI stability value uses IPCA1/IPCA2 weighted by their %.
        ip1 = g_scores[:, 0]
        ip2 = g_scores[:, 1] if n_axes >= 2 else np.zeros(g)
        w1 = pct[0] / pct[1] if n_axes >= 2 and pct[1] > 1e-12 else 1.0
        asv = np.sqrt((w1 * ip1) ** 2 + ip2 ** 2)   # AMMI stability value (smaller = more stable)
        inter_norm = np.sqrt((inter ** 2).sum(axis=1))  # genotype interaction residual norm
        geno_tbl = pd.DataFrame({
            "genotype": [str(x) for x in mat.index],
            "mean": M.mean(axis=1),
            "main_effect": g_main,
            "IPCA1": ip1,
            "IPCA2": ip2,
            "ASV_stability": asv,            # smaller = more stable across environments
            "interaction_norm": inter_norm,
        }).sort_values("ASV_stability")
        geno_tbl.to_csv(d / "ammi_genotype_stability.csv", index=False, encoding="utf-8")
        files.append("ammi_genotype_stability.csv")

        estimates["n_genotypes"] = float(g)
        estimates["n_environments"] = float(e)
        estimates["n_ipca_axes"] = float(n_axes)
        estimates["interaction_ss"] = total_inter_ss
        estimates["IPCA1_pct"] = float(pct[0])
        if n_axes >= 2:
            estimates["IPCA2_pct"] = float(pct[1])
            estimates["IPCA1_IPCA2_pct"] = float(pct[0] + pct[1])
        # most stable genotype = smallest ASV
        most_stable = str(geno_tbl.iloc[0]["genotype"])

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6.5, 5.5))
            ax.axhline(0, color="#999", lw=0.8)
            ax.axvline(0, color="#999", lw=0.8)
            ax.scatter(g_scores[:, 0], ip2, color="#4C72B0", marker="o", label="genotype")
            for i, lab in enumerate(mat.index):
                ax.annotate(str(lab), (g_scores[i, 0], ip2[i]), fontsize=8, color="#1f3a66")
            e_ip2 = e_scores[:, 1] if n_axes >= 2 else np.zeros(e)
            ax.scatter(e_scores[:, 0], e_ip2, color="#C44E52", marker="^", label="environment")
            for j, lab in enumerate(mat.columns):
                ax.annotate(str(lab), (e_scores[j, 0], e_ip2[j]), fontsize=8, color="#7a2a2c")
            ax.set_xlabel(f"IPCA1 ({pct[0]:.1f}% of interaction)")
            ax.set_ylabel(f"IPCA2 ({pct[1]:.1f}% of interaction)" if n_axes >= 2 else "IPCA2")
            ax.set_title(f"AMMI-2 biplot — {y}")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "ammi_biplot.png", dpi=150)
            plt.close(fig)
            files.append("ammi_biplot.png")
        except Exception:
            pass

        role_note = "（角色自动猜测，建议 config 明确 genotype/environment）" if guessed else ""
        two = f"，IPCA1+IPCA2 共 {pct[0]+pct[1]:.1f}%" if n_axes >= 2 else ""
        summary.append(
            f"{entry.method} 完成{role_note}：{g} 基因型 × {e} 环境（结果 {y}）。"
            f"加性主效应(基因型+环境)分离后，对 G×E 交互残差做 SVD：共 {n_axes} 条 IPCA 轴，"
            f"IPCA1 解释 {pct[0]:.1f}% 的交互平方和{two}。"
            f"最稳定基因型（AMMI 稳定值 ASV 最小）：{most_stable}。"
            " ⚠ AMMI 是对 G×E 交互的**描述性乘法分解**（非推断检验）；ASV/IPCA 是稳定性度量、口径多样"
            "（也可用 Gauch F 检验定保留轴数，此处仅报 % 解释）；biplot 远离原点=交互大/不稳定，需谨慎解读；"
            "需重复的多环境试验、每 基因型×环境 格有均值；缺格已删整行/列。"
        )
        code += [
            "import numpy as np",
            "M = df.groupby([genotype, environment])[y].mean().unstack().dropna().to_numpy()",
            "inter = M - (M.mean() + (M.mean(1)-M.mean())[:,None] + (M.mean(0)-M.mean())[None,:])",
            "U,S,Vt = np.linalg.svd(inter, full_matrices=False)  # IPCA: % = S**2/sum(S**2)",
        ]
    except Exception as err:
        summary.append(f"AMMI 失败：{err}")


@register("gge_biplot")
def _branch_gge_biplot(ctx: Ctx) -> None:
    """GGE biplot = SVD of the environment-centered Genotype + Genotype×Environment means.
    Unlike AMMI, the genotype main effect is KEPT (only the environment main effect is
    removed by column-centering). Reports PC1/PC2 variance explained and a which-won-where
    readout (winning genotype per environment from the biplot approximation)."""
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    _excl = {fp.unit_col, fp.time_col}
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
    y = cfg["outcome"] if cfg.get("outcome") in cont else (cont[0] if cont else None)
    genotype, environment, guessed = _pick_geno_env(fp, df, cfg, y)

    if y is None or genotype is None or environment is None or genotype == environment:
        summary.append(
            "GGE biplot 失败：需要 1 个连续结果 + 基因型 genotype + 环境 environment 两个因子。"
            '用 config={"outcome":..,"genotype":..,"environment":..} 指定角色。'
        )
        return

    sub = df[[y, genotype, environment]].dropna()
    mat = _ge_means_matrix(sub, genotype, environment, y)
    g, e = mat.shape
    if g < 3 or e < 3:
        summary.append(
            f"GGE biplot 失败：完整 G×E 均值表为 {g} 基因型 × {e} 环境（需各 ≥3）。"
            "缺格已删行/列；请提供更完整的多环境试验。"
        )
        return

    try:
        M = mat.to_numpy(dtype=float)
        # environment-centering: remove each environment (column) mean → retains G + G×E
        col_mean = M.mean(axis=0)
        centered = M - col_mean[None, :]

        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        rank = min(g, e)
        S = S[:rank]
        U = U[:, :rank]
        Vt = Vt[:rank, :]
        ss = S ** 2
        total = float(ss.sum())
        pct = (ss / total * 100.0) if total > 1e-12 else np.zeros_like(ss)

        # symmetric scaling for biplot coordinates
        sqrtS = np.sqrt(S)
        g_scores = U * sqrtS[None, :]      # genotype coords
        e_scores = (Vt.T) * sqrtS[None, :]  # environment coords
        n_pc = int(rank)

        var_df = pd.DataFrame({"PC": [f"PC{i+1}" for i in range(n_pc)],
                               "pct_variance": pct})
        var_df.to_csv(d / "gge_variance.csv", index=False, encoding="utf-8")
        files.append("gge_variance.csv")

        # which-won-where: winning genotype per environment = genotype with the highest
        # biplot projection onto that environment vector (rank-2 GGE approximation).
        g2 = g_scores[:, :2] if n_pc >= 2 else np.column_stack([g_scores[:, 0], np.zeros(g)])
        e2 = e_scores[:, :2] if n_pc >= 2 else np.column_stack([e_scores[:, 0], np.zeros(e)])
        proj = g2 @ e2.T                  # g × e projection
        winners_idx = proj.argmax(axis=0)
        www = pd.DataFrame({
            "environment": [str(c) for c in mat.columns],
            "winning_genotype": [str(mat.index[i]) for i in winners_idx],
        })
        www.to_csv(d / "gge_which_won_where.csv", index=False, encoding="utf-8")
        files.append("gge_which_won_where.csv")

        estimates["n_genotypes"] = float(g)
        estimates["n_environments"] = float(e)
        estimates["PC1_pct"] = float(pct[0])
        if n_pc >= 2:
            estimates["PC2_pct"] = float(pct[1])
            estimates["PC1_PC2_pct"] = float(pct[0] + pct[1])
        estimates["n_winning_genotypes"] = float(len(set(winners_idx.tolist())))

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            gy = g2[:, 1]
            ey = e2[:, 1]
            fig, ax = plt.subplots(figsize=(6.5, 5.5))
            ax.axhline(0, color="#999", lw=0.8)
            ax.axvline(0, color="#999", lw=0.8)
            ax.scatter(g2[:, 0], gy, color="#4C72B0", marker="o", label="genotype")
            for i, lab in enumerate(mat.index):
                ax.annotate(str(lab), (g2[i, 0], gy[i]), fontsize=8, color="#1f3a66")
            # environments drawn as vectors from origin
            for j, lab in enumerate(mat.columns):
                ax.annotate("", xy=(e2[j, 0], ey[j]), xytext=(0, 0),
                            arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.0))
                ax.annotate(str(lab), (e2[j, 0], ey[j]), fontsize=8, color="#7a2a2c")
            ax.set_xlabel(f"PC1 ({pct[0]:.1f}%)")
            ax.set_ylabel(f"PC2 ({pct[1]:.1f}%)" if n_pc >= 2 else "PC2")
            ax.set_title(f"GGE biplot — {y}")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "gge_biplot.png", dpi=150)
            plt.close(fig)
            files.append("gge_biplot.png")
        except Exception:
            pass

        role_note = "（角色自动猜测，建议 config 明确 genotype/environment）" if guessed else ""
        two = f"，PC1+PC2 共 {pct[0]+pct[1]:.1f}%" if n_pc >= 2 else ""
        n_win = len(set(winners_idx.tolist()))
        win_note = (f"{n_win} 个基因型在不同环境胜出（存在 which-won-where 分区）"
                    if n_win > 1 else "单一基因型在所有环境胜出（无明显分区）")
        summary.append(
            f"{entry.method} 完成{role_note}：{g} 基因型 × {e} 环境（结果 {y}）。"
            f"环境中心化(去环境主效应、保留 G+G×E)后做 SVD：PC1 解释 {pct[0]:.1f}% 变异{two}。"
            f"Which-won-where：{win_note}（详见 gge_which_won_where.csv）。"
            " ⚠ GGE 是对 基因型+G×E 的**描述性双标图**（非推断检验）；中心化/标度方式（环境中心化、"
            "对称标度）会改变图形与解读；biplot 仅是低秩近似，which-won-where 是近似投影、非显著性判定；"
            "需重复的多环境试验、每 基因型×环境 格有均值；缺格已删整行/列。"
        )
        code += [
            "import numpy as np",
            "M = df.groupby([genotype, environment])[y].mean().unstack().dropna().to_numpy()",
            "centered = M - M.mean(0)[None,:]  # environment-centered → keeps G + GxE",
            "U,S,Vt = np.linalg.svd(centered, full_matrices=False)  # PC%% = S**2/sum(S**2)",
        ]
    except Exception as err:
        summary.append(f"GGE biplot 失败：{err}")


@register("response_surface")
def _branch_response_surface(ctx: Ctx) -> None:
    """RSM = Response Surface Methodology. Fits a second-order polynomial
    response ~ x1 + x2 + ... + x1² + ... + x1:x2 + ..., locates the stationary point
    (∇=0 → solve 2B·x = -b), and classifies it via the Hessian (2B) eigenvalues
    (all<0 max / all>0 min / mixed saddle). Draws a contour plot over the first two
    factors. Config: outcome + factors (list of ≥2 continuous columns)."""
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import itertools

    import numpy as np
    import pandas as pd

    _excl = {fp.unit_col, fp.time_col}
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]

    y = cfg["outcome"] if cfg.get("outcome") in cont else (cont[0] if cont else None)
    # factors: config list (any continuous columns ≠ outcome), else remaining continuous
    fac_cfg = cfg.get("factors")
    if isinstance(fac_cfg, (list, tuple)):
        factors = [c for c in fac_cfg if c in df.columns and c != y]
    else:
        factors = []
    guessed = not factors
    if not factors:
        factors = [c for c in cont if c != y]

    if y is None or len(factors) < 2:
        summary.append(
            "响应面 RSM 失败：需要 1 个连续结果 + ≥2 个连续因子列。"
            '用 config={"outcome":..,"factors":["x1","x2",...]} 指定。'
        )
        return

    sub = df[[y] + factors].dropna()
    # need enough rows + variation in each factor for a second-order fit
    n_params = 1 + 2 * len(factors) + len(factors) * (len(factors) - 1) // 2  # const+lin+quad+cross
    nuniq = {f: int(sub[f].nunique()) for f in factors}
    if len(sub) <= n_params or any(v < 3 for v in nuniq.values()):
        summary.append(
            f"响应面 RSM 失败：二阶模型需 {n_params} 个参数、有效行 {len(sub)}（需 > 参数数），"
            f"且每个因子至少 3 个不同水平（当前 {nuniq}）。请提供设计过的因子水平（如 CCD/Box-Behnken）。"
        )
        return

    try:
        import statsmodels.formula.api as smf

        # center factors to stabilize the quadratic fit; map back to raw scale for the
        # stationary point. Build a design matrix manually (robust to odd column names).
        Xc = sub[factors].to_numpy(dtype=float)
        ctr = Xc.mean(axis=0)
        Xcen = Xc - ctr[None, :]
        yv = sub[y].to_numpy(dtype=float)
        kf = len(factors)

        # design: intercept, linear (k), quadratic (k), cross terms (k choose 2)
        cols = [np.ones(len(sub))]
        names = ["const"]
        for i in range(kf):
            cols.append(Xcen[:, i]); names.append(f"L{i}")
        for i in range(kf):
            cols.append(Xcen[:, i] ** 2); names.append(f"Q{i}")
        cross_pairs = list(itertools.combinations(range(kf), 2))
        for (i, j) in cross_pairs:
            cols.append(Xcen[:, i] * Xcen[:, j]); names.append(f"X{i}_{j}")
        D = np.column_stack(cols)
        Ddf = pd.DataFrame(D, columns=names)
        Ddf["__y__"] = yv
        model = smf.ols("__y__ ~ " + " + ".join(names[1:]), data=Ddf).fit()
        r2 = float(model.rsquared)
        if not np.isfinite(r2):
            summary.append("响应面 RSM 失败：二阶模型不可估（设计矩阵奇异/共线）。需设计过的因子水平。")
            return

        beta = model.params
        b = np.array([float(beta.get(f"L{i}", 0.0)) for i in range(kf)])     # linear coefs
        # Hessian / B matrix: diagonal = 2*quad? In y = b0 + b'x + x'B x, quad coef = B_ii,
        # cross coef = 2 B_ij. Stationary point: x_s = -0.5 * B^{-1} b.
        B = np.zeros((kf, kf))
        for i in range(kf):
            B[i, i] = float(beta.get(f"Q{i}", 0.0))
        for (i, j) in cross_pairs:
            half = 0.5 * float(beta.get(f"X{i}_{j}", 0.0))
            B[i, j] = half
            B[j, i] = half

        # gradient = b + 2 B x = 0 → x_s = -0.5 B^{-1} b (in centered coords)
        eig = np.linalg.eigvalsh(B)
        stationary_ok = bool(np.all(np.abs(eig) > 1e-9))
        if stationary_ok:
            x_s_cen = np.linalg.solve(2.0 * B, -b)
            x_s = x_s_cen + ctr
            # predicted response at stationary point: b0 + b'x_s + x_s' B x_s (centered)
            b0 = float(beta.get("const", beta.get("Intercept", 0.0)))
            y_s = b0 + float(b @ x_s_cen) + float(x_s_cen @ B @ x_s_cen)
        else:
            x_s_cen = np.full(kf, np.nan)
            x_s = np.full(kf, np.nan)
            y_s = float("nan")

        # classify via Hessian (2B) eigenvalues — same sign pattern as B
        if np.all(eig < -1e-9):
            kind = "maximum"; kind_zh = "极大值(凸顶)"
        elif np.all(eig > 1e-9):
            kind = "minimum"; kind_zh = "极小值(凹底)"
        elif stationary_ok:
            kind = "saddle"; kind_zh = "鞍点(混合曲率)"
        else:
            kind = "ridge/degenerate"; kind_zh = "脊/退化(近零特征值)"

        # in-region check: is the stationary point inside the observed factor box?
        in_region = bool(np.all((x_s >= Xc.min(axis=0)) & (x_s <= Xc.max(axis=0)))) if stationary_ok else False

        rows = []
        for i, f in enumerate(factors):
            rows.append({"factor": f,
                         "stationary_point": float(x_s[i]) if stationary_ok else float("nan"),
                         "factor_min": float(Xc[:, i].min()),
                         "factor_max": float(Xc[:, i].max())})
            estimates[f"stationary_{f}"] = float(x_s[i]) if stationary_ok else float("nan")
        pd.DataFrame(rows).to_csv(d / "rsm_stationary_point.csv", index=False, encoding="utf-8")
        files.append("rsm_stationary_point.csv")

        coef_tbl = pd.DataFrame({"term": list(model.params.index),
                                 "coef": [float(v) for v in model.params.values],
                                 "p_value": [float(v) for v in model.pvalues.values]})
        coef_tbl.to_csv(d / "rsm_coefficients.csv", index=False, encoding="utf-8")
        files.append("rsm_coefficients.csv")

        estimates["r_squared"] = r2
        estimates["n_factors"] = float(kf)
        estimates["stationary_response"] = float(y_s) if y_s == y_s else float("nan")
        estimates["stationary_in_region"] = 1.0 if in_region else 0.0
        for i, ev in enumerate(eig):
            estimates[f"hessian_eig{i+1}"] = float(ev)

        # contour plot over the first two factors (others held at their stationary/centered value)
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            i0, i1 = 0, 1
            x0 = np.linspace(Xc[:, i0].min(), Xc[:, i0].max(), 60)
            x1 = np.linspace(Xc[:, i1].min(), Xc[:, i1].max(), 60)
            XX, YY = np.meshgrid(x0, x1)
            # build grid design rows; other factors fixed at center (=0 in centered coords)
            base = np.zeros(kf)
            if stationary_ok:
                base = x_s_cen.copy()
            grid_cen = np.zeros((XX.size, kf))
            grid_cen[:] = base[None, :]
            grid_cen[:, i0] = XX.ravel() - ctr[i0]
            grid_cen[:, i1] = YY.ravel() - ctr[i1]
            gcols = {f"L{i}": grid_cen[:, i] for i in range(kf)}
            gcols.update({f"Q{i}": grid_cen[:, i] ** 2 for i in range(kf)})
            for (i, j) in cross_pairs:
                gcols[f"X{i}_{j}"] = grid_cen[:, i] * grid_cen[:, j]
            Zpred = model.predict(pd.DataFrame(gcols)).to_numpy().reshape(XX.shape)
            fig, ax = plt.subplots(figsize=(6.5, 5))
            cs = ax.contourf(XX, YY, Zpred, levels=15, cmap="viridis")
            fig.colorbar(cs, ax=ax, label=f"predicted {y}")
            ax.scatter(Xc[:, i0], Xc[:, i1], c="white", s=12, edgecolors="k", lw=0.4, label="design points")
            if stationary_ok and in_region:
                ax.scatter([x_s[i0]], [x_s[i1]], c="red", s=80, marker="*", label=f"stationary ({kind})")
            ax.set_xlabel(factors[i0])
            ax.set_ylabel(factors[i1])
            ax.set_title(f"Response surface — {y}")
            ax.legend(fontsize=8, loc="best")
            fig.tight_layout()
            fig.savefig(d / "rsm_contour.png", dpi=150)
            plt.close(fig)
            files.append("rsm_contour.png")
        except Exception:
            pass

        role_note = "（factors 自动取全部连续列，建议 config 明确 factors 列表）" if guessed else ""
        if stationary_ok:
            sp_txt = "、".join(f"{f}={x_s[i]:.4g}" for i, f in enumerate(factors))
            region_txt = "在设计区域内" if in_region else "⚠ 落在设计区域外（外推，不可信）"
            sp_summary = f"驻点 {kind_zh} 于 {sp_txt}（预测 {y}≈{y_s:.4g}，{region_txt}）。"
        else:
            sp_summary = "驻点不可解（Hessian 近奇异 → 脊系统/退化曲面，无唯一最优）。"
        summary.append(
            f"{entry.method} 完成{role_note}：{y} ~ 二阶多项式({kf} 因子：线性+平方+交互项)；R²={r2:.3f}。"
            f"{sp_summary}Hessian 特征值={np.array2string(eig, precision=3)}。"
            " ⚠ RSM 是设计区域内的**局部二次近似**；驻点仅在因子水平覆盖的区域内有效（区域外为外推、不可信）；"
            "需**设计过的因子水平**(如中心复合 CCD / Box-Behnken)，观测性数据的因子常共线导致曲面不可估或脊；"
            "等高线图固定其余因子于驻点/中心，残差正态/等方差假定。"
        )
        code += [
            "import numpy as np, statsmodels.formula.api as smf",
            "# fit y ~ x1+x2+...+ x1^2+...+ x1:x2+...  (centered factors)",
            "# stationary point: x_s = -0.5 * B^{-1} b ; classify via eig(B): <0 max / >0 min / mixed saddle",
        ]
    except Exception as err:
        summary.append(f"响应面 RSM 失败：{err}")
