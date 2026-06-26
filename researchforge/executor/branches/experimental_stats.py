"""Branch handlers for the experimental_stats family — classical experimental-design
ANOVA methods (pure Python: statsmodels / scipy / numpy / pandas, no R).

  * anova_oneway            — one-way ANOVA across one grouping factor: F + df, η²/ω²
                              effect sizes, Levene homogeneity check, Welch's robust F,
                              and Tukey HSD post-hoc pairwise comparisons.
  * ancova                  — ANCOVA: outcome ~ C(factor) + covariate(s); adjusted (EMM)
                              group means at the covariate mean, factor F + partial η²,
                              the covariate coefficient, and the homogeneity-of-slopes
                              (factor×covariate interaction) check.
  * repeated_measures_anova — within-subjects RM-ANOVA (long OR wide input). AnovaRM
                              within-factor F + partial η², Mauchly's sphericity test, and
                              Greenhouse-Geisser / Huynh-Feldt epsilon + GG-corrected p.

Each handler resolves roles (config overrides win, else infer), degrades honestly
(<2 groups/conditions, too few rows, non-numeric outcome, unbalanced RM, singular,
missing import -> append a Chinese "<方法>跳过/失败：<原因>" and return — never crash),
writes CSV + PNG (matplotlib Agg, ENGLISH plot labels), fills float `estimates`, appends
a Chinese `summary` ending with ⚠ disclosures, and MUTATES ctx (never rebinds).
See executor/_branch_api.py and CLAUDE.md.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register

# Likert/integer-coded grouping factors profile as count/id, not categorical (profiler
# "id" trap) — accept those kinds so a config-free run can still pick a group factor.
_FACTOR_KINDS = {"categorical", "binary", "count", "id"}
_MAX_GROUP_LEVELS = 10  # auto-pick a grouping factor only when 2..~10 distinct levels


def _continuous(fp) -> list[str]:
    _excl = {fp.unit_col, fp.time_col}
    return [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]


def _factor_candidates(fp, df, y: str | None) -> list[str]:
    """Low-cardinality grouping-factor candidates (2..~10 levels), excluding the outcome
    and unit/time roles. Sorted by ascending cardinality (prefer a clean small factor)."""
    _excl = {fp.unit_col, fp.time_col}
    out = []
    for c in fp.columns:
        if c.kind not in _FACTOR_KINDS or c.name in _excl or c.name == y:
            continue
        try:
            k = int(df[c.name].nunique(dropna=True))
        except Exception:
            continue
        if 2 <= k <= _MAX_GROUP_LEVELS:
            out.append((c.name, k))
    out.sort(key=lambda t: t[1])
    return [name for name, _ in out]


# ─────────────────────────────────────────────────────────────────────────────
# (A) anova_oneway — one continuous outcome across the levels of ONE factor.
# ─────────────────────────────────────────────────────────────────────────────
@register("anova_oneway")
def _branch_anova_oneway(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np

    cont = _continuous(fp)
    y = cfg["outcome"] if cfg.get("outcome") in cont else (cont[0] if cont else None)
    # config group override accepts ANY column (a factor may profile as count/id)
    group = cfg.get("group") if cfg.get("group") in df.columns else None
    guessed = group is None
    if group is None:
        cands = _factor_candidates(fp, df, y)
        group = cands[0] if cands else None

    if y is None or group is None or group == y:
        summary.append(
            "单因素方差分析失败：需要 1 个连续结果 + 1 个分组因子（2..~10 水平）。"
            'config={"outcome":..,"group":..} 指定。'
        )
        return

    sub = df[[y, group]].dropna()
    try:
        sub = sub.astype({y: float})
    except (TypeError, ValueError):
        summary.append(f"单因素方差分析失败：结果列 {y} 非数值，无法做 ANOVA。")
        return
    k = int(sub[group].nunique())
    n = int(len(sub))
    if k < 2:
        summary.append(f"单因素方差分析失败：分组水平={k}（需 ≥2）。")
        return
    if n <= k:
        summary.append(f"单因素方差分析失败：有效行={n} ≤ 组数={k}，无残差自由度。")
        return

    try:
        import pandas as pd
        from scipy import stats

        groups = [g[y].to_numpy(dtype=float) for _, g in sub.groupby(group, observed=True)]
        # drop empty groups (defensive) and require >=2 non-empty
        groups = [g for g in groups if g.size > 0]
        if len(groups) < 2:
            summary.append("单因素方差分析失败：去缺后有效组 <2。")
            return

        # classic one-way ANOVA (equal-variance F)
        f_stat, p_value = stats.f_oneway(*groups)
        f_stat, p_value = float(f_stat), float(p_value)
        df_between = float(k - 1)
        df_within = float(n - k)

        # sums of squares for η² / ω²
        grand = float(sub[y].mean())
        ss_total = float(((sub[y] - grand) ** 2).sum())
        gmeans = sub.groupby(group, observed=True)[y].mean()
        gns = sub.groupby(group, observed=True)[y].count()
        ss_between = float((gns * (gmeans - grand) ** 2).sum())
        ss_within = ss_total - ss_between
        ms_within = ss_within / df_within if df_within > 0 else float("nan")
        eta_sq = ss_between / ss_total if ss_total > 1e-12 else float("nan")
        # ω² = (SS_between - df_between*MS_within) / (SS_total + MS_within)
        denom_w = ss_total + ms_within
        omega_sq = ((ss_between - df_between * ms_within) / denom_w) if denom_w > 1e-12 else float("nan")

        # Levene's test for homogeneity of variance (center='median' = Brown-Forsythe, robust)
        try:
            _, levene_p = stats.levene(*groups, center="median")
            levene_p = float(levene_p)
        except Exception:
            levene_p = float("nan")

        # Welch's ANOVA (robust to unequal variances) — computed by hand (scipy lacks it directly)
        welch_f, welch_p = _welch_anova(groups)

        estimates.update({
            "f_stat": f_stat, "p_value": p_value,
            "eta_squared": float(eta_sq), "omega_squared": float(omega_sq),
            "levene_p": levene_p, "welch_p": float(welch_p),
            "n_groups": float(k), "n": float(n),
        })

        # group means / sd / n table
        gstats = sub.groupby(group, observed=True)[y].agg(["mean", "std", "count"])
        gstats.to_csv(d / "group_stats.csv", encoding="utf-8")
        files.append("group_stats.csv")

        # Tukey HSD post-hoc
        tukey_done, tukey_sig_pairs = False, []
        try:
            from statsmodels.stats.multicomp import pairwise_tukeyhsd

            tuk = pairwise_tukeyhsd(sub[y].to_numpy(dtype=float), sub[group].astype(str).to_numpy())
            tuk_df = pd.DataFrame(tuk.summary().data[1:], columns=tuk.summary().data[0])
            tuk_df.to_csv(d / "tukey_hsd.csv", index=False, encoding="utf-8")
            files.append("tukey_hsd.csv")
            tukey_done = True
            for _, r in tuk_df.iterrows():
                if str(r.get("reject")).strip().lower() in {"true", "1"}:
                    tukey_sig_pairs.append(f"{r['group1']}↔{r['group2']}")
        except Exception:
            pass

        # PNG: group means with 95% CI
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            m = gstats["mean"]
            se = gstats["std"] / np.sqrt(gstats["count"].clip(lower=1))
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.errorbar(range(len(m)), m.to_numpy(), yerr=(1.96 * se).to_numpy(),
                        fmt="o", capsize=4, color="#4C72B0")
            ax.set_xticks(range(len(m)))
            ax.set_xticklabels([str(i) for i in m.index], rotation=30, ha="right")
            ax.set_xlabel(f"group ({group})")
            ax.set_ylabel(f"mean {y} (95% CI)")
            ax.set_title(f"One-way ANOVA group means — {y}")
            fig.tight_layout()
            fig.savefig(d / "group_means.png", dpi=150)
            plt.close(fig)
            files.append("group_means.png")
        except Exception:
            pass

        role_note = "（分组因子自动猜测，建议 config 明确 group）" if guessed else ""
        sig = "显著" if (p_value == p_value and p_value < 0.05) else "不显著"
        lev_note = (
            "Levene 检验提示方差不齐(p<0.05)，建议看 Welch F" if (levene_p == levene_p and levene_p < 0.05)
            else "Levene 检验未拒绝方差齐性"
        )
        if tukey_done:
            tukey_note = (f"Tukey HSD 显著差异对：{', '.join(tukey_sig_pairs)}。" if tukey_sig_pairs
                          else "Tukey HSD 未发现显著两两差异。")
        else:
            tukey_note = ""
        summary.append(
            f"{entry.method} 完成{role_note}：{y} 跨 {group} 的 {k} 组比较（n={n}）；"
            f"F({df_between:.0f},{df_within:.0f})={f_stat:.3f}, p={p_value:.3g}（{sig}）；"
            f"η²={eta_sq:.3f}, ω²={omega_sq:.3f}；Welch F p={welch_p:.3g}。{lev_note}。{tukey_note}"
            " ⚠ 经典 ANOVA 假定残差正态 + 各组等方差（Levene 查后者；若违反优先用已报的 Welch F）；"
            "η² 偏高估、ω² 偏差更小；Tukey HSD 控制族错误率（family-wise）。config 可指定 outcome/group。"
        )
        code += [
            "from scipy import stats",
            f"groups = [g['{y}'].values for _, g in df.groupby('{group}')]",
            "F, p = stats.f_oneway(*groups)  # 经典单因素 ANOVA",
            "from statsmodels.stats.multicomp import pairwise_tukeyhsd",
            f"print(pairwise_tukeyhsd(df['{y}'], df['{group}'].astype(str)))  # Tukey HSD 事后比较",
        ]
    except Exception as err:
        summary.append(f"单因素方差分析失败：{err}")


def _welch_anova(groups):
    """Welch's ANOVA F + p (robust to unequal variances). Pure numpy/scipy.
    Returns (F, p); (nan, nan) if not computable."""
    import numpy as np
    from scipy import stats

    try:
        k = len(groups)
        ni = np.array([g.size for g in groups], dtype=float)
        means = np.array([g.mean() for g in groups], dtype=float)
        # ddof=1 sample variances; guard zero-variance / size-1 groups
        varis = np.array([g.var(ddof=1) if g.size > 1 else np.nan for g in groups], dtype=float)
        if np.any(~np.isfinite(varis)) or np.any(varis <= 0):
            return float("nan"), float("nan")
        wi = ni / varis
        sw = wi.sum()
        xbar = float((wi * means).sum() / sw)
        num = float((wi * (means - xbar) ** 2).sum()) / (k - 1)
        tmp = float((((1 - wi / sw) ** 2) / (ni - 1)).sum())
        denom = 1.0 + (2.0 * (k - 2) / (k ** 2 - 1)) * tmp
        f = num / denom
        df1 = k - 1
        df2 = (k ** 2 - 1) / (3.0 * tmp)
        p = float(stats.f.sf(f, df1, df2))
        return float(f), p
    except Exception:
        return float("nan"), float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# (B) ancova — outcome ~ C(factor) + covariate(s); adjusted group means.
# ─────────────────────────────────────────────────────────────────────────────
@register("ancova")
def _branch_ancova(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np

    cont = _continuous(fp)
    y = cfg["outcome"] if cfg.get("outcome") in cont else (cont[0] if cont else None)
    group = cfg.get("group") if cfg.get("group") in df.columns else None
    guessed_g = group is None
    if group is None:
        cands = _factor_candidates(fp, df, y)
        group = cands[0] if cands else None

    # covariate(s): config "covariates" (str or list), else remaining continuous columns
    cov_cfg = cfg.get("covariates") or cfg.get("covariate")
    if isinstance(cov_cfg, str):
        cov_cfg = [cov_cfg]
    if cov_cfg:
        covariates = [c for c in cov_cfg if c in df.columns and c not in {y, group}]
        guessed_c = False
    else:
        covariates = [c for c in cont if c not in {y} and c != group]
        guessed_c = True

    if y is None or group is None or group == y or not covariates:
        summary.append(
            "ANCOVA 失败：需要 1 个连续结果 + 1 个分类因子 + ≥1 个连续协变量。"
            'config={"outcome":..,"group":..,"covariates":[..]} 指定。'
        )
        return

    cols = [y, group] + covariates
    sub = df[cols].dropna()
    try:
        sub = sub.astype({c: float for c in [y] + covariates})
    except (TypeError, ValueError):
        summary.append("ANCOVA 失败：结果/协变量含非数值，无法拟合。")
        return
    k = int(sub[group].nunique())
    n = int(len(sub))
    n_params = k + len(covariates)  # group dummies (k-1) + intercept + covariates
    if k < 2:
        summary.append(f"ANCOVA 失败：因子水平={k}（需 ≥2）。")
        return
    if n <= n_params + 1:
        summary.append(f"ANCOVA 失败：有效行={n} 不足以拟合 {len(covariates)} 协变量 + {k} 组。")
        return

    try:
        import pandas as pd
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm

        gq = f'C(Q("{group}"))'
        cov_terms = " + ".join(f'Q("{c}")' for c in covariates)
        main_formula = f'Q("{y}") ~ {gq} + {cov_terms}'
        model = smf.ols(main_formula, data=sub).fit()
        aov = anova_lm(model, typ=2)
        aov.to_csv(d / "ancova_table.csv", encoding="utf-8")
        files.append("ancova_table.csv")

        # factor F + p (exact term key, substring fallback)
        f_term = gq if gq in aov.index else next(
            (t for t in aov.index if t != "Residual" and group in t), None)
        factor_f = float(aov.loc[f_term, "F"]) if f_term else float("nan")
        factor_p = float(aov.loc[f_term, "PR(>F)"]) if f_term else float("nan")
        # partial η² for the factor = SS_factor / (SS_factor + SS_residual)
        ss_factor = float(aov.loc[f_term, "sum_sq"]) if f_term else float("nan")
        ss_res = float(aov.loc["Residual", "sum_sq"]) if "Residual" in aov.index else float("nan")
        partial_eta = (ss_factor / (ss_factor + ss_res)) if (ss_factor + ss_res) > 1e-12 else float("nan")

        # covariate p (first covariate, the canonical one) from the model params.
        # Match the EXACT constructed term Q("<cov>") first (a bare substring match can
        # mis-key when one covariate name contains another, e.g. "x" in "income_x").
        first_cov = covariates[0]
        exact_key = f'Q("{first_cov}")'
        if exact_key in model.pvalues.index:
            cov_key = exact_key
        else:
            cov_key = next((nm for nm in model.pvalues.index if first_cov in nm), None)
        covariate_p = float(model.pvalues[cov_key]) if cov_key else float("nan")
        covariate_b = float(model.params[cov_key]) if cov_key else float("nan")

        # homogeneity-of-regression-slopes check: add factor×covariate interaction
        inter_terms = " + ".join(f'{gq}:Q("{c}")' for c in covariates)
        slopes_formula = f"{main_formula} + {inter_terms}"
        slopes_p = float("nan")
        try:
            m_slopes = smf.ols(slopes_formula, data=sub).fit()
            aov_s = anova_lm(m_slopes, typ=2)
            # collect all interaction rows (contain ':' and the group key)
            inter_rows = [t for t in aov_s.index if ":" in t and group in t]
            if inter_rows:
                # combined interaction test: pool via min p (any sig slope difference flags violation)
                ps = [float(aov_s.loc[t, "PR(>F)"]) for t in inter_rows
                      if pd.notna(aov_s.loc[t, "PR(>F)"])]
                slopes_p = float(min(ps)) if ps else float("nan")
        except Exception:
            pass

        # adjusted (EMM) group means at the covariate mean(s): predict at each level
        cov_means = {c: float(sub[c].mean()) for c in covariates}
        levels = list(sub[group].dropna().unique())
        adj_rows = []
        for lv in levels:
            pred_df = pd.DataFrame({group: [lv], **{c: [cov_means[c]] for c in covariates}})
            try:
                adj = float(model.predict(pred_df).iloc[0])
            except Exception:
                adj = float("nan")
            unadj = float(sub.loc[sub[group] == lv, y].mean())
            adj_rows.append({"group": str(lv), "unadjusted_mean": unadj, "adjusted_mean": adj})
        means_tbl = pd.DataFrame(adj_rows)
        means_tbl.to_csv(d / "adjusted_means.csv", index=False, encoding="utf-8")
        files.append("adjusted_means.csv")

        estimates.update({
            "factor_f": factor_f, "factor_p": factor_p,
            "partial_eta_sq": float(partial_eta),
            "covariate_p": covariate_p, "slopes_interaction_p": slopes_p,
            "n_groups": float(k), "n": float(n),
        })

        # PNG: covariate vs outcome scatter colored by group + per-group fit lines
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            xcov = covariates[0]
            fig, ax = plt.subplots(figsize=(6.2, 4.5))
            palette = ["#4C72B0", "#C44E52", "#55A868", "#8172B2", "#CCB974",
                       "#64B5CD", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"]
            for i, lv in enumerate(levels):
                gsub = sub[sub[group] == lv]
                col = palette[i % len(palette)]
                ax.scatter(gsub[xcov], gsub[y], s=20, alpha=0.7, color=col, label=str(lv))
                if len(gsub) >= 2:
                    b, a0 = np.polyfit(gsub[xcov].to_numpy(dtype=float),
                                       gsub[y].to_numpy(dtype=float), 1)
                    xs = np.linspace(gsub[xcov].min(), gsub[xcov].max(), 50)
                    ax.plot(xs, a0 + b * xs, color=col, lw=1.3)
            ax.set_xlabel(f"covariate ({xcov})")
            ax.set_ylabel(str(y))
            ax.set_title(f"ANCOVA: {y} vs covariate by {group}")
            ax.legend(fontsize=8, title=str(group))
            fig.tight_layout()
            fig.savefig(d / "ancova_scatter.png", dpi=150)
            plt.close(fig)
            files.append("ancova_scatter.png")
        except Exception:
            pass

        role_note = ""
        if guessed_g or guessed_c:
            role_note = "（因子/协变量部分自动推断，建议 config 明确 group/covariates）"
        sig = "显著" if (factor_p == factor_p and factor_p < 0.05) else "不显著"
        if slopes_p == slopes_p and slopes_p < 0.05:
            slope_note = (f"⚠ 同质回归斜率假定被违反（因子×协变量交互 p={slopes_p:.3g}<0.05）—— "
                          "各组斜率不同，ANCOVA 的平行斜率前提不成立，调整均值解读须谨慎。")
        elif slopes_p == slopes_p:
            slope_note = f"同质回归斜率检验未拒绝（交互 p={slopes_p:.3g}），平行斜率假定可接受。"
        else:
            slope_note = "（同质斜率交互检验不可估）"
        summary.append(
            f"{entry.method} 完成{role_note}：{y} ~ {group}（{k} 组）+ 协变量 {', '.join(covariates)}（n={n}）；"
            f"控制协变量后因子效应 F={factor_f:.3f}, p={factor_p:.3g}（{sig}），偏 η²={partial_eta:.3f}；"
            f"协变量 {first_cov} 系数={covariate_b:.4g}(p={covariate_p:.3g})。"
            f"调整均值(协变量取均值处)见 adjusted_means.csv。{slope_note}"
            " ⚠ ANCOVA 假定同质回归斜率（交互检验已查）、协变量与结果线性、协变量测量无误差且不受因子影响"
            "（否则会过度/不足调整）。config 可指定 outcome/group/covariates。"
        )
        code += [
            "import statsmodels.formula.api as smf",
            "from statsmodels.stats.anova import anova_lm",
            f"m = smf.ols('Q(\"{y}\") ~ C(Q(\"{group}\")) + {cov_terms}', data=df).fit()",
            "print(anova_lm(m, typ=2))  # ANCOVA：控制协变量后的因子 F",
            f"# 斜率同质检验：加 C(Q(\"{group}\")):协变量 交互项，看是否显著",
        ]
    except Exception as err:
        summary.append(f"ANCOVA 失败：{err}")


# ─────────────────────────────────────────────────────────────────────────────
# (C) repeated_measures_anova — within-subjects RM-ANOVA (long or wide input).
# ─────────────────────────────────────────────────────────────────────────────
@register("repeated_measures_anova")
def _branch_repeated_measures_anova(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np

    try:
        pass
    except Exception:
        summary.append("重复测量方差分析失败：pandas 不可用。")
        return

    # ── resolve long-format roles (config wins), else try to detect wide format ──
    subject = cfg.get("subject") if cfg.get("subject") in df.columns else None
    within = cfg.get("within") if cfg.get("within") in df.columns else None
    cont = _continuous(fp)
    outcome = cfg.get("outcome") if cfg.get("outcome") in df.columns else (cont[0] if cont else None)

    long_df, n_subj, n_cond, wide_used = None, 0, 0, False

    if subject is not None and within is not None and outcome is not None:
        # explicit long format
        long_df = df[[subject, within, outcome]].dropna().copy()
    else:
        # try WIDE format: config "measures" (list of repeated-measure columns) or infer
        measures = cfg.get("measures")
        if isinstance(measures, str):
            measures = [measures]
        if measures:
            measures = [c for c in measures if c in df.columns]
        else:
            # infer: >=2 continuous columns are the repeated measures; subject = unit_col or row index
            measures = list(cont)
        if measures and len(measures) >= 2:
            wide_used = True
            subj_col = (cfg.get("subject") if cfg.get("subject") in df.columns
                        else (fp.unit_col if fp.unit_col in df.columns else None))
            wdf = df.copy()
            if subj_col is None:
                wdf = wdf.reset_index().rename(columns={"index": "_subject_"})
                subj_col = "_subject_"
            keep = [subj_col] + measures
            wdf = wdf[keep].dropna()
            long_df = wdf.melt(id_vars=[subj_col], value_vars=measures,
                               var_name="_condition_", value_name="_value_")
            subject, within, outcome = subj_col, "_condition_", "_value_"
        elif subject is not None or within is not None or outcome is not None:
            # partial long config but missing pieces
            long_df = None

    if long_df is None or subject is None or within is None or outcome is None:
        summary.append(
            "重复测量方差分析失败：需要 长表(subject + within 条件 + outcome) 或 宽表(每受试者多列重复测量)。"
            'config={"subject":..,"within":..,"outcome":..} 或 {"measures":[列..]} 指定。'
        )
        return

    # numeric outcome
    try:
        long_df[outcome] = long_df[outcome].astype(float)
    except (TypeError, ValueError):
        summary.append(f"重复测量方差分析失败：结果 {outcome} 非数值。")
        return

    long_df = long_df.dropna(subset=[subject, within, outcome])
    n_cond = int(long_df[within].nunique())
    if n_cond < 2:
        summary.append(f"重复测量方差分析失败：within 条件数={n_cond}（需 ≥2）。")
        return

    # balance: keep only subjects observed once per condition (AnovaRM needs balanced data)
    cell = long_df.groupby([subject, within], observed=True).size().unstack(fill_value=0)
    complete_subjects = cell.index[(cell == 1).all(axis=1)].tolist()
    n_dropped = int(cell.shape[0] - len(complete_subjects))
    long_df = long_df[long_df[subject].isin(complete_subjects)]
    n_subj = int(long_df[subject].nunique())
    if n_subj < 2:
        summary.append(
            f"重复测量方差分析失败：去掉缺条件/重复观测后只剩 {n_subj} 名完整受试者（需 ≥2）。"
            "RM-ANOVA 需平衡（每受试者每条件恰 1 次）。"
        )
        return

    try:
        from statsmodels.stats.anova import AnovaRM

        rm = AnovaRM(long_df, depvar=outcome, subject=subject, within=[within]).fit()
        tbl = rm.anova_table
        tbl.to_csv(d / "rm_anova_table.csv", encoding="utf-8")
        files.append("rm_anova_table.csv")

        f_stat = float(tbl.loc[within, "F Value"])
        p_value = float(tbl.loc[within, "Pr > F"])
        num_df = float(tbl.loc[within, "Num DF"])
        den_df = float(tbl.loc[within, "Den DF"])

        # build the subject×condition matrix for partial η² + sphericity diagnostics
        wide = long_df.pivot_table(index=subject, columns=within, values=outcome,
                                   observed=True).dropna()
        M = wide.to_numpy(dtype=float)
        ns, kc = M.shape

        # partial η² (within factor) = SS_condition / (SS_condition + SS_error)
        grand = float(M.mean())
        cond_means = M.mean(axis=0)
        subj_means = M.mean(axis=1)
        ss_cond = float(ns * ((cond_means - grand) ** 2).sum())
        # error SS = total - subjects - condition (two-way no-interaction RM model)
        ss_total = float(((M - grand) ** 2).sum())
        ss_subj = float(kc * ((subj_means - grand) ** 2).sum())
        ss_error = ss_total - ss_subj - ss_cond
        partial_eta = ss_cond / (ss_cond + ss_error) if (ss_cond + ss_error) > 1e-12 else float("nan")

        # sphericity: GG / HF epsilon from the covariance matrix of the conditions
        gg_eps, hf_eps, mauchly_p = _sphericity(M)

        # GG-corrected p (adjust df by epsilon)
        from scipy import stats as _st
        if gg_eps == gg_eps:
            gg_p = float(_st.f.sf(f_stat, num_df * gg_eps, den_df * gg_eps))
        else:
            gg_p = float("nan")

        estimates.update({
            "f_stat": f_stat, "p_value": p_value,
            "partial_eta_sq": float(partial_eta),
            "mauchly_p": float(mauchly_p), "gg_epsilon": float(gg_eps),
            "gg_corrected_p": gg_p,
            "n_subjects": float(n_subj), "n_conditions": float(n_cond),
        })

        # condition means table
        cmeans = long_df.groupby(within, observed=True)[outcome].agg(["mean", "std", "count"])
        cmeans.to_csv(d / "condition_means.csv", encoding="utf-8")
        files.append("condition_means.csv")

        # PNG: per-condition means with within-subject error bars + faint subject spaghetti
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            conds = list(wide.columns)
            x = range(len(conds))
            # within-subject (Cousineau-Morey style) SE: remove between-subject variability
            adj = M - subj_means[:, None] + grand
            wse = adj.std(axis=0, ddof=1) / np.sqrt(max(ns, 1))
            fig, ax = plt.subplots(figsize=(6.2, 4.5))
            for r in range(ns):  # subject spaghetti
                ax.plot(x, M[r, :], color="#BBBBBB", lw=0.6, alpha=0.5)
            ax.errorbar(x, cond_means, yerr=1.96 * wse, fmt="o-", capsize=4,
                        color="#4C72B0", lw=2, label="condition mean ±95% within-subj CI")
            ax.set_xticks(list(x))
            ax.set_xticklabels([str(c) for c in conds], rotation=30, ha="right")
            ax.set_xlabel(f"within-subject condition ({within})")
            ax.set_ylabel(f"mean {outcome}")
            ax.set_title(f"Repeated-measures ANOVA — {outcome}")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "rm_means.png", dpi=150)
            plt.close(fig)
            files.append("rm_means.png")
        except Exception:
            pass

        fmt = "宽表(已自动 melt)" if wide_used else "长表"
        drop_note = f"（去除 {n_dropped} 名缺条件/重复的受试者）" if n_dropped > 0 else ""
        sig = "显著" if (p_value == p_value and p_value < 0.05) else "不显著"
        if mauchly_p == mauchly_p and mauchly_p < 0.05:
            sph_note = (f"⚠ Mauchly 球形度检验被拒绝(p≈{mauchly_p:.3g})——球形假定违反，"
                        f"采用 GG 校正 p={gg_p:.3g}（ε_GG={gg_eps:.3f}）。")
        elif mauchly_p == mauchly_p:
            sph_note = f"Mauchly 球形度检验未拒绝(p≈{mauchly_p:.3g})，未校正 p 可用；GG 校正 p={gg_p:.3g} 备查。"
        else:
            sph_note = f"（Mauchly 近似不可估，仍报 GG 校正 p={gg_p:.3g}, ε_GG={gg_eps:.3f}）"
        summary.append(
            f"{entry.method} 完成（{fmt}{drop_note}）：{n_subj} 名受试者 × {n_cond} 个条件（{within}，结果 {outcome}）；"
            f"组内主效应 F({num_df:.0f},{den_df:.0f})={f_stat:.3f}, p={p_value:.3g}（{sig}），偏 η²={partial_eta:.3f}。"
            f"{sph_note}"
            " ⚠ RM-ANOVA 假定球形度（Mauchly 查；违反时用已报的 GG 校正 p）；需 ≥2 条件且平衡（缺条件的受试者已删并披露）；"
            "Mauchly p 由卡方近似得出（小样本近似），ε 由条件协方差阵算。config 可指定 subject/within/outcome（或 measures 宽表）。"
        )
        code += [
            "from statsmodels.stats.anova import AnovaRM",
            f"rm = AnovaRM(long_df, depvar='{outcome}', subject='{subject}', within=['{within}']).fit()",
            "print(rm.anova_table)  # 组内因子 F 检验",
            "# 球形度违反时用 Greenhouse-Geisser ε 校正 df：p = f.sf(F, df1*eps, df2*eps)",
        ]
    except Exception as err:
        summary.append(f"重复测量方差分析失败：{err}")


def _sphericity(M):
    """Greenhouse-Geisser & Huynh-Feldt epsilon + an approximate Mauchly p, from the
    subjects×conditions matrix M (rows = subjects, cols = conditions, balanced/complete).

    Returns (gg_epsilon, hf_epsilon, mauchly_p). Any value is nan if not computable.
    GG ε is computed from the eigenvalues of the covariance matrix of the conditions; the
    Mauchly p uses the standard chi-square approximation to W = det(S*)/(tr(S*)/(k-1))^(k-1)
    on the (k-1) transformed contrasts. Disclosed as an approximation in the summary.
    """
    import numpy as np

    try:
        ns, k = M.shape
        if k < 2 or ns < 2:
            return float("nan"), float("nan"), float("nan")
        # covariance of the conditions (k×k), then project onto an orthonormal contrast
        # basis of the (k-1)-dim difference space -> S* is (k-1)×(k-1).
        S = np.cov(M, rowvar=False, ddof=1)
        # orthonormal contrast matrix C: (k-1)×k, rows orthogonal to the all-ones vector
        C = _orthonormal_contrasts(k)
        Sstar = C @ S @ C.T
        eig = np.linalg.eigvalsh(Sstar)
        eig = eig[eig > 1e-12]
        m = len(eig)
        if m == 0:
            return float("nan"), float("nan"), float("nan")
        # Greenhouse-Geisser epsilon = (sum λ)^2 / ((k-1) * sum λ^2)
        gg = float((eig.sum() ** 2) / ((k - 1) * (eig ** 2).sum()))
        gg = max(1.0 / (k - 1), min(1.0, gg))  # bound to [1/(k-1), 1]
        # Huynh-Feldt epsilon (less conservative); standard formula
        n = ns
        hf_num = n * (k - 1) * gg - 2.0
        hf_den = (k - 1) * (n - 1 - (k - 1) * gg)
        hf = float(hf_num / hf_den) if abs(hf_den) > 1e-12 else float("nan")
        if hf == hf:
            hf = min(1.0, hf)

        # Mauchly's W + chi-square approximation on S* (k-1 eigenvalues)
        from scipy import stats as _st
        detS = float(np.prod(eig))
        trS = float(eig.sum())
        p = k - 1
        W = detS / ((trS / p) ** p) if trS > 1e-12 else float("nan")
        mauchly_p = float("nan")
        if W == W and W > 0:
            dfree = p * (p + 1) // 2 - 1
            d_corr = 1.0 - (2.0 * p ** 2 + p + 2.0) / (6.0 * p * (n - 1))
            chi = -(n - 1) * d_corr * np.log(W)
            if dfree >= 1 and np.isfinite(chi) and chi >= 0:
                mauchly_p = float(_st.chi2.sf(chi, dfree))
        return gg, hf, mauchly_p
    except Exception:
        return float("nan"), float("nan"), float("nan")


def _orthonormal_contrasts(k):
    """An orthonormal (k-1)×k contrast matrix whose rows span the space orthogonal to the
    all-ones vector (via QR of the centering matrix). Used to project the condition
    covariance into the difference space for sphericity diagnostics."""
    import numpy as np

    # start from a basis of the orthogonal complement of 1_k
    A = np.eye(k) - np.ones((k, k)) / k  # centering -> rank k-1
    # QR of the centering matrix to get orthonormal columns spanning its column space
    Q, _ = np.linalg.qr(A)
    # Q has k columns; the first k-1 are an orthonormal basis of the centered space
    C = Q[:, : k - 1].T
    return C
