"""Branch handlers for the panel_extra family — completes the panel toolkit.

Three time-honoured panel-econometrics methods (all pure-Python via linearmodels,
NO R), each resolving the panel roles exactly like econometrics.py's
``random_effects`` / ``spatial_panel`` (unit/entity col, time col, outcome = first
continuous, predictors = remaining continuous/binary; config can override
unit/time/outcome/predictors):

  * mundlak          — Mundlak (1978) correlated random effects: augment RE with the
                       ENTITY MEANS of the time-varying covariates; the joint Wald that
                       all mean-coefs = 0 is the (robust) Mundlak FE-vs-RE test.
  * hausman_test     — formal Hausman (1978) FE-vs-RE test: H = (b_FE-b_RE)'
                       pinv(V_FE-V_RE) (b_FE-b_RE) ~ χ²_k (unadjusted/efficient covs).
  * first_difference — First-difference (FD) estimator (linearmodels FirstDifferenceOLS):
                       Δy = β·Δx + Δε, removing entity effects; contrast with FE (within).

Each handler degrades honestly (no linearmodels / not a panel / too few periods or
entities / no time-varying predictors / singular fit -> Chinese "<方法>跳过：<原因>"
+ return), writes CSV + PNG (matplotlib Agg, ENGLISH plot labels), fills float
`estimates`, appends a Chinese `summary` ending with ⚠ disclosures, and MUTATES ctx
(never rebinds). See executor/_branch_api.py, econometrics.py, and CLAUDE.md.

linearmodels + statsmodels are installed.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


# ─────────────────────────────────────────────────────────────────────────────
# Shared panel role-resolution (same idiom as econometrics.random_effects, plus
# config overrides like spatial_panel). Returns
#   (unit, time, outcome, predictors, problem_msg)
# When problem_msg is not None the caller appends it to summary and returns.
# `min_periods` enforces the per-method time-period requirement.
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_panel(ctx: Ctx, label: str, min_periods: int = 2):
    fp, cfg, df = ctx.fp, ctx.cfg, ctx.df

    unit = cfg.get("unit") or fp.unit_col
    time = cfg.get("time") or fp.time_col
    if not (unit and time):
        return None, None, None, [], f"{label}跳过：需要面板数据（单位列 + 时间列）。"

    _exc = {unit, time}
    outcome = cfg.get("outcome") or next(
        (c.name for c in fp.columns if c.kind == "continuous" and c.name not in _exc),
        None,
    )
    if outcome is None:
        return unit, time, None, [], f"{label}跳过：需要 1 个连续结果变量（outcome）。"

    predictors = cfg.get("predictors") or [
        c.name
        for c in fp.columns
        if c.kind in {"continuous", "binary"} and c.name not in _exc and c.name != outcome
    ][:6]
    if not predictors:
        return unit, time, outcome, [], f"{label}跳过：需要 ≥1 个预测变量。"

    if unit not in df.columns or time not in df.columns or outcome not in df.columns:
        return unit, time, outcome, [], f"{label}跳过：指定的列不在数据中。"

    n_periods = int(df[time].nunique())
    if n_periods < min_periods:
        return (
            unit, time, outcome, predictors,
            f"{label}跳过：需要 ≥{min_periods} 个时间期（当前 {n_periods}）。",
        )
    return unit, time, outcome, predictors, None


def _panel_frame(df, unit, time, outcome, predictors):
    """Build a linearmodels MultiIndex (entity, time) panel: drop NA, dedup the
    (unit,time) key, set_index([unit,time]). linearmodels needs a 2-level index
    whose first level is the entity and second the time — exactly what
    econometrics.random_effects builds."""
    return (
        df[[unit, time, outcome, *predictors]]
        .dropna()
        .drop_duplicates([unit, time])
        .set_index([unit, time])
    )


def _time_varying(dd, predictors):
    """Predictors that actually vary WITHIN at least one entity. Time-invariant
    regressors are collinear with the entity effect / mean and must be dropped
    (FD differences them out; the Mundlak mean is collinear; FE drops them)."""
    keep = []
    for p in predictors:
        # within-entity std summed across entities; >0 means some within variation
        wsd = dd.groupby(level=0)[p].std(ddof=0).fillna(0.0)
        if float(wsd.sum()) > 1e-12:
            keep.append(p)
    return keep


def _coef_plot(d, files, terms, coefs, ses, xlabel, title):
    """Best-effort coefficient plot with 95% CIs (Agg, ENGLISH labels)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        ax.errorbar(coefs, range(len(terms)), xerr=[1.96 * s for s in ses], fmt="o")
        ax.axvline(0, color="grey", ls="--")
        ax.set_yticks(range(len(terms)))
        ax.set_yticklabels(terms)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(d / "coefficients.png", dpi=150)
        plt.close(fig)
        files.append("coefficients.png")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# (A) mundlak — Mundlak (1978) correlated random effects + robust FE-vs-RE test
# ─────────────────────────────────────────────────────────────────────────────
@register("mundlak")
def _branch_mundlak(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    unit, time, outcome, predictors, problem = _resolve_panel(ctx, "Mundlak CRE", min_periods=2)
    if problem is not None:
        summary.append(problem)
        return
    try:
        import numpy as np
        import pandas as pd
        from scipy.stats import chi2

        from linearmodels.panel import RandomEffects

        dd = _panel_frame(df, unit, time, outcome, predictors)
        n_entities = int(dd.index.get_level_values(0).nunique())
        if n_entities < 2:
            summary.append("Mundlak CRE 跳过：需要 ≥2 个单位（实体）。")
            return

        tv = _time_varying(dd, predictors)
        dropped = [p for p in predictors if p not in tv]
        if not tv:
            summary.append(
                "Mundlak CRE 跳过：所有预测变量在单位内都不随时间变化（实体均值与之共线，无法识别）。"
            )
            return

        # entity means of the time-varying covariates (the Mundlak device)
        means = dd.groupby(level=0)[tv].transform("mean")
        mean_terms = [f"{c}_mean" for c in tv]
        means.columns = mean_terms
        aug = pd.concat([dd[tv], means], axis=1)
        aug["const"] = 1.0

        # RE on the augmented regressors: β on x is within-consistent, γ on x̄ are the
        # mean coefs. Clustered SE for reporting; the Mundlak Wald on γ uses these.
        re = RandomEffects(dd[outcome], aug).fit(cov_type="clustered", cluster_entity=True)

        b_mean = re.params[mean_terms].values
        v_mean = re.cov.loc[mean_terms, mean_terms].values
        # robust to near-singular V (collinear means): pseudo-inverse
        try:
            vinv = np.linalg.inv(v_mean)
            pinv_used = False
        except np.linalg.LinAlgError:
            vinv = np.linalg.pinv(v_mean)
            pinv_used = True
        wald = max(0.0, float(b_mean @ vinv @ b_mean))
        wp = float(chi2.sf(wald, len(mean_terms)))
        reject = wp < 0.05
        rec = "FE（固定效应）" if reject else "RE（随机效应）可接受"

        # per-predictor β (within) and γ (mean) table
        rows = []
        for p in tv:
            mt = f"{p}_mean"
            rows.append(
                {
                    "term": p,
                    "beta_within": round(float(re.params[p]), 4),
                    "beta_se": round(float(re.std_errors[p]), 4),
                    "beta_p": round(float(re.pvalues[p]), 4),
                    "gamma_mean": round(float(re.params[mt]), 4),
                    "gamma_se": round(float(re.std_errors[mt]), 4),
                    "gamma_p": round(float(re.pvalues[mt]), 4),
                }
            )
        tab = pd.DataFrame(rows)
        tab.to_csv(d / "mundlak_coefficients.csv", index=False, encoding="utf-8")
        files.append("mundlak_coefficients.csv")

        # plot: β (within) vs γ (mean) per predictor, with 95% CIs
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            ypos = np.arange(len(tv))
            bse = [1.96 * float(re.std_errors[p]) for p in tv]
            gse = [1.96 * float(re.std_errors[f"{p}_mean"]) for p in tv]
            bco = [float(re.params[p]) for p in tv]
            gco = [float(re.params[f"{p}_mean"]) for p in tv]
            fig, ax = plt.subplots(figsize=(5.8, 3.4))
            ax.errorbar(bco, ypos + 0.12, xerr=bse, fmt="o", color="#4C72B0",
                        label="beta (within)")
            ax.errorbar(gco, ypos - 0.12, xerr=gse, fmt="s", color="#DD8452",
                        label="gamma (entity mean)")
            ax.axvline(0, color="grey", ls="--")
            ax.set_yticks(ypos)
            ax.set_yticklabels(tv)
            ax.set_xlabel("coefficient (95% CI)")
            ax.set_title("Mundlak CRE: within (beta) vs entity-mean (gamma)")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "coefficients.png", dpi=150)
            plt.close(fig)
            files.append("coefficients.png")
        except Exception:
            pass

        estimates["mundlak_wald_chi2"] = round(wald, 4)
        estimates["mundlak_p"] = round(wp, 4)
        estimates["n_predictors"] = float(len(tv))
        estimates["n_entities"] = float(n_entities)
        estimates["n_obs"] = float(len(dd))
        for p in tv:
            estimates[p] = round(float(re.params[p]), 4)
            estimates[f"{p}_mean"] = round(float(re.params[f"{p}_mean"]), 4)

        dropped_note = (
            f"（时不变变量 {dropped} 因与实体均值共线已剔除）" if dropped else ""
        )
        summary.append(
            f"{entry.method} 完成：面板 {n_entities} 单位 × "
            f"{int(dd.index.get_level_values(1).nunique())} 期；结果 {outcome}，"
            f"{len(tv)} 个时变预测变量{dropped_note}。"
            f"Mundlak 检验（γ=0 联合 Wald）χ²={wald:.3f}（df={len(mean_terms)}），"
            f"p={wp:.3g} → {rec}。β（within 一致）与 γ（实体均值系数）见 mundlak_coefficients.csv。"
            f"\n⚠ 披露：(1) 显著的 Mundlak 检验（p<0.05）表示实体效应与回归元相关、"
            "RE 不一致，应改用 FE（β 已是 within 一致估计）；不显著则 RE 可接受、更有效。"
            "(2) Mundlak 检验与 Hausman 渐近等价（同假设下）；本实现用 clustered SE，是其**稳健化**版本"
            "（基于辅助回归、避免协方差差非正定，对异方差/序列相关稳健，故与经典 Hausman 数值不同但更可靠）。"
            f"(3) 只有时变协变量有意义的实体均值；时不变变量与均值共线、被剔除{dropped_note}。"
            "(4) 这是模型设定下的估计，非结构因果；clustered SE 假定单位间独立。"
            + ("（V(γ) 近奇异，已用伪逆，检验对小样本敏感）" if pinv_used else "")
        )
        code += [
            "from linearmodels.panel import RandomEffects  # Mundlak (1978) CRE",
            "# aug = [x, x_mean(=entity mean of x), const]; RE(y ~ aug)",
            "# Mundlak test = joint Wald that all x_mean coefs = 0 (robust FE-vs-RE)",
        ]
    except Exception as err:
        summary.append(f"Mundlak CRE 跳过：{err}")


# ─────────────────────────────────────────────────────────────────────────────
# (B) hausman_test — formal Hausman (1978) FE-vs-RE specification test
# ─────────────────────────────────────────────────────────────────────────────
@register("hausman_test")
def _branch_hausman_test(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    unit, time, outcome, predictors, problem = _resolve_panel(ctx, "Hausman 检验", min_periods=2)
    if problem is not None:
        summary.append(problem)
        return
    try:
        import numpy as np
        import pandas as pd
        from scipy.stats import chi2

        from linearmodels.panel import PanelOLS, RandomEffects

        dd = _panel_frame(df, unit, time, outcome, predictors)
        n_entities = int(dd.index.get_level_values(0).nunique())
        if n_entities < 2:
            summary.append("Hausman 检验跳过：需要 ≥2 个单位（实体）。")
            return

        tv = _time_varying(dd, predictors)
        if not tv:
            summary.append(
                "Hausman 检验跳过：所有预测变量在单位内都不随时间变化（FE 会全部剔除，无可比系数）。"
            )
            return

        # clustered fits for the reported coefficients/SE
        fe = PanelOLS(dd[outcome], dd[tv], entity_effects=True).fit(
            cov_type="clustered", cluster_entity=True
        )
        re = RandomEffects(dd[outcome], dd[tv].assign(const=1.0)).fit(
            cov_type="clustered", cluster_entity=True
        )
        # classic Hausman needs the EFFICIENT (unadjusted) covariances so that
        # V_FE - V_RE is PSD; clustered/robust covs break the χ² basis. Use the
        # unadjusted fits for the test, the clustered fits only for reporting SEs.
        fe_u = PanelOLS(dd[outcome], dd[tv], entity_effects=True).fit()
        re_u = RandomEffects(dd[outcome], dd[tv].assign(const=1.0)).fit()

        common = [p for p in tv if p in fe_u.params.index and p in re_u.params.index]
        if not common:
            summary.append("Hausman 检验跳过：FE 与 RE 无共同（时变）系数可比较。")
            return
        diff = (fe_u.params[common] - re_u.params[common]).values
        vdiff = (fe_u.cov.loc[common, common] - re_u.cov.loc[common, common]).values
        # The classic variance-difference Hausman is only valid when V_FE - V_RE is
        # positive-definite. In finite samples V_FE can be < V_RE (vdiff non-PSD),
        # making H undefined — clipping it to 0 would FALSELY say "use RE" on data
        # where RE is clearly inconsistent. So use the classic form only when PSD;
        # otherwise fall back to the regression-based (Mundlak) Hausman, which is
        # algebraically equivalent and ALWAYS well-defined.
        eigmin = float(np.linalg.eigvalsh(vdiff).min()) if len(common) else 0.0
        if len(common) and np.linalg.cond(vdiff) < 1e12 and eigmin > 0:
            h_stat = max(0.0, float(diff @ np.linalg.inv(vdiff) @ diff))
            h_df = len(common)
            h_p = float(chi2.sf(h_stat, h_df))
            method_note = "classic"
        else:
            # regression-based Hausman: RE augmented with entity means of the
            # time-varying regressors; the joint Wald that those means = 0 IS the
            # robust Hausman (Mundlak/Wooldridge), well-defined even when vdiff is non-PSD.
            # Use the UNADJUSTED covariance (like the classic branch) so the fallback is
            # the exact asymptotic equivalent of the classic Hausman on the SAME basis
            # (the cluster-robust variant lives in the separate `mundlak` method).
            emeans = dd.groupby(level=0)[tv].transform("mean")
            mcols = [f"{c}__m" for c in tv]
            emeans.columns = mcols
            aug = pd.concat([dd[tv], emeans], axis=1).assign(const=1.0)
            re_aux = RandomEffects(dd[outcome], aug).fit()
            bm = re_aux.params[mcols].values
            vm = re_aux.cov.loc[mcols, mcols].values
            h_stat = max(0.0, float(bm @ np.linalg.pinv(vm) @ bm))
            h_df = len(mcols)
            h_p = float(chi2.sf(h_stat, h_df))
            method_note = "regression"
        use_fe = h_p < 0.05
        rec = "FE（固定效应）" if use_fe else "RE（随机效应，更有效且一致）"

        tab = pd.DataFrame(
            {
                "term": common,
                "FE_coef": [round(float(fe.params[c]), 4) for c in common],
                "FE_se": [round(float(fe.std_errors[c]), 4) for c in common],
                "RE_coef": [round(float(re.params[c]), 4) for c in common],
                "RE_se": [round(float(re.std_errors[c]), 4) for c in common],
                "difference": [round(float(fe.params[c] - re.params[c]), 4) for c in common],
            }
        )
        tab.to_csv(d / "hausman_fe_re_coefficients.csv", index=False, encoding="utf-8")
        files.append("hausman_fe_re_coefficients.csv")

        # plot: FE vs RE coefficient comparison with CIs (ENGLISH labels)
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            ypos = np.arange(len(common))
            fe_se = [1.96 * float(fe.std_errors[c]) for c in common]
            re_se = [1.96 * float(re.std_errors[c]) for c in common]
            fe_co = [float(fe.params[c]) for c in common]
            re_co = [float(re.params[c]) for c in common]
            fig, ax = plt.subplots(figsize=(5.8, 3.4))
            ax.errorbar(fe_co, ypos + 0.12, xerr=fe_se, fmt="o", color="#4C72B0",
                        label="FE (fixed effects)")
            ax.errorbar(re_co, ypos - 0.12, xerr=re_se, fmt="s", color="#DD8452",
                        label="RE (random effects)")
            ax.axvline(0, color="grey", ls="--")
            ax.set_yticks(ypos)
            ax.set_yticklabels(common)
            ax.set_xlabel("coefficient (95% CI)")
            ax.set_title("Hausman test: FE vs RE coefficients")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "coefficients.png", dpi=150)
            plt.close(fig)
            files.append("coefficients.png")
        except Exception:
            pass

        estimates["hausman_chi2"] = round(h_stat, 4)
        estimates["hausman_df"] = float(h_df)
        estimates["hausman_p"] = round(h_p, 4)
        estimates["n_coefs_compared"] = float(len(common))
        estimates["n_entities"] = float(n_entities)
        for c in common:
            estimates[f"{c}_FE"] = round(float(fe.params[c]), 4)
            estimates[f"{c}_RE"] = round(float(re.params[c]), 4)

        summary.append(
            f"{entry.method} 完成：面板 {n_entities} 单位 × "
            f"{int(dd.index.get_level_values(1).nunique())} 期；结果 {outcome}，"
            f"比较 {len(common)} 个共同（时变）系数。"
            f"Hausman H={h_stat:.3f}（df={h_df}），p={h_p:.3g} → 推荐 {rec}"
            f"（p<0.05：FE 与 RE 系数显著有别、实体效应与回归元相关、RE 不一致 → 用 FE；"
            "否则 RE 既一致又更有效）。FE/RE 系数对比见 hausman_fe_re_coefficients.csv。"
            + ("\n⚠ 披露：(1) 经典 Hausman 用无调整（高效）协方差的 V_FE−V_RE（正定，理论基础）；"
               if method_note == "classic" else
               "\n⚠ 披露：(1) 经典 V_FE−V_RE 在本样本非正定（FE 未必比 RE 方差大），经典式失效——"
               "已改用**回归式(Mundlak)Hausman**（RE 加协变量的实体均值、对均值系数做联合 Wald），"
               "其与经典式渐近等价且恒可定义；")
            + "报告表里的 SE 是 clustered 稳健 SE（仅供阅读）。"
            "(2) Hausman 在 H0 下假定 RE 高效；本族的 mundlak 方法是更稳健的等价替代（可交叉验证）。"
            "(3) 时不变变量在 FE 下被剔除、不参与比较；这是设定检验，非结构因果。"
        )
        code += [
            "from linearmodels.panel import PanelOLS, RandomEffects  # Hausman (1978)",
            "# FE=PanelOLS(entity_effects); RE=RandomEffects; use UNADJUSTED covs for the test",
            "# H=(b_FE-b_RE)' pinv(V_FE-V_RE) (b_FE-b_RE) ~ chi2_k",
        ]
    except Exception as err:
        summary.append(f"Hausman 检验跳过：{err}")


# ─────────────────────────────────────────────────────────────────────────────
# (C) first_difference — First-difference (FD) estimator
# ─────────────────────────────────────────────────────────────────────────────
@register("first_difference")
def _branch_first_difference(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    unit, time, outcome, predictors, problem = _resolve_panel(ctx, "一阶差分(FD)", min_periods=2)
    if problem is not None:
        summary.append(problem)
        return
    try:
        import numpy as np
        import pandas as pd

        from linearmodels.panel import FirstDifferenceOLS, PanelOLS

        dd = _panel_frame(df, unit, time, outcome, predictors)
        n_entities = int(dd.index.get_level_values(0).nunique())
        if n_entities < 2:
            summary.append("一阶差分(FD)跳过：需要 ≥2 个单位（实体）。")
            return

        tv = _time_varying(dd, predictors)
        dropped = [p for p in predictors if p not in tv]
        if not tv:
            summary.append(
                "一阶差分(FD)跳过：所有预测变量在单位内都不随时间变化（差分后恒为 0，无法识别）。"
            )
            return

        # FirstDifferenceOLS: NO constant allowed (it differences out); clustered SE.
        fd = FirstDifferenceOLS(dd[outcome], dd[tv]).fit(
            cov_type="clustered", cluster_entity=True
        )
        # FE (within) contrast on the same spec
        try:
            fe = PanelOLS(dd[outcome], dd[tv], entity_effects=True).fit(
                cov_type="clustered", cluster_entity=True
            )
            fe_ok = True
        except Exception:
            fe = None
            fe_ok = False

        ci = fd.conf_int()
        rows = []
        for p in tv:
            row = {
                "term": p,
                "FD_coef": round(float(fd.params[p]), 4),
                "FD_se": round(float(fd.std_errors[p]), 4),
                "FD_p": round(float(fd.pvalues[p]), 4),
                "ci_low": round(float(ci.loc[p, "lower"]), 4),
                "ci_high": round(float(ci.loc[p, "upper"]), 4),
            }
            if fe_ok and p in fe.params.index:
                row["FE_within_coef"] = round(float(fe.params[p]), 4)
            rows.append(row)
        tab = pd.DataFrame(rows)
        tab.to_csv(d / "first_difference_coefficients.csv", index=False, encoding="utf-8")
        files.append("first_difference_coefficients.csv")

        _coef_plot(
            d, files, tv,
            [float(fd.params[p]) for p in tv],
            [float(fd.std_errors[p]) for p in tv],
            "FD coefficient (95% CI)",
            f"First-difference estimator — {outcome}",
        )

        coefs = np.array([float(fd.params[p]) for p in tv])
        estimates["n_predictors"] = float(len(tv))
        estimates["n_entities"] = float(n_entities)
        estimates["n_obs_differenced"] = float(int(fd.nobs))
        estimates["max_abs_coef"] = round(float(np.max(np.abs(coefs))), 4)
        estimates["rsquared"] = round(float(fd.rsquared), 4)
        for p in tv:
            estimates[p] = round(float(fd.params[p]), 4)

        n_periods = int(dd.index.get_level_values(1).nunique())
        fe_note = ""
        if fe_ok:
            maxgap = max(
                abs(float(fd.params[p]) - float(fe.params[p]))
                for p in tv if p in fe.params.index
            )
            if n_periods == 2:
                fe_note = "（T=2，FD 与 FE within 估计应一致）"
            else:
                fe_note = f"（T>2，FD 与 FE 一般不同；最大系数差={maxgap:.3f}）"
        dropped_note = f"（时不变变量 {dropped} 差分后为 0、已剔除）" if dropped else ""
        summary.append(
            f"{entry.method} 完成：面板 {n_entities} 单位 × {n_periods} 期；结果 {outcome}，"
            f"{len(tv)} 个时变预测变量{dropped_note}；差分后 {int(fd.nobs)} 个观测，"
            f"R²={fd.rsquared:.3f}。系数（含 95% CI 与 FE within 对照）见 "
            f"first_difference_coefficients.csv{fe_note}。"
            "\n⚠ 披露：(1) FD 仅移除实体（单位）效应，**不移除时间效应**——若有共同时间冲击需另控时间固定效应；"
            "(2) FD 需每个单位 ≥2 期且预测变量随时间变化（时不变变量差分后恒为 0、被剔除）"
            f"{dropped_note}；"
            "(3) FD 与 FE 的相对效率取决于误差的序列相关：误差近随机游走时 FD 更有效、"
            "误差无序列相关时 FE 更有效，T=2 时二者重合；"
            "(4) clustered SE 假定单位间独立；这是模型设定下的估计，非结构因果。"
        )
        code += [
            "from linearmodels.panel import FirstDifferenceOLS  # FD estimator (no constant)",
            "# FirstDifferenceOLS(y, X).fit(cov_type='clustered', cluster_entity=True)",
            "# delta y = beta * delta x + delta eps  (entity effects differenced out)",
        ]
    except Exception as err:
        summary.append(f"一阶差分(FD)跳过：{err}")
