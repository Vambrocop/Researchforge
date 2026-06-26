"""Branch handlers for the psychometrics / reliability family.

Pure-Python (numpy / scipy / pandas / factor_analyzer) — no R bridge. Three
reliability estimators:

  * ``cronbach_alpha``  — Cronbach's coefficient alpha (Cronbach 1951) with a
    Feldt (1965) F-based 95% CI, corrected item-total correlations and
    alpha-if-item-dropped diagnostics.
  * ``mcdonald_omega``  — McDonald's omega_total (McDonald 1999) from a single
    common-factor (congeneric) loading solution.
  * ``icc``             — Intraclass correlation coefficients ICC(1,1)/(2,1)/(3,1)
    and their average-of-k forms, per Shrout & Fleiss (1979).

Engine conventions (see CLAUDE.md「引擎约定」): handlers mutate
summary/estimates/files/code (never rebind); items default to continuous/count
columns excluding unit_col/time_col and are overridable via config
``items``/``columns``; products are CSV + (best-effort) English-labelled Agg PNG;
estimates hold floats; summary is Chinese and ends with ⚠ disclosures; on failure
append a Chinese "<方法>失败/跳过：<reason>" line and return.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


# ---------------------------------------------------------------------------
# Shared item-resolution helper (local to this family)
# ---------------------------------------------------------------------------

def _resolve_items(ctx: Ctx) -> list[str]:
    """Pick the item / rater columns.

    Priority: config ``items`` then config ``columns`` (any columns the user
    names, validated against the frame); otherwise the auto default of
    continuous + count columns excluding the panel unit/time columns. Likert
    items are frequently profiled as ``count`` (small-integer) — both kinds are
    accepted so ordinal scales are not silently dropped.
    """
    df, fp, cfg = ctx.df, ctx.fp, ctx.cfg
    forced = cfg.get("items") or cfg.get("columns")
    if forced:
        return [c for c in forced if c in df.columns]
    excl = {fp.unit_col, fp.time_col}
    return [
        c.name
        for c in fp.columns
        if c.kind in {"continuous", "count"} and c.name not in excl
    ]


# ===========================================================================
# 1. Cronbach's alpha
# ===========================================================================

@register("cronbach_alpha")
def _branch_cronbach_alpha(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    items = _resolve_items(ctx)
    if len(items) < 3:
        summary.append("Cronbach's α 跳过：需要 ≥3 个数值题项（连续/计数列），当前不足。")
        return
    try:
        sub = df[items].apply(pd.to_numeric, errors="coerce").dropna()
        if sub.shape[0] < 3:
            summary.append("Cronbach's α 跳过：成列删除缺失后样本不足 3 行，无法估计。")
            return
        X = sub.to_numpy(dtype=float)
        n, k = X.shape  # n = persons, k = items

        # --- Cronbach (1951): alpha = k/(k-1) * (1 - sum(var_item)/var_total) ---
        # ddof=1 (sample variance); ratio is scale-free so ddof choice cancels,
        # but we keep ddof=1 throughout for consistency with the item-total stats.
        item_var = X.var(axis=0, ddof=1)
        total_score = X.sum(axis=1)
        total_var = total_score.var(ddof=1)
        if total_var <= 0:
            summary.append("Cronbach's α 失败：总分方差为 0（题项全无变异）。")
            return
        alpha = (k / (k - 1.0)) * (1.0 - item_var.sum() / total_var)

        # --- Feldt (1965) F-based 95% CI for alpha ---
        # 1 - alpha ~ F-ratio with df1 = n-1, df2 = (n-1)(k-1); the CI is
        #   [1 - (1-a)*F_upper , 1 - (1-a)*F_lower] using the F quantiles.
        from scipy import stats

        df1 = n - 1
        df2 = (n - 1) * (k - 1)
        f_l = stats.f.ppf(0.975, df1, df2)
        f_u = stats.f.ppf(0.025, df1, df2)
        ci_low = 1.0 - (1.0 - alpha) * f_l
        ci_high = 1.0 - (1.0 - alpha) * f_u

        # --- corrected item-total correlation + alpha-if-item-dropped ---
        rows = []
        for j, name in enumerate(items):
            rest = np.delete(X, j, axis=1)
            rest_sum = rest.sum(axis=1)
            # corrected (item vs sum-of-rest); guard zero-variance item
            if X[:, j].std(ddof=1) == 0 or rest_sum.std(ddof=1) == 0:
                itc = float("nan")
            else:
                itc = float(np.corrcoef(X[:, j], rest_sum)[0, 1])
            # alpha with this item dropped
            if k - 1 >= 2:
                rv = rest.var(axis=0, ddof=1)
                tv = rest_sum.var(ddof=1)
                a_drop = (
                    ((k - 1) / (k - 2.0)) * (1.0 - rv.sum() / tv)
                    if tv > 0
                    else float("nan")
                )
            else:
                a_drop = float("nan")
            rows.append(
                {
                    "item": name,
                    "item_total_corr_corrected": round(itc, 4) if itc == itc else None,
                    "alpha_if_dropped": round(a_drop, 4) if a_drop == a_drop else None,
                    "raises_alpha_if_dropped": bool(a_drop > alpha) if a_drop == a_drop else False,
                }
            )
        item_df = pd.DataFrame(rows)
        item_df.to_csv(d / "cronbach_item_stats.csv", index=False, encoding="utf-8")
        files.append("cronbach_item_stats.csv")

        # flags
        problem_drop = [r["item"] for r in rows if r["raises_alpha_if_dropped"]]
        neg_itc = [
            r["item"]
            for r in rows
            if r["item_total_corr_corrected"] is not None
            and r["item_total_corr_corrected"] < -0.1
        ]

        # --- plot: alpha-if-dropped vs overall alpha ---
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7, max(3, len(items) * 0.4)))
            vals = [r["alpha_if_dropped"] if r["alpha_if_dropped"] is not None else np.nan for r in rows]
            colors = ["#C44E52" if (v == v and v > alpha) else "#4C72B0" for v in vals]
            ax.barh(items[::-1], vals[::-1], color=colors[::-1])
            ax.axvline(alpha, color="black", ls="--", lw=1.0, label=f"overall alpha={alpha:.3f}")
            ax.set_xlabel("Cronbach's alpha if item dropped")
            ax.set_title("Alpha-if-item-dropped (red = removal RAISES alpha)")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "cronbach_alpha_if_dropped.png", dpi=150)
            plt.close(fig)
            files.append("cronbach_alpha_if_dropped.png")
        except Exception:
            pass

        estimates["cronbach_alpha"] = round(float(alpha), 4)
        estimates["alpha_ci_low"] = round(float(ci_low), 4)
        estimates["alpha_ci_high"] = round(float(ci_high), 4)
        estimates["n_items"] = float(k)
        estimates["n_persons"] = float(n)

        grade = (
            "优秀(≥0.9)" if alpha >= 0.9 else
            "良好(≥0.8)" if alpha >= 0.8 else
            "可接受(≥0.7)" if alpha >= 0.7 else
            "偏低(<0.7,⚠)"
        )
        msg = (
            f"{entry.method} 完成：{k} 个题项 × {n} 个被试，α={alpha:.3f}"
            f"（{grade}），Feldt 95% CI=[{ci_low:.3f}, {ci_high:.3f}]。"
        )
        if problem_drop:
            msg += f" ⚠ 删除以下题项会提高 α（建议复核/删除）：{', '.join(problem_drop)}。"
        if neg_itc:
            msg += f" ⚠ 题项 {', '.join(neg_itc)} 校正题总相关为负，可能是反向计分未重编码。"
        msg += (
            " ⚠ α 假设题项 tau 等价(等载荷)且单维——违反时低估真实信度(改看 McDonald ω)；"
            "需 ≥3 个题项；反向题须先重编码。"
        )
        summary.append(msg)
        code += [
            "import numpy as np  # Cronbach (1951) alpha",
            f"X = df[{items!r}].apply(pd.to_numeric, errors='coerce').dropna().to_numpy(float)",
            "n, k = X.shape",
            "alpha = k/(k-1) * (1 - X.var(0, ddof=1).sum() / X.sum(1).var(ddof=1))",
            "# Feldt(1965) CI: 1-alpha ~ F(n-1,(n-1)(k-1))",
        ]
    except Exception as err:
        summary.append(f"Cronbach's α 失败：{err}")


# ===========================================================================
# 2. McDonald's omega_total
# ===========================================================================

@register("mcdonald_omega")
def _branch_mcdonald_omega(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    items = _resolve_items(ctx)
    if len(items) < 3:
        summary.append("McDonald's ω 跳过：需要 ≥3 个数值题项（连续/计数列），当前不足。")
        return
    try:
        sub = df[items].apply(pd.to_numeric, errors="coerce").dropna()
        if sub.shape[0] < max(10, len(items) + 1):
            summary.append("McDonald's ω 跳过：样本太少，单因子解不稳定（建议 ≥10 行且 >题项数）。")
            return

        from factor_analyzer import FactorAnalyzer

        # Single common factor (congeneric model), minres extraction. Items are
        # auto-standardised by FactorAnalyzer (correlation-matrix solution), so
        # loadings/communalities are on the standardised metric, and residual
        # variance of each standardised item is 1 - h^2.
        fa = FactorAnalyzer(n_factors=1, rotation=None, method="minres")
        fa.fit(sub.to_numpy(dtype=float))
        load = fa.loadings_[:, 0].astype(float)  # lambda_i
        comm = fa.get_communalities().astype(float)  # h^2_i

        # Sign-align: a 1-factor solution is sign-indeterminate; flip so the
        # factor points with the majority of items (omega itself is invariant
        # because it uses (sum lambda)^2, but reported loadings read cleaner).
        if np.sum(load) < 0:
            load = -load
        resid = 1.0 - comm  # residual var on standardised items
        resid = np.clip(resid, 0.0, None)

        sum_load = float(load.sum())
        num = sum_load**2
        # McDonald (1999) omega_total = (sum lambda)^2 / ((sum lambda)^2 + sum(1-h^2))
        denom = num + float(resid.sum())
        if denom <= 0:
            summary.append("McDonald's ω 失败：方差分解分母非正（退化解）。")
            return
        omega = num / denom

        # contrast with Cronbach alpha on the same standardised items
        Z = (sub - sub.mean()) / sub.std(ddof=1)
        Z = Z.to_numpy(dtype=float)
        n, k = Z.shape
        iv = Z.var(axis=0, ddof=1)
        tv = Z.sum(axis=1).var(ddof=1)
        alpha = (k / (k - 1.0)) * (1.0 - iv.sum() / tv) if tv > 0 else float("nan")

        load_df = pd.DataFrame(
            {
                "item": items,
                "loading": np.round(load, 4),
                "communality_h2": np.round(comm, 4),
                "residual_var": np.round(resid, 4),
            }
        )
        load_df.to_csv(d / "omega_loadings.csv", index=False, encoding="utf-8")
        files.append("omega_loadings.csv")

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7, max(3, len(items) * 0.4)))
            ax.barh(items[::-1], load[::-1], color="#55A868")
            ax.axvline(0.0, color="black", lw=0.8)
            ax.set_xlabel("1-factor loading (lambda)")
            ax.set_title(f"McDonald's omega = {omega:.3f}  (alpha = {alpha:.3f})")
            fig.tight_layout()
            fig.savefig(d / "omega_loadings.png", dpi=150)
            plt.close(fig)
            files.append("omega_loadings.png")
        except Exception:
            pass

        estimates["mcdonald_omega"] = round(float(omega), 4)
        # standardized (correlation-matrix) alpha — distinct from the cronbach_alpha branch's
        # RAW-variance alpha; keep a separate key so the two are not silently conflated.
        estimates["cronbach_alpha_standardized"] = round(float(alpha), 4) if alpha == alpha else -1.0
        estimates["sum_loadings"] = round(sum_load, 4)
        estimates["n_items"] = float(k)

        neg = [items[i] for i in range(k) if load[i] < 0]
        grade = (
            "优秀(≥0.9)" if omega >= 0.9 else
            "良好(≥0.8)" if omega >= 0.8 else
            "可接受(≥0.7)" if omega >= 0.7 else
            "偏低(<0.7,⚠)"
        )
        msg = (
            f"{entry.method} 完成：{k} 个题项的单因子(同源 congeneric)解，"
            f"ω_total={omega:.3f}（{grade}）；同题项标准化 Cronbach α={alpha:.3f}（ω 通常 ≥ α；"
            f"此 α 为相关矩阵口径，与 cronbach_alpha 分析的原始方差 α 略有差异）。"
        )
        if neg:
            msg += f" ⚠ 题项 {', '.join(neg)} 载荷为负——可能反向计分未重编码或非单维。"
        msg += (
            " ⚠ ω 放松了 α 的 tau 等价假设(允许载荷不等的同源模型)，一般更优先；"
            "仍要求单维(单公因子)；载荷符号不定，已对齐为多数正向。"
        )
        summary.append(msg)
        code += [
            "from factor_analyzer import FactorAnalyzer  # McDonald (1999) omega_total",
            f"fa = FactorAnalyzer(n_factors=1, rotation=None, method='minres').fit(df[{items!r}].dropna())",
            "lam = fa.loadings_[:,0]; h2 = fa.get_communalities()",
            "omega = lam.sum()**2 / (lam.sum()**2 + (1 - h2).sum())",
        ]
    except Exception as err:
        summary.append(f"McDonald's ω 失败：{err}")


# ===========================================================================
# 3. Intraclass correlation (Shrout & Fleiss 1979)
# ===========================================================================

@register("icc")
def _branch_icc(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import pandas as pd

    # Wide layout: rows = subjects/targets, columns = raters/measurements.
    items = _resolve_items(ctx)
    if len(items) < 2:
        summary.append("ICC 跳过：需要 ≥2 个评分者/测量列（宽表 被试×评分者）。")
        return
    try:
        sub = df[items].apply(pd.to_numeric, errors="coerce").dropna()
        if sub.shape[0] < 2:
            summary.append("ICC 跳过：成列删除缺失后被试不足 2 行，无法做 ANOVA 分解。")
            return
        Y = sub.to_numpy(dtype=float)
        n, k = Y.shape  # n = subjects (rows), k = raters (cols)

        # --- Two-way ANOVA mean-squares decomposition (Shrout & Fleiss 1979) ---
        grand = Y.mean()
        row_means = Y.mean(axis=1)  # per subject
        col_means = Y.mean(axis=0)  # per rater
        # Sum of squares
        SST = ((Y - grand) ** 2).sum()
        SSR = k * ((row_means - grand) ** 2).sum()  # between subjects
        SSC = n * ((col_means - grand) ** 2).sum()  # between raters
        SSE = SST - SSR - SSC  # residual (subject x rater interaction)
        SSW = SST - SSR  # within subjects (one-way error = SSC + SSE)

        df_r = n - 1
        df_c = k - 1
        df_e = (n - 1) * (k - 1)
        df_w = n * (k - 1)

        MSR = SSR / df_r if df_r > 0 else float("nan")  # between-subjects
        MSC = SSC / df_c if df_c > 0 else float("nan")  # between-raters
        MSE = SSE / df_e if df_e > 0 else float("nan")  # residual
        MSW = SSW / df_w if df_w > 0 else float("nan")  # within (one-way)

        # --- ICC forms (Shrout & Fleiss 1979, Table) ---
        # ICC(1,1): one-way random,   (MSR - MSW) / (MSR + (k-1) MSW)
        icc1 = (MSR - MSW) / (MSR + (k - 1) * MSW) if (MSR + (k - 1) * MSW) != 0 else float("nan")
        icc1k = (MSR - MSW) / MSR if MSR != 0 else float("nan")
        # ICC(2,1): two-way random, absolute agreement
        denom21 = MSR + (k - 1) * MSE + (k / n) * (MSC - MSE)
        icc2 = (MSR - MSE) / denom21 if denom21 != 0 else float("nan")
        denom2k = MSR + (MSC - MSE) / n
        icc2k = (MSR - MSE) / denom2k if denom2k != 0 else float("nan")
        # ICC(3,1): two-way mixed, consistency
        icc3 = (MSR - MSE) / (MSR + (k - 1) * MSE) if (MSR + (k - 1) * MSE) != 0 else float("nan")
        icc3k = (MSR - MSE) / MSR if MSR != 0 else float("nan")

        # F-test that ICC > 0 (between-subjects variance is significant).
        # For ICC(1,1): F = MSR/MSW, df = (n-1, n(k-1)).
        # For ICC(2,1)/(3,1): F = MSR/MSE, df = (n-1, (n-1)(k-1)).
        from scipy import stats

        f1 = MSR / MSW if MSW not in (0, float("nan")) and MSW > 0 else float("nan")
        p1 = float(stats.f.sf(f1, df_r, df_w)) if f1 == f1 else float("nan")
        f3 = MSR / MSE if MSE not in (0, float("nan")) and MSE > 0 else float("nan")
        p3 = float(stats.f.sf(f3, df_r, df_e)) if f3 == f3 else float("nan")

        def _r(x: float) -> float:
            return round(float(x), 4) if x == x else float("nan")

        anova_df = pd.DataFrame(
            [
                {"source": "between-subjects (rows)", "SS": _r(SSR), "df": df_r, "MS": _r(MSR)},
                {"source": "between-raters (cols)", "SS": _r(SSC), "df": df_c, "MS": _r(MSC)},
                {"source": "residual (interaction)", "SS": _r(SSE), "df": df_e, "MS": _r(MSE)},
                {"source": "within-subjects (one-way error)", "SS": _r(SSW), "df": df_w, "MS": _r(MSW)},
                {"source": "total", "SS": _r(SST), "df": n * k - 1, "MS": None},
            ]
        )
        anova_df.to_csv(d / "icc_anova.csv", index=False, encoding="utf-8")
        files.append("icc_anova.csv")

        icc_df = pd.DataFrame(
            [
                {"form": "ICC(1,1)", "type": "one-way random, single", "ICC": _r(icc1)},
                {"form": "ICC(1,k)", "type": "one-way random, average", "ICC": _r(icc1k)},
                {"form": "ICC(2,1)", "type": "two-way random, single, absolute agreement", "ICC": _r(icc2)},
                {"form": "ICC(2,k)", "type": "two-way random, average, absolute agreement", "ICC": _r(icc2k)},
                {"form": "ICC(3,1)", "type": "two-way mixed, single, consistency", "ICC": _r(icc3)},
                {"form": "ICC(3,k)", "type": "two-way mixed, average, consistency", "ICC": _r(icc3k)},
            ]
        )
        icc_df.to_csv(d / "icc_estimates.csv", index=False, encoding="utf-8")
        files.append("icc_estimates.csv")

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7, 4))
            forms = list(icc_df["form"])
            vals = [v if v == v else 0.0 for v in icc_df["ICC"]]
            ax.bar(forms, vals, color="#8172B3")
            ax.set_ylabel("ICC")
            ax.set_ylim(min(0.0, min(vals) - 0.05), 1.0)
            ax.axhline(0.0, color="black", lw=0.8)
            ax.set_title("Intraclass correlation forms (Shrout & Fleiss 1979)")
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            fig.tight_layout()
            fig.savefig(d / "icc_forms.png", dpi=150)
            plt.close(fig)
            files.append("icc_forms.png")
        except Exception:
            pass

        estimates["icc_1_1"] = _r(icc1)
        estimates["icc_2_1"] = _r(icc2)
        estimates["icc_3_1"] = _r(icc3)
        estimates["icc_1_k"] = _r(icc1k)
        estimates["icc_2_k"] = _r(icc2k)
        estimates["icc_3_k"] = _r(icc3k)
        estimates["f_consistency"] = _r(f3)
        estimates["p_consistency"] = _r(p3)
        estimates["n_subjects"] = float(n)
        estimates["n_raters"] = float(k)

        def _grade(v: float) -> str:
            if v != v:
                return "不可用"
            return (
                "优秀(≥0.9)" if v >= 0.9 else
                "良好(0.75–0.9)" if v >= 0.75 else
                "中等(0.5–0.75)" if v >= 0.5 else
                "差(<0.5,⚠)"
            )

        summary.append(
            f"{entry.method} 完成：{n} 个被试 × {k} 个评分者。"
            f"ICC(1,1)={icc1:.3f}、ICC(2,1)={icc2:.3f}（绝对一致）、ICC(3,1)={icc3:.3f}（一致性，{_grade(icc3)}）；"
            f"平均测量形式 ICC(1,k)={icc1k:.3f}/(2,k)={icc2k:.3f}/(3,k)={icc3k:.3f}。"
            f"一致性 F={f3:.2f}, p={p3:.2g}。"
            " ⚠ 选用哪种 ICC 取决于设计：评分者随机 vs 固定（→2 型 vs 3 型）、单个 vs 平均评分（→ (·,1) vs (·,k)）、"
            "绝对一致 vs 一致性（→2 型 vs 3 型）；需 被试×评分者 宽表布局。"
        )
        code += [
            "import numpy as np  # ICC — Shrout & Fleiss (1979) two-way ANOVA decomposition",
            f"Y = df[{items!r}].apply(pd.to_numeric, errors='coerce').dropna().to_numpy(float)",
            "n, k = Y.shape; g = Y.mean()",
            "SSR = k*((Y.mean(1)-g)**2).sum(); SSC = n*((Y.mean(0)-g)**2).sum()",
            "SST = ((Y-g)**2).sum(); SSE = SST-SSR-SSC; SSW = SST-SSR",
            "MSR=SSR/(n-1); MSC=SSC/(k-1); MSE=SSE/((n-1)*(k-1)); MSW=SSW/(n*(k-1))",
            "icc_3_1 = (MSR-MSE)/(MSR+(k-1)*MSE)  # consistency, single",
            "icc_2_1 = (MSR-MSE)/(MSR+(k-1)*MSE+(k/n)*(MSC-MSE))  # absolute agreement",
            "icc_1_1 = (MSR-MSW)/(MSR+(k-1)*MSW)  # one-way",
        ]
    except Exception as err:
        summary.append(f"ICC 失败：{err}")
