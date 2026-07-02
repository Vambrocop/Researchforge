"""Branch handlers for the distribution family.

Three probability-distribution methods that work on ONE numeric column:

  * distribution_fit  — MLE-fit several candidate continuous distributions, rank by AIC/BIC/KS
  * goodness_of_fit   — formal GoF tests (KS / Anderson-Darling / Cramer-von Mises / Shapiro)
  * qq_analysis       — Q-Q + P-P diagnostic plots with the PPCC (Hazen positions) and reference line

Each handler resolves the numeric column (cfg.get("column") else first continuous col),
degrades honestly (no scipy / too few rows / non-numeric / all-constant -> append a Chinese
⚠ message and return), writes CSV + PNG (matplotlib Agg, ENGLISH plot labels), fills float
`estimates`, appends a Chinese `summary` with ⚠ disclosures, and mutates ctx (never rebinds).
See executor/_branch_api.py and CLAUDE.md.

Pure Python: numpy / scipy.stats / pandas (no R).
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register

# Candidate distributions for distribution_fit. Those needing strictly positive
# support are skipped when the data contains values <= 0.
_POSITIVE_ONLY = {"lognorm", "gamma", "weibull_min", "expon"}
_CANDIDATES = ["norm", "lognorm", "gamma", "weibull_min", "expon"]

# Distributions scipy.stats.anderson supports, mapped to its `dist` argument.
_AD_SUPPORTED = {"norm": "norm", "expon": "expon", "logistic": "logistic",
                 "gumbel_r": "gumbel_r", "gumbel": "gumbel", "weibull_min": "weibull_min"}


def _resolve_column(ctx: Ctx):
    """Pick the numeric column to analyse (cfg['column'] else first continuous).

    Returns (column_name, values_ndarray, problem_msg). When problem_msg is not None
    the caller should append it to summary and return (honest degrade)."""
    import importlib.util

    if importlib.util.find_spec("scipy") is None:
        return None, None, "分布分析需要 scipy 包（未检测到）。安装：pip install scipy。"

    import numpy as np

    df, fp, cfg = ctx.df, ctx.fp, ctx.cfg
    col = cfg.get("column")
    if col is not None:
        if col not in df.columns:
            return None, None, f"分布分析失败：指定列 {col!r} 不在数据中。"
    else:
        # first continuous column; fall back to any numeric (count/id) so float-like
        # integer columns the profiler tagged otherwise are still usable
        cont = [c.name for c in fp.columns if c.kind == "continuous"]
        numeric = [c.name for c in fp.columns if c.kind in {"continuous", "count", "id"}]
        pick = cont or numeric
        if not pick:
            return None, None, "分布分析失败：未找到可用的数值列。用 config={\"column\":\"<列>\"} 指定。"
        col = pick[0]

    s = df[col]
    if not np.issubdtype(s.dropna().to_numpy().dtype, np.number):
        try:
            s = s.astype(float)
        except Exception:
            return None, None, f"分布分析失败：列 {col!r} 不是数值列。"
    x = s.dropna().to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return None, None, f"分布分析跳过：列 {col!r} 有效样本不足（n={x.size}<8）。"
    if np.nanstd(x) == 0:
        return None, None, f"分布分析跳过：列 {col!r} 为常数列（无方差）。"
    return col, x, None


# ─────────────────────────────────────────────────────────────────────────────
# 1. distribution_fit — MLE-fit candidates, rank by AIC/BIC/KS
# ─────────────────────────────────────────────────────────────────────────────
@register("distribution_fit")
def _branch_distribution_fit(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    col, x, problem = _resolve_column(ctx)
    if problem is not None:
        summary.append(problem)
        return
    try:
        import numpy as np
        import pandas as pd
        from scipy import stats

        n = x.size
        has_nonpos = bool(np.any(x <= 0))
        skipped = []
        rows = []
        for name in _CANDIDATES:
            if name in _POSITIVE_ONLY and has_nonpos:
                skipped.append(name)
                continue
            dist = getattr(stats, name)
            try:
                params = dist.fit(x)
                logL = float(np.sum(dist.logpdf(x, *params)))
                if not np.isfinite(logL):
                    skipped.append(name)
                    continue
                k = len(params)
                aic = 2.0 * k - 2.0 * logL
                bic = k * np.log(n) - 2.0 * logL
                ks = float(stats.kstest(x, dist.cdf, args=params).statistic)
                rows.append({
                    "dist": name,
                    "n_params": k,
                    "params": ", ".join(f"{p:.5g}" for p in params),
                    "logL": round(logL, 4),
                    "AIC": round(float(aic), 4),
                    "BIC": round(float(bic), 4),
                    "KS": round(ks, 5),
                    "_params_raw": params,
                })
            except Exception:
                skipped.append(name)
                continue

        if not rows:
            summary.append(f"分布拟合跳过：列 {col!r} 没有任何候选分布拟合成功。")
            return

        res_df = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
        best = res_df.iloc[0]
        best_name = str(best["dist"])
        best_params = best["_params_raw"]
        delta_aic_second = (
            float(res_df.iloc[1]["AIC"] - best["AIC"]) if len(res_df) > 1 else float("nan")
        )

        out_df = res_df.drop(columns=["_params_raw"])
        out_df.to_csv(d / "distribution_fit.csv", index=False, encoding="utf-8")
        files.append("distribution_fit.csv")

        # histogram (density) + fitted PDFs overlaid, best highlighted
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7.5, 5))
            ax.hist(x, bins="auto", density=True, alpha=0.4, color="#999999",
                    label="data (density)")
            grid = np.linspace(float(np.min(x)), float(np.max(x)), 400)
            for _, r in res_df.iterrows():
                dist = getattr(stats, str(r["dist"]))
                pdf = dist.pdf(grid, *r["_params_raw"])
                is_best = str(r["dist"]) == best_name
                ax.plot(grid, pdf, lw=2.4 if is_best else 1.2,
                        alpha=1.0 if is_best else 0.6,
                        label=f"{r['dist']}" + (" (best AIC)" if is_best else ""))
            ax.set_xlabel(f"{col}")
            ax.set_ylabel("density")
            ax.set_title(f"Distribution fit: {col} (best = {best_name} by AIC)")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "distribution_fit.png", dpi=150)
            plt.close(fig)
            files.append("distribution_fit.png")
        except Exception:
            pass

        estimates["best_aic"] = round(float(best["AIC"]), 4)
        estimates["best_bic"] = round(float(best["BIC"]), 4)
        estimates["best_ks"] = round(float(best["KS"]), 5)
        estimates["delta_aic_second"] = (
            round(delta_aic_second, 4) if delta_aic_second == delta_aic_second else float("nan")
        )
        estimates["n"] = float(n)

        skip_note = (f"（因数据含 ≤0 值，跳过正支撑分布：{', '.join(skipped)}）"
                     if (skipped and has_nonpos) else
                     (f"（跳过未能拟合的：{', '.join(skipped)}）" if skipped else ""))
        gap_txt = (f"{round(delta_aic_second, 2)}" if delta_aic_second == delta_aic_second else "NA")
        strong = (delta_aic_second == delta_aic_second and delta_aic_second >= 10)
        rank_note = "（与次优 ΔAIC≥10，胜出明显）" if strong else "（与次优差距不大，结论不稳健）"

        fitted_names = set(res_df["dist"].tolist())
        shifted_fitted = [nm for nm in ("lognorm", "gamma", "weibull_min") if nm in fitted_names]
        loc_note = (
            f"⚠ {', '.join(shifted_fitted)} 用 scipy 默认的**自由 loc**（三参数平移形式）拟合，"
            "而非经典两参数族——正支撑族的自由-loc MLE 可能把 loc 拖向 min(x)，须留意。"
            if shifted_fitted else ""
        )

        (d / "distribution_fit_summary.txt").write_text(
            f"分布拟合（MLE，scipy.stats）：列 {col}，n={n}\n"
            f"候选分布：{', '.join(_CANDIDATES)} {skip_note}\n"
            f"最优（按 AIC 越小越好）：{best_name}，AIC={round(float(best['AIC']), 3)}，"
            f"BIC={round(float(best['BIC']), 3)}，KS={round(float(best['KS']), 4)}\n"
            f"与次优的 AIC 差 ΔAIC={gap_txt} {rank_note}\n"
            "注：AIC/BIC 比较的是候选分布之间的**相对**拟合优劣，并不证明最优分布在绝对意义上"
            "「足够好」——请配合 goodness_of_fit 做正式检验确认；正支撑分布在数据含 ≤0 时被跳过；"
            "列可用 config={\"column\":\"...\"} 指定。\n"
            + (loc_note + "\n" if loc_note else "")
            + "\n全部候选（按 AIC 排序）：\n" + out_df.to_string(index=False),
            encoding="utf-8",
        )
        files.append("distribution_fit_summary.txt")

        summary.append(
            f"{entry.method} 完成（MLE）：列 {col}，n={n}；候选 {len(res_df)} 个分布，"
            f"最优={best_name}（AIC={round(float(best['AIC']), 2)}，KS={round(float(best['KS']), 4)}）；"
            f"与次优 ΔAIC={gap_txt}{rank_note}。"
            + (f" {skip_note}" if skip_note else "")
            + "⚠ AIC/BIC 只比较候选间的相对拟合，不证明绝对充分性——请配合 goodness_of_fit 确认；"
            "数据含 ≤0 时正支撑分布被跳过；列可经 config 指定。"
            + (f" {loc_note}" if loc_note else "")
        )
        code += [
            "import numpy as np",
            "from scipy import stats",
            f"x = df[{col!r}].dropna().to_numpy(float)",
            "for name in ['norm','lognorm','gamma','weibull_min','expon']:",
            "    d = getattr(stats, name); p = d.fit(x)",
            "    logL = np.sum(d.logpdf(x, *p)); k = len(p)",
            "    AIC = 2*k - 2*logL; BIC = k*np.log(len(x)) - 2*logL",
            "    KS = stats.kstest(x, d.cdf, args=p).statistic  # rank by AIC (lower=better)",
        ]
    except Exception as err:
        summary.append(f"分布拟合失败：{err}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. goodness_of_fit — KS / Anderson-Darling / Cramer-von Mises / Shapiro
# ─────────────────────────────────────────────────────────────────────────────
@register("goodness_of_fit")
def _branch_goodness_of_fit(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    col, x, problem = _resolve_column(ctx)
    if problem is not None:
        summary.append(problem)
        return
    try:
        import numpy as np
        import pandas as pd
        from scipy import stats

        n = x.size
        dist_name = str(cfg.get("dist", "norm")).lower()
        if not hasattr(stats, dist_name):
            summary.append(f"拟合优度检验失败：未知分布 {dist_name!r}（scipy.stats 无此分布）。")
            return
        dist = getattr(stats, dist_name)

        # positive-support distributions cannot be tested on non-positive data
        if dist_name in _POSITIVE_ONLY and bool(np.any(x <= 0)):
            summary.append(
                f"拟合优度检验跳过：分布 {dist_name} 需正支撑，但列 {col!r} 含 ≤0 值。"
                "请改用 config={\"dist\":\"norm\"} 或先变换数据。"
            )
            return

        try:
            params = dist.fit(x)
        except Exception as err:
            summary.append(f"拟合优度检验失败：分布 {dist_name} 参数估计失败（{err}）。")
            return

        rows = []
        ks_stat = ks_p = float("nan")
        ad_stat = float("nan")
        cvm_stat = cvm_p = float("nan")
        shapiro_p = float("nan")

        # KS test with FITTED params (Lilliefors caveat: estimated params -> conservative)
        try:
            kr = stats.kstest(x, dist.cdf, args=params)
            ks_stat, ks_p = float(kr.statistic), float(kr.pvalue)
            rows.append({"test": "Kolmogorov-Smirnov (fitted params)",
                         "statistic": round(ks_stat, 5),
                         "p_value": round(ks_p, 5),
                         "critical_5pct": float("nan"),
                         "reject_at_0.05": bool(ks_p < 0.05)})
        except Exception:
            pass

        # Anderson-Darling (only defined for specific dists)
        ad_crit5 = float("nan")
        ad_reject = None
        if dist_name in _AD_SUPPORTED:
            try:
                import warnings

                with warnings.catch_warnings():
                    # SciPy 1.17 added a `method` param to anderson and warns about a
                    # future default change; we read critical_values defensively below.
                    warnings.simplefilter("ignore", FutureWarning)
                    ar = stats.anderson(x, dist=_AD_SUPPORTED[dist_name])
                ad_stat = float(ar.statistic)
                # critical value at 5% significance. scipy reports significance_level in
                # PERCENT ([15,10,5,2.5,1]) for norm/expon/logistic/gumbel_r, but for
                # weibull_min it instead reports CONFIDENCE levels (1-alpha) in
                # [0.5,0.75,...,0.995] -> the 5% critical value is the 0.95 entry. Both
                # tables have max <= 1.0, so disambiguate by whether it looks like a
                # confidence-level table (min >= 0.5) vs a hypothetical small-alpha
                # fraction table ([.01..0.15], which would need 0.05).
                sigs = list(ar.significance_level)
                if sigs and max(sigs) <= 1.0:
                    target = 0.95 if min(sigs) >= 0.5 else 0.05
                else:
                    target = 5.0
                if target in sigs:
                    ad_crit5 = float(ar.critical_values[sigs.index(target)])
                else:
                    # nearest to the 5% level
                    j = int(np.argmin([abs(s - target) for s in sigs]))
                    ad_crit5 = float(ar.critical_values[j])
                ad_reject = bool(ad_stat > ad_crit5)
                rows.append({"test": f"Anderson-Darling ({dist_name})",
                             "statistic": round(ad_stat, 5),
                             "p_value": float("nan"),
                             "critical_5pct": round(ad_crit5, 5),
                             "reject_at_0.05": ad_reject})
            except Exception:
                pass

        # Cramer-von Mises with fitted cdf
        try:
            cr = stats.cramervonmises(x, dist.cdf, args=params)
            cvm_stat, cvm_p = float(cr.statistic), float(cr.pvalue)
            rows.append({"test": "Cramer-von Mises (fitted cdf)",
                         "statistic": round(cvm_stat, 5),
                         "p_value": round(cvm_p, 5),
                         "critical_5pct": float("nan"),
                         "reject_at_0.05": bool(cvm_p < 0.05)})
        except Exception:
            pass

        # Shapiro-Wilk only when testing normality
        if dist_name == "norm":
            try:
                sr = stats.shapiro(x)
                shapiro_p = float(sr.pvalue)
                rows.append({"test": "Shapiro-Wilk (normality)",
                             "statistic": round(float(sr.statistic), 5),
                             "p_value": round(shapiro_p, 5),
                             "critical_5pct": float("nan"),
                             "reject_at_0.05": bool(shapiro_p < 0.05)})
            except Exception:
                pass

        if not rows:
            summary.append(f"拟合优度检验跳过：列 {col!r} 对分布 {dist_name} 没有可用的检验。")
            return

        gof_df = pd.DataFrame(rows)
        gof_df.to_csv(d / "goodness_of_fit.csv", index=False, encoding="utf-8")
        files.append("goodness_of_fit.csv")

        # ECDF vs fitted CDF
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            xs = np.sort(x)
            ecdf = np.arange(1, n + 1) / n
            grid = np.linspace(float(xs[0]), float(xs[-1]), 400)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.step(xs, ecdf, where="post", label="empirical CDF", color="#4C72B0")
            ax.plot(grid, dist.cdf(grid, *params), label=f"fitted {dist_name} CDF",
                    color="#C44E52", lw=2)
            ax.set_xlabel(f"{col}")
            ax.set_ylabel("cumulative probability")
            ax.set_title(f"Goodness of fit: {col} vs {dist_name} (ECDF vs CDF)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(d / "goodness_of_fit.png", dpi=150)
            plt.close(fig)
            files.append("goodness_of_fit.png")
        except Exception:
            pass

        estimates["ks_stat"] = round(ks_stat, 5) if ks_stat == ks_stat else float("nan")
        estimates["ks_p"] = round(ks_p, 5) if ks_p == ks_p else float("nan")
        estimates["ad_stat"] = round(ad_stat, 5) if ad_stat == ad_stat else float("nan")
        estimates["cvm_stat"] = round(cvm_stat, 5) if cvm_stat == cvm_stat else float("nan")
        estimates["cvm_p"] = round(cvm_p, 5) if cvm_p == cvm_p else float("nan")
        estimates["shapiro_p"] = round(shapiro_p, 5) if shapiro_p == shapiro_p else float("nan")

        any_reject = any(bool(r["reject_at_0.05"]) for r in rows)
        verdict = ("至少一项检验在 0.05 拒绝 → 数据**不**符合该分布"
                   if any_reject else
                   "各项检验均未在 0.05 拒绝 → 与该分布**不矛盾**（但不证明就是该分布）")

        (d / "goodness_of_fit_summary.txt").write_text(
            f"拟合优度检验（scipy.stats）：列 {col}（n={n}）对分布 {dist_name}\n"
            f"拟合参数：{', '.join(f'{p:.5g}' for p in params)}\n"
            f"结论：{verdict}\n"
            "注：KS 与 Cramér-von Mises 都用**估计参数**的 cdf，其 p 值偏保守（理论假定参数已知；"
            "估自同一数据时真 p 值不同——本检验更难拒绝，即更易「看起来符合」；严格应配模拟/"
            "Stephens 校正临界值）；Anderson-Darling 仅对特定分布有定义（norm/expon/logistic/"
            "gumbel/weibull）；不显著(p 大)**不**证明数据来自该分布，只是没有足够证据拒绝；"
            "列/分布可用 config={\"column\":\"...\",\"dist\":\"...\"} 指定。\n\n"
            "各检验结果：\n" + gof_df.to_string(index=False),
            encoding="utf-8",
        )
        files.append("goodness_of_fit_summary.txt")

        bits = []
        if ks_p == ks_p:
            bits.append(f"KS={round(ks_stat, 4)}(p={round(ks_p, 4)})")
        if ad_stat == ad_stat:
            bits.append(f"AD={round(ad_stat, 4)}(crit5%={round(ad_crit5, 4)})")
        if cvm_p == cvm_p:
            bits.append(f"CvM={round(cvm_stat, 4)}(p={round(cvm_p, 4)})")
        if shapiro_p == shapiro_p:
            bits.append(f"Shapiro p={round(shapiro_p, 4)}")

        summary.append(
            f"{entry.method} 完成：列 {col}（n={n}）对 {dist_name}；" + "，".join(bits)
            + f"。{verdict}。"
            + "⚠ KS 与 CvM 用估计参数时 p 偏保守（真 p 不同）；AD 仅对特定分布有定义；"
            "p 大不证明就是该分布（只是没足够证据拒绝）；列/分布可经 config 指定。"
        )
        code += [
            "import numpy as np",
            "from scipy import stats",
            f"x = df[{col!r}].dropna().to_numpy(float)",
            f"dist = stats.{dist_name}; params = dist.fit(x)",
            "ks = stats.kstest(x, dist.cdf, args=params)  # fitted params -> conservative (Lilliefors)",
            f"ad = stats.anderson(x, dist={_AD_SUPPORTED.get(dist_name, 'norm')!r})  # if supported",
            "cvm = stats.cramervonmises(x, dist.cdf, args=params)",
            "sw = stats.shapiro(x)  # only meaningful for normality",
        ]
    except Exception as err:
        summary.append(f"拟合优度检验失败：{err}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. qq_analysis — Q-Q + P-P diagnostics, PPCC (Hazen positions), reference line
# ─────────────────────────────────────────────────────────────────────────────
@register("qq_analysis")
def _branch_qq_analysis(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    col, x, problem = _resolve_column(ctx)
    if problem is not None:
        summary.append(problem)
        return
    try:
        import numpy as np
        import pandas as pd
        from scipy import stats

        n = x.size
        dist_name = str(cfg.get("dist", "norm")).lower()
        if not hasattr(stats, dist_name):
            summary.append(f"Q-Q 分析失败：未知分布 {dist_name!r}（scipy.stats 无此分布）。")
            return
        dist = getattr(stats, dist_name)

        if dist_name in _POSITIVE_ONLY and bool(np.any(x <= 0)):
            summary.append(
                f"Q-Q 分析跳过：分布 {dist_name} 需正支撑，但列 {col!r} 含 ≤0 值。"
                "请改用 config={\"dist\":\"norm\"} 或先变换数据。"
            )
            return

        try:
            params = dist.fit(x)
        except Exception as err:
            summary.append(f"Q-Q 分析失败：分布 {dist_name} 参数估计失败（{err}）。")
            return

        ordered = np.sort(x)
        # plotting positions (i - 0.5)/n  (Hazen)
        pp = (np.arange(1, n + 1) - 0.5) / n
        theo_q = dist.ppf(pp, *params)

        # guard against +/-inf at the tails (ppf at 0/1) -> drop those points for the line/PPCC
        finite = np.isfinite(theo_q) & np.isfinite(ordered)
        oq = ordered[finite]
        tq = theo_q[finite]

        # probability-plot correlation coefficient (PPCC) at Hazen positions
        # (NOT Filliben's median-rank PPCC, which uses different plotting positions)
        if oq.size >= 2 and np.std(tq) > 0 and np.std(oq) > 0:
            ppcc = float(np.corrcoef(oq, tq)[0, 1])
        else:
            ppcc = float("nan")

        # reference line: least squares ordered_data ~ slope * theoretical_quantile + intercept
        if oq.size >= 2 and np.std(tq) > 0:
            slope, intercept = np.polyfit(tq, oq, 1)
            slope, intercept = float(slope), float(intercept)
        else:
            slope = intercept = float("nan")

        # empirical vs theoretical CDF for the P-P plot
        emp_cdf = pp  # empirical cdf at each order stat (the plotting position)
        theo_cdf = dist.cdf(ordered, *params)

        ppdf = pd.DataFrame({
            "order": np.arange(1, n + 1),
            "data_sorted": np.round(ordered, 6),
            "theoretical_quantile": np.round(theo_q, 6),
            "empirical_cdf": np.round(emp_cdf, 6),
            "theoretical_cdf": np.round(theo_cdf, 6),
        })
        ppdf.to_csv(d / "qq_analysis.csv", index=False, encoding="utf-8")
        files.append("qq_analysis.csv")

        # 1x2 panel: Q-Q with reference line + P-P
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, (axq, axp) = plt.subplots(1, 2, figsize=(11, 5))
            axq.scatter(theo_q, ordered, s=16, color="#4C72B0", alpha=0.7, label="data")
            if slope == slope:
                lx = np.array([np.nanmin(tq), np.nanmax(tq)])
                axq.plot(lx, slope * lx + intercept, color="#C44E52", lw=2,
                         label=f"fit: y={slope:.3g}x+{intercept:.3g}")
            axq.set_xlabel(f"theoretical quantiles ({dist_name})")
            axq.set_ylabel("ordered data")
            axq.set_title(f"Q-Q plot: {col} vs {dist_name} (PPCC={ppcc:.4f})")
            axq.legend(fontsize=8)

            axp.scatter(theo_cdf, emp_cdf, s=16, color="#55A868", alpha=0.7, label="data")
            axp.plot([0, 1], [0, 1], color="#C44E52", lw=1.5, ls="--", label="y=x")
            axp.set_xlabel(f"theoretical CDF ({dist_name})")
            axp.set_ylabel("empirical CDF")
            axp.set_title(f"P-P plot: {col} vs {dist_name}")
            axp.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "qq_analysis.png", dpi=150)
            plt.close(fig)
            files.append("qq_analysis.png")
        except Exception:
            pass

        estimates["ppcc"] = round(ppcc, 5) if ppcc == ppcc else float("nan")
        estimates["qq_slope"] = round(slope, 5) if slope == slope else float("nan")
        estimates["qq_intercept"] = round(intercept, 5) if intercept == intercept else float("nan")
        estimates["n"] = float(n)

        quality = ("接近 1，直线吻合好（图示上分布契合较好）" if (ppcc == ppcc and ppcc >= 0.99)
                   else ("尚可但有偏离（看图尾部）" if (ppcc == ppcc and ppcc >= 0.95)
                         else "明显偏离直线（分布契合差）"))

        (d / "qq_analysis_summary.txt").write_text(
            f"Q-Q / P-P 诊断（scipy.stats）：列 {col}（n={n}）对分布 {dist_name}\n"
            f"拟合参数：{', '.join(f'{p:.5g}' for p in params)}\n"
            f"PPCC（Hazen 绘图位；有序数据 vs 理论分位的相关系数）={round(ppcc, 5) if ppcc == ppcc else 'NaN'}"
            f" —— {quality}\n"
            f"Q-Q 参考线（最小二乘 有序数据~理论分位）：斜率={round(slope, 5) if slope == slope else 'NaN'}，"
            f"截距={round(intercept, 5) if intercept == intercept else 'NaN'}\n"
            "注：PPCC 接近 1 表示契合好，但它是**可视化/启发式**诊断，不是正式检验"
            "（请配合 goodness_of_fit 做正式 GoF）；理论分位用绘图位 (i-0.5)/n（Hazen）；"
            "尾部 ppf 可能为 ±inf，这些点已从相关/参考线中剔除；列/分布可用 "
            "config={\"column\":\"...\",\"dist\":\"...\"} 指定。\n\n"
            "前 20 个序：\n" + ppdf.head(20).to_string(index=False),
            encoding="utf-8",
        )
        files.append("qq_analysis_summary.txt")

        summary.append(
            f"{entry.method} 完成：列 {col}（n={n}）对 {dist_name}；"
            f"PPCC（Hazen）={round(ppcc, 5) if ppcc == ppcc else 'NaN'}（{quality}）；"
            f"Q-Q 参考线 斜率={round(slope, 4) if slope == slope else 'NaN'}、"
            f"截距={round(intercept, 4) if intercept == intercept else 'NaN'}。"
            "⚠ PPCC 接近 1 表示契合好，但属可视化/启发式诊断（非正式检验），请配合 goodness_of_fit；"
            "列/分布可经 config 指定。"
        )
        code += [
            "import numpy as np",
            "from scipy import stats",
            f"x = np.sort(df[{col!r}].dropna().to_numpy(float)); n = len(x)",
            f"dist = stats.{dist_name}; params = dist.fit(x)",
            "pp = (np.arange(1, n+1) - 0.5)/n  # Hazen plotting positions",
            "theo_q = dist.ppf(pp, *params)",
            "ppcc = np.corrcoef(x, theo_q)[0,1]  # probability-plot corr coef (PPCC, Hazen positions)",
            "slope, intercept = np.polyfit(theo_q, x, 1)  # Q-Q reference line",
        ]
    except Exception as err:
        summary.append(f"Q-Q 分析失败：{err}")
