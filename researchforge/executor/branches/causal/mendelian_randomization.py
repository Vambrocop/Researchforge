"""Causal family branch handler: mendelian_randomization — two-sample summary-data MR.

Mendelian randomization treats genetic variants as instrumental variables for an
exposure, estimating its causal effect on an outcome from per-variant summary
statistics (one row per genetic instrument): the SNP→exposure effect (beta_exposure,
se_exposure) and the SNP→outcome effect (beta_outcome, se_outcome). Three standard
estimators with complementary robustness:

  * IVW (inverse-variance weighted) — the efficient pooled Wald-ratio estimate
    (a weighted regression of by on bx through the origin); a multiplicative
    random-effects SE corrects for between-instrument heterogeneity (Cochran's Q).
    Assumes NO directional pleiotropy.
  * MR-Egger — a weighted regression of by on bx WITH an intercept; the slope is the
    causal effect adjusted for directional pleiotropy, and the intercept is a formal
    test for it (intercept ≠ 0 ⇒ directional pleiotropy / InSIDE violation).
  * Weighted median — the inverse-variance-weighted median of the per-variant Wald
    ratios; consistent if ≥50% of the instrument weight comes from valid instruments
    (SE by parametric bootstrap).

Pure Python (numpy / scipy). Honest degrade when the four summary columns can't be
resolved or there are too few instruments.
"""
from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


def _pick(cols, *needles):
    """First column whose lower-cased name contains ALL needle groups (each group is a
    tuple of synonyms — any one matches)."""
    for c in cols:
        low = str(c).lower()
        if all(any(s in low for s in grp) for grp in needles):
            return c
    return None


@register("mendelian_randomization")
def _branch_mendelian_randomization(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    import numpy as np
    import pandas as pd
    from scipy import stats

    cols = list(df.columns)
    # resolve the 4 summary columns: config first, else name heuristics.
    bx_c = cfg.get("beta_exposure") if cfg.get("beta_exposure") in df.columns else _pick(cols, ("beta", "b_", "effect"), ("exp", "_x"))
    by_c = cfg.get("beta_outcome") if cfg.get("beta_outcome") in df.columns else _pick(cols, ("beta", "b_", "effect"), ("out", "_y"))
    sex_c = cfg.get("se_exposure") if cfg.get("se_exposure") in df.columns else _pick(cols, ("se", "stderr"), ("exp", "_x"))
    sey_c = cfg.get("se_outcome") if cfg.get("se_outcome") in df.columns else _pick(cols, ("se", "stderr"), ("out", "_y"))
    if not all([bx_c, by_c, sex_c, sey_c]) or len({bx_c, by_c, sex_c, sey_c}) < 4:
        summary.append(
            "孟德尔随机化跳过：需要按工具(每行=一个遗传变异)的四列汇总统计——SNP→暴露效应及其 SE、"
            "SNP→结局效应及其 SE。请用 config={\"beta_exposure\":..,\"se_exposure\":..,"
            "\"beta_outcome\":..,\"se_outcome\":..} 指定（或把列名含 beta/se + exposure/outcome）。"
        )
        return

    sub = df[[bx_c, by_c, sex_c, sey_c]].apply(lambda s: pd.to_numeric(s, errors="coerce")).dropna()
    sub = sub[(sub[sex_c] > 0) & (sub[sey_c] > 0) & (sub[bx_c] != 0)]
    K = len(sub)
    if K < 3:
        summary.append(f"孟德尔随机化跳过：有效工具变量过少（{K}<3；MR-Egger 至少需要 3 个）。")
        return

    bx = sub[bx_c].to_numpy(float)
    by = sub[by_c].to_numpy(float)
    sey = sub[sey_c].to_numpy(float)
    sex = sub[sex_c].to_numpy(float)

    try:
        # ── IVW: weighted regression of by on bx through the origin, w = 1/sey^2 ──
        w = 1.0 / sey ** 2
        ivw = float(np.sum(w * bx * by) / np.sum(w * bx ** 2))
        se_fixed = float(np.sqrt(1.0 / np.sum(w * bx ** 2)))
        # Cochran's Q heterogeneity + multiplicative random-effects SE (inflate if Q>df).
        Q = float(np.sum(w * (by - ivw * bx) ** 2))
        q_df = K - 1
        q_p = float(stats.chi2.sf(Q, q_df)) if q_df > 0 else float("nan")
        se_ivw = se_fixed * max(1.0, np.sqrt(Q / q_df)) if q_df > 0 else se_fixed
        ivw_p = float(2 * stats.norm.sf(abs(ivw / se_ivw))) if se_ivw > 0 else float("nan")

        # ── MR-Egger: weighted LS of by ~ 1 + bx, w = 1/sey^2 ──
        # MR-Egger REQUIRES each variant oriented to a POSITIVE exposure effect (flip the
        # sign of bx AND by where bx<0): unlike IVW / weighted-median (sign-invariant), the
        # Egger intercept and slope are NOT sign-invariant, and real GWAS betas are mixed-
        # sign (coded-allele dependent). Orient before fitting (Bowden et al. 2015).
        egint = egslope = egint_se = egslope_se = egint_p = egslope_p = float("nan")
        if K >= 3:
            flip = bx < 0
            bx_o = np.where(flip, -bx, bx)
            by_o = np.where(flip, -by, by)
            X = np.column_stack([np.ones(K), bx_o])
            WX = X * w[:, None]
            XtWX = X.T @ WX
            beta = np.linalg.solve(XtWX, WX.T @ by_o)
            resid = by_o - X @ beta
            dof = K - 2
            # multiplicative dispersion FLOORED at 1 (never deflate below fixed-effect WLS
            # SE; matches the IVW random-effects treatment + the MendelianRandomization pkg).
            phi = max(1.0, float((w * resid ** 2).sum() / dof))
            cov = phi * np.linalg.inv(XtWX)
            se = np.sqrt(np.diag(cov))
            egint, egslope = float(beta[0]), float(beta[1])
            egint_se, egslope_se = float(se[0]), float(se[1])
            egint_p = float(2 * stats.t.sf(abs(egint / egint_se), dof)) if egint_se > 0 else float("nan")
            egslope_p = float(2 * stats.t.sf(abs(egslope / egslope_se), dof)) if egslope_se > 0 else float("nan")

        # ── weighted median of Wald ratios (IV weights), SE by parametric bootstrap ──
        ratio = by / bx
        rw = (bx ** 2) / sey ** 2

        def _wmedian(v, ww):
            o = np.argsort(v)
            v2, w2 = v[o], ww[o]
            cw = (np.cumsum(w2) - 0.5 * w2) / np.sum(w2)
            return float(np.interp(0.5, cw, v2))

        wmed = _wmedian(ratio, rw)
        rng = np.random.default_rng(int(cfg.get("seed", 42)) if str(cfg.get("seed", 42)).isdigit() else 42)
        B = 1000
        boot = np.empty(B)
        for b in range(B):
            bxb = rng.normal(bx, sex)
            byb = rng.normal(by, sey)
            with np.errstate(divide="ignore", invalid="ignore"):
                rb = byb / bxb
            # keep the ORIGINAL inverse-variance weights fixed across bootstrap draws
            # (Bowden et al. 2016 weighted_median_bootstrap) — don't re-derive from bxb.
            ok = np.isfinite(rb)
            boot[b] = _wmedian(rb[ok], rw[ok]) if ok.sum() >= 3 else np.nan
        wmed_se = float(np.nanstd(boot, ddof=1))
        wmed_p = float(2 * stats.norm.sf(abs(wmed / wmed_se))) if wmed_se > 0 else float("nan")

        z = stats.norm.ppf(0.975)
        estimates["ivw_estimate"] = round(ivw, 5)
        estimates["ivw_se"] = round(se_ivw, 5)
        estimates["ivw_ci_low"] = round(ivw - z * se_ivw, 5)
        estimates["ivw_ci_high"] = round(ivw + z * se_ivw, 5)
        estimates["ivw_p"] = round(ivw_p, 5)
        estimates["egger_slope"] = round(egslope, 5)
        estimates["egger_slope_se"] = round(egslope_se, 5)
        estimates["egger_slope_p"] = round(egslope_p, 5)
        estimates["egger_intercept"] = round(egint, 5)
        estimates["egger_intercept_se"] = round(egint_se, 5)
        estimates["egger_intercept_p"] = round(egint_p, 5)          # pleiotropy test
        estimates["weighted_median"] = round(wmed, 5)
        estimates["weighted_median_se"] = round(wmed_se, 5)
        estimates["weighted_median_p"] = round(wmed_p, 5)
        estimates["cochran_q"] = round(Q, 4)
        estimates["cochran_q_p"] = round(q_p, 5)
        estimates["n_instruments"] = float(K)

        # per-instrument table + plot
        pd.DataFrame({
            "beta_exposure": np.round(bx, 6), "se_exposure": np.round(sex, 6),
            "beta_outcome": np.round(by, 6), "se_outcome": np.round(sey, 6),
            "wald_ratio": np.round(ratio, 6),
        }).to_csv(d / "mr_instruments.csv", index=False, encoding="utf-8")
        files.append("mr_instruments.csv")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5.2, 4))
            ax.errorbar(bx, by, xerr=sex, yerr=sey, fmt="o", ms=4, color="#4C72B0",
                        ecolor="#bbb", elinewidth=0.8, capsize=0, label="instruments")
            xs = np.array([0, bx.max() * 1.05])
            ax.plot(xs, ivw * xs, "-", color="#2f6f4f", lw=1.8, label=f"IVW slope={ivw:.3f}")
            ax.plot(xs, egint + egslope * xs, "--", color="#C44E52", lw=1.5,
                    label=f"MR-Egger slope={egslope:.3f}")
            ax.axhline(0, color="grey", lw=0.6)
            ax.set_xlabel("SNP -> exposure effect")
            ax.set_ylabel("SNP -> outcome effect")
            ax.set_title("Mendelian randomization")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "mr_scatter.png", dpi=150)
            plt.close(fig)
            files.append("mr_scatter.png")
        except Exception:
            pass

        het = (q_p == q_p and q_p < 0.05)
        pleio = (egint_p == egint_p and egint_p < 0.05)
        summary.append(
            f"{entry.method} 完成（两样本汇总数据 MR，{K} 个工具变量）："
            f"IVW 因果估计={ivw:.4g}（95% CI [{ivw - z * se_ivw:.3g}, {ivw + z * se_ivw:.3g}]，p={ivw_p:.3g}）；"
            f"MR-Egger 斜率={egslope:.4g}（截距={egint:.4g}，p={egint_p:.3g}"
            f"{'，⚠ 检出方向性多效性→IVW 可能有偏，优先看 Egger/加权中位数' if pleio else '，未检出方向性多效性'}）；"
            f"加权中位数={wmed:.4g}（p={wmed_p:.3g}）。"
            f"Cochran Q={Q:.3g}（p={q_p:.3g}{'，⚠ 工具间异质（已用随机效应放大 IVW SE）' if het else '，异质性不显著'}）。"
            "明细见 mr_instruments.csv、散点见 mr_scatter.png。"
            " ⚠ MR 三大假设：①工具与暴露强相关 ②工具与混杂独立 ③工具仅经暴露影响结局（排除约束/无多效性）——"
            "后者不可直接检验；IVW 假设无方向性多效性，MR-Egger 截距是其形式检验、斜率作校正（但效率较低）；"
            "加权中位数在≥50% 权重来自有效工具时一致。弱工具会使估计偏向观察性混杂；本法为两样本汇总数据 MR。"
            "（MR-Egger 已将各变异定向为正暴露效应；IVW 与 Egger 的 SE 均为随机效应/截断离散≥1 形式。）"
        )
        code += [
            "import numpy as np  # 两样本汇总数据 MR",
            "w = 1/sey**2",
            "ivw = np.sum(w*bx*by)/np.sum(w*bx**2)                 # IVW (过原点加权回归)",
            "# MR-Egger: 加权 LS  by ~ 1 + bx (截距=方向性多效性检验, 斜率=校正后因果效应)",
            "# 加权中位数: Wald 比 by/bx 的逆方差加权中位数 (参数自助 SE)",
        ]
    except Exception as err:
        summary.append(f"孟德尔随机化失败：{err}")
