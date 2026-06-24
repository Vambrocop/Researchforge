"""Branch handlers for the correlation_suite family — pure-Python bivariate /
partial / matrix correlation (scipy / numpy / pandas; NO R).

Three methods that quantify association between numeric columns:

  * pearson_correlation — bivariate r between two columns: Pearson r + p + Fisher-z
    95% CI, plus Spearman rho and Kendall tau (linear vs monotonic at a glance).
  * partial_correlation — partial r between x and y CONTROLLING for covariates z
    (residualisation), contrasted against the zero-order r (confounding/suppression).
  * correlation_matrix — full correlation matrix over all numeric columns with
    per-pair p-values, a BH-FDR significance flag, and the strongest pairs.

Engine conventions (CLAUDE.md): config overrides (x/y/covariates/method/columns);
honest degrade -> Chinese "<方法>跳过：<原因>" + RETURN (never crash/fabricate);
CSV + PNG best-effort try/except (matplotlib Agg, ENGLISH plot labels); float
`estimates`; Chinese `summary` with ⚠ disclosures; MUTATE ctx.* (never rebind).
See executor/_branch_api.py.

This family is DISTINCT from the existing `correlation` branch (statistics.py), which
only dumps a bare corr() matrix + heatmap with no inferential statistics.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


# ───────────────────────────────────────────────────────────────────────────── #
# Shared: numeric-column resolution (engine convention — continuous first, then  #
# count/binary as fallback so the family still works on Likert-ish data).        #
# ───────────────────────────────────────────────────────────────────────────── #
def _numeric_columns(ctx: Ctx) -> list[str]:
    fp, df = ctx.fp, ctx.df
    excl = {fp.unit_col, fp.time_col}
    cols = [
        c.name
        for c in fp.columns
        if c.kind in {"continuous", "count", "binary"} and c.name not in excl
    ]
    # keep only columns actually present + numeric-coercible
    return [c for c in cols if c in df.columns]


def _fisher_ci(r: float, n: int, alpha: float = 0.05):
    """Fisher z-transform 95% CI for a correlation coefficient.

    z = atanh(r), SE = 1/sqrt(n-3); back-transform the z ± 1.96·SE band with tanh.
    Returns (low, high); (nan, nan) when undefined (|r|≈1 or n≤3)."""
    import numpy as np
    from scipy import stats

    if not np.isfinite(r) or n <= 3 or abs(r) >= 1.0:
        return float("nan"), float("nan")
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    crit = stats.norm.ppf(1.0 - alpha / 2.0)
    lo, hi = np.tanh(z - crit * se), np.tanh(z + crit * se)
    return float(lo), float(hi)


# ───────────────────────────────────────────────────────────────────────────── #
# (A) pearson_correlation — bivariate r between two numeric columns              #
# ───────────────────────────────────────────────────────────────────────────── #
@register("pearson_correlation")
def _branch_pearson_correlation(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np

    try:
        from scipy import stats
    except Exception as err:  # noqa: BLE001 — degrade, never crash
        summary.append(f"Pearson 相关跳过：导入 scipy 失败（{err}）。")
        return

    num = _numeric_columns(ctx)
    x = cfg.get("x") if cfg.get("x") in df.columns else (num[0] if len(num) >= 1 else None)
    # y = first numeric column that is NOT x
    y = cfg.get("y") if (cfg.get("y") in df.columns and cfg.get("y") != x) else None
    if y is None:
        y = next((c for c in num if c != x), None)
    if x is None or y is None or x == y:
        summary.append(
            "Pearson 相关跳过：需要两个数值列（x、y）。"
            "用 config={\"x\":\"<列>\",\"y\":\"<列>\"} 指定。"
        )
        return

    sub = df[[x, y]].apply(lambda s: __import__("pandas").to_numeric(s, errors="coerce")).dropna()
    n = int(sub.shape[0])
    if n < 3:
        summary.append(f"Pearson 相关跳过：有效样本不足（n={n}，需 ≥3）。")
        return
    xv = sub[x].to_numpy(dtype=float)
    yv = sub[y].to_numpy(dtype=float)
    if np.std(xv) == 0 or np.std(yv) == 0:
        summary.append(f"Pearson 相关跳过：列 {x if np.std(xv)==0 else y} 为常数（方差为 0），相关无定义。")
        return

    pr, pp = stats.pearsonr(xv, yv)
    sr, sp = stats.spearmanr(xv, yv)
    kt, kp = stats.kendalltau(xv, yv)
    ci_lo, ci_hi = _fisher_ci(float(pr), n)

    estimates["pearson_r"] = round(float(pr), 6)
    estimates["pearson_p"] = round(float(pp), 6)
    estimates["ci_low"] = round(float(ci_lo), 6) if np.isfinite(ci_lo) else float("nan")
    estimates["ci_high"] = round(float(ci_hi), 6) if np.isfinite(ci_hi) else float("nan")
    estimates["spearman_rho"] = round(float(sr), 6)
    estimates["kendall_tau"] = round(float(kt), 6)
    estimates["n"] = float(n)

    # CSV: the three coefficients + p + CI
    try:
        import pandas as pd

        tab = pd.DataFrame(
            {
                "coefficient": ["pearson_r", "spearman_rho", "kendall_tau"],
                "value": [round(float(pr), 6), round(float(sr), 6), round(float(kt), 6)],
                "p_value": [round(float(pp), 6), round(float(sp), 6), round(float(kp), 6)],
                "ci_low": [round(float(ci_lo), 6) if np.isfinite(ci_lo) else float("nan"), float("nan"), float("nan")],
                "ci_high": [round(float(ci_hi), 6) if np.isfinite(ci_hi) else float("nan"), float("nan"), float("nan")],
            }
        )
        tab.to_csv(d / "correlation_coefficients.csv", index=False, encoding="utf-8")
        files.append("correlation_coefficients.csv")
    except Exception:  # noqa: BLE001
        pass

    # PNG: scatter + OLS fit line + r annotation
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        slope, intercept = np.polyfit(xv, yv, 1)  # OLS line y = slope·x + intercept
        xs = np.linspace(float(xv.min()), float(xv.max()), 100)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.scatter(xv, yv, s=22, alpha=0.6, color="#4C72B0", edgecolor="white", linewidth=0.3)
        ax.plot(xs, slope * xs + intercept, color="#C44E52", lw=1.8, label="OLS fit")
        ax.set_xlabel(str(x))
        ax.set_ylabel(str(y))
        ax.set_title("Pearson scatter with OLS fit")
        ax.annotate(
            f"Pearson r = {pr:.3f}\np = {pp:.2g}\nn = {n}",
            xy=(0.04, 0.96), xycoords="axes fraction", va="top", ha="left",
            fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="#888", alpha=0.85),
        )
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(d / "scatter_fit.png", dpi=150)
        plt.close(fig)
        files.append("scatter_fit.png")
    except Exception:  # noqa: BLE001
        pass

    strength = "强" if abs(pr) >= 0.7 else ("中等" if abs(pr) >= 0.4 else ("弱" if abs(pr) >= 0.1 else "几乎无"))
    sig = "显著" if pp < 0.05 else "不显著"
    lin_vs_mono = (
        "Spearman/Kendall 明显大于 Pearson → 关系可能是单调但非线性的（如曲线/有异常值）。"
        if (abs(sr) - abs(pr)) > 0.1
        else "三个系数量级相近 → 关系大体线性。"
    )
    ci_txt = f"，Fisher-z 95% CI [{ci_lo:.3f}, {ci_hi:.3f}]" if np.isfinite(ci_lo) else ""
    summary.append(
        f"{entry.method} 完成（{x} vs {y}，n={n}）：Pearson r={pr:.3f}（p={pp:.2g}，{sig}，{strength}相关）"
        f"{ci_txt}；Spearman ρ={sr:.3f}（p={sp:.2g}），Kendall τ={kt:.3f}（p={kp:.2g}）。{lin_vs_mono}"
        " ⚠ Pearson 测的是【线性】关联，对异常值/非正态敏感（单调/有序关系看 Spearman/Kendall）；"
        "相关 ≠ 因果；可用 config x/y 指定列。"
    )
    code += [
        "from scipy import stats  # bivariate correlation",
        f"x, y = df['{x}'].astype(float), df['{y}'].astype(float)",
        "r, p = stats.pearsonr(x, y); rho, _ = stats.spearmanr(x, y); tau, _ = stats.kendalltau(x, y)",
        "# Fisher-z 95% CI: tanh(atanh(r) ± 1.96/sqrt(n-3))",
    ]


# ───────────────────────────────────────────────────────────────────────────── #
# (B) partial_correlation — partial r between x,y controlling for covariates z   #
# ───────────────────────────────────────────────────────────────────────────── #
@register("partial_correlation")
def _branch_partial_correlation(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np

    try:
        from scipy import stats
    except Exception as err:  # noqa: BLE001
        summary.append(f"偏相关跳过：导入 scipy 失败（{err}）。")
        return

    num = _numeric_columns(ctx)
    x = cfg.get("x") if cfg.get("x") in df.columns else (num[0] if len(num) >= 1 else None)
    y = cfg.get("y") if (cfg.get("y") in df.columns and cfg.get("y") != x) else None
    if y is None:
        y = next((c for c in num if c != x), None)
    # covariates: config list, else all remaining numeric columns
    cov_cfg = cfg.get("covariates")
    if cov_cfg:
        covariates = [c for c in cov_cfg if c in df.columns and c not in {x, y}]
    else:
        covariates = [c for c in num if c not in {x, y}]
    if x is None or y is None or x == y:
        summary.append(
            "偏相关跳过：需要两个数值列（x、y）。"
            "用 config={\"x\":\"<列>\",\"y\":\"<列>\",\"covariates\":[\"<列>\"]} 指定。"
        )
        return
    if not covariates:
        summary.append(
            "偏相关跳过：未找到要控制的协变量 z（至少需要 1 个）。"
            "用 config={\"covariates\":[\"<列>\"]} 指定。"
        )
        return

    allcols = [x, y] + covariates
    sub = df[allcols].apply(lambda s: __import__("pandas").to_numeric(s, errors="coerce")).dropna()
    n = int(sub.shape[0])
    n_controls = len(covariates)
    if n < n_controls + 3:
        summary.append(f"偏相关跳过：有效样本不足（n={n}，控制变量 {n_controls} 个，需 n ≥ {n_controls + 3}）。")
        return
    xv = sub[x].to_numpy(dtype=float)
    yv = sub[y].to_numpy(dtype=float)
    Z = sub[covariates].to_numpy(dtype=float)
    if np.std(xv) == 0 or np.std(yv) == 0:
        summary.append("偏相关跳过：x 或 y 为常数（方差为 0），相关无定义。")
        return

    # zero-order (uncontrolled) r for the contrast
    zero_r, zero_p = stats.pearsonr(xv, yv)

    # ── Partial correlation via RESIDUALISATION ──
    # Regress x on [1, Z] and y on [1, Z] by OLS; the partial r is the Pearson
    # correlation of the two residual vectors (numerically equivalent to the
    # precision-matrix off-diagonal but more transparent and robust here). The
    # control rank-deficiency / singular design degrades honestly.
    Zc = np.column_stack([np.ones(n), Z])  # design with intercept
    if np.linalg.matrix_rank(Zc) < Zc.shape[1]:
        summary.append("偏相关跳过：协变量设计矩阵奇异（协变量完全共线 / 含常数列）。")
        return
    try:
        bx, *_ = np.linalg.lstsq(Zc, xv, rcond=None)
        by, *_ = np.linalg.lstsq(Zc, yv, rcond=None)
    except Exception as err:  # noqa: BLE001
        summary.append(f"偏相关跳过：残差化回归失败（{err}）。")
        return
    rx = xv - Zc @ bx  # residual of x after partialling out Z
    ry = yv - Zc @ by  # residual of y after partialling out Z
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        summary.append("偏相关跳过：x 或 y 被协变量完全解释（残差方差≈0），偏相关无定义。")
        return

    pr, _ = stats.pearsonr(rx, ry)
    # p-value & CI on the partial r: df = n - 2 - k controls (residual df loses k+intercept,
    # then 1 more for the residual-residual correlation slope).
    dof = n - 2 - n_controls
    if dof < 1:
        summary.append(f"偏相关跳过：自由度不足（df={dof}）。")
        return
    t_stat = pr * np.sqrt(dof / max(1e-12, 1.0 - pr * pr))
    p_partial = 2.0 * stats.t.sf(abs(t_stat), dof)
    # Fisher-z CI for a partial r controlling k vars: SE = 1/sqrt(n-k-3) = 1/sqrt(dof-1).
    # _fisher_ci does 1/sqrt(n_eff-3), so pass n_eff = dof + 2 (= n - k); at k=0 this
    # reduces to the zero-order Pearson SE 1/sqrt(n-3). (dof = n-2-k.)
    ci_lo, ci_hi = _fisher_ci(float(pr), dof + 2)

    estimates["partial_r"] = round(float(pr), 6)
    estimates["partial_p"] = round(float(p_partial), 6)
    estimates["zero_order_r"] = round(float(zero_r), 6)
    estimates["ci_low"] = round(float(ci_lo), 6) if np.isfinite(ci_lo) else float("nan")
    estimates["ci_high"] = round(float(ci_hi), 6) if np.isfinite(ci_hi) else float("nan")
    estimates["n_controls"] = float(n_controls)
    estimates["n"] = float(n)

    # CSV: zero-order vs partial
    try:
        import pandas as pd

        tab = pd.DataFrame(
            {
                "type": ["zero_order", "partial"],
                "r": [round(float(zero_r), 6), round(float(pr), 6)],
                "p_value": [round(float(zero_p), 6), round(float(p_partial), 6)],
                "ci_low": [float("nan"), round(float(ci_lo), 6) if np.isfinite(ci_lo) else float("nan")],
                "ci_high": [float("nan"), round(float(ci_hi), 6) if np.isfinite(ci_hi) else float("nan")],
                "controls": ["", "; ".join(covariates)],
            }
        )
        tab.to_csv(d / "partial_vs_zero_order.csv", index=False, encoding="utf-8")
        files.append("partial_vs_zero_order.csv")
    except Exception:  # noqa: BLE001
        pass

    # PNG: residual-vs-residual scatter
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        slope, intercept = np.polyfit(rx, ry, 1)
        xs = np.linspace(float(rx.min()), float(rx.max()), 100)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.scatter(rx, ry, s=22, alpha=0.6, color="#55A868", edgecolor="white", linewidth=0.3)
        ax.plot(xs, slope * xs + intercept, color="#C44E52", lw=1.8, label="partial fit")
        ax.set_xlabel(f"{x} residual (after controls)")
        ax.set_ylabel(f"{y} residual (after controls)")
        ax.set_title("Partial correlation: residual vs residual")
        ax.annotate(
            f"partial r = {pr:.3f}\nzero-order r = {zero_r:.3f}\np = {p_partial:.2g}, n = {n}",
            xy=(0.04, 0.96), xycoords="axes fraction", va="top", ha="left",
            fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="#888", alpha=0.85),
        )
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(d / "residual_scatter.png", dpi=150)
        plt.close(fig)
        files.append("residual_scatter.png")
    except Exception:  # noqa: BLE001
        pass

    drop = abs(zero_r) - abs(pr)
    if drop > 0.1:
        interp = f"控制 z 后相关大幅下降（|r|: {abs(zero_r):.3f}→{abs(pr):.3f}）→ 提示【混淆】：z 同时驱动 x 和 y。"
    elif drop < -0.1:
        interp = f"控制 z 后相关反而增强（|r|: {abs(zero_r):.3f}→{abs(pr):.3f}）→ 提示【抑制效应】（suppression）。"
    else:
        interp = f"控制 z 前后相关变化不大（|r|: {abs(zero_r):.3f}→{abs(pr):.3f}）→ z 对 x–y 关系影响有限。"
    sig = "显著" if p_partial < 0.05 else "不显著"
    ci_txt = f"，95% CI [{ci_lo:.3f}, {ci_hi:.3f}]" if np.isfinite(ci_lo) else ""
    summary.append(
        f"{entry.method} 完成（{x} vs {y}，控制 {n_controls} 个协变量，n={n}）："
        f"零阶 r={zero_r:.3f}，偏相关 r={pr:.3f}（p={p_partial:.2g}，{sig}）{ci_txt}。{interp}"
        " ⚠ 偏相关假定各变量间关系【线性】；config 可指定 x/y/covariates。"
    )
    code += [
        "import numpy as np; from scipy import stats  # partial correlation (residualisation)",
        "Zc = np.column_stack([np.ones(n), Z])  # controls + intercept",
        "rx = x - Zc @ np.linalg.lstsq(Zc, x, rcond=None)[0]  # residualise x on Z",
        "ry = y - Zc @ np.linalg.lstsq(Zc, y, rcond=None)[0]  # residualise y on Z",
        "partial_r, _ = stats.pearsonr(rx, ry)  # correlation of residuals",
    ]


# ───────────────────────────────────────────────────────────────────────────── #
# (C) correlation_matrix — full matrix + per-pair p + BH-FDR over all numeric    #
# ───────────────────────────────────────────────────────────────────────────── #
@register("correlation_matrix")
def _branch_correlation_matrix(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np

    try:
        import pandas as pd
        from scipy import stats
    except Exception as err:  # noqa: BLE001
        summary.append(f"相关矩阵跳过：导入失败（{err}）。")
        return

    method = str(cfg.get("method", "pearson")).lower()
    if method not in {"pearson", "spearman"}:
        method = "pearson"
    num = _numeric_columns(ctx)
    if cfg.get("columns"):
        cols = [c for c in cfg["columns"] if c in df.columns]
    else:
        cols = num
    if len(cols) < 2:
        summary.append(f"相关矩阵跳过：需要 ≥2 个数值列（当前 {len(cols)} 个）。")
        return

    sub = df[cols].apply(lambda s: pd.to_numeric(s, errors="coerce")).dropna()
    n = int(sub.shape[0])
    if n < 3:
        summary.append(f"相关矩阵跳过：有效样本不足（n={n}，需 ≥3）。")
        return
    # drop constant columns (correlation undefined)
    keep = [c for c in cols if sub[c].std() > 0]
    dropped = [c for c in cols if c not in keep]
    if len(keep) < 2:
        summary.append(f"相关矩阵跳过：去掉常数列后剩 {len(keep)} 个数值列（需 ≥2）。")
        return
    sub = sub[keep]
    p_vars = len(keep)

    corr_fn = (lambda a, b: stats.spearmanr(a, b)) if method == "spearman" else (lambda a, b: stats.pearsonr(a, b))

    corr = pd.DataFrame(np.eye(p_vars), index=keep, columns=keep)
    pmat = pd.DataFrame(np.zeros((p_vars, p_vars)), index=keep, columns=keep)
    pairs = []  # (var1, var2, r, p)
    for i in range(p_vars):
        for j in range(i + 1, p_vars):
            a = sub[keep[i]].to_numpy(dtype=float)
            b = sub[keep[j]].to_numpy(dtype=float)
            r, p = corr_fn(a, b)
            r = float(r) if np.isfinite(r) else 0.0
            p = float(p) if np.isfinite(p) else 1.0
            corr.iloc[i, j] = corr.iloc[j, i] = round(r, 6)
            pmat.iloc[i, j] = pmat.iloc[j, i] = round(p, 6)
            pairs.append((keep[i], keep[j], r, p))

    # Benjamini-Hochberg FDR over the m = p*(p-1)/2 unique off-diagonal tests.
    m = len(pairs)
    pvals = np.array([pr[3] for pr in pairs], dtype=float)
    order = np.argsort(pvals)
    ranks = np.empty(m, dtype=int)
    ranks[order] = np.arange(1, m + 1)
    p_fdr_raw = pvals * m / ranks  # BH adjusted (before monotone enforcement)
    # enforce monotonicity of BH-adjusted p-values (step-up)
    p_fdr = np.empty(m, dtype=float)
    running = 1.0
    for idx in order[::-1]:
        running = min(running, p_fdr_raw[idx])
        p_fdr[idx] = min(1.0, running)

    long = pd.DataFrame(
        {
            "var1": [pr[0] for pr in pairs],
            "var2": [pr[1] for pr in pairs],
            "r": [round(pr[2], 6) for pr in pairs],
            "abs_r": [round(abs(pr[2]), 6) for pr in pairs],
            "p": [round(pr[3], 6) for pr in pairs],
            "p_fdr": [round(float(v), 6) for v in p_fdr],
            "sig_fdr": [bool(v < 0.05) for v in p_fdr],
        }
    ).sort_values("abs_r", ascending=False).reset_index(drop=True)

    n_sig_fdr = int(long["sig_fdr"].sum())
    abs_offdiag = long["abs_r"].to_numpy(dtype=float)
    max_abs = float(abs_offdiag.max()) if m else 0.0
    mean_abs = float(abs_offdiag.mean()) if m else 0.0

    estimates["max_abs_corr"] = round(max_abs, 6)
    estimates["mean_abs_corr"] = round(mean_abs, 6)
    estimates["n_vars"] = float(p_vars)
    estimates["n_sig_pairs_fdr"] = float(n_sig_fdr)
    estimates["n"] = float(n)

    # CSV: matrix + long pairwise table
    try:
        corr.to_csv(d / "correlation_matrix.csv", encoding="utf-8")
        files.append("correlation_matrix.csv")
        long.to_csv(d / "pairwise_correlations.csv", index=False, encoding="utf-8")
        files.append("pairwise_correlations.csv")
    except Exception:  # noqa: BLE001
        pass

    # PNG: correlation heatmap
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(max(4.5, 0.6 * p_vars + 2), max(4.0, 0.6 * p_vars + 1.5)))
        cmat = corr.to_numpy(dtype=float)
        im = ax.imshow(cmat, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(p_vars))
        ax.set_yticks(range(p_vars))
        ax.set_xticklabels(keep, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(keep, fontsize=8)
        ax.set_title(f"{method.capitalize()} correlation matrix")
        if p_vars <= 12:  # annotate cells only when legible
            for i in range(p_vars):
                for j in range(p_vars):
                    ax.text(j, i, f"{cmat[i, j]:.2f}", ha="center", va="center",
                            fontsize=7, color="black" if abs(cmat[i, j]) < 0.6 else "white")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="correlation")
        fig.tight_layout()
        fig.savefig(d / "correlation_heatmap.png", dpi=150)
        plt.close(fig)
        files.append("correlation_heatmap.png")
    except Exception:  # noqa: BLE001
        pass

    top = long.iloc[0]
    collinear = long[long["abs_r"] >= 0.9]
    coll_txt = ""
    if not collinear.empty:
        cp = collinear.iloc[0]
        coll_txt = f" ⚠ 发现近共线对（如 {cp['var1']}–{cp['var2']}，|r|={cp['abs_r']:.2f}≥0.9）。"
    drop_txt = f"（已剔除 {len(dropped)} 个常数列）" if dropped else ""
    summary.append(
        f"{entry.method} 完成（{method}，{p_vars} 个数值变量{drop_txt}，n={n}，{m} 对）："
        f"最强关联 {top['var1']}–{top['var2']}（r={top['r']:.3f}），平均 |r|={mean_abs:.3f}；"
        f"BH-FDR 控制后 {n_sig_fdr}/{m} 对显著（α=0.05）。{coll_txt}"
        " ⚠ 大量两两检验会膨胀假阳性（已用 BH-FDR 校正，见 p_fdr 列）；"
        f" ⚠ n={n} 为完整个案数（complete-case，对所选列做 listwise 删除）——"
        "某列缺失多会同时拉低每一对的 n（所有相关都基于同一批完整行）；"
        "pearson=线性、spearman=单调；config 可指定 method（pearson/spearman）与 columns。"
    )
    code += [
        "import pandas as pd, numpy as np; from scipy import stats  # correlation matrix + BH-FDR",
        f"sub = df[cols].apply(pd.to_numeric, errors='coerce').dropna()  # method={method}",
        "# pairwise r/p over upper triangle -> Benjamini-Hochberg FDR over m=k(k-1)/2 tests",
    ]
