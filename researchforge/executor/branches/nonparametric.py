"""Branch handlers for the nonparametric / resampling family.

Three distribution-free / resampling methods that lean on the empirical
distribution rather than parametric assumptions:

* ``permutation_test``  — distribution-free two-/k-sample test by relabelling.
* ``bootstrap_ci``      — BCa (bias-corrected & accelerated) bootstrap CI.
* ``robust_regression`` — outlier-resistant regression (Huber M + Theil-Sen).

Engine conventions (see CLAUDE.md「引擎约定」): each handler is
``@register("<id>") def _branch_<id>(ctx)``; it unpacks ctx into
df/fp/entry/cfg/d + files/summary/estimates/code and **mutates** them. Outcome =
first continuous column; group = lowest-cardinality categorical/binary; predictors
= remaining continuous/binary. Any role can be overridden via ``config``.

Reproducibility: every resampling routine uses a FIXED seed
``np.random.default_rng(0)`` (overridable via ``config["seed"]``) and the seed is
disclosed in the summary. Pure Python (scipy/numpy/statsmodels) — no R.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register

# Hard cap on resampling iterations so a stray config can't wedge the engine.
_MAX_RESAMPLES = 200_000


def _continuous(fp, df) -> list[str]:
    """Continuous columns (engine convention), excluding unit/time id columns."""
    excl = {fp.unit_col, fp.time_col}
    return [c.name for c in fp.columns if c.kind == "continuous" and c.name not in excl]


# ----------------------------------------------------------------------------- #
# 1. permutation_test — distribution-free two-/k-sample test                    #
# ----------------------------------------------------------------------------- #
@register("permutation_test")
def _branch_permutation_test(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np

    # --- column resolution (config overrides; else engine convention) -------- #
    excl = {fp.unit_col, fp.time_col}
    bin_cols = [c.name for c in fp.columns if c.kind == "binary" and c.name not in excl]
    cat_cols = [c.name for c in fp.columns if c.kind == "categorical" and c.name not in excl]
    cat_cols.sort(key=lambda name: int(df[name].nunique()))  # lowest-cardinality first
    group_candidates = bin_cols + cat_cols
    cont_cols = _continuous(fp, df)

    outcome = cfg.get("outcome") if cfg.get("outcome") in df.columns else (
        cont_cols[0] if cont_cols else None
    )
    group_col = cfg.get("group") if cfg.get("group") in df.columns else (
        group_candidates[0] if group_candidates else None
    )

    if outcome is None or group_col is None:
        summary.append("置换检验失败：未找到连续结果变量或分组变量。")
        return

    sub = df[[group_col, outcome]].dropna()
    sub = sub[np.isfinite(sub[outcome].to_numpy(dtype=float))]
    # keep only levels that actually have >=1 observation, preserving order
    levels = [lv for lv in sub[group_col].unique().tolist()
              if int((sub[group_col] == lv).sum()) > 0]
    groups = [sub.loc[sub[group_col] == lv, outcome].to_numpy(dtype=float) for lv in levels]
    n_groups = len(groups)
    if n_groups < 2:
        summary.append("置换检验失败：有效分组少于 2 个。")
        return

    seed = int(cfg.get("seed", 0))
    rng = np.random.default_rng(seed)
    n_perm = int(cfg.get("n_perm", 9999))
    n_perm = max(99, min(n_perm, _MAX_RESAMPLES))

    labels = sub[group_col].to_numpy()
    values = sub[outcome].to_numpy(dtype=float)
    group_means = {str(lv): float(values[labels == lv].mean()) for lv in levels}

    # --- observed statistic + permutation null ------------------------------- #
    # Standard exact/Monte-Carlo permutation test (Fisher 1935; Good, "Permutation
    # Tests" 2005): under H0 the group labels are exchangeable, so we recompute the
    # statistic on n_perm random relabellings of the SAME pooled values.
    if n_groups == 2:
        stat_name = "mean difference"
        a, b = groups[0], groups[1]

        def _stat(perm_labels):
            ga = values[perm_labels == levels[0]]
            gb = values[perm_labels == levels[1]]
            return ga.mean() - gb.mean()

        obs = float(a.mean() - b.mean())
        perm_stats = np.empty(n_perm)
        for i in range(n_perm):
            perm_stats[i] = _stat(rng.permutation(labels))
        # two-sided p with +1 correction (Phipson & Smyth 2010): the observed
        # permutation IS one of the equally-likely arrangements, so it is counted.
        more_extreme = int(np.sum(np.abs(perm_stats) >= abs(obs) - 1e-12))
        pval = (more_extreme + 1) / (n_perm + 1)
        tail = "two-sided |perm| >= |obs|"
    else:
        stat_name = "F-ratio"
        # one-way F statistic (between-group MS / within-group MS), recomputed on
        # each relabelling. F is one-sided (large F => group separation).
        grand = values.mean()

        def _fstat(perm_labels):
            gm = np.array([values[perm_labels == lv].mean() for lv in levels])
            gn = np.array([np.sum(perm_labels == lv) for lv in levels], dtype=float)
            ss_between = float(np.sum(gn * (gm - grand) ** 2))
            ss_within = 0.0
            for lv in levels:
                gv = values[perm_labels == lv]
                ss_within += float(np.sum((gv - gv.mean()) ** 2))
            df_b = n_groups - 1
            df_w = values.size - n_groups
            if df_w <= 0 or ss_within <= 0:
                return np.inf
            return (ss_between / df_b) / (ss_within / df_w)

        obs = float(_fstat(labels))
        perm_stats = np.empty(n_perm)
        for i in range(n_perm):
            perm_stats[i] = _fstat(rng.permutation(labels))
        # one-sided >= for F
        more_extreme = int(np.sum(perm_stats >= obs - 1e-12))
        pval = (more_extreme + 1) / (n_perm + 1)
        tail = "one-sided perm >= obs (F)"

    estimates["observed_stat"] = float(obs)
    estimates["p_value"] = float(pval)
    estimates["n_perm"] = float(n_perm)
    estimates["n_groups"] = float(n_groups)

    # --- products: means CSV + permutation-null histogram -------------------- #
    import pandas as pd

    means_tab = pd.DataFrame(
        {"group": list(group_means.keys()), "mean": list(group_means.values())}
    )
    means_tab["n"] = [int(np.sum(labels == lv)) for lv in levels]
    means_tab.to_csv(d / "group_means.csv", index=False, encoding="utf-8")
    files.append("group_means.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(perm_stats, bins=40, color="#6c8ebf", alpha=0.8, edgecolor="white")
        ax.axvline(obs, color="#b85450", lw=2, label=f"observed = {obs:.4g}")
        ax.set_xlabel(f"permutation null: {stat_name}")
        ax.set_ylabel("count")
        ax.set_title(f"Permutation test (n_perm={n_perm}, p={pval:.4g})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(d / "permutation_null.png", dpi=150)
        plt.close(fig)
        files.append("permutation_null.png")
    except Exception:
        pass

    means_str = "，".join(f"{k}={v:.4g}" for k, v in group_means.items())
    summary.append(
        f"{entry.method} 完成：{outcome} 按 {group_col} 分 {n_groups} 组，"
        f"统计量({stat_name})={obs:.4g}，置换 p={pval:.4g}（n_perm={n_perm}，种子={seed}）。"
        f" 组均值：{means_str}。"
    )
    summary.append(
        "⚠ 假定：H0 下仅组标签可交换（exchangeability）；本检验只检验所选统计量"
        f"（{stat_name}），不是分布整体差异；p 仅精确到重采样分辨率，最小 p="
        f"1/(n_perm+1)={1.0 / (n_perm + 1):.2g}（{tail}，含 +1 校正）。"
    )

    code += [
        "import numpy as np",
        f"rng = np.random.default_rng({seed})  # fixed seed, disclosed",
        f"sub = df[['{group_col}', '{outcome}']].dropna()",
        f"labels = sub['{group_col}'].to_numpy(); values = sub['{outcome}'].to_numpy(float)",
        "# 2-group: mean diff; k-group: F-ratio. Null by permuting labels.",
        "obs = values[labels==L0].mean() - values[labels==L1].mean()  # 2-group example",
        f"perm = [ (values[(p:=rng.permutation(labels))==L0].mean()-values[p==L1].mean()) "
        f"for _ in range({n_perm}) ]",
        "p = (np.sum(np.abs(perm) >= abs(obs)) + 1) / (len(perm) + 1)  # two-sided, +1",
    ]


# ----------------------------------------------------------------------------- #
# 2. bootstrap_ci — BCa (bias-corrected & accelerated) bootstrap CI             #
# ----------------------------------------------------------------------------- #
@register("bootstrap_ci")
def _branch_bootstrap_ci(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    from scipy import stats as _sps

    cont_cols = _continuous(fp, df)
    if not cont_cols:
        summary.append("Bootstrap CI 失败：未找到连续型列。")
        return

    statistic = str(cfg.get("statistic", "mean")).lower()
    valid_stats = {"mean", "median", "std", "correlation"}
    if statistic not in valid_stats:
        statistic = "mean"

    # column resolution
    col = cfg.get("column") if cfg.get("column") in df.columns else cont_cols[0]
    col2 = None
    if statistic == "correlation":
        col2 = cfg.get("column2") if cfg.get("column2") in df.columns else next(
            (c for c in cont_cols if c != col), None
        )
        if col2 is None:
            summary.append("Bootstrap CI（相关）失败：需要两个连续列。")
            return

    # --- assemble the sample ------------------------------------------------- #
    if statistic == "correlation":
        sub = df[[col, col2]].dropna()
        x = sub[col].to_numpy(dtype=float)
        y = sub[col2].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        n = x.size

        def _stat_paired(ix):
            return float(np.corrcoef(x[ix], y[ix])[0, 1])

        obs = float(np.corrcoef(x, y)[0, 1]) if n >= 2 else float("nan")
        stat_label = f"corr({col}, {col2})"
    else:
        s = df[col].to_numpy(dtype=float)
        s = s[np.isfinite(s)]
        n = s.size
        _fn = {"mean": np.mean, "median": np.median, "std": lambda a: np.std(a, ddof=1)}[statistic]

        def _stat_paired(ix):
            return float(_fn(s[ix]))

        obs = float(_fn(s)) if n >= 1 else float("nan")
        stat_label = f"{statistic}({col})"

    if n < 3 or not np.isfinite(obs):
        summary.append(f"Bootstrap CI 失败：有效样本不足（n={n}）。")
        return

    seed = int(cfg.get("seed", 0))
    rng = np.random.default_rng(seed)
    n_boot = int(cfg.get("n_boot", 9999))
    n_boot = max(199, min(n_boot, _MAX_RESAMPLES))
    ci_level = float(cfg.get("ci", 0.95))
    if not (0.0 < ci_level < 1.0):
        ci_level = 0.95
    alpha = 1.0 - ci_level

    # --- bootstrap distribution --------------------------------------------- #
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)  # resample WITH replacement
        boot[i] = _stat_paired(idx)
    boot_finite = boot[np.isfinite(boot)]

    # ===================== BCa interval (Efron 1987) ======================== #
    # Reference: Efron & Tibshirani, "An Introduction to the Bootstrap" (1993), ch.14.
    # z0  = Phi^{-1}( #{theta*_b < theta_hat} / B )   -- bias-correction
    # a   = sum( (mean(jk) - jk_i)^3 ) / ( 6 * (sum( (mean(jk) - jk_i)^2 ))^1.5 )  -- accel (jackknife)
    # alpha1 = Phi( z0 + (z0 + z_{alpha/2}) / (1 - a*(z0 + z_{alpha/2})) )
    # alpha2 = Phi( z0 + (z0 + z_{1-alpha/2}) / (1 - a*(z0 + z_{1-alpha/2})) )
    # CI = ( percentile(boot, 100*alpha1), percentile(boot, 100*alpha2) )
    # tie-aware "mean" rank convention (matches scipy's percentile-of-score): plain
    # strict-`<` biases z0 when bootstrap values tie the estimate (discrete stats like
    # median/std on integer data) and shifts the whole interval the wrong way.
    n_eff = boot_finite.size
    prop_less = (
        np.count_nonzero(boot_finite < obs) + np.count_nonzero(boot_finite <= obs)
    ) / (2.0 * n_eff)
    # clamp away from 0/1 so the inverse-normal is finite (Efron's convention)
    prop_less = min(max(prop_less, 1.0 / (2 * n_boot)), 1.0 - 1.0 / (2 * n_boot))
    z0 = float(_sps.norm.ppf(prop_less))

    # jackknife acceleration: leave-one-out recomputation of the statistic
    jk = np.empty(n)
    if statistic == "correlation":
        for i in range(n):
            ix = np.concatenate((np.arange(i), np.arange(i + 1, n)))
            jk[i] = float(np.corrcoef(x[ix], y[ix])[0, 1])
    else:
        for i in range(n):
            ix = np.concatenate((np.arange(i), np.arange(i + 1, n)))
            jk[i] = float(_fn(s[ix]))
    jk_mean = float(np.mean(jk))
    sumsq_jk = float(np.sum((jk_mean - jk) ** 2))
    num = float(np.sum((jk_mean - jk) ** 3))
    den = 6.0 * (sumsq_jk ** 1.5)
    a = num / den if den != 0 else 0.0
    # BCa is ill-founded when the jackknife is constant (sumsq=0 -> a forced to 0) or the
    # bootstrap distribution is near-degenerate (≤2 distinct values) — typical of highly
    # tied/discrete statistics or a near-constant column. scipy raises/returns NaN here;
    # we still return the percentile-shifted interval but flag it as unreliable.
    bca_degenerate = (sumsq_jk == 0.0) or (int(np.unique(boot_finite).size) < 3)

    z_lo = float(_sps.norm.ppf(alpha / 2.0))
    z_hi = float(_sps.norm.ppf(1.0 - alpha / 2.0))

    def _adjust(zq: float) -> float:
        denom = 1.0 - a * (z0 + zq)
        if denom == 0:
            denom = 1e-12
        return float(_sps.norm.cdf(z0 + (z0 + zq) / denom))

    a1 = _adjust(z_lo)
    a2 = _adjust(z_hi)
    ci_lo = float(np.percentile(boot_finite, 100.0 * a1))
    ci_hi = float(np.percentile(boot_finite, 100.0 * a2))

    estimates["estimate"] = float(obs)
    estimates["ci_low"] = ci_lo
    estimates["ci_high"] = ci_hi
    estimates["z0"] = z0
    estimates["acceleration"] = float(a)
    estimates["n_boot"] = float(n_boot)
    estimates["ci_level"] = ci_level

    # --- products ------------------------------------------------------------ #
    import pandas as pd

    tab = pd.DataFrame(
        {
            "statistic": [stat_label],
            "estimate": [obs],
            "ci_low": [ci_lo],
            "ci_high": [ci_hi],
            "ci_level": [ci_level],
            "z0": [z0],
            "acceleration": [a],
            "n_boot": [n_boot],
            "method": ["BCa"],
        }
    )
    tab.to_csv(d / "bootstrap_ci.csv", index=False, encoding="utf-8")
    files.append("bootstrap_ci.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(boot_finite, bins=40, color="#82b366", alpha=0.8, edgecolor="white")
        ax.axvline(obs, color="#b85450", lw=2, label=f"estimate = {obs:.4g}")
        ax.axvline(ci_lo, color="#6c8ebf", ls="--", lw=1.5, label=f"BCa {int(ci_level*100)}% CI")
        ax.axvline(ci_hi, color="#6c8ebf", ls="--", lw=1.5)
        ax.set_xlabel(f"bootstrap distribution: {stat_label}")
        ax.set_ylabel("count")
        ax.set_title(f"BCa bootstrap (n_boot={n_boot})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(d / "bootstrap_dist.png", dpi=150)
        plt.close(fig)
        files.append("bootstrap_dist.png")
    except Exception:
        pass

    estimates["bca_reliable"] = 0.0 if bca_degenerate else 1.0
    summary.append(
        f"{entry.method} 完成：{stat_label} 点估计={obs:.4g}，"
        f"BCa {int(ci_level*100)}% CI=[{ci_lo:.4g}, {ci_hi:.4g}]"
        f"（z0={z0:.3g}，加速度 a={a:.3g}，n_boot={n_boot}，种子={seed}）。"
    )
    if bca_degenerate:
        summary.append(
            "⚠ BCa 退化警告：刀切法/重采样分布近常数（统计量在数据上高度离散/有结，或列近常数），"
            "偏差/加速度校正不可靠，区间应谨慎解读（scipy 在此情形会拒绝返回 BCa 区间）。"
        )
    summary.append(
        "⚠ 假定：BCa 校正了 bootstrap 分布的偏差(z0)与偏态(加速度 a)，优于朴素百分位法；"
        "假定重采样能模拟原抽样过程——样本极小或强依赖(时序/聚类)时会失效；"
        f"置信水平由 config['ci'] 控制（默认 0.95，本次 {ci_level}）。"
    )

    code += [
        "import numpy as np; from scipy import stats",
        f"rng = np.random.default_rng({seed})  # fixed seed, disclosed",
        f"s = df['{col}'].dropna().to_numpy(float); n = s.size; obs = np.mean(s)",
        f"boot = np.array([np.mean(s[rng.integers(0,n,n)]) for _ in range({n_boot})])",
        "z0 = stats.norm.ppf(np.mean(boot < obs))  # bias-correction",
        "jk = np.array([np.mean(np.delete(s,i)) for i in range(n)])  # jackknife",
        "m = jk.mean(); a = np.sum((m-jk)**3) / (6*np.sum((m-jk)**2)**1.5)  # accel",
        "zl, zh = stats.norm.ppf([0.025, 0.975])",
        "a1 = stats.norm.cdf(z0 + (z0+zl)/(1-a*(z0+zl)))",
        "a2 = stats.norm.cdf(z0 + (z0+zh)/(1-a*(z0+zh)))",
        "ci = np.percentile(boot, [100*a1, 100*a2])  # BCa interval",
    ]


# ----------------------------------------------------------------------------- #
# 3. robust_regression — Huber M-estimator + Theil-Sen (high breakdown)         #
# ----------------------------------------------------------------------------- #
@register("robust_regression")
def _branch_robust_regression(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    excl = {fp.unit_col, fp.time_col}
    cont_cols = _continuous(fp, df)
    outcome = cfg.get("outcome") if cfg.get("outcome") in df.columns else (
        cont_cols[0] if cont_cols else None
    )
    if outcome is None:
        summary.append("稳健回归失败：未找到连续型结果变量。")
        return

    exclude = {outcome, fp.unit_col, fp.time_col}
    if cfg.get("predictors"):
        predictors = [c for c in cfg["predictors"] if c in df.columns and c not in exclude]
    else:
        predictors = [
            c.name
            for c in fp.columns
            if c.kind in {"continuous", "binary", "count"} and c.name not in exclude
        ][:5]
    if not predictors:
        summary.append("稳健回归失败：未找到预测变量。")
        return

    sub = df[[outcome] + predictors].dropna()
    for c in [outcome] + predictors:
        sub = sub[np.isfinite(sub[c].to_numpy(dtype=float))]
    if sub.shape[0] < len(predictors) + 2:
        summary.append(f"稳健回归失败：有效样本不足（n={sub.shape[0]}）。")
        return

    y = sub[outcome].to_numpy(dtype=float)
    X = sub[predictors].to_numpy(dtype=float)
    X_const = sm.add_constant(X, has_constant="add")
    names = ["const"] + predictors

    try:
        # Robust Linear Model — Huber's T M-estimator (statsmodels RLM, IRLS).
        # Reference: Huber (1964); Holland & Welsch (1977). Downweights large
        # residuals in Y. Robust SEs are asymptotic.
        rlm = sm.RLM(y, X_const, M=sm.robust.norms.HuberT()).fit()
        ols = sm.OLS(y, X_const).fit()
    except Exception as err:
        summary.append(f"稳健回归失败：{err}")
        return

    # coefficient comparison table: robust vs OLS exposes outlier influence
    comp = pd.DataFrame(
        {
            "term": names,
            "robust_coef": [float(v) for v in rlm.params],
            "robust_se": [float(v) for v in rlm.bse],
            "ols_coef": [float(v) for v in ols.params],
            "ols_se": [float(v) for v in ols.bse],
        }
    )
    comp["coef_diff"] = comp["robust_coef"] - comp["ols_coef"]

    theil_slope = theil_lo = theil_hi = None
    if len(predictors) == 1:
        # Theil-Sen: median of pairwise slopes, ~29.3% breakdown point (resists
        # high-leverage X that Huber does not). scipy.stats.theilslopes.
        from scipy import stats as _sps

        ts = _sps.theilslopes(y, X[:, 0])
        theil_slope, theil_intercept = float(ts[0]), float(ts[1])
        theil_lo, theil_hi = float(ts[2]), float(ts[3])
        comp.loc[comp["term"] == predictors[0], "theilsen_slope"] = theil_slope
        estimates["theilsen_slope"] = theil_slope

    comp.to_csv(d / "robust_vs_ols.csv", index=False, encoding="utf-8")
    files.append("robust_vs_ols.csv")

    for nm, val in zip(names, rlm.params):
        if nm != "const":
            estimates[nm] = float(val)

    # --- plot: scatter + robust/OLS fitted lines (1-predictor case) ---------- #
    if len(predictors) == 1:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            xv = X[:, 0]
            order = np.argsort(xv)
            xs = xv[order]
            grid = np.linspace(float(xs.min()), float(xs.max()), 100)
            rlm_pred = rlm.params[0] + rlm.params[1] * grid
            ols_pred = ols.params[0] + ols.params[1] * grid

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(xv, y, s=14, color="#888888", alpha=0.6, label="data")
            ax.plot(grid, rlm_pred, color="#82b366", lw=2,
                    label=f"Huber robust (slope={rlm.params[1]:.3g})")
            ax.plot(grid, ols_pred, color="#b85450", lw=2, ls="--",
                    label=f"OLS (slope={ols.params[1]:.3g})")
            if theil_slope is not None:
                ts_pred = theil_intercept + theil_slope * grid
                ax.plot(grid, ts_pred, color="#6c8ebf", lw=1.8, ls=":",
                        label=f"Theil-Sen (slope={theil_slope:.3g})")
            ax.set_xlabel(predictors[0])
            ax.set_ylabel(outcome)
            ax.set_title(f"Robust vs OLS regression: {outcome} ~ {predictors[0]}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(d / "robust_vs_ols.png", dpi=150)
            plt.close(fig)
            files.append("robust_vs_ols.png")
        except Exception:
            pass

    # summary
    key = predictors[0]
    rlm_b = float(rlm.params[1])
    ols_b = float(ols.params[1])
    ts_note = ""
    if theil_slope is not None:
        ts_note = f"，Theil-Sen 斜率={theil_slope:.4g}（95% CI [{theil_lo:.4g}, {theil_hi:.4g}]）"
    summary.append(
        f"{entry.method} 完成：结果 {outcome}，{len(predictors)} 个预测变量。"
        f" 关键变量 {key}：Huber 稳健斜率={rlm_b:.4g}，OLS 斜率={ols_b:.4g}"
        f"（差={rlm_b - ols_b:+.4g}，对比见 robust_vs_ols.csv）{ts_note}。"
    )
    summary.append(
        "⚠ 假定：M-估计(Huber)对 Y 方向离群点降权，但不防高杠杆 X（需 MM 估计或 Theil-Sen）；"
        "Theil-Sen 击穿点约 29%；稳健标准误是渐近的；与 OLS 对比可看出离群点把斜率拉偏了多少。"
    )

    code += [
        "import statsmodels.api as sm; import numpy as np",
        f"y = df['{outcome}'].to_numpy(float); X = sm.add_constant(df[{predictors}].to_numpy(float))",
        "rlm = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()  # Huber M-estimator",
        "ols = sm.OLS(y, X).fit()  # compare to see outlier pull",
        "from scipy import stats",
        f"slope, intercept, lo, hi = stats.theilslopes(y, df['{key}'])  # ~29% breakdown",
    ]
