"""Branch handlers for the conditional_process family.

Hayes-style conditional process analysis (PROCESS-flavoured) on continuous
columns, all OLS via statsmodels — no R. Two methods:

  * moderated_mediation — first-stage moderated mediation (PROCESS model 7):
      X->M moderated by W, then M->Y controlling X. Reports the index of moderated
      mediation (a3*b) and conditional indirect effects at W = mean-SD/mean/mean+SD,
      with bootstrap percentile 95% CIs.
  * johnson_neyman — probe a two-way interaction X*W on outcome Y; conditional
      effect theta(w)=b1+b3*w with its analytic SE; Johnson-Neyman boundaries (the w
      where |theta/SE| = t_crit) by solving the quadratic; simple slopes at W=mean+/-SD.

Each handler resolves variable roles from the continuous columns (config
x/m/y/w overridable), degrades honestly (too few continuous cols / too few rows /
constant column / import missing -> append a Chinese "<method>跳过：<reason>" message and
RETURN), writes CSV + PNG (matplotlib Agg, ENGLISH plot labels), fills float
`estimates`, appends a Chinese `summary` ending with disclosures, and MUTATES
ctx (never rebinds). See executor/_branch_api.py and CLAUDE.md.

statsmodels / numpy / scipy / pandas are installed.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register

# Fixed bootstrap / RNG seed (disclosed in summaries) for reproducibility.
_SEED = 20240607
_N_BOOT = 2000


# ---------------------------------------------------------------------------
# Shared continuous-column resolution.  Returns (cont, problem_msg); when
# problem_msg is not None the caller appends it and returns (honest degrade).
# `min_cont` continuous columns are required.
# ---------------------------------------------------------------------------
def _resolve_continuous(ctx: Ctx, min_cont: int, method_label: str):
    import importlib.util

    fp = ctx.fp
    if importlib.util.find_spec("statsmodels") is None:
        return None, f"{method_label}跳过：需要 statsmodels 包（未检测到）。安装：pip install statsmodels。"

    excl = {fp.unit_col, fp.time_col}
    cont = [
        c.name
        for c in fp.columns
        if c.kind == "continuous" and c.name not in excl
    ]
    if len(cont) < min_cont:
        return None, (
            f"{method_label}跳过：需要 >={min_cont} 个连续列"
            f"（X/调节变量/中介/结果），当前仅 {len(cont)} 个。"
        )
    return cont, None


def _pick(cfg_val, cont, used, fallback_pool):
    """Resolve one role: use cfg_val if it's a valid unused continuous col,
    else take the first unused col from fallback_pool. Returns (name, auto_flag)."""
    if cfg_val and cfg_val in cont and cfg_val not in used:
        return cfg_val, False
    for c in fallback_pool:
        if c not in used:
            return c, True
    return None, True


# ---------------------------------------------------------------------------
# 1. moderated_mediation — PROCESS model 7 (first-stage moderated mediation)
# ---------------------------------------------------------------------------
@register("moderated_mediation")
def _branch_moderated_mediation(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    import numpy as np
    import pandas as pd

    cont, msg = _resolve_continuous(ctx, 4, "调节中介分析")
    if msg is not None:
        summary.append(msg)
        return

    import statsmodels.api as sm

    # -- role resolution ----------------------------------------------------
    used: list[str] = []
    y_col, y_auto = _pick(cfg.get("y"), cont, used, cont)
    used.append(y_col)
    x_col, x_auto = _pick(cfg.get("x"), cont, used, [c for c in cont if c not in used])
    used.append(x_col)
    m_col, m_auto = _pick(cfg.get("m"), cont, used, [c for c in cont if c not in used])
    used.append(m_col)
    w_col, w_auto = _pick(cfg.get("w"), cont, used, [c for c in cont if c not in used])
    used.append(w_col)

    if None in (y_col, x_col, m_col, w_col):
        summary.append("调节中介分析跳过：无法解析 4 个不同的连续角色（X/M/Y/W）。")
        return

    sub = df[[y_col, x_col, m_col, w_col]].apply(
        lambda s: pd.to_numeric(s, errors="coerce")
    ).dropna()
    if len(sub) < 30:
        summary.append(f"调节中介分析跳过：有效行数 {len(sub)} < 30，bootstrap 不可靠。")
        return
    # constant-column guard
    for c in (y_col, x_col, m_col, w_col):
        s = sub[c].std(ddof=1)
        if s == 0 or not np.isfinite(s):
            summary.append(f"调节中介分析跳过：列 {c} 为常数/无方差。")
            return

    Y = sub[y_col].to_numpy(float)
    Xr = sub[x_col].to_numpy(float)
    Mr = sub[m_col].to_numpy(float)
    Wr = sub[w_col].to_numpy(float)
    n = len(sub)

    w_mean = float(Wr.mean())
    w_sd = float(Wr.std(ddof=1))

    def _fit_indices(Xa, Ma, Wa, Ya):
        """Mean-center X and W (reduces X*W collinearity), fit both OLS models,
        return (a1, a3, b, c_prime, w_mean_local). a1 is the X->M slope at
        W = w_mean_local, so the caller MUST evaluate conditional effects as
        (a1 + a3*(w - w_mean_local)) using THIS replicate's own centering anchor.
        Raises on singular fit."""
        w_mean_local = float(Wa.mean())
        xc = Xa - Xa.mean()
        wc = Wa - w_mean_local
        xw = xc * wc
        # mediator model: M = i1 + a1*Xc + a2*Wc + a3*(Xc*Wc)
        Dm = np.column_stack([np.ones(len(Xa)), xc, wc, xw])
        mres = sm.OLS(Ma, Dm).fit()
        a1 = float(mres.params[1])
        a3 = float(mres.params[3])
        # outcome model: Y = i2 + c'*Xc + b*M
        Do = np.column_stack([np.ones(len(Xa)), xc, Ma])
        ores = sm.OLS(Ya, Do).fit()
        cp = float(ores.params[1])
        b = float(ores.params[2])
        return a1, a3, b, cp, w_mean_local

    try:
        a1, a3, b, cp, _ = _fit_indices(Xr, Mr, Wr, Y)  # full-sample anchor == w_mean
    except Exception as err:  # noqa: BLE001
        summary.append(f"调节中介分析跳过：OLS 拟合失败（{err}）。")
        return

    index_mm = a3 * b

    # conditional indirect effect (a1 + a3*(w-anchor))*b ; X,W mean-centered so the
    # absolute moderator value w enters as deviation from the centering anchor.
    def _cond_indirect(a1_, a3_, b_, w_abs, anchor):
        return (a1_ + a3_ * (w_abs - anchor)) * b_

    w_levels = {
        "mean-SD": w_mean - w_sd,
        "mean": w_mean,
        "mean+SD": w_mean + w_sd,
    }
    ind_point = {k: _cond_indirect(a1, a3, b, v, w_mean) for k, v in w_levels.items()}

    # -- bootstrap (row resampling, refit both models) ----------------------
    # Each replicate recenters W on its OWN resampled mean, so conditional effects
    # MUST be evaluated with that replicate's anchor (w_mean_b), not the global one
    # — otherwise a per-replicate offset a3*(w_mean - w_mean_b)*b inflates the CIs.
    rng = np.random.default_rng(_SEED)
    grid = np.linspace(Wr.min(), Wr.max(), 25)
    boot_index = np.full(_N_BOOT, np.nan)
    boot_levels = {k: np.full(_N_BOOT, np.nan) for k in w_levels}
    boot_grid = np.full((_N_BOOT, grid.size), np.nan)
    for bi in range(_N_BOOT):
        idx = rng.integers(0, n, n)
        try:
            ba1, ba3, bb, _, w_mean_b = _fit_indices(Xr[idx], Mr[idx], Wr[idx], Y[idx])
        except Exception:  # noqa: BLE001
            continue
        boot_index[bi] = ba3 * bb
        for k, wv in w_levels.items():
            boot_levels[k][bi] = _cond_indirect(ba1, ba3, bb, wv, w_mean_b)
        boot_grid[bi] = (ba1 + ba3 * (grid - w_mean_b)) * bb

    def _ci(arr):
        a = arr[np.isfinite(arr)]
        if a.size < 100:
            return float("nan"), float("nan")
        return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

    idx_lo, idx_hi = _ci(boot_index)
    lvl_ci = {k: _ci(boot_levels[k]) for k in w_levels}

    estimates["index_mod_med"] = round(index_mm, 6)
    estimates["index_ci_low"] = round(idx_lo, 6)
    estimates["index_ci_high"] = round(idx_hi, 6)
    estimates["indirect_lo"] = round(ind_point["mean-SD"], 6)
    estimates["indirect_mean"] = round(ind_point["mean"], 6)
    estimates["indirect_hi"] = round(ind_point["mean+SD"], 6)
    estimates["a3"] = round(a3, 6)
    estimates["b"] = round(b, 6)

    # grid bootstrap CIs reused by CSV + PNG
    g_lo = np.array([_ci(boot_grid[:, j])[0] for j in range(grid.size)])
    g_hi = np.array([_ci(boot_grid[:, j])[1] for j in range(grid.size)])
    g_pt = (a1 + a3 * (grid - w_mean)) * b

    # -- CSV products -------------------------------------------------------
    try:
        rows = [
            {"quantity": "index_mod_med (a3*b)", "estimate": index_mm,
             "ci_low": idx_lo, "ci_high": idx_hi, "W_value": np.nan},
        ]
        for k, wv in w_levels.items():
            lo, hi = lvl_ci[k]
            rows.append({"quantity": f"conditional_indirect@{k}", "estimate": ind_point[k],
                         "ci_low": lo, "ci_high": hi, "W_value": wv})
        pd.DataFrame(rows).to_csv(d / "moderated_mediation_summary.csv",
                                  index=False, encoding="utf-8")
        files.append("moderated_mediation_summary.csv")

        pd.DataFrame({"W": grid, "indirect_effect": g_pt,
                      "ci_low": g_lo, "ci_high": g_hi}).to_csv(
            d / "moderated_mediation_grid.csv", index=False, encoding="utf-8")
        files.append("moderated_mediation_grid.csv")
    except Exception:  # noqa: BLE001
        pass

    # -- PNG ----------------------------------------------------------------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        ax.plot(grid, g_pt, color="C0", label="conditional indirect effect")
        ax.fill_between(grid, g_lo, g_hi, color="C0", alpha=0.2, label="95% bootstrap CI")
        ax.axhline(0, color="grey", ls="--", lw=1)
        for k, wv in w_levels.items():
            ax.axvline(wv, color="grey", ls=":", lw=0.8)
        ax.set_xlabel(f"Moderator W ({w_col})")
        ax.set_ylabel("Indirect effect of X on Y via M")
        ax.set_title(f"Moderated mediation: {x_col}->{m_col}->{y_col} by {w_col}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(d / "moderated_mediation_plot.png", dpi=150)
        plt.close(fig)
        files.append("moderated_mediation_plot.png")
    except Exception:  # noqa: BLE001
        pass

    code += [
        "import numpy as np, statsmodels.api as sm",
        "# mediator: M = i1 + a1*Xc + a2*Wc + a3*(Xc*Wc)  (X,W mean-centered)",
        "# outcome:  Y = i2 + c'*Xc + b*M",
        "# index of moderated mediation = a3*b ; bootstrap rows B=%d, seed=%d" % (_N_BOOT, _SEED),
    ]

    sig = "" if not (np.isfinite(idx_lo) and np.isfinite(idx_hi)) else (
        "（95% CI 不含 0，间接效应随 W 变化显著）" if (idx_lo > 0 or idx_hi < 0)
        else "（95% CI 含 0，无证据表明间接效应随 W 变化）"
    )
    method = entry.method or "调节中介分析"
    auto_note = ""
    if any([y_auto, x_auto, m_auto, w_auto]):
        auto_note = (
            f"⚠ 角色按列序自动指派（首连续列=Y，其后依次为 X、M、W）：Y={y_col}、X={x_col}、"
            f"M={m_col}、W={w_col}——**顺序极其重要**（X/M/W 不对称，换一组指派就是另一模型），"
            "请用 config x/m/y/w 核对你的理论路径。 "
        )
    summary.append(
        f"{method} 完成（PROCESS 模型 7，X->M 受 W 调节，M->Y 控制 X）：路径 "
        f"{x_col}->{m_col}->{y_col}，调节变量 W={w_col}。调节中介指数 a3*b={index_mm:.4f}"
        f"（bootstrap 95% CI [{idx_lo:.4f}, {idx_hi:.4f}]）{sig}；a3={a3:.4f}、b={b:.4f}。"
        f"条件间接效应：W=mean-SD 时 {ind_point['mean-SD']:.4f}（CI [{lvl_ci['mean-SD'][0]:.4f}, {lvl_ci['mean-SD'][1]:.4f}]）、"
        f"W=mean 时 {ind_point['mean']:.4f}（CI [{lvl_ci['mean'][0]:.4f}, {lvl_ci['mean'][1]:.4f}]）、"
        f"W=mean+SD 时 {ind_point['mean+SD']:.4f}（CI [{lvl_ci['mean+SD'][0]:.4f}, {lvl_ci['mean+SD'][1]:.4f}]）。"
        f"X、W 已中心化后构造 X*W（降低共线性）。" + auto_note +
        f"⚠ 因果解读需序贯可忽略性（X-M、M-Y、X-Y 均无未测混杂）——否则这是**关联性**分解而非因果；"
        f"index CI 不含 0 即表明间接效应依赖 W；bootstrap B={_N_BOOT}、固定随机种子 seed={_SEED}（可复现）；"
        f"可用 config x/m/y/w 指定角色。"
    )


# ---------------------------------------------------------------------------
# 2. johnson_neyman — probe a two-way interaction X*W on outcome Y
# ---------------------------------------------------------------------------
@register("johnson_neyman")
def _branch_johnson_neyman(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    import numpy as np
    import pandas as pd

    cont, msg = _resolve_continuous(ctx, 3, "Johnson-Neyman 分析")
    if msg is not None:
        summary.append(msg)
        return

    import statsmodels.api as sm
    from scipy import stats

    # -- role resolution ----------------------------------------------------
    used: list[str] = []
    y_col, y_auto = _pick(cfg.get("y"), cont, used, cont)
    used.append(y_col)
    x_col, x_auto = _pick(cfg.get("x"), cont, used, [c for c in cont if c not in used])
    used.append(x_col)
    w_col, w_auto = _pick(cfg.get("w"), cont, used, [c for c in cont if c not in used])
    used.append(w_col)

    if None in (y_col, x_col, w_col):
        summary.append("Johnson-Neyman 分析跳过：无法解析 3 个不同的连续角色（X/W/Y）。")
        return

    sub = df[[y_col, x_col, w_col]].apply(
        lambda s: pd.to_numeric(s, errors="coerce")
    ).dropna()
    if len(sub) < 10:
        summary.append(f"Johnson-Neyman 分析跳过：有效行数 {len(sub)} < 10。")
        return
    for c in (y_col, x_col, w_col):
        s = sub[c].std(ddof=1)
        if s == 0 or not np.isfinite(s):
            summary.append(f"Johnson-Neyman 分析跳过：列 {c} 为常数/无方差。")
            return

    Y = sub[y_col].to_numpy(float)
    Xr = sub[x_col].to_numpy(float)
    Wr = sub[w_col].to_numpy(float)
    n = len(sub)
    if n - 4 <= 0:
        summary.append("Johnson-Neyman 分析跳过：样本不足以估计 4 个参数（df=n-4<=0）。")
        return

    w_mean = float(Wr.mean())
    w_sd = float(Wr.std(ddof=1))

    # mean-center X and W for numerical stability; W enters theta(w) on the centered
    # scale, then boundaries are mapped back to the ORIGINAL W scale for reporting.
    xc = Xr - Xr.mean()
    wc = Wr - w_mean
    xw = xc * wc
    D = np.column_stack([np.ones(n), xc, wc, xw])  # b0, b1(X), b2(W), b3(X*W)
    try:
        res = sm.OLS(Y, D).fit()
    except Exception as err:  # noqa: BLE001
        summary.append(f"Johnson-Neyman 分析跳过：OLS 拟合失败（{err}）。")
        return

    b1 = float(res.params[1])
    b3 = float(res.params[3])
    p_int = float(res.pvalues[3])
    cov = res.cov_params()
    var_b1 = float(cov[1, 1])
    var_b3 = float(cov[3, 3])
    cov_13 = float(cov[1, 3])
    t_crit = float(stats.t.ppf(1 - 0.05 / 2, df=n - 4))

    # theta(wc) = b1 + b3*wc  (wc = W - w_mean) ; Var = var_b1 + wc^2*var_b3 + 2*wc*cov_13
    def _theta(wc_val):
        return b1 + b3 * wc_val

    def _se(wc_val):
        v = var_b1 + wc_val ** 2 * var_b3 + 2 * wc_val * cov_13
        return float(np.sqrt(v)) if v > 0 else float("nan")

    # Johnson-Neyman: solve |theta/SE| = t_crit  ->  theta^2 = t_crit^2 * Var
    # (b1 + b3*wc)^2 = t^2 (var_b1 + wc^2 var_b3 + 2 wc cov_13)
    # A*wc^2 + B*wc + C = 0
    t2 = t_crit ** 2
    A = b3 ** 2 - t2 * var_b3
    B = 2.0 * b1 * b3 - 2.0 * t2 * cov_13
    C = b1 ** 2 - t2 * var_b1

    roots_centered: list[float] = []
    if abs(A) < 1e-12:
        if abs(B) > 1e-12:
            roots_centered = [-C / B]
    else:
        disc = B ** 2 - 4 * A * C
        if disc >= 0:
            sq = np.sqrt(disc)
            roots_centered = sorted([(-B - sq) / (2 * A), (-B + sq) / (2 * A)])

    # map centered roots back to ORIGINAL W scale
    roots_w = sorted(r + w_mean for r in roots_centered)
    jn_lower = float(roots_w[0]) if len(roots_w) >= 1 else float("nan")
    jn_upper = float(roots_w[1]) if len(roots_w) >= 2 else float("nan")

    # simple slopes of X at W = mean +/- SD (and mean) — on centered W scale
    slope_lo = _theta(-w_sd)
    slope_mean = _theta(0.0)
    slope_hi = _theta(w_sd)

    def _slope_p(wc_val):
        se = _se(wc_val)
        if not np.isfinite(se) or se == 0:
            return float("nan")
        t = _theta(wc_val) / se
        return float(2 * stats.t.sf(abs(t), df=n - 4))

    p_lo, p_mn, p_hi = _slope_p(-w_sd), _slope_p(0.0), _slope_p(w_sd)

    estimates["jn_lower"] = round(jn_lower, 6)
    estimates["jn_upper"] = round(jn_upper, 6)
    estimates["b3_interaction"] = round(b3, 6)
    estimates["p_interaction"] = round(p_int, 6)
    estimates["slope_lo"] = round(slope_lo, 6)
    estimates["slope_mean"] = round(slope_mean, 6)
    estimates["slope_hi"] = round(slope_hi, 6)

    # -- CSV: theta(w), SE, 95% CI, significance flag across observed W range
    grid_w = np.linspace(float(Wr.min()), float(Wr.max()), 60)
    grid_wc = grid_w - w_mean
    theta_g = np.array([_theta(wc_) for wc_ in grid_wc])
    se_g = np.array([_se(wc_) for wc_ in grid_wc])
    ci_lo_g = theta_g - t_crit * se_g
    ci_hi_g = theta_g + t_crit * se_g
    sig_g = (ci_lo_g > 0) | (ci_hi_g < 0)  # CI excludes 0
    try:
        pd.DataFrame({"W": grid_w, "theta_slope_X": theta_g, "SE": se_g,
                      "ci_low": ci_lo_g, "ci_high": ci_hi_g,
                      "significant": sig_g.astype(int)}).to_csv(
            d / "johnson_neyman_slopes.csv", index=False, encoding="utf-8")
        files.append("johnson_neyman_slopes.csv")
    except Exception:  # noqa: BLE001
        pass

    # -- PNG: theta(w) with CI band + shaded region(s) of significance ------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        ax.plot(grid_w, theta_g, color="C0", label="slope of X on Y")
        ax.fill_between(grid_w, ci_lo_g, ci_hi_g, color="C0", alpha=0.18, label="95% CI")
        ax.axhline(0, color="grey", ls="--", lw=1)
        ylo, yhi = ax.get_ylim()
        # shade significant region(s)
        ax.fill_between(grid_w, ylo, yhi, where=sig_g, color="C2", alpha=0.10,
                        label="region of significance")
        ax.set_ylim(ylo, yhi)
        for r in roots_w:
            if Wr.min() <= r <= Wr.max():
                ax.axvline(r, color="C3", ls=":", lw=1.2)
        ax.set_xlabel(f"Moderator W ({w_col})")
        ax.set_ylabel(f"Conditional slope of X ({x_col}) on Y ({y_col})")
        ax.set_title("Johnson-Neyman: conditional effect vs moderator")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(d / "johnson_neyman_plot.png", dpi=150)
        plt.close(fig)
        files.append("johnson_neyman_plot.png")
    except Exception:  # noqa: BLE001
        pass

    code += [
        "import numpy as np, statsmodels.api as sm; from scipy import stats",
        "# Y = b0 + b1*Xc + b2*Wc + b3*(Xc*Wc)  (X,W mean-centered)",
        "# theta(w)=b1+b3*w ; Var=var_b1 + w^2 var_b3 + 2w cov_13 ; |theta/SE|=t_crit -> quadratic in w",
    ]

    # -- disclosures --------------------------------------------------------
    w_lo_obs, w_hi_obs = float(Wr.min()), float(Wr.max())

    def _range_note(val):
        if not np.isfinite(val):
            return "不存在"
        inside = w_lo_obs <= val <= w_hi_obs
        return f"{val:.4f}" + ("" if inside else "（⚠ 落在观测范围外，属外推）")

    jn_txt = f"下界 {_range_note(jn_lower)}、上界 {_range_note(jn_upper)}"
    int_note = ("交互项 b3 显著" if p_int < 0.05
                else "⚠ 交互项 b3 不显著（p>=0.05），J-N 区域探测意义有限")
    method = entry.method or "Johnson-Neyman 分析"
    auto_note = ""
    if any([y_auto, x_auto, w_auto]):
        auto_note = (
            f"⚠ 角色按列序自动指派（首连续列=Y，其后为 X、W）：Y={y_col}、X={x_col}、W={w_col}"
            "——请用 config x/w/y 核对。 "
        )
    summary.append(
        f"{method} 完成：拟合 {y_col} ~ {x_col} * {w_col}（X、W 已中心化以稳定估计）。"
        f"交互项 b3={b3:.4f}（p={p_int:.3g}，{int_note}）。Johnson-Neyman 显著性边界："
        f"{jn_txt}（W 原始尺度；观测范围 [{w_lo_obs:.3g}, {w_hi_obs:.3g}]）。"
        f"X 的简单斜率：W=mean-SD 时 {slope_lo:.4f}（p={p_lo:.3g}）、"
        f"W=mean 时 {slope_mean:.4f}（p={p_mn:.3g}）、W=mean+SD 时 {slope_hi:.4f}（p={p_hi:.3g}）。"
        + auto_note +
        f"⚠ 需连续调节变量；J-N 区域可能落在观测 W 范围外（已标注为外推）；"
        f"交互不显著时 J-N 探测意义有限；可用 config x/w/y 指定角色。"
    )
