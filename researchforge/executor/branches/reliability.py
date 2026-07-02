"""Branch handlers for the RELIABILITY-ENGINEERING family
(``family="reliability"``, ``domain="engineering"``) — the industrial sibling of
the SPC / quality family.

A coherent reliability trio — **single-unit life → repairable-system growth →
life-stress extrapolation**:

  - weibull_life_analysis  — life-data (Weibull) analysis with right-censoring:
                             2-parameter Weibull MLE (+ lognormal, AIC-selected),
                             B10/B50 life, MTTF, hazard-shape interpretation,
                             Weibull probability plot.
  - reliability_growth     — Crow-AMSAA / NHPP power-law for ONE repairable system
                             under a test-fix program: growth slope β, cumulative &
                             instantaneous MTBF, Duane plot.
  - accelerated_life_test  — life-stress extrapolation (Arrhenius / inverse power
                             law): acceleration factor and extrapolated MTTF/B10 at a
                             use stress, activation energy (Arrhenius).

Conventions (CLAUDE.md「引擎约定」):
  * Honest degrade -> Chinese "<方法>跳过：<原因>" appended to summary + return
    (never crash, never fabricate).
  * Products: CSV + PNG (matplotlib Agg, ENGLISH plot labels, best-effort
    try/except), a FLAT float ``estimates`` dict (NaN for an index that does not
    apply), Chinese ``summary`` with ⚠ assumption / bias disclosures.
  * Reproducible analysis code appended to ``code``.

Profiler traps handled (CLAUDE.md): a time-to-failure column may profile as
``count`` (cycles), ``id`` (all-distinct values), or be the ``time_col`` — for
reliability the time-to-failure IS the point, so column resolution accepts
count / continuous / id / time_col for the life column. A stress column likewise
accepts continuous / count / id. The censoring indicator is a binary 0/1 column.

Pure Python (numpy / scipy / pandas / matplotlib). NO R.

NOTES / STOP-AND-REPORT design choices (see also each handler):
  * Crow-AMSAA MLE divisor follows the truncation type (Rigdon & Basu): FAILURE-
    truncated by default (T = last failure time t_n → β̂ = (n-1)/Σ ln(t_n/t_i)), and
    TIME-truncated when config ``termination_time`` T>t_n is given (β̂ = n/Σ ln(T/t_i)).
    λ̂ = n/T^β̂ in both. Input is cumulative failure times unless config
    ``input_type=interarrival`` (then cumsum'd); inter-arrival can't be told from
    cumulative for an improving system, so the default is cumulative.
  * Arrhenius units: if the stress values look like Celsius (0–300, not already a
    plausible Kelvin range) we ADD 273.15 and disclose it; if they already look
    like Kelvin (>= ~200 and the column name hints kelvin) we use them as-is.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register

# Boltzmann constant in eV/K, for the Arrhenius activation energy Ea = b * k.
_BOLTZMANN_EV_K = 8.617e-5


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _num(series):
    """Coerce a pandas Series to a float numpy array (never raises on dirty data)."""
    import pandas as pd

    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _life_col(ctx: Ctx):
    """Resolve the numeric life / time-to-failure column.

    Priority: config ``life`` / ``time`` -> a name-hinted column -> the profiler's
    time_col -> the first continuous/count/id column. For reliability the
    time-to-failure IS the analysis target, so (unlike most families) we do NOT
    exclude fp.time_col, and we accept count (cycles) / id (all-distinct) kinds.
    Returns the column name or None."""
    fp, cfg, df = ctx.fp, ctx.cfg, ctx.df
    for key in ("life", "time"):
        c = cfg.get(key)
        if c in df.columns:
            return c
    life_kws = ("life", "ttf", "fail", "time", "cycle", "hour", "hrs", "km",
                "mile", "duration", "age")
    # name-hinted continuous/count/id column first
    for c in fp.columns:
        if c.kind in ("continuous", "count", "id") and any(
            k in c.name.lower() for k in life_kws
        ):
            return c.name
    # the profiler's time column (a life column often profiles as the time_col)
    if fp.time_col in df.columns:
        return fp.time_col
    for c in fp.columns:
        if c.kind in ("continuous", "count", "id"):
            return c.name
    return None


def _event_col(ctx: Ctx, exclude):
    """Resolve an optional censoring / event indicator (1=failure, 0=right-censored).

    config ``event`` / ``status`` overrides; else a binary 0/1 column whose name
    hints event/status/censor. Returns the column name or None (treat all as
    failures when None)."""
    fp, cfg, df = ctx.fp, ctx.cfg, ctx.df
    for key in ("event", "status"):
        c = cfg.get(key)
        if c in df.columns and c not in exclude:
            return c
    ev_kws = ("event", "status", "censor", "fail", "suspen", "observed")
    for c in fp.columns:
        if c.name in exclude:
            continue
        if c.kind == "binary" and any(k in c.name.lower() for k in ev_kws):
            return c.name
    # any binary column not the life column, as a last resort
    for c in fp.columns:
        if c.name in exclude:
            continue
        if c.kind == "binary":
            return c.name
    return None


def _stress_col(ctx: Ctx, exclude):
    """Resolve the stress column (temperature / voltage / load …).

    config ``stress`` overrides; else a name-hinted numeric column; else the first
    continuous/count/id column that is not the life column / excluded set."""
    fp, cfg, df = ctx.fp, ctx.cfg, ctx.df
    c = cfg.get("stress")
    if c in df.columns and c not in exclude:
        return c
    stress_kws = ("temp", "stress", "volt", "load", "pressure", "kelvin",
                  "celsius", "current", "humid", "cycle_rate")
    for col in fp.columns:
        if col.name in exclude:
            continue
        if col.kind in ("continuous", "count", "id") and any(
            k in col.name.lower() for k in stress_kws
        ):
            return col.name
    for col in fp.columns:
        if col.name in exclude:
            continue
        if col.kind in ("continuous", "count", "id"):
            return col.name
    return None


def _save_fig(d, fname, files, build):
    """best-effort matplotlib figure (Agg). build(plt) draws on the current figure."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        build(plt)
        plt.tight_layout()
        plt.savefig(d / fname, dpi=150)
        plt.close("all")
        files.append(fname)
    except Exception:
        pass


def _median_ranks(n):
    """Bernard's median-rank plotting positions F_i = (i - 0.3) / (n + 0.4),
    i = 1..n (the standard approximation used on Weibull probability paper)."""
    import numpy as np

    i = np.arange(1, n + 1, dtype=float)
    return (i - 0.3) / (n + 0.4)


def _weibull_probplot_start(t_fail):
    """Robust start (β0, η0) from a probability-plot least-squares regression of
    ln(-ln(1-F)) on ln(t) over the FAILURES only (median-rank F). Slope = β,
    intercept = -β·ln(η). Falls back to moment-ish defaults on degeneracy."""
    import numpy as np

    t = np.sort(np.asarray(t_fail, dtype=float))
    t = t[t > 0]
    n = t.size
    if n < 2:
        m = float(t.mean()) if n else 1.0
        return 1.0, max(m, 1e-6)
    F = _median_ranks(n)
    x = np.log(t)
    y = np.log(-np.log(1.0 - F))
    sx = float(x.std())
    if not np.isfinite(sx) or sx <= 0:
        return 1.0, max(float(t.mean()), 1e-6)
    beta, intercept = np.polyfit(x, y, 1)
    beta = float(beta)
    if not np.isfinite(beta) or beta <= 0:
        beta = 1.0
    eta = float(np.exp(-intercept / beta)) if beta > 0 else float(t.mean())
    if not np.isfinite(eta) or eta <= 0:
        eta = max(float(t.mean()), 1e-6)
    return beta, eta


def _weibull_mle(t_fail, t_cens):
    """2-parameter Weibull MLE on possibly right-censored data.

    Maximises the censored log-likelihood
        L = Σ_failures log f(t) + Σ_censored log S(t)
    where (shape β, scale η)
        f(t) = (β/η)(t/η)^(β-1) exp(-(t/η)^β),   S(t) = exp(-(t/η)^β).
    Optimised over (ln β, ln η) so the parameters stay positive; started from a
    probability-plot regression on the failures. Returns (β̂, η̂, loglik)."""
    import numpy as np
    from scipy.optimize import minimize

    tf = np.asarray(t_fail, dtype=float)
    tc = np.asarray(t_cens, dtype=float)
    tf = tf[np.isfinite(tf) & (tf > 0)]
    tc = tc[np.isfinite(tc) & (tc > 0)]

    def negloglik(theta):
        beta = np.exp(theta[0])
        eta = np.exp(theta[1])
        # failures: log f = log β - log η + (β-1)(log t - log η) - (t/η)^β
        zf = tf / eta
        ll = np.sum(
            np.log(beta) - np.log(eta) + (beta - 1.0) * np.log(zf) - zf ** beta
        )
        if tc.size:
            zc = tc / eta
            ll += np.sum(-(zc ** beta))  # log S = -(t/η)^β
        if not np.isfinite(ll):
            return 1e18
        return -ll

    b0, e0 = _weibull_probplot_start(tf)
    theta0 = np.array([np.log(max(b0, 1e-3)), np.log(max(e0, 1e-6))])
    res = minimize(negloglik, theta0, method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000})
    beta = float(np.exp(res.x[0]))
    eta = float(np.exp(res.x[1]))
    loglik = float(-res.fun)
    return beta, eta, loglik


def _lognormal_mle(t_fail, t_cens):
    """Lognormal MLE on possibly right-censored data (mu, sigma on log scale).

    For an uncensored sample the closed form is mu = mean(log t),
    sigma = std(log t); with right-censored observations we maximise
        Σ_failures log φ((ln t - mu)/σ)/(σ t) + Σ_censored log (1 - Φ((ln t - mu)/σ)).
    Returns (mu, sigma, loglik)."""
    import numpy as np
    from scipy.optimize import minimize
    from scipy.stats import norm

    tf = np.asarray(t_fail, dtype=float)
    tc = np.asarray(t_cens, dtype=float)
    tf = tf[np.isfinite(tf) & (tf > 0)]
    tc = tc[np.isfinite(tc) & (tc > 0)]
    lf = np.log(tf)

    if tc.size == 0:
        mu = float(lf.mean())
        sigma = float(lf.std(ddof=0)) if lf.size > 1 else 1.0
        sigma = max(sigma, 1e-6)
        ll = float(np.sum(norm.logpdf(lf, mu, sigma) - lf))  # -lf: Jacobian d ln t
        return mu, sigma, ll

    lc = np.log(tc)

    def negloglik(theta):
        mu = theta[0]
        sigma = np.exp(theta[1])
        ll = np.sum(norm.logpdf(lf, mu, sigma) - lf)
        ll += np.sum(norm.logsf(lc, mu, sigma))
        if not np.isfinite(ll):
            return 1e18
        return -ll

    mu0 = float(lf.mean())
    s0 = float(lf.std(ddof=0)) if lf.size > 1 else 1.0
    theta0 = np.array([mu0, np.log(max(s0, 1e-3))])
    res = minimize(negloglik, theta0, method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000})
    mu = float(res.x[0])
    sigma = float(np.exp(res.x[1]))
    return mu, sigma, float(-res.fun)


# ===========================================================================
# 1) weibull_life_analysis — life-data (Weibull) analysis with censoring
#    Refs: Abernethy "The New Weibull Handbook"; Nelson "Applied Life Data
#          Analysis"; ReliaSoft Life Data Analysis (Weibull++) methodology.
# ===========================================================================
@register("weibull_life_analysis")
def _branch_weibull_life_analysis(ctx: Ctx) -> None:
    d, files, summary, estimates, code = ctx.d, ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np

    cfg = ctx.cfg
    life = _life_col(ctx)
    if life is None:
        summary.append(
            "威布尔寿命分析 跳过：找不到数值寿命/失效时间列"
            "（用 config['life'] 指定）。"
        )
        return
    try:
        df = ctx.df
        event = _event_col(ctx, exclude={life})
        t_all = _num(df[life])
        if event is not None:
            ev = _num(df[event])
            mask = np.isfinite(t_all) & np.isfinite(ev) & (t_all > 0)
            t_all, ev = t_all[mask], ev[mask]
            # 1 = failure/event observed, 0 = right-censored / suspension
            t_fail = t_all[ev >= 0.5]
            t_cens = t_all[ev < 0.5]
        else:
            mask = np.isfinite(t_all) & (t_all > 0)
            t_all = t_all[mask]
            t_fail = t_all
            t_cens = np.array([], dtype=float)

        n_fail = int(t_fail.size)
        n_cens = int(t_cens.size)
        if n_fail < 2:
            summary.append(
                "威布尔寿命分析 跳过：有效失效观测过少（至少需 2 个失效；"
                f"当前失效={n_fail}、删失={n_cens}）。"
            )
            return

        # ---- Weibull MLE (censored) -----------------------------------------
        beta, eta, ll_w = _weibull_mle(t_fail, t_cens)
        k_w = 2  # (β, η)
        aic_w = 2 * k_w - 2 * ll_w

        # ---- Lognormal MLE (censored), for AIC comparison -------------------
        try:
            mu_ln, sigma_ln, ll_ln = _lognormal_mle(t_fail, t_cens)
            k_ln = 2
            aic_ln = 2 * k_ln - 2 * ll_ln
        except Exception:
            mu_ln = sigma_ln = float("nan")
            aic_ln = float("nan")

        # ---- Weibull life metrics -------------------------------------------
        from scipy.special import gamma as _gammafn

        def _weib_quantile(p):  # F(t)=p -> t = η * (-ln(1-p))^(1/β)
            return eta * (-np.log(1.0 - p)) ** (1.0 / beta)

        b10 = float(_weib_quantile(0.10))
        b50 = float(_weib_quantile(0.50))  # = median
        mttf = float(eta * _gammafn(1.0 + 1.0 / beta))
        # Weibull hazard h(t) = (β/η)(t/η)^(β-1); at t=η -> h(η) = β/η.
        hazard_at_eta = float(beta / eta)

        if beta < 1:
            shape_txt = "β<1：递减失效率（早期失效 / 婴儿期 infant mortality）"
        elif abs(beta - 1.0) < 0.05:
            shape_txt = "β≈1：恒定失效率（随机失效，等价指数分布）"
        else:
            shape_txt = "β>1：递增失效率（磨损期 wear-out）"

        best = "Weibull" if (not np.isfinite(aic_ln) or aic_w <= aic_ln) else "Lognormal"

        # ---- products: life-percentiles CSV ---------------------------------
        import pandas as pd

        ps = [0.01, 0.05, 0.10, 0.50, 0.632, 0.90, 0.95, 0.99]
        life_tbl = pd.DataFrame({
            "percentile": ps,
            "B_life": [round(float(_weib_quantile(p)), 6) for p in ps],
        })
        life_tbl.to_csv(d / "weibull_life_percentiles.csv", index=False, encoding="utf-8")
        files.append("weibull_life_percentiles.csv")

        estimates.update({
            "weibull_beta": round(float(beta), 4),
            "weibull_eta": round(float(eta), 4),
            "b10_life": round(b10, 4),
            "b50_life": round(b50, 4),
            "mttf": round(mttf, 4),
            "failure_rate_at_eta": round(hazard_at_eta, 8),
            "aic_weibull": round(float(aic_w), 4),
            "aic_lognormal": round(float(aic_ln), 4) if np.isfinite(aic_ln) else float("nan"),
            "n_failures": float(n_fail),
            "n_censored": float(n_cens),
        })

        # ---- Weibull probability plot ---------------------------------------
        def _plot(plt):
            tf_sorted = np.sort(t_fail)
            nf = tf_sorted.size
            F = _median_ranks(nf)
            xp = np.log(tf_sorted)
            yp = np.log(-np.log(1.0 - F))
            fig, ax = plt.subplots(figsize=(7.6, 5.2))
            ax.scatter(xp, yp, color="#4C72B0", s=28, zorder=3,
                       label="failures (median rank)")
            # fitted line: ln(-ln(1-F)) = β·ln t - β·ln η
            xs = np.linspace(xp.min(), xp.max(), 100)
            ax.plot(xs, beta * xs - beta * np.log(eta), color="#C44E52", lw=1.6,
                    label=f"Weibull fit (beta={beta:.3g}, eta={eta:.4g})")
            ax.set_xlabel("ln(time-to-failure)")
            ax.set_ylabel("ln(-ln(1 - F))")
            ax.set_title("Weibull probability plot")
            ax.legend(fontsize=8, loc="best")
            ax.grid(True, ls=":", alpha=0.4)

        _save_fig(d, "weibull_probability_plot.png", files, _plot)

        cens_txt = (f"，含 {n_cens} 个右删失（悬挂）观测，已用删失似然 Σlog f + Σlog S 处理"
                    if n_cens else "（全部为失效观测，无删失）")
        aic_txt = (f"AIC：Weibull={aic_w:.3f}、Lognormal={aic_ln:.3f} → 选 {best}"
                   if np.isfinite(aic_ln) else f"AIC(Weibull)={aic_w:.3f}（对数正态拟合失败，仅报威布尔）")
        summary.append(
            f"{ctx.entry.method} 完成：寿命列={life}（{n_fail} 失效{cens_txt}）。"
            f"两参数威布尔 MLE：形状 β={beta:.4g}（{shape_txt}）、尺度 η={eta:.4g}"
            f"（特征寿命，第 63.2 百分位）；B10 寿命={b10:.4g}、B50（中位）寿命={b50:.4g}、"
            f"MTTF=η·Γ(1+1/β)={mttf:.4g}；η 处失效率 h(η)=β/η={hazard_at_eta:.4g}。{aic_txt}。"
            "明细见 weibull_life_percentiles.csv 与威布尔概率图。"
            " ⚠ 本方法与 parametric_survival（参数 AFT 回归，含协变量、估加速因子）互补而非重复："
            "这里聚焦单分布的删失寿命指标（B 寿命 / MTTF / 失效率形状 / 概率图），不含协变量。"
            " ⚠ β 区分 婴儿期(β<1)/随机(β≈1)/磨损(β>1)，是可靠性的工程判断核心。"
            " ⚠ 小样本或重度删失会显著放宽所有估计（β、η、B 寿命、MTTF 的不确定性都增大）；"
            "AIC 仅作分布选择参考，并不保证所选分布外推可靠。"
        )
        code += [
            "import numpy as np; from scipy.optimize import minimize",
            "from scipy.special import gamma",
            "# 删失威布尔 MLE：L = Σ_fail log f(t) + Σ_cens log S(t)",
            "def nll(theta):",
            "    b, eta = np.exp(theta)",
            "    zf = t_fail/eta",
            "    ll = np.sum(np.log(b)-np.log(eta)+(b-1)*np.log(zf)-zf**b)",
            "    if t_cens.size: ll += np.sum(-(t_cens/eta)**b)   # log S",
            "    return -ll",
            "b, eta = np.exp(minimize(nll, x0, method='Nelder-Mead').x)",
            "b10 = eta*(-np.log(1-0.10))**(1/b); b50 = eta*(-np.log(0.5))**(1/b)",
            "mttf = eta*gamma(1+1/b)                       # 平均失效前时间",
        ]
    except Exception as exc:
        summary.append(f"威布尔寿命分析 计算失败：{exc}")


# ===========================================================================
# 2) reliability_growth — Crow-AMSAA (NHPP power law) for a repairable system
#    Refs: Crow (1975) AMSAA reliability-growth model; MIL-HDBK-189;
#          IEC 61164; Duane (1964) plotting method.
# ===========================================================================
@register("reliability_growth")
def _branch_reliability_growth(ctx: Ctx) -> None:
    d, files, summary, estimates, code = ctx.d, ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np

    cfg = ctx.cfg
    life = _life_col(ctx)
    if life is None:
        summary.append(
            "可靠性增长(Crow-AMSAA) 跳过：找不到数值失效时间列"
            "（用 config['life'] 指定累积失效时间或到达间隔）。"
        )
        return
    try:
        df = ctx.df
        raw = _num(df[life])
        raw = raw[np.isfinite(raw)]
        raw = raw[raw > 0]
        if raw.size < 3:
            summary.append(
                "可靠性增长(Crow-AMSAA) 跳过：失效时间过少（至少需 ≥3 个失效时间）。"
            )
            return

        # Input is CUMULATIVE failure times by default. Inter-arrival times can't be
        # told apart from cumulative reliably (for an IMPROVING system β<1 the gaps are
        # themselves monotone increasing), so we honour an explicit config flag and only
        # auto-cumsum when the data is clearly NON-monotone (an unambiguous gaps signal).
        it = str(cfg.get("input_type") or "").strip().lower()
        # cumulative failure times are non-decreasing by construction; only a CLEAR
        # decrease is unambiguous inter-arrival data (a tied pair must NOT flip it).
        has_decrease = bool(np.any(np.diff(raw) < 0))
        if it in {"interarrival", "inter-arrival", "inter_arrival", "gap", "gaps"}:
            t, input_kind = np.cumsum(raw), "inter-arrival"
        elif it in {"cumulative", "cum", "arrival"}:
            t, input_kind = raw.copy(), "cumulative"
        elif not has_decrease:
            t, input_kind = raw.copy(), "cumulative"
        else:
            t, input_kind = np.cumsum(raw), "inter-arrival"
        t = np.sort(t)  # safety
        n = int(t.size)

        # total test time T: config termination_time (time-terminated stop) else the
        # last observed failure time (failure-terminated -> our time-terminated MLE
        # uses T = t_n; documented NOTE below).
        # Truncation type sets the MLE divisor (Rigdon & Basu):
        #   time-truncated  (explicit termination_time T > t_n): β̂ = n   / Σ_{i=1..n} ln(T/t_i)
        #   failure-truncated (default, T = last failure t_n):    β̂ = (n-1)/ Σ_{i=1..n-1} ln(t_n/t_i)
        # In the failure-truncated case the t_n term is ln(1)=0, so the same Σ works; only
        # the numerator differs (n vs n-1). λ̂ = n / T^β̂ in both.
        term = cfg.get("termination_time")
        time_truncated = term is not None
        try:
            T = float(term) if time_truncated else float(t[-1])
        except (TypeError, ValueError):
            T, time_truncated = float(t[-1]), False
        if T <= 0 or not np.isfinite(T) or T < t[-1]:
            summary.append("可靠性增长(Crow-AMSAA) 跳过：总试验时间 T 无效（需 ≥ 末次失效时间）。")
            return

        denom = float(np.sum(np.log(T / t)))
        if denom <= 0 or not np.isfinite(denom):
            summary.append(
                "可靠性增长(Crow-AMSAA) 跳过：Σ ln(T/t_i)≤0，无法估计幂律斜率"
                "（可能所有失效时间集中在末端或 T 设置不当）。"
            )
            return
        beta = (n if time_truncated else (n - 1)) / denom
        lam = n / (T ** beta)

        # MTBF metrics at T:
        #   cumulative MTBF  = T / N(T) = T / (λ T^β) = 1/(λ T^(β-1))
        #   instantaneous MTBF = 1 / intensity(T) = 1/(λ β T^(β-1))
        intensity_T = lam * beta * T ** (beta - 1.0)  # ROCOF u(T)=dN/dt
        mtbf_inst = 1.0 / intensity_T if intensity_T > 0 else float("inf")
        n_expected_T = lam * T ** beta
        mtbf_cum = T / n_expected_T if n_expected_T > 0 else float("inf")

        # projected (instantaneous) MTBF at a future time (config projection_time),
        # assuming the SAME growth law continues: MTBF_inst(t*) = 1/(λ β t*^(β-1)).
        proj_t = cfg.get("projection_time")
        try:
            tstar = float(proj_t) if proj_t is not None else 1.5 * T
        except (TypeError, ValueError):
            tstar = 1.5 * T
        intensity_star = lam * beta * tstar ** (beta - 1.0)
        mtbf_proj = 1.0 / intensity_star if intensity_star > 0 else float("inf")

        if beta < 1:
            trend_txt = "β<1：可靠性正在改善（瞬时 MTBF 增长，test-fix 见效）"
        elif abs(beta - 1.0) < 0.05:
            trend_txt = "β≈1：齐次泊松过程（无增长也无退化，失效率恒定）"
        else:
            trend_txt = "β>1：可靠性在恶化（失效越来越频繁）"

        # ---- products: cumulative table CSV ---------------------------------
        import pandas as pd

        idx = np.arange(1, n + 1, dtype=float)
        cum_mtbf_obs = t / idx  # observed cumulative MTBF at each failure
        gtbl = pd.DataFrame({
            "failure_number": idx.astype(int),
            "cumulative_time": np.round(t, 6),
            "cumulative_mtbf_obs": np.round(cum_mtbf_obs, 6),
            "cumulative_mtbf_fit": np.round((1.0 / lam) * t ** (1.0 - beta), 6),
        })
        gtbl.to_csv(d / "reliability_growth.csv", index=False, encoding="utf-8")
        files.append("reliability_growth.csv")

        estimates.update({
            "camsaa_beta": round(float(beta), 4),
            "camsaa_lambda": round(float(lam), 8),
            "mtbf_cumulative": round(float(mtbf_cum), 4),
            "mtbf_instantaneous": round(float(mtbf_inst), 4),
            "mtbf_projected": round(float(mtbf_proj), 4),
            "n_failures": float(n),
            "total_time": round(float(T), 4),
        })

        # ---- Duane plot (log cumulative MTBF vs log cumulative time) ---------
        def _plot(plt):
            fig, ax = plt.subplots(figsize=(7.6, 5.2))
            ax.scatter(t, cum_mtbf_obs, color="#4C72B0", s=28, zorder=3,
                       label="observed cumulative MTBF")
            # Duane fitted line: cum MTBF = (1/λ) t^(1-β) -> straight on log-log
            xs = np.linspace(t.min(), t.max(), 100)
            ax.plot(xs, (1.0 / lam) * xs ** (1.0 - beta), color="#C44E52", lw=1.6,
                    label=f"Crow-AMSAA fit (beta={beta:.3g})")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("cumulative test time (log)")
            ax.set_ylabel("cumulative MTBF (log)")
            ax.set_title("Duane / Crow-AMSAA reliability-growth plot")
            ax.legend(fontsize=8, loc="best")
            ax.grid(True, which="both", ls=":", alpha=0.4)

        _save_fig(d, "duane_plot.png", files, _plot)

        kind_txt = ("输入按累积失效时间处理" if input_kind == "cumulative"
                    else "输入按到达间隔时间处理，已累加为累积失效时间")
        trunc_txt = "时间截尾" if time_truncated else "失效截尾"
        inst_cmp = "改善时(β<1)高于、退化时(β>1)低于累积"
        # ⚠ note must describe what was ACTUALLY computed (not a hard-coded default
        # description) — the truncation type (and hence the β numerator, n vs n-1)
        # switches on whether config['termination_time'] was supplied.
        _mle_txt = (
            "时间截尾 MLE（β̂=n/Σln(T/t_i)，λ̂=n/T^β̂，T=给定的 termination_time）"
            if time_truncated else
            "失效截尾 MLE（β̂=(n-1)/Σln(t_n/t_i)，λ̂=n/t_n^β̂，T=末次失效时间；"
            "未给 termination_time 时的默认）"
        )
        summary.append(
            f"{ctx.entry.method} 完成（Crow-AMSAA / NHPP 幂律，{trunc_txt} MLE）：失效时间列={life}"
            f"（{n} 个失效，{kind_txt}，总试验时间 T={T:.4g}）。增长斜率 β={beta:.4g}（{trend_txt}）、"
            f"λ={lam:.4g}（N(t)=λ·t^β）。T 处累积 MTBF={mtbf_cum:.4g}、瞬时 MTBF={mtbf_inst:.4g}"
            f"（瞬时是当前真实水平，{inst_cmp}）；外推到 t*={tstar:.4g} 的预计瞬时 MTBF≈{mtbf_proj:.4g}。"
            "明细见 reliability_growth.csv 与 Duane 图。"
            " ⚠ 本模型假定单一可修系统在 test-fix 增长计划下（NHPP 幂律强度 u(t)=λβt^(β-1)）；"
            "β 是『增长斜率』，与方法 1(weibull_life_analysis) 的威布尔形状参数含义不同（勿混用）。"
            f" ⚠ 本次采用{_mle_txt}；两种截尾的 β 分子不同（时间截尾=n，失效截尾=n-1），"
            "给/不给 termination_time 会切换截尾类型，差异在小样本时明显。"
            " ⚠ 投影 MTBF 假定相同增长律持续，外推有风险。"
        )
        if time_truncated:
            code += [
                "import numpy as np",
                "t = np.sort(failure_times)            # 累积失效时间（若为间隔则先 cumsum）",
                "n = t.size; T = termination_time       # 时间截尾：给定的终止时间 T（>=末次失效时间）",
                "beta = n / np.sum(np.log(T / t))      # Crow-AMSAA 增长斜率 (β<1=改善)；时间截尾分子=n",
                "lam  = n / T**beta                    # N(t)=λ t^β",
                "mtbf_cum  = T / (lam * T**beta)       # 累积 MTBF",
                "mtbf_inst = 1.0 / (lam*beta*T**(beta-1))  # 瞬时 MTBF",
            ]
        else:
            code += [
                "import numpy as np",
                "t = np.sort(failure_times)            # 累积失效时间（若为间隔则先 cumsum）",
                "n = t.size; T = t[-1]                 # 失效截尾（默认）：T=末次失效时间",
                "beta = (n - 1) / np.sum(np.log(T / t))  # Crow-AMSAA 增长斜率 (β<1=改善)；失效截尾分子=n-1",
                "lam  = n / T**beta                    # N(t)=λ t^β",
                "mtbf_cum  = T / (lam * T**beta)       # 累积 MTBF",
                "mtbf_inst = 1.0 / (lam*beta*T**(beta-1))  # 瞬时 MTBF",
            ]
    except Exception as exc:
        summary.append(f"可靠性增长(Crow-AMSAA) 计算失败：{exc}")


# ===========================================================================
# 3) accelerated_life_test — life-stress extrapolation (Arrhenius / inverse power)
#    Refs: Nelson "Accelerated Testing"; Escobar & Meeker; ReliaSoft ALTA.
# ===========================================================================
@register("accelerated_life_test")
def _branch_accelerated_life_test(ctx: Ctx) -> None:
    d, files, summary, estimates, code = ctx.d, ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np

    cfg = ctx.cfg
    life = _life_col(ctx)
    if life is None:
        summary.append(
            "加速寿命试验(ALT) 跳过：找不到数值寿命/失效时间列（用 config['life'] 指定）。"
        )
        return
    try:
        df = ctx.df
        stress = _stress_col(ctx, exclude={life})
        if stress is None or stress == life:
            summary.append(
                "加速寿命试验(ALT) 跳过：找不到应力列（温度/电压/载荷…）——用 config['stress'] 指定。"
            )
            return

        event = _event_col(ctx, exclude={life, stress})
        L = _num(df[life])
        S = _num(df[stress])
        mask = np.isfinite(L) & np.isfinite(S) & (L > 0)
        L, S = L[mask], S[mask]
        if event is not None:
            ev = _num(df[event])[mask]
            fail_mask = ev >= 0.5
        else:
            fail_mask = np.ones(L.size, dtype=bool)

        # use the FAILURES for the life-stress regression (censoring-aware Tobit fit is
        # overkill here — disclosed; censored points are still used to set the use stress
        # default range / counts).
        Lf, Sf = L[fail_mask], S[fail_mask]
        n_fail = int(Lf.size)
        n_cens = int((~fail_mask).sum())
        distinct_stress = int(np.unique(np.round(S, 9)).size)
        if distinct_stress < 2:
            summary.append(
                f"加速寿命试验(ALT) 跳过：至少需 2 个不同应力水平（当前 {distinct_stress} 个）。"
            )
            return
        if n_fail < 2:
            summary.append(
                f"加速寿命试验(ALT) 跳过：有效失效观测过少（至少需 2 个失效；当前 {n_fail}）。"
            )
            return

        # ---- choose the life-stress model -----------------------------------
        # config model = "arrhenius" | "inverse_power"; default: temperature-looking
        # stress -> Arrhenius, else inverse power law.
        sname = stress.lower()
        looks_temp = (
            any(k in sname for k in ("temp", "°", "kelvin", "celsius", "degc", "degk"))
            or bool(np.all((S >= 200) & (S <= 1000)))   # plausible Kelvin
            or bool(np.all((S >= 0) & (S <= 400)))      # plausible Celsius
        )
        model = cfg.get("model")
        if model not in ("arrhenius", "inverse_power"):
            model = "arrhenius" if looks_temp else "inverse_power"

        celsius_converted = False
        celsius_ambiguity_note = ""
        if model == "arrhenius":
            # Arrhenius uses absolute temperature (Kelvin). STOP-AND-REPORT:
            # if values look like Celsius (0–300 and not an obvious Kelvin band) add
            # 273.15 and disclose; otherwise assume already Kelvin.
            import re as _re

            name_says_kelvin = ("kelvin" in sname or "degk" in sname
                                or bool(_re.search(r"(^|[^a-z])k($|[^a-z])", sname))
                                or "_k" in sname or "(k)" in sname)
            looks_kelvin = name_says_kelvin or bool(np.all(S >= 200))
            looks_celsius = bool(np.all((S >= -50) & (S <= 300)))
            if looks_celsius and not looks_kelvin:
                S_used = S + 273.15
                Sf_used = Sf + 273.15
                celsius_converted = True
            else:
                S_used = S.copy()
                Sf_used = Sf.copy()
            # ⚠ genuine ambiguity, disclosure only (do NOT change the Kelvin default):
            # a column that only generically hints "temperature" (no explicit unit) with
            # values that fall in a range a real high-temp Celsius test (HTOL/HAST, e.g.
            # 200/250/300 °C) would ALSO produce gets silently treated as Kelvin above via
            # the `S >= 200` fallback. Flag it so the user can confirm/override.
            if not celsius_converted and not name_says_kelvin:
                _temp_hint = any(k in sname for k in ("temp", "°c", "celsius", "degc", "℃"))
                if _temp_hint and bool(np.all(S <= 400)):
                    celsius_ambiguity_note = (
                        f" ⚠ 应力列『{stress}』名称仅泛称『温度』、未标明单位，且数值范围（≤400，本次 "
                        f"{float(np.min(S)):.4g}–{float(np.max(S)):.4g}）与摄氏度高温加速试验"
                        "（如 HTOL/HAST 的 200/250/300°C）同样吻合——本次按开尔文原样使用（未 +273.15）。"
                        "若实际是摄氏度，请把 config['stress'] 对应列改名含'celsius'/'°C'或换算后重跑，"
                        "否则活化能 Ea 与外推寿命会系统性偏差。"
                    )
            # ln(L) = a + b*(1/T)
            x_fit = 1.0 / Sf_used
        else:
            # inverse power law: ln(L) = a + b*ln(stress)
            S_used = S.copy()
            Sf_used = Sf.copy()
            if np.any(Sf_used <= 0):
                summary.append(
                    "加速寿命试验(ALT) 跳过：逆幂律要求应力>0（应力含非正值，"
                    "若为温度请改用 config['model']='arrhenius'）。"
                )
                return
            x_fit = np.log(Sf_used)

        y_fit = np.log(Lf)
        # OLS log-linear life-stress regression: y = a + b*x
        b, a = np.polyfit(x_fit, y_fit, 1)  # slope, intercept
        b = float(b)
        a = float(a)

        # ---- use stress + acceleration factor -------------------------------
        # config use_stress; else the MINIMUM observed stress (the mildest = closest
        # to a field/use condition) and disclose.
        use_raw = cfg.get("use_stress")
        use_supplied = use_raw is not None
        try:
            use_stress = float(use_raw) if use_supplied else float(np.min(S))
        except (TypeError, ValueError):
            use_stress = float(np.min(S))
            use_supplied = False
        high_stress = float(np.max(S))  # most severe tested stress

        def _life_at(stress_value):
            """Predicted MTTF-scale life at a raw stress value (handle K conversion)."""
            if model == "arrhenius":
                temp = stress_value + 273.15 if celsius_converted else stress_value
                return float(np.exp(a + b * (1.0 / temp)))
            return float(np.exp(a + b * np.log(stress_value)))

        life_use = _life_at(use_stress)
        life_high = _life_at(high_stress)
        # acceleration factor = life at use / life at highest tested stress (>1 means
        # the use condition is gentler -> longer life).
        accel_factor = life_use / life_high if life_high > 0 else float("inf")

        # extrapolated metrics at use stress. The log-linear fit gives the life SCALE;
        # report it as the characteristic/MTTF proxy and a B10 using the residual
        # log-scatter (sigma) under a lognormal-of-life assumption.
        resid = y_fit - (a + b * x_fit)
        sigma_log = float(np.std(resid, ddof=1)) if n_fail > 2 else float(np.std(resid))
        from scipy.stats import norm

        # life_use = exp(a + b*x) = exp(E[ln L]) is the MEDIAN (characteristic) life of
        # the fitted lognormal-of-life, NOT the mean. Report both explicitly: the true
        # mean of a lognormal is median * exp(sigma_log^2 / 2) (>= median), so labelling
        # life_use as "MTTF" (a mean) would be systematically too low.
        median_life_use = life_use  # median/characteristic life at use stress
        mean_life_use = (float(median_life_use * np.exp(0.5 * sigma_log ** 2))
                          if np.isfinite(sigma_log) else float("nan"))
        b10_use = float(life_use * np.exp(norm.ppf(0.10) * sigma_log)) if np.isfinite(sigma_log) else float("nan")

        activation_energy = float("nan")
        if model == "arrhenius":
            # ln L = a + Ea/k * (1/T)  => b = Ea/k => Ea = b*k (eV)
            activation_energy = float(b * _BOLTZMANN_EV_K)

        # ---- products: life-stress CSV --------------------------------------
        import pandas as pd

        alt_tbl = pd.DataFrame({
            "stress": np.round(S, 6),
            "life": np.round(L, 6),
            "is_failure": fail_mask.astype(int),
        })
        alt_tbl.to_csv(d / "accelerated_life_test.csv", index=False, encoding="utf-8")
        files.append("accelerated_life_test.csv")

        estimates.update({
            "model_intercept": round(a, 6),
            "model_slope": round(b, 6),
            "acceleration_factor": round(float(accel_factor), 4),
            "median_life_use": round(float(median_life_use), 4),
            "mean_life_use": round(float(mean_life_use), 4) if np.isfinite(mean_life_use) else float("nan"),
            "b10_use": round(float(b10_use), 4) if np.isfinite(b10_use) else float("nan"),
            "use_stress": round(float(use_stress), 6),
            "sigma_log": round(float(sigma_log), 6) if np.isfinite(sigma_log) else float("nan"),
            "n_failures": float(n_fail),
            "n_stress_levels": float(distinct_stress),
            "activation_energy": round(activation_energy, 6) if np.isfinite(activation_energy) else float("nan"),
        })

        # ---- life-stress plot -----------------------------------------------
        def _plot(plt):
            fig, ax = plt.subplots(figsize=(7.8, 5.2))
            ax.scatter(S[fail_mask], L[fail_mask], color="#4C72B0", s=30, zorder=3,
                       label="failures")
            if (~fail_mask).any():
                ax.scatter(S[~fail_mask], L[~fail_mask], facecolors="none",
                           edgecolors="#888888", s=34, zorder=3, label="censored")
            xs = np.linspace(min(S.min(), use_stress), S.max(), 120)
            ys = np.array([_life_at(v) for v in xs])
            ax.plot(xs, ys, color="#C44E52", lw=1.6, label=f"{model} fit")
            ax.axvline(use_stress, color="#55A868", ls="--", lw=1.3,
                       label=f"use stress={use_stress:.4g}")
            ax.scatter([use_stress], [life_use], color="#55A868", marker="*", s=160,
                       zorder=5, label=f"extrapolated life={life_use:.3g}")
            ax.set_yscale("log")
            ax.set_xlabel(f"stress ({stress})")
            ax.set_ylabel("life (log)")
            ax.set_title(f"Accelerated life test — {model} life-stress model")
            ax.legend(fontsize=7.5, loc="best")
            ax.grid(True, which="both", ls=":", alpha=0.4)

        _save_fig(d, "accelerated_life_test.png", files, _plot)

        model_txt = ("Arrhenius：ln(L)=a+b·(1/T)" if model == "arrhenius"
                     else "逆幂律：ln(L)=a+b·ln(stress)")
        unit_txt = ("（应力按摄氏识别，已 +273.15 转为开尔文）" if celsius_converted
                    else ("（应力按开尔文使用）" if model == "arrhenius" else ""))
        use_txt = ("（用户指定）" if use_supplied else "（未指定，取最低实测应力作使用应力并披露）")
        ea_txt = (f"活化能 Ea=b·k={activation_energy:.4g} eV（k=8.617e-5 eV/K）；"
                  if model == "arrhenius" and np.isfinite(activation_energy) else "")
        cens_txt = (f"，{n_cens} 个删失点仅在拟合中按失效外处理（未做 Tobit 加权，已披露）"
                    if n_cens else "")
        mean_txt = (f"、均值≈{mean_life_use:.4g}" if np.isfinite(mean_life_use) else "")
        summary.append(
            f"{ctx.entry.method} 完成（{model_txt}{unit_txt}）：寿命列={life}、应力列={stress}"
            f"（{distinct_stress} 个应力水平、{n_fail} 个失效{cens_txt}）。"
            f"系数 a={a:.4g}、b={b:.4g}；{ea_txt}"
            f"使用应力={use_stress:.4g}{use_txt}，相对最严酷实测应力({high_stress:.4g})的加速因子 AF={accel_factor:.4g}；"
            f"使用应力下外推寿命：中位/特征寿命≈{median_life_use:.4g}{mean_txt}、B10≈{b10_use:.4g}"
            f"（用残差对数散度 σ={sigma_log:.4g} 估 B10）。"
            "明细见 accelerated_life_test.csv 与寿命-应力图。"
            " ⚠ **外推到实测应力范围之外是模型相关且有风险的**——AF 与使用应力寿命强烈依赖所选寿命-应力模型"
            "（Arrhenius vs 逆幂律）是否正确，错选模型会系统性高估/低估使用寿命。"
            " ⚠ 拟合在失效点上做对数线性回归（删失点未做删失感知 Tobit 加权）；B10 假定寿命服从对数正态、"
            "散度在各应力水平相同（常数 σ 假定）。 ⚠ 至少应有 ≥2 个应力水平、每水平有足够失效，估计才稳。"
            " ⚠ 报告的『中位/特征寿命』=exp(a+b·x)，是对数正态假定下的中位数而非均值；"
            "均值=中位·exp(σ_log²/2)（对数正态下均值恒 ≥ 中位）。"
            f"{celsius_ambiguity_note}"
        )
        code += [
            "import numpy as np",
            "# Arrhenius: ln(L)=a+b*(1/T)（T 用开尔文）；逆幂律: ln(L)=a+b*ln(stress)",
            "x = 1.0/T_kelvin    # 或 np.log(stress) 对逆幂律",
            "b, a = np.polyfit(x, np.log(life_fail), 1)",
            "life = lambda s: np.exp(a + b*(1.0/s))        # 在应力 s 处的寿命尺度 = 中位数（非均值）",
            "AF = life(use_stress) / life(high_stress)     # 加速因子",
            "Ea = b * 8.617e-5                             # Arrhenius 活化能 (eV)",
            "mean_life = life(use_stress) * np.exp(0.5*sigma_log**2)  # 对数正态均值（>=中位）",
        ]
    except Exception as exc:
        summary.append(f"加速寿命试验(ALT) 计算失败：{exc}")
