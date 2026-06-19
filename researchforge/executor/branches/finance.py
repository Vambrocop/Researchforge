"""Branch handlers for the FINANCE / RISK family.

Three single-series risk methods, each taking ONE numeric series (a returns or
price column):
  - value_at_risk        — VaR + Expected Shortfall (CVaR): historical, parametric
                           Gaussian, and Cornish-Fisher modified VaR.
  - extreme_value        — Extreme Value Theory, Peaks-Over-Threshold (GPD fit).
  - risk_adjusted_return — Sharpe / Sortino / max drawdown / Calmar performance metrics.

Conventions (see CLAUDE.md「引擎约定」):
  * Series resolution: first continuous column (excluding unit_col/time_col);
    config `column` / `returns` overrides the name.
  * If the series looks like PRICES (all positive AND trending), we compute
    log-returns and DISCLOSE; config `is_returns` (bool) forces the choice.
  * LOSS-SIGN convention: losses are POSITIVE, gains NEGATIVE (loss = -return).
    VaR/ES are reported as positive loss magnitudes.
  * Time-ordered by fp.time_col when present (matches _branch_garch).
  * Products: CSV + PNG (matplotlib Agg, English labels, best-effort), float
    estimates, Chinese summary with ⚠ disclosures; honest failure + return.

Pure Python (numpy / pandas / scipy) — no R.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


# --------------------------------------------------------------------------- #
# Shared series resolution: returns (series, name, is_returns_flag, price_note)
# or (None, None, _, _, msg) on failure.
# --------------------------------------------------------------------------- #
def _resolve_series(ctx: Ctx, min_n: int):
    """Resolve the single numeric series to analyse and convert prices→log-returns.

    Returns (y, name, used_returns, note, None) on success, where `y` is a 1-D
    numpy float array of RETURNS, or (None, None, None, None, message) on failure.
    """
    import numpy as np

    df, fp, cfg = ctx.df, ctx.fp, ctx.cfg
    excl = {fp.unit_col, fp.time_col}
    # config column override: accept either `column` or `returns` key (alias).
    name = cfg.get("column") or cfg.get("returns")
    if name not in df.columns:
        name = next(
            (c.name for c in fp.columns if c.kind == "continuous" and c.name not in excl),
            None,
        )
    if name is None:
        return None, None, None, None, "需要一个连续数值序列（收益或价格列）。config['column'] 可指定。"

    # time-order to match GARCH idiom (ordering matters for prices→log-returns)
    d2 = df.sort_values(fp.time_col) if (fp.time_col and fp.time_col in df.columns) else df
    raw = d2[name].astype(float).dropna().to_numpy()
    if raw.size < 2:
        return None, None, None, None, f"序列 {name} 有效观测过少（<2）。"

    # PRICES vs RETURNS: config is_returns forces; else infer. Prices are all-positive
    # LEVELS — either visibly trending (|corr with time| high) OR a near-random-walk
    # (lag-1 autocorrelation ≈ 1). A weakly-drifting price can have low trend-corr yet
    # ρ₁≈0.99, whereas returns are ~white (ρ₁≈0); so we accept all>0 AND (|trend_r|≥0.30
    # OR ρ₁≥0.9) — the autocorrelation test catches level series the trend test misses.
    forced = cfg.get("is_returns")
    if isinstance(forced, bool):
        used_returns = forced
        inferred_price = not forced
    else:
        idx = np.arange(raw.size)
        trend_r = abs(np.corrcoef(idx, raw)[0, 1]) if np.std(raw) > 0 else 0.0
        lag1_ac = (
            abs(np.corrcoef(raw[:-1], raw[1:])[0, 1])
            if raw.size > 2 and np.std(raw) > 0 else 0.0
        )
        inferred_price = bool(np.all(raw > 0) and (trend_r >= 0.30 or lag1_ac >= 0.90))
        used_returns = not inferred_price

    note = ""
    if inferred_price:
        if np.any(raw <= 0):
            return None, None, None, None, (
                f"序列 {name} 被判为价格但含非正值，无法取对数收益；"
                "若它本就是收益请设 config['is_returns']=true。"
            )
        y = np.diff(np.log(raw))  # log-returns r_t = ln(P_t / P_{t-1})
        note = (
            f"（⚠ 序列 {name} 被判为价格[全正且呈水平/趋势特征：趋势相关高或近随机游走]，已转为对数收益 "
            f"r_t=ln(P_t/P_{{t-1}})，n {raw.size}→{y.size}；若它本就是收益请设 "
            "config['is_returns']=true）"
        )
    else:
        y = raw

    y = y[np.isfinite(y)]
    if y.size < min_n:
        return None, None, None, None, (
            f"序列 {name} 收益观测不足（{y.size}<{min_n}），无法稳定估计。"
        )
    return y, name, used_returns, note, None


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


def _gpd_tail_var(u: float, sigma: float, xi: float, ratio: float, p: float) -> float:
    """GPD peaks-over-threshold tail VaR at level p (McNeil-Frey-Embrechts QRM 7.18).
    ``ratio = n / Nu``. The xi->0 branch is the EXACT limit of the general formula
    (since (a^(-xi)-1)/xi -> -ln a with a = ratio*(1-p)); a regression test pins it.
    """
    import numpy as np

    if abs(xi) < 1e-8:
        return u - sigma * np.log(ratio * (1.0 - p))
    return u + (sigma / xi) * (((ratio * (1.0 - p)) ** (-xi)) - 1.0)


# ===========================================================================
# 1) value_at_risk — VaR + Expected Shortfall (CVaR)
#    Refs: Jorion (2007) "Value at Risk"; Cornish & Fisher (1938) expansion;
#          Artzner et al. (1999) on coherence (ES coherent, VaR not).
# ===========================================================================
@register("value_at_risk")
def _branch_value_at_risk(ctx: Ctx) -> None:
    d, files, summary, estimates, code = ctx.d, ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd
    from scipy import stats

    y, name, used_returns, note, err = _resolve_series(ctx, min_n=30)
    if err is not None:
        summary.append(f"VaR/ES 失败：{err}")
        return
    try:
        # LOSS-SIGN convention: loss = -return (losses positive, gains negative).
        losses = -y
        n = losses.size
        mu_l = float(losses.mean())
        sd_l = float(losses.std(ddof=1))
        # sample skewness / excess kurtosis OF THE LOSSES (Fisher: excess, normal→0)
        S = float(stats.skew(losses, bias=False))
        K = float(stats.kurtosis(losses, fisher=True, bias=False))

        # alpha confidence levels (default 0.95, 0.99). VaR_alpha = quantile at alpha
        # of the loss distribution; tail probability p = 1 - alpha.
        alphas = ctx.cfg.get("alpha", [0.95, 0.99])
        if isinstance(alphas, (int, float)):
            alphas = [float(alphas)]
        alphas = sorted({float(a) for a in alphas if 0.0 < float(a) < 1.0})
        if not alphas:
            alphas = [0.95, 0.99]

        rows = []
        for a in alphas:
            # z is the standard-normal quantile at the confidence level alpha
            z = float(stats.norm.ppf(a))

            # (a) HISTORICAL: empirical quantile of the loss sample
            var_hist = float(np.quantile(losses, a))
            tail_h = losses[losses >= var_hist]
            es_hist = float(tail_h.mean()) if tail_h.size else var_hist

            # (b) PARAMETRIC GAUSSIAN: VaR = mu + z*sigma.
            # ES (Gaussian) = mu + sigma * phi(z) / (1-alpha)  [closed form]
            var_gauss = mu_l + z * sd_l
            es_gauss = mu_l + sd_l * float(stats.norm.pdf(z)) / (1.0 - a)

            # (c) CORNISH-FISHER modified VaR: adjust the z-quantile for skew S &
            # excess kurtosis K, then VaR = mu + z_cf * sigma.
            #   z_cf = z + (z^2-1) S/6 + (z^3-3z) K/24 - (2 z^3 - 5 z) S^2/36
            z_cf = (
                z
                + (z * z - 1.0) * S / 6.0
                + (z * z * z - 3.0 * z) * K / 24.0
                - (2.0 * z * z * z - 5.0 * z) * S * S / 36.0
            )
            var_cf = mu_l + z_cf * sd_l
            # ES under Cornish-Fisher: mean loss beyond the CF VaR using the
            # empirical tail (coherent, data-anchored).
            tail_cf = losses[losses >= var_cf]
            es_cf = float(tail_cf.mean()) if tail_cf.size else var_cf

            rows.append({
                "alpha": a,
                "var_historical": round(var_hist, 6),
                "es_historical": round(es_hist, 6),
                "var_gaussian": round(var_gauss, 6),
                "es_gaussian": round(es_gauss, 6),
                "var_cornish_fisher": round(var_cf, 6),
                "es_cornish_fisher": round(es_cf, 6),
                "z": round(z, 4),
                "z_cf": round(z_cf, 4),
                "n_tail_obs": int(tail_h.size),
            })

        tbl = pd.DataFrame(rows)
        tbl.to_csv(d / "var_es.csv", index=False, encoding="utf-8")
        files.append("var_es.csv")

        # flat estimates dict (one alpha → suffix _NN where NN = round(alpha*100))
        for r in rows:
            sfx = f"_{int(round(r['alpha'] * 100))}"
            estimates.update({
                f"var_hist{sfx}": r["var_historical"],
                f"es_hist{sfx}": r["es_historical"],
                f"var_gauss{sfx}": r["var_gaussian"],
                f"es_gauss{sfx}": r["es_gaussian"],
                f"var_cf{sfx}": r["var_cornish_fisher"],
                f"es_cf{sfx}": r["es_cornish_fisher"],
            })
        estimates.update({
            "skewness": round(S, 4), "excess_kurtosis": round(K, 4),
            "mean_loss": round(mu_l, 6), "sd_loss": round(sd_l, 6), "n_obs": float(n),
        })

        # distribution plot with VaR lines (returns axis: VaR drawn at -VaR on returns)
        def _plot(plt):
            fig, ax = plt.subplots(figsize=(8, 4.0))
            ax.hist(y, bins=min(60, max(10, n // 10)), color="#bbbbbb",
                    edgecolor="white", alpha=0.85)
            colors = {"var_historical": "#4C72B0", "var_gaussian": "#55A868",
                      "var_cornish_fisher": "#C44E52"}
            r0 = rows[-1]  # most extreme alpha
            for key, col in colors.items():
                ax.axvline(-r0[key], color=col, ls="--", lw=1.4,
                           label=f"{key.replace('var_', 'VaR ')} (a={r0['alpha']})")
            ax.set_xlabel(f"{name} returns")
            ax.set_ylabel("frequency")
            ax.set_title("Return distribution with VaR thresholds (losses on left)")
            ax.legend(fontsize=8)

        _save_fig(d, "var_distribution.png", files, _plot)

        worst = max(rows, key=lambda r: r["alpha"])
        a_txt = "、".join(f"{r['alpha']:.0%}" for r in rows)
        summary.append(
            f"{ctx.entry.method} 完成：序列 {name}（n={n}，损失符号约定=损失为正）{note}；"
            f"置信水平 {a_txt}。在最高水平 α={worst['alpha']:.0%}："
            f"历史 VaR={worst['var_historical']:.4g}/ES={worst['es_historical']:.4g}，"
            f"高斯 VaR={worst['var_gaussian']:.4g}/ES={worst['es_gaussian']:.4g}，"
            f"Cornish-Fisher VaR={worst['var_cornish_fisher']:.4g}/ES={worst['es_cornish_fisher']:.4g}"
            f"（偏度 S={S:.3f}、超额峰度 K={K:.3f}）。三法对比见 var_es.csv 与图。"
            " ⚠ VaR 是分位数、非一致性风险度量（不满足次可加性）；ES(CVaR) 是一致的，超额损失的均值。"
            " ⚠ 高斯 VaR 对厚尾收益低估尾部风险（Cornish-Fisher 用偏度/峰度部分修正）；"
            "历史 VaR 受样本最坏观测限制（无法外推到样本之外）。"
        )
        code += [
            "import numpy as np; from scipy import stats",
            "losses = -returns  # 损失符号: 损失为正",
            "z = stats.norm.ppf(alpha)",
            "var_hist = np.quantile(losses, alpha)            # 历史法",
            "var_gauss = losses.mean() + z*losses.std(ddof=1) # 参数高斯法",
            "z_cf = z + (z**2-1)*S/6 + (z**3-3*z)*K/24 - (2*z**3-5*z)*S**2/36  # Cornish-Fisher",
            "var_cf = losses.mean() + z_cf*losses.std(ddof=1)",
            "es = losses[losses>=var].mean()  # ES/CVaR = 超额损失均值",
        ]
    except Exception as exc:
        summary.append(f"VaR/ES 计算失败：{exc}")


# ===========================================================================
# 2) extreme_value — EVT, Peaks-Over-Threshold (GPD)
#    Refs: Pickands (1975); Balkema-de Haan; McNeil-Frey-Embrechts (2005)
#          "Quantitative Risk Management" §7 (POT / GPD tail estimator).
# ===========================================================================
@register("extreme_value")
def _branch_extreme_value(ctx: Ctx) -> None:
    d, files, summary, estimates, code = ctx.d, ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd
    from scipy import stats

    y, name, used_returns, note, err = _resolve_series(ctx, min_n=50)
    if err is not None:
        summary.append(f"极值理论(EVT) 失败：{err}")
        return
    try:
        losses = -y  # loss-sign convention: model the upper tail of losses
        n = losses.size

        # threshold u: config `threshold` (absolute loss level) or 95th pct of losses.
        thr_cfg = ctx.cfg.get("threshold")
        if isinstance(thr_cfg, (int, float)):
            u = float(thr_cfg)
        else:
            q = float(ctx.cfg.get("threshold_quantile", 0.95))
            q = q if 0.5 <= q < 1.0 else 0.95
            u = float(np.quantile(losses, q))

        exceed = losses[losses > u]
        excesses = exceed - u  # peaks over threshold (y = loss - u > 0)
        Nu = excesses.size
        if Nu < 30:
            summary.append(
                f"极值理论(EVT) 失败：阈值 u={u:.4g} 上的超阈观测仅 {Nu} 个（<30，不足以稳定拟合 GPD）。"
                " 降低 config['threshold'] 或 config['threshold_quantile']。"
            )
            return

        # Fit GPD to the EXCESSES with loc fixed at 0 (excesses are >=0).
        # scipy.stats.genpareto: shape c == xi; scale == sigma. floc=0.
        xi, loc, sigma = stats.genpareto.fit(excesses, floc=0.0)
        xi = float(xi)
        sigma = float(sigma)

        # goodness-of-fit: KS test of the excesses against the fitted GPD
        ks_stat, ks_p = stats.kstest(excesses, "genpareto", args=(xi, 0.0, sigma))
        ks_stat, ks_p = float(ks_stat), float(ks_p)

        # EVT tail VaR / return levels at extreme alphas via the GPD tail formula
        #   VaR_p = u + (sigma/xi) * [ ((n / Nu) * (1 - p))^(-xi) - 1 ]    (xi != 0)
        #   VaR_p = u - sigma * ln( (n / Nu) * (1 - p) )                   (xi -> 0 limit:
        #     since (a^(-xi)-1)/xi -> -ln a with a=(n/Nu)(1-p); == u + sigma*ln((Nu/n)/(1-p)))
        ratio = n / Nu
        evt_alphas = ctx.cfg.get("evt_alpha", [0.99, 0.999])
        if isinstance(evt_alphas, (int, float)):
            evt_alphas = [float(evt_alphas)]
        evt_alphas = sorted({float(a) for a in evt_alphas if 0.0 < float(a) < 1.0})
        if not evt_alphas:
            evt_alphas = [0.99, 0.999]

        rows = []
        for p in evt_alphas:
            var_p = _gpd_tail_var(u, sigma, xi, ratio, p)
            # GPD-based ES (xi<1): ES_p = VaR_p/(1-xi) + (sigma - xi*u)/(1-xi)
            es_p = (var_p / (1.0 - xi) + (sigma - xi * u) / (1.0 - xi)) if xi < 1.0 else float("nan")
            rows.append({"alpha": p, "evt_var": round(float(var_p), 6),
                         "evt_es": round(float(es_p), 6) if es_p == es_p else float("nan")})

        pd.DataFrame(rows).to_csv(d / "evt_return_levels.csv", index=False, encoding="utf-8")
        files.append("evt_return_levels.csv")

        for r in rows:
            sfx = f"_{int(round(r['alpha'] * 1000))}"  # 0.99→990, 0.999→999
            estimates[f"evt_var{sfx}"] = r["evt_var"]
            estimates[f"evt_es{sfx}"] = r["evt_es"]
        estimates.update({
            "xi_shape": round(xi, 4), "sigma_scale": round(sigma, 6),
            "threshold_u": round(u, 6), "n_exceedances": float(Nu),
            "ks_stat": round(ks_stat, 4), "ks_pvalue": round(ks_p, 4), "n_obs": float(n),
        })

        # mean-excess plot: e(v) = mean(loss - v | loss > v) over a grid of v;
        # a GPD tail is approximately LINEAR in v (slope sign tracks xi).
        def _plot(plt):
            fig, ax = plt.subplots(1, 2, figsize=(10, 4.0))
            sl = np.sort(losses)
            vs = sl[(sl >= np.quantile(losses, 0.50)) & (sl < sl[-1])]
            uniq = np.unique(vs)
            step = max(1, len(uniq) // 60)
            vs = uniq[::step]
            me = [float((losses[losses > v] - v).mean()) for v in vs if (losses > v).any()]
            vs = [v for v in vs if (losses > v).any()]
            ax[0].plot(vs, me, color="#4C72B0", lw=1.2)
            ax[0].axvline(u, color="#C44E52", ls="--", lw=1.2, label=f"threshold u={u:.3g}")
            ax[0].set_xlabel("threshold v (loss)")
            ax[0].set_ylabel("mean excess e(v)")
            ax[0].set_title("Mean-excess plot (linear -> GPD tail)")
            ax[0].legend(fontsize=8)
            # exceedance fit: empirical vs fitted GPD density on the excesses
            xs = np.linspace(0, excesses.max(), 200)
            ax[1].hist(excesses, bins=min(40, max(8, Nu // 8)), density=True,
                       color="#bbbbbb", edgecolor="white", alpha=0.85, label="excesses")
            ax[1].plot(xs, stats.genpareto.pdf(xs, xi, 0.0, sigma),
                       color="#C44E52", lw=1.6, label=f"GPD fit xi={xi:.3f}")
            ax[1].set_xlabel("excess = loss - u")
            ax[1].set_ylabel("density")
            ax[1].set_title(f"GPD fit to {Nu} exceedances")
            ax[1].legend(fontsize=8)

        _save_fig(d, "evt_fit.png", files, _plot)

        tail_txt = ("ξ>0：重尾(Fréchet 吸引域)" if xi > 0.02 else
                    ("ξ<0：有界尾(短尾)" if xi < -0.02 else "ξ≈0：轻尾(Gumbel/指数)"))
        ks_note = ("；⚠ KS 拒绝 GPD 拟合(p<0.05)，阈值/分布或不合适" if ks_p < 0.05 else "")
        lvl_txt = "、".join(f"VaR_{r['alpha']:.1%}={r['evt_var']:.4g}" for r in rows)
        summary.append(
            f"{ctx.entry.method} 完成：序列 {name}（n={n}）POT 法，阈值 u={u:.4g}、超阈 Nu={Nu}{note}；"
            f"GPD 形状 ξ={xi:.3f}（{tail_txt}）、尺度 σ={sigma:.4g}；KS 拟合优度 p={ks_p:.3g}{ks_note}。"
            f"EVT 尾部风险水平：{lvl_txt}（见 evt_return_levels.csv 与图）。"
            " ⚠ 阈值选择是关键敏感点（过低→偏差，过高→方差；均值超额图近线性段指引选取）；"
            "需足够超阈观测（≥~30）；EVT 向尾部数据之外外推，越深不确定性越大；"
            "假定超阈值独立同分布（未做序列极值的去丛聚/declustering）。"
        )
        code += [
            "from scipy import stats; import numpy as np",
            "losses = -returns; u = np.quantile(losses, 0.95)  # 阈值",
            "excesses = losses[losses>u] - u                    # 超阈值",
            "xi, _, sigma = stats.genpareto.fit(excesses, floc=0.0)  # GPD 拟合",
            "VaR_p = u + (sigma/xi)*(((n/Nu)*(1-p))**(-xi) - 1)  # GPD 尾部 VaR",
        ]
    except Exception as exc:
        summary.append(f"极值理论(EVT) 计算失败：{exc}")


# ===========================================================================
# 3) risk_adjusted_return — Sharpe / Sortino / max drawdown / Calmar
#    Refs: Sharpe (1966, 1994); Sortino & van der Meer (1991); Young (1991, Calmar).
# ===========================================================================
@register("risk_adjusted_return")
def _branch_risk_adjusted_return(ctx: Ctx) -> None:
    d, files, summary, estimates, code = ctx.d, ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    y, name, used_returns, note, err = _resolve_series(ctx, min_n=20)
    if err is not None:
        summary.append(f"风险调整收益 失败：{err}")
        return
    try:
        r = np.asarray(y, dtype=float)
        n = r.size
        ppy = ctx.cfg.get("periods_per_year", 252)
        try:
            ppy = float(ppy)
            if ppy <= 0:
                ppy = 252.0
        except (TypeError, ValueError):
            ppy = 252.0
        rf = ctx.cfg.get("rf", 0.0)  # risk-free per PERIOD
        try:
            rf = float(rf)
        except (TypeError, ValueError):
            rf = 0.0

        mu = float(r.mean())
        sd = float(r.std(ddof=1))
        # annualization (iid assumption): mean*ppy, vol*sqrt(ppy)
        ann_return = mu * ppy
        ann_vol = sd * np.sqrt(ppy)

        # Sharpe = (mean excess return / sd) annualized by sqrt(ppy).
        excess = r - rf
        sharpe = (float(excess.mean()) / sd * np.sqrt(ppy)) if sd > 0 else float("nan")

        # Sortino: downside deviation = sqrt(mean(min(excess,0)^2)) (target=rf,
        # full-sample denominator — target-semideviation convention).
        downside = np.minimum(excess, 0.0)
        dd_dev = float(np.sqrt(np.mean(downside ** 2)))
        sortino = (float(excess.mean()) / dd_dev * np.sqrt(ppy)) if dd_dev > 0 else float("nan")

        # cumulative wealth curve (compounded) and maximum drawdown.
        wealth = np.cumprod(1.0 + r)
        running_max = np.maximum.accumulate(wealth)
        drawdown = wealth / running_max - 1.0  # <= 0
        max_dd = float(drawdown.min())  # most negative
        # Calmar = annualized return / |max drawdown|
        calmar = (ann_return / abs(max_dd)) if max_dd < 0 else float("nan")
        total_return = float(wealth[-1] - 1.0)

        estimates.update({
            "ann_return": round(ann_return, 6),
            "ann_volatility": round(ann_vol, 6),
            "sharpe": round(sharpe, 4) if sharpe == sharpe else float("nan"),
            "sortino": round(sortino, 4) if sortino == sortino else float("nan"),
            "max_drawdown": round(max_dd, 6),
            "calmar": round(calmar, 4) if calmar == calmar else float("nan"),
            "total_return": round(total_return, 6),
            "downside_deviation": round(dd_dev, 6),
            "mean_return": round(mu, 6), "sd_return": round(sd, 6),
            "periods_per_year": ppy, "rf_per_period": rf, "n_obs": float(n),
        })

        pd.DataFrame({
            "period": np.arange(n),
            "return": np.round(r, 6),
            "wealth": np.round(wealth, 6),
            "drawdown": np.round(drawdown, 6),
        }).to_csv(d / "risk_metrics.csv", index=False, encoding="utf-8")
        files.append("risk_metrics.csv")

        def _plot(plt):
            fig, ax = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
            ax[0].plot(wealth, color="#4C72B0", lw=1.3, label="cumulative wealth (1+r) compounded")
            ax[0].plot(running_max, color="#999999", lw=0.8, ls="--", label="running peak")
            ax[0].set_ylabel("wealth")
            ax[0].set_title(f"Cumulative return & drawdown — {name}")
            ax[0].legend(fontsize=8)
            ax[1].fill_between(np.arange(n), drawdown, 0.0, color="#C44E52", alpha=0.6)
            ax[1].set_ylabel("drawdown")
            ax[1].set_xlabel("period index")
            ax[1].axhline(max_dd, color="#8B0000", ls=":", lw=1.0,
                          label=f"max DD = {max_dd:.2%}")
            ax[1].legend(fontsize=8)

        _save_fig(d, "drawdown.png", files, _plot)

        def _fmt(x):
            return f"{x:.3f}" if x == x else "NA"

        summary.append(
            f"{ctx.entry.method} 完成：序列 {name}（n={n}，每年 {ppy:g} 期，rf={rf:g}/期）{note}；"
            f"年化收益={ann_return:.3%}、年化波动={ann_vol:.3%}；"
            f"Sharpe={_fmt(sharpe)}、Sortino={_fmt(sortino)}；"
            f"最大回撤={max_dd:.2%}、Calmar={_fmt(calmar)}；累计收益={total_return:.2%}。"
            "（指标表见 risk_metrics.csv，回撤图见 drawdown.png）"
            " ⚠ Sharpe 假定收益近正态（Sortino 用下行偏差、Calmar 用最大回撤补充刻画下行/回撤风险）；"
            "年化用 √T 缩放假定收益独立同分布（自相关下会失真）；rf 与 periods_per_year 为假定值；"
            "样本内指标偏乐观（此处未做 Deflated-Sharpe 多重检验校正）。"
        )
        code += [
            "import numpy as np",
            "mu, sd = returns.mean(), returns.std(ddof=1)",
            "sharpe = (returns - rf).mean()/sd * np.sqrt(ppy)",
            "dd = np.sqrt(np.mean(np.minimum(returns-rf,0)**2))",
            "sortino = (returns-rf).mean()/dd * np.sqrt(ppy)",
            "w = np.cumprod(1+returns); max_dd = (w/np.maximum.accumulate(w)-1).min()",
            "calmar = (mu*ppy)/abs(max_dd)",
        ]
    except Exception as exc:
        summary.append(f"风险调整收益 计算失败：{exc}")
