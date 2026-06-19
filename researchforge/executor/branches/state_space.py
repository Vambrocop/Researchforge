"""Branch handlers for the state-space / regime time-series family.

Three statsmodels state-space / regime models, each a real-inference model (the risk
is correct model construction + result extraction + honest disclosure):

- unobserved_components  -> statsmodels.tsa.statespace.structural.UnobservedComponents
- markov_switching       -> statsmodels.tsa.regime_switching.markov_regression.MarkovRegression
                            (or MarkovAutoregression when config order>0)
- dynamic_factor         -> statsmodels.tsa.statespace.dynamic_factor.DynamicFactor

See executor/_branch_api.py for the Ctx / register contract. This NEW family file
auto-registers via branches/__init__ walk_packages — no edits elsewhere.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


def _periodogram_period(x, n):
    """Dominant seasonal period via the periodogram, or None if no SIGNIFICANT periodicity.

    Linearly detrends first (a trend's low-frequency power otherwise dominates), requires >=3
    cycles (period <= n/3), and applies Fisher's g-test (alpha=0.05) so pure noise/trend -> None.
    (Mirrors timeseries._periodogram_period; kept local so this family is self-contained.)
    """
    import numpy as np

    x = np.asarray(x, dtype=float)
    idx = np.arange(n)
    c = np.polyfit(idx, x, 1)  # remove linear trend
    x = x - (c[0] * idx + c[1])
    if np.std(x) == 0:
        return None
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(n)
    mask = freqs >= 3.0 / n  # candidate seasonal freqs (period <= n/3)
    if not mask.any():
        return None
    pm = power[mask]
    m = len(pm)
    if m < 2 or pm.sum() <= 0:
        return None
    g = float(pm.max() / pm.sum())  # Fisher's g statistic
    g_crit = 1.0 - (0.05 / m) ** (1.0 / (m - 1))  # alpha=0.05 critical value
    if g <= g_crit:  # no significant periodicity
        return None
    freq = freqs[mask][int(np.argmax(pm))]
    if freq <= 0:
        return None
    per = int(round(1.0 / freq))
    return per if 2 <= per <= n // 3 else None


@register("unobserved_components")
def _branch_unobserved_components(ctx: Ctx) -> None:
    # Unobserved-components (structural time series) state-space model:
    # statsmodels.tsa.statespace.structural.UnobservedComponents decomposes a single series into
    # unobserved stochastic level / trend (slope) / seasonal + irregular components, with INFERENCE
    # (a model-based STL alternative). The estimated component VARIANCES tell which components are
    # stochastic (sigma2 > 0) vs effectively deterministic (sigma2 ~= 0).
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np

    _excl = {fp.unit_col, fp.time_col}
    value = cfg.get("column") if cfg.get("column") in df.columns else next(
        (c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl), None)
    if value is None:
        summary.append("UC（无观测分量）失败：需要一个连续序列。config['column'] 可指定。")
        return
    try:
        import pandas as pd
        from statsmodels.tsa.statespace.structural import UnobservedComponents

        d2 = df.sort_values(fp.time_col) if (fp.time_col and fp.time_col in df.columns) else df
        y = d2[value].astype(float).dropna().reset_index(drop=True)
        n = len(y)
        if n < 30 or y.nunique() < 5:
            summary.append("UC（无观测分量）失败：观测不足（<30）或近常数序列——UC 需较长序列。")
            return

        # --- spec selection ---------------------------------------------------
        # level: default 'local linear trend' = stochastic level + stochastic slope (most general
        # smooth-trend spec). config['level'] overrides (e.g. 'local level', 'smooth trend').
        level = cfg.get("level") or "local linear trend"
        # 'trend' kept as an accepted alias key but the slope already lives inside the level spec;
        # we expose stochastic-slope info via the estimated trend (slope) variance below.
        # seasonal: only added if a period is supplied or auto-detected.
        seasonal = None
        seas_src = "无"
        try:
            sp = cfg.get("seasonal_period")
            seasonal = int(sp) if sp is not None and 2 <= int(sp) <= n // 2 else None
            if seasonal:
                seas_src = f"config 指定={seasonal}"
        except (TypeError, ValueError):
            seasonal = None
        if seasonal is None and cfg.get("seasonal", True) is not False:
            auto = _periodogram_period(y.to_numpy(), n)
            if auto and 2 <= auto <= n // 2:
                seasonal, seas_src = auto, f"周期图自动检出={auto}"

        kwargs = dict(level=level)
        if seasonal:
            kwargs["seasonal"] = seasonal

        model = UnobservedComponents(y.to_numpy(), **kwargs)
        # disp=False; cov_type 'oim' is robust; bump maxiter for the seasonal spec which is heavier.
        res = model.fit(disp=False, maxiter=int(cfg.get("maxiter", 200)))

        # --- component variances (the diagnostic of WHICH components are stochastic) -----------
        # statsmodels names the variance params 'sigma2.<component>': sigma2.irregular,
        # sigma2.level, sigma2.trend (the slope), sigma2.seasonal. Pull by name (not by index) so
        # we never grab the wrong cell when a component is absent.
        params = res.params
        names = list(res.param_names)
        def _var(key):
            for nm in names:
                if nm == f"sigma2.{key}" or nm.endswith(f"sigma2.{key}"):
                    return float(params[names.index(nm)])
            return float("nan")

        var_irregular = _var("irregular")
        var_level = _var("level")
        var_trend = _var("trend")
        var_seasonal = _var("seasonal") if seasonal else float("nan")

        # smoothed components (level/trend/seasonal) for the plot + CSV
        sm = res.smoothed_state  # rows = states in canonical order: level, [trend], [seasonal...]
        level_hat = np.asarray(sm[0], dtype=float)
        trend_hat = None
        if level == "local linear trend" or "trend" in str(level).lower():
            # for 'local linear trend' the slope state is row 1
            if sm.shape[0] > 1:
                trend_hat = np.asarray(sm[1], dtype=float)
        # seasonal smoothed component (sum of seasonal dummy states): use the model's filtered
        # decomposition via get_smoothed -> fall back to reconstructing from states if needed.
        seasonal_hat = None
        if seasonal:
            try:
                # the seasonal component contribution = design row for seasonal states dotted in;
                # statsmodels exposes it via .seasonal smoothed attribute on the results.
                seas_res = getattr(res, "seasonal", None)
                if seas_res is not None and getattr(seas_res, "smoothed", None) is not None:
                    seasonal_hat = np.asarray(seas_res.smoothed, dtype=float)
            except Exception:
                seasonal_hat = None

        estimates.update({
            "var_irregular": round(var_irregular, 6) if var_irregular == var_irregular else float("nan"),
            "var_level": round(var_level, 6) if var_level == var_level else float("nan"),
            "var_trend": round(var_trend, 6) if var_trend == var_trend else float("nan"),
            "var_seasonal": round(var_seasonal, 6) if var_seasonal == var_seasonal else float("nan"),
            "aic": round(float(res.aic), 2), "bic": round(float(res.bic), 2),
            "loglik": round(float(res.llf), 2), "n_obs": float(n),
            "has_seasonal": 1.0 if seasonal else 0.0,
        })

        # CSV of smoothed components
        cols = {"index": range(n), "observed": np.round(y.to_numpy(), 4),
                "level": np.round(level_hat, 4)}
        if trend_hat is not None:
            cols["trend_slope"] = np.round(trend_hat, 4)
        if seasonal_hat is not None and len(seasonal_hat) == n:
            cols["seasonal"] = np.round(seasonal_hat, 4)
        pd.DataFrame(cols).to_csv(d / "uc_components.csv", index=False, encoding="utf-8")
        files.append("uc_components.csv")
        (d / "uc_summary.txt").write_text(str(res.summary()), encoding="utf-8")
        files.append("uc_summary.txt")

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            npanel = 2 + (1 if (seasonal_hat is not None and len(seasonal_hat) == n) else 0)
            fig, axes = plt.subplots(npanel, 1, figsize=(8, 2.3 * npanel), sharex=True)
            axes = np.atleast_1d(axes)
            axes[0].plot(y.to_numpy(), color="#999999", lw=0.9, label="observed")
            axes[0].plot(level_hat, color="#4C72B0", lw=1.6, label="smoothed level")
            axes[0].legend(fontsize=8)
            axes[0].set_ylabel(value, fontsize=9)
            if trend_hat is not None:
                axes[1].plot(trend_hat, color="#55A868", lw=1.4, label="smoothed trend (slope)")
            else:
                axes[1].plot(level_hat, color="#55A868", lw=1.4, label="smoothed level")
            axes[1].axhline(0.0, color="grey", ls="--", lw=0.8)
            axes[1].legend(fontsize=8)
            axes[1].set_ylabel("trend", fontsize=9)
            if npanel == 3:
                axes[2].plot(seasonal_hat, color="#C44E52", lw=1.2, label="smoothed seasonal")
                axes[2].legend(fontsize=8)
                axes[2].set_ylabel("seasonal", fontsize=9)
            axes[-1].set_xlabel("period index")
            axes[0].set_title(f"Unobserved Components — {value} (level={level}"
                              + (f", seasonal={seasonal}" if seasonal else "") + ")")
            fig.tight_layout()
            fig.savefig(d / "uc_components.png", dpi=150)
            plt.close(fig)
            files.append("uc_components.png")
        except Exception:
            pass

        # interpretation: a component variance ~0 means that component is effectively DETERMINISTIC.
        thr = 1e-7 * max(float(np.var(y.to_numpy())), 1e-12)
        def _det(v):
            return (v == v) and v <= thr
        det_notes = []
        if _det(var_level):
            det_notes.append("水平近确定性")
        if trend_hat is not None and _det(var_trend):
            det_notes.append("斜率近确定性")
        if seasonal and _det(var_seasonal):
            det_notes.append("季节近确定性")
        det_txt = ("；".join(det_notes) + "（对应 σ²≈0）" ) if det_notes else "各分量均含随机成分"

        summary.append(
            f"{entry.method} 完成：{value}（n={n}）UC 状态空间分解，spec level={level}"
            + (f"、seasonal={seasonal}（{seas_src}）" if seasonal else "（无季节项）") + "；"
            f"分量方差 σ²: 水平={var_level:.4g}、斜率="
            f"{(trend_hat is not None and f'{var_trend:.4g}') or '—'}、"
            f"季节={(seasonal and f'{var_seasonal:.4g}') or '—'}、不规则={var_irregular:.4g}；"
            f"{det_txt}。AIC={res.aic:.1f}、BIC={res.bic:.1f}、loglik={res.llf:.1f}；"
            "平滑分量见 uc_components.csv 与图。"
            " ⚠ UC 把序列分解为无观测的随机分量（带推断的 STL 替代）；spec（哪些分量随机）很关键，"
            "由方差判断（σ²≈0 ⇒ 该分量近似确定性）；需较长序列；假定高斯状态空间（线性+正态新息）。"
        )
        code += [
            "from statsmodels.tsa.statespace.structural import UnobservedComponents  # UC 结构时序",
            f"# UnobservedComponents(y, level='{level}'"
            + (f", seasonal={seasonal}" if seasonal else "") + ").fit()",
            "# 分量方差 res.params['sigma2.level'/'sigma2.trend'/'sigma2.seasonal'/'sigma2.irregular']",
        ]
    except Exception as err:
        summary.append(f"UC（无观测分量）拟合失败：{err}")


@register("markov_switching")
def _branch_markov_switching(ctx: Ctx) -> None:
    # Markov-switching regime model:
    # statsmodels.tsa.regime_switching.markov_regression.MarkovRegression (or MarkovAutoregression
    # when config order>0). The series mean (and, by default, variance) switches between k latent
    # regimes governed by a hidden Markov chain. We report per-regime means/variances, the transition
    # matrix, expected durations 1/(1-p_ii), and the smoothed regime probabilities.
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np

    _excl = {fp.unit_col, fp.time_col}
    value = cfg.get("column") if cfg.get("column") in df.columns else next(
        (c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl), None)
    if value is None:
        summary.append("马尔可夫切换失败：需要一个连续序列。config['column'] 可指定。")
        return
    try:
        import pandas as pd
        from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

        d2 = df.sort_values(fp.time_col) if (fp.time_col and fp.time_col in df.columns) else df
        y = d2[value].astype(float).dropna().reset_index(drop=True)
        n = len(y)
        try:
            k = int(cfg.get("k_regimes", 2))
        except (TypeError, ValueError):
            k = 2
        k = max(2, min(k, 4))
        if n < 30 * k // 2 or y.nunique() < 5:
            summary.append(f"马尔可夫切换失败：观测不足（n={n}，{k} 个状态需要更多数据）或近常数序列。")
            return
        sw_var = cfg.get("switching_variance", True) is not False
        try:
            order = int(cfg.get("order", 0))
        except (TypeError, ValueError):
            order = 0

        yv = y.to_numpy()
        if order > 0:
            model = MarkovAutoregression(yv, k_regimes=k, order=order,
                                         switching_variance=sw_var)
            mclass = "MarkovAutoregression"
        else:
            model = MarkovRegression(yv, k_regimes=k, switching_variance=sw_var)
            mclass = "MarkovRegression"
        # local optima are common -> fit with statsmodels' EM-based search (multiple starts via
        # search_reps). We assert on robust quantities (orderings) downstream, not exact params.
        res = model.fit(search_reps=int(cfg.get("search_reps", 20)),
                        maxiter=int(cfg.get("maxiter", 100)), disp=False)

        # --- per-regime means: the intercept 'const[i]' (mean regime i). switching variance ->
        # 'sigma2[i]'. Pull by name. MarkovRegressionResults has no `param_names` attr (unlike the
        # statespace results) — the canonical names live on the MODEL. -------------------------
        pnames = list(res.model.param_names)
        pvals = np.asarray(res.params, dtype=float)
        def _by_prefix(prefix):
            out = {}
            for nm, v in zip(pnames, pvals):
                if nm.startswith(prefix):
                    # trailing index e.g. 'const[0]'
                    try:
                        i = int(nm[nm.index("[") + 1:nm.index("]")])
                    except Exception:
                        continue
                    out[i] = float(v)
            return out
        means = _by_prefix("const")
        if not means:  # AR spec names the intercept differently in some versions
            means = _by_prefix(f"{value}")
        variances = _by_prefix("sigma2")

        regime_mean = np.array([means.get(i, float("nan")) for i in range(k)], dtype=float)
        # variance: if not switching there is a single sigma2 -> broadcast
        if sw_var and len(variances) >= k:
            regime_var = np.array([variances.get(i, float("nan")) for i in range(k)], dtype=float)
        else:
            single = next(iter(variances.values()), float("nan"))
            regime_var = np.full(k, single)

        # --- LABEL-SWITCHING fix: regime identity is arbitrary. Order regimes by mean ascending
        # and remap EVERYTHING (transition matrix rows/cols, smoothed probs) consistently. -----
        order_idx = np.argsort(regime_mean)
        remap = {old: new for new, old in enumerate(order_idx)}
        regime_mean_s = regime_mean[order_idx]
        regime_var_s = regime_var[order_idx]

        # transition matrix: res.regime_transition has shape (k, k[, nobs]); element [i, j] is
        # P(next=i | curr=j) i.e. columns are the FROM-state. We transpose to a conventional
        # row=from, col=to matrix, then reorder by the mean-sorted labels.
        rt = np.asarray(res.regime_transition, dtype=float)
        if rt.ndim == 3:
            rt = rt[:, :, 0]
        P_to_from = rt                      # [to, from]
        P = P_to_from.T                      # [from, to]
        P = P[np.ix_(order_idx, order_idx)]  # reorder both axes
        # numerical clean-up: rows should sum to 1
        P = np.clip(P, 0.0, 1.0)
        rowsum = P.sum(axis=1, keepdims=True)
        P = np.where(rowsum > 0, P / rowsum, P)

        durations = np.array([1.0 / (1.0 - P[i, i]) if P[i, i] < 1.0 else float("inf")
                              for i in range(k)])

        # smoothed regime probabilities reordered to the sorted labels. statsmodels returns
        # shape (nobs, k_regimes); orient to (k, nobs). Key on the k-sized axis explicitly so it
        # is unambiguous even in the degenerate nobs==k case (the k axis is the SHORT one here).
        smp = np.asarray(res.smoothed_marginal_probabilities, dtype=float)
        if smp.ndim != 2:
            smp = np.atleast_2d(smp)
        if smp.shape[1] == k and smp.shape[0] != k:
            smp = smp.T                      # (nobs, k) -> (k, nobs)
        elif smp.shape[0] != k and smp.shape[1] == k:
            smp = smp.T
        smp_s = smp[order_idx, :]
        most_likely = np.argmax(smp_s, axis=0)  # 0..k-1 in sorted-label space

        # --- estimates ---
        for i in range(k):
            estimates[f"regime{i}_mean"] = round(float(regime_mean_s[i]), 4)
            estimates[f"regime{i}_var"] = round(float(regime_var_s[i]), 4) if regime_var_s[i] == regime_var_s[i] else float("nan")
            estimates[f"regime{i}_p_stay"] = round(float(P[i, i]), 4)
            estimates[f"regime{i}_expected_duration"] = round(float(durations[i]), 3) if np.isfinite(durations[i]) else float("inf")
        estimates.update({"k_regimes": float(k), "aic": round(float(res.aic), 2),
                          "loglik": round(float(res.llf), 2), "n_obs": float(n)})

        # transition matrix CSV (sorted labels)
        Pdf = pd.DataFrame(np.round(P, 4),
                           index=[f"from_regime{i}" for i in range(k)],
                           columns=[f"to_regime{i}" for i in range(k)])
        Pdf.to_csv(d / "transition_matrix.csv", encoding="utf-8")
        files.append("transition_matrix.csv")
        # smoothed probs CSV
        prob_cols = {"index": range(smp_s.shape[1])}
        for i in range(k):
            prob_cols[f"P_regime{i}"] = np.round(smp_s[i], 4)
        prob_cols["most_likely_regime"] = most_likely
        pd.DataFrame(prob_cols).to_csv(d / "regime_probabilities.csv", index=False, encoding="utf-8")
        files.append("regime_probabilities.csv")
        (d / "markov_summary.txt").write_text(str(res.summary()), encoding="utf-8")
        files.append("markov_summary.txt")

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # AR drops the first `order` obs from probabilities; align the x-axis.
            offset = n - smp_s.shape[1]
            x = np.arange(offset, n)
            palette = ["#4C72B0", "#C44E52", "#55A868", "#CCB974"]
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            ax1.plot(np.arange(n), yv, color="#333333", lw=0.8)
            # shade by most-likely regime
            for t in range(len(x)):
                ax1.axvspan(x[t] - 0.5, x[t] + 0.5,
                            color=palette[int(most_likely[t]) % len(palette)], alpha=0.18, lw=0)
            ax1.set_ylabel(value)
            ax1.set_title(f"Markov switching ({mclass}, k={k}) — series shaded by most-likely regime")
            for i in range(k):
                ax2.plot(x, smp_s[i], color=palette[i % len(palette)], lw=1.2,
                         label=f"regime {i} (mean={regime_mean_s[i]:.2f})")
            ax2.set_ylim(-0.02, 1.02)
            ax2.set_ylabel("smoothed P(regime)")
            ax2.set_xlabel("period index")
            ax2.legend(fontsize=8, ncol=k)
            fig.tight_layout()
            fig.savefig(d / "markov_regimes.png", dpi=150)
            plt.close(fig)
            files.append("markov_regimes.png")
        except Exception:
            pass

        dur_txt = "、".join(
            f"R{i}≈{durations[i]:.1f}期" if np.isfinite(durations[i]) else f"R{i}≈∞"
            for i in range(k))
        mean_txt = "、".join(f"R{i}={regime_mean_s[i]:.3f}" for i in range(k))
        var_txt = "、".join(
            f"R{i}={regime_var_s[i]:.3f}" if regime_var_s[i] == regime_var_s[i] else f"R{i}=—"
            for i in range(k))
        thin = any(int((most_likely == i).sum()) < 5 for i in range(k))
        thin_note = "；⚠ 某状态被分配的期数过少(<5)，该状态参数不稳" if thin else ""
        summary.append(
            f"{entry.method} 完成：{value}（n={n}）拟合 {k}-状态马尔可夫切换（{mclass}"
            + (f", order={order}" if order else "") + f", switching_variance={sw_var}）；"
            f"各状态均值（已按均值升序排列）{mean_txt}；方差 {var_txt}；"
            f"自留概率 p_ii={'、'.join(f'{P[i, i]:.3f}' for i in range(k))}，"
            f"预期驻留 {dur_txt}；转移矩阵见 transition_matrix.csv、平滑状态概率见图。"
            f" AIC={res.aic:.1f}、loglik={res.llf:.1f}。{thin_note}"
            " ⚠ regime 是潜在状态的「解释」，非外部验证的真实状态；存在 label-switching"
            "（状态编号本身任意——本结果已按均值升序固定编号并说明）；每状态需足够数据；"
            "EM/优化易陷局部最优（已用 search_reps 多起点，结果仍可能因起点而异）。"
        )
        code += [
            "from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression  # 马尔可夫切换",
            f"# MarkovRegression(y, k_regimes={k}, switching_variance={sw_var}).fit(search_reps=20)",
            "# 均值 res.params['const[i]']、方差 'sigma2[i]'、转移阵 res.regime_transition（[to,from]，需转置）",
            "# 平滑概率 res.smoothed_marginal_probabilities；驻留期=1/(1-p_ii)；按均值排序解 label-switching",
        ]
    except Exception as err:
        summary.append(f"马尔可夫切换拟合失败：{err}")


@register("dynamic_factor")
def _branch_dynamic_factor(ctx: Ctx) -> None:
    # Dynamic factor model (DFM):
    # statsmodels.tsa.statespace.dynamic_factor.DynamicFactor extracts k_factors common factor(s)
    # (each following an AR(factor_order) process) driving the comovement of several STANDARDIZED
    # series, plus idiosyncratic noise. A dynamic PCA WITH inference. We report factor loadings,
    # the fraction of each series' variance explained by the common factor, loglik/AIC, and a plot
    # of the extracted factor vs the standardized series.
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np

    _excl = {fp.unit_col, fp.time_col}
    forced = [c for c in (cfg.get("columns") or []) if c in df.columns and c not in _excl]
    series = (forced if len(forced) >= 2 else
              [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl])[:8]
    if len(series) < 3:
        summary.append("动态因子模型失败：需要 ≥3 个连续序列（共同因子才有意义）。config['columns'] 可指定。")
        return
    try:
        import pandas as pd
        from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

        d2 = df.sort_values(fp.time_col) if (fp.time_col and fp.time_col in df.columns) else df
        data = d2[series].dropna().reset_index(drop=True).astype(float)
        n = len(data)
        if n < 40:
            summary.append("动态因子模型失败：观测不足（<40），状态空间难稳健估计。")
            return
        try:
            k_factors = int(cfg.get("k_factors", 1))
        except (TypeError, ValueError):
            k_factors = 1
        k_factors = max(1, min(k_factors, len(series) - 1))
        try:
            factor_order = int(cfg.get("factor_order", 1))
        except (TypeError, ValueError):
            factor_order = 1
        factor_order = max(1, factor_order)

        # STANDARDIZE inputs (DFM loadings/variance-explained are only meaningful on a common scale).
        mu = data.mean()
        sd = data.std(ddof=0).replace(0, 1.0)
        Z = (data - mu) / sd

        # pass the DataFrame (NOT .to_numpy()) so statsmodels keeps the real column names —
        # otherwise loadings are named loading.f1.y1/y2/... and the by-name parse below misses
        # every series (loadings all NaN -> variance-explained collapses to 0).
        model = DynamicFactor(Z, k_factors=k_factors, factor_order=factor_order,
                              error_order=0)
        res = model.fit(disp=False, maxiter=int(cfg.get("maxiter", 200)))

        # --- factor loadings: param names 'loading.f<j>.<varname>' (one per series per factor).
        # Pull by name into a (series x k_factors) matrix; never index blindly. ----------------
        pnames = list(res.param_names)
        pvals = np.asarray(res.params, dtype=float)
        loadings = np.full((len(series), k_factors), np.nan)
        for idx, nm in enumerate(pnames):
            if nm.startswith("loading."):
                # format loading.f1.<var> (1-based factor) or loading.<var>.f1 across versions
                parts = nm.split(".")
                fj = None
                var = None
                for p in parts[1:]:
                    if p.startswith("f") and p[1:].isdigit():
                        fj = int(p[1:]) - 1
                    elif p in series:
                        var = p
                if var is None:
                    # last resort: trailing token is the variable
                    cand = parts[-1]
                    if cand in series:
                        var = cand
                if fj is None:
                    fj = 0
                if var in series and 0 <= fj < k_factors:
                    loadings[series.index(var), fj] = float(pvals[idx])

        # primary (first) factor for reporting/plot
        factor_states = np.asarray(res.smoothed_state, dtype=float)  # rows: factor lags then errors
        factor1 = factor_states[0]  # first factor's level state

        # --- SIGN identification: factor is only identified up to sign (and scale). Convention:
        # make the SUM of first-factor loadings non-negative; flip factor & its loadings together. -
        col0 = loadings[:, 0]
        if np.nansum(col0) < 0:
            loadings[:, 0] = -loadings[:, 0]
            factor1 = -factor1
            col0 = loadings[:, 0]

        # --- fraction of each series' variance explained by the COMMON factor(s) -------------
        # Model on standardized data: z_i = sum_j loading_ij * f_j + eps_i, so the fraction of
        # series i's (unit) variance explained by the common factors ~= sum_j loading_ij^2 * Var(f_j)
        # (factors approx uncorrelated). We use the SMOOTHED factor variances (honest about scale)
        # rather than assuming Var(f)=1. Factors are the LEADING k_factors state rows (factor_order
        # only adds lag rows after them), so factor j's level series is factor_states[j].
        var_explained = np.full(len(series), np.nan)
        for i in range(len(series)):
            ve = 0.0
            for j in range(k_factors):
                lij = loadings[i, j]
                if lij == lij:  # not NaN
                    fvar = float(np.var(factor_states[j])) if factor_states.shape[0] > j else 1.0
                    ve += (lij ** 2) * fvar
            var_explained[i] = min(max(ve, 0.0), 1.0)  # standardized series have unit variance

        # correlation of each standardized series with the extracted factor (diagnostic + plot)
        corr_with_factor = np.array([
            float(np.corrcoef(Z.iloc[:, i].to_numpy(), factor1)[0, 1]) if np.std(factor1) > 0 else float("nan")
            for i in range(len(series))])

        for i, s in enumerate(series):
            estimates[f"loading_{s}"] = round(float(loadings[i, 0]), 4) if loadings[i, 0] == loadings[i, 0] else float("nan")
            estimates[f"var_explained_{s}"] = round(float(var_explained[i]), 4) if var_explained[i] == var_explained[i] else float("nan")
        estimates.update({
            "k_factors": float(k_factors), "factor_order": float(factor_order),
            "mean_var_explained": round(float(np.nanmean(var_explained)), 4),
            "aic": round(float(res.aic), 2), "loglik": round(float(res.llf), 2), "n_obs": float(n),
        })

        # CSV of loadings + variance explained
        pd.DataFrame({
            "series": series,
            "loading_factor1": np.round(loadings[:, 0], 4),
            "var_explained_common": np.round(var_explained, 4),
            "corr_with_factor1": np.round(corr_with_factor, 4),
        }).to_csv(d / "factor_loadings.csv", index=False, encoding="utf-8")
        files.append("factor_loadings.csv")
        # CSV of the extracted factor(s)
        fac_cols = {"index": range(len(factor1))}
        for j in range(k_factors):
            fcol = factor_states[j] if factor_states.shape[0] > j else factor1
            if j == 0:
                fcol = factor1  # sign-fixed primary factor
            fac_cols[f"factor{j + 1}"] = np.round(fcol, 4)
        pd.DataFrame(fac_cols).to_csv(d / "common_factor.csv", index=False, encoding="utf-8")
        files.append("common_factor.csv")
        (d / "dfm_summary.txt").write_text(str(res.summary()), encoding="utf-8")
        files.append("dfm_summary.txt")

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 4.2))
            for i in range(len(series)):
                ax.plot(Z.iloc[:, i].to_numpy(), color="#cccccc", lw=0.7, alpha=0.8,
                        label="standardized series" if i == 0 else None)
            ax.plot(factor1, color="#C44E52", lw=2.0, label="extracted common factor (sign-fixed)")
            ax.set_xlabel("period index")
            ax.set_ylabel("standardized value")
            ax.set_title(f"Dynamic Factor — {k_factors} factor(s), factor_order={factor_order} "
                         f"(mean var explained={np.nanmean(var_explained):.2f})")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "common_factor.png", dpi=150)
            plt.close(fig)
            files.append("common_factor.png")
        except Exception:
            pass

        load_txt = "、".join(f"{s}={loadings[i, 0]:.2f}" for i, s in enumerate(series))
        ve_txt = "、".join(f"{s}={var_explained[i]:.2f}" for i, s in enumerate(series))
        weak = float(np.nanmean(var_explained)) < 0.2
        weak_note = "；⚠ 共同因子解释的方差很低(<0.2)：序列共动弱，单因子或不合适" if weak else ""
        summary.append(
            f"{entry.method} 完成：{len(series)} 个标准化序列 × {n} 期，提取 {k_factors} 个共同因子"
            f"（factor_order={factor_order}）；因子载荷（第一因子）{load_txt}；"
            f"各序列被共同因子解释的方差比例 {ve_txt}（均值={np.nanmean(var_explained):.2f}）；"
            f"AIC={res.aic:.1f}、loglik={res.llf:.1f}；载荷见 factor_loadings.csv、因子序列见图。{weak_note}"
            " ⚠ DFM 假定共动由少数共同因子 + 各自特异噪声驱动（带推断的动态 PCA）；输入已标准化（已报告）；"
            "因子个数是一种选择；因子的符号/尺度仅在旋转意义下可识别（本结果按「第一因子载荷之和≥0」固定符号并披露）。"
        )
        code += [
            "from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor  # 动态因子模型",
            f"# Z=(data-mean)/std; DynamicFactor(Z, k_factors={k_factors}, factor_order={factor_order}).fit()",
            "# 载荷 res.params['loading.f1.<var>']；因子 res.smoothed_state[0]；符号按载荷之和≥0 固定",
        ]
    except Exception as err:
        summary.append(f"动态因子模型拟合失败：{err}")
