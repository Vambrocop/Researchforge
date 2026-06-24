"""Branch handlers for the forecasting family — univariate time-series forecasting.

Three pure-Python (statsmodels / numpy / pandas) forecasters for a SINGLE numeric
series (no R):

  * exponential_smoothing — Holt-Winters (statsmodels ExponentialSmoothing): auto-pick
    trend/seasonal (additive), report fitted params + in-sample fit (AIC, SSE) + an
    h-step forecast with the model's prediction interval.
  * theta_method          — the Theta method (Assimakopoulos & Nikolopoulos 2000, the
    M3 winner): SES on the theta=0 line + linear drift (theta=2 line), recombined 50/50,
    with optional (de)seasonalization; h-step forecast.
  * croston               — Croston's method for INTERMITTENT (zero-heavy) demand:
    separately SES-smooth nonzero demand sizes and inter-arrival intervals; forecast
    = size / interval. Also the SBA (Syntetos-Boylan) bias-corrected variant.

Each handler resolves the series (config column else first continuous; time order via
fp.time_col if present), degrades honestly (series too short / non-numeric / all-constant
/ import missing -> Chinese "<方法>跳过：<原因>" + RETURN; never crashes/fabricates),
writes CSV + PNG (matplotlib Agg, ENGLISH plot labels) best-effort, fills float
`estimates`, appends a Chinese `summary` ending in ⚠ disclosures, and MUTATES ctx
(never rebinds). See executor/_branch_api.py and CLAUDE.md.

statsmodels / numpy / pandas are installed; matplotlib best-effort.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register

_MIN_N = 12  # honest-degrade floor: too short to forecast


def _resolve_value_col(ctx: Ctx):
    """Resolve the series column: cfg['column']/['value'] override else first continuous."""
    fp, cfg, df = ctx.fp, ctx.cfg, ctx.df
    _excl = {fp.unit_col, fp.time_col}
    chosen = cfg.get("column") or cfg.get("value")
    if chosen and chosen in df.columns:
        return chosen
    return next(
        (c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl),
        None,
    )


def _series_array(ctx: Ctx, value_col, method_zh: str):
    """Materialize the ordered float series for `value_col`, with honest-degrade checks.

    Returns (y, problem). y is a 1-D float numpy array (NaNs dropped, ordered by
    fp.time_col if present); problem is a Chinese skip message or None.
    """
    import numpy as np
    import pandas as pd

    fp, df = ctx.fp, ctx.df
    if value_col is None:
        return None, f"{method_zh}跳过：未找到数值序列列（config['column'] 可指定）。"
    d2 = df.sort_values(fp.time_col) if (fp.time_col and fp.time_col in df.columns) else df
    s = pd.to_numeric(d2[value_col], errors="coerce").dropna().reset_index(drop=True)
    y = s.to_numpy(dtype=float)
    if y.size < _MIN_N:
        return None, f"{method_zh}跳过：有效观测不足（n={y.size}<{_MIN_N}），序列太短无法预测。"
    if np.nanstd(y) == 0 or pd.Series(y).nunique() < 2:
        return None, f"{method_zh}跳过：序列近常数（无变化），无法预测。"
    return y, None


def _ses(y, alpha: float):
    """Simple exponential smoothing levels. Returns (levels, last_level).

    levels[t] is the one-step-ahead forecast made at the end of period t-1 (so
    levels[0] = y[0]). last_level is the level after the final observation = the
    flat SES forecast for all future steps.
    """
    import numpy as np

    y = np.asarray(y, dtype=float)
    lvl = np.empty_like(y)
    lvl[0] = y[0]
    for t in range(1, len(y)):
        lvl[t] = alpha * y[t - 1] + (1 - alpha) * lvl[t - 1]
    last = alpha * y[-1] + (1 - alpha) * lvl[-1]
    return lvl, float(last)


def _detect_period(ctx: Ctx, y):
    """Seasonal period: config seasonal_periods else periodogram auto-detect, else None."""
    cfg = ctx.cfg
    n = len(y)
    sp = cfg.get("seasonal_periods")
    try:
        sp = int(sp) if sp is not None else None
    except (TypeError, ValueError):
        sp = None
    if sp and 2 <= sp <= n // 2:
        return sp
    # reuse the timeseries family's periodogram detector (>=3 cycles, Fisher g-test)
    try:
        from researchforge.executor.branches.timeseries import _periodogram_period

        per = _periodogram_period(y, n)
        if per and 2 <= per <= n // 2:
            return per
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# A. exponential_smoothing — Holt-Winters (statsmodels ExponentialSmoothing)
# ─────────────────────────────────────────────────────────────────────────────
@register("exponential_smoothing")
def _branch_exponential_smoothing(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import importlib.util

    if importlib.util.find_spec("statsmodels") is None:
        summary.append("指数平滑跳过：需要 statsmodels 包（未检测到）。安装：pip install statsmodels。")
        return
    value_col = _resolve_value_col(ctx)
    y, problem = _series_array(ctx, value_col, "指数平滑")
    if problem is not None:
        summary.append(problem)
        return
    try:
        import numpy as np
        import pandas as pd
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        n = len(y)
        try:
            h = int(cfg.get("h", 10))
        except (TypeError, ValueError):
            h = 10
        h = max(1, h)

        # trend: config override else auto (additive if a clear linear slope vs noise)
        trend = cfg.get("trend")
        if trend in {"additive"}:
            trend = "add"
        elif trend in {"multiplicative"}:
            trend = "mul"
        if trend not in {"add", "mul", None}:
            trend = None
        auto_trend = cfg.get("trend") is None
        if auto_trend:
            idx = np.arange(n)
            slope, intercept = np.polyfit(idx, y, 1)
            fit = slope * idx + intercept
            ss_res = float(np.sum((y - fit) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            trend = "add" if r2 > 0.10 else None

        # seasonal: config override else periodogram auto-detect (additive only here)
        seasonal_cfg = cfg.get("seasonal")
        sp = _detect_period(ctx, y)
        if seasonal_cfg in {"none", "no", "off"}:
            seasonal, sp = None, None
        elif sp and n >= 2 * sp:  # need >=2 full cycles for HW seasonal estimation
            seasonal = "add"
        else:
            seasonal, sp = None, None

        model = ExponentialSmoothing(
            y, trend=trend, seasonal=seasonal, seasonal_periods=sp,
            initialization_method="estimated",
        )
        res = model.fit()
        fitted = np.asarray(res.fittedvalues, dtype=float)
        resid = y - fitted
        sse = float(np.sum(resid ** 2))
        sigma = float(np.std(resid, ddof=1)) if n > 1 else 0.0

        # Prediction interval: prefer statsmodels' state-space ETS, whose
        # get_prediction(...).summary_frame gives a MODEL-CONSISTENT (analytic
        # state-space) 95% interval that widens correctly under the fitted
        # trend/seasonal structure. Same trend/seasonal/seasonal_periods spec as
        # the Holt-Winters fit above. Fall back to the SES √step approximation iff
        # ETSModel import/fit fails (disclosed in the summary).
        pi_method = "state-space"
        try:
            from statsmodels.tsa.exponential_smoothing.ets import ETSModel

            error_spec = "mul" if (trend == "mul" or seasonal == "mul") else "add"
            ets_model = ETSModel(
                y, error=error_spec, trend=trend, seasonal=seasonal,
                seasonal_periods=sp if seasonal else None,
                initialization_method="estimated",
            )
            ets_res = ets_model.fit(disp=False)
            pred = ets_res.get_prediction(start=n, end=n + h - 1)
            sf = pred.summary_frame(alpha=0.05)
            fc = np.asarray(sf["mean"], dtype=float)
            # column names vary by statsmodels version (pi_lower/pi_upper or lower/upper)
            lo_col = next((c for c in sf.columns if "lower" in c.lower()), None)
            hi_col = next((c for c in sf.columns if "upper" in c.lower()), None)
            if lo_col is None or hi_col is None:
                raise RuntimeError("ETS summary_frame missing interval columns")
            lower = np.asarray(sf[lo_col], dtype=float)
            upper = np.asarray(sf[hi_col], dtype=float)
            if not (np.all(np.isfinite(fc)) and np.all(np.isfinite(lower))
                    and np.all(np.isfinite(upper))):
                raise RuntimeError("ETS interval produced non-finite values")
        except Exception:
            # honest degrade: SES-style sqrt(step) widening of the residual sd (95%).
            pi_method = "approx"
            fc = np.asarray(res.forecast(h), dtype=float)
            z = 1.959963985
            steps = np.arange(1, h + 1)
            widen = z * sigma * np.sqrt(steps)
            lower = fc - widen
            upper = fc + widen

        params = res.params
        alpha = float(params.get("smoothing_level", float("nan")))
        beta = float(params.get("smoothing_trend", float("nan")))
        gamma = float(params.get("smoothing_seasonal", float("nan")))
        level0 = float(params.get("initial_level", float("nan")))
        trend0 = float(params.get("initial_trend", float("nan")))
        aic = float(res.aic) if np.isfinite(res.aic) else float("nan")

        estimates.update({
            "alpha": round(alpha, 4) if np.isfinite(alpha) else float("nan"),
            "aic": round(aic, 2) if np.isfinite(aic) else float("nan"),
            "sse": round(sse, 4),
            "forecast_next": round(float(fc[0]), 4),
            "n": float(n),
            "seasonal_periods": float(sp) if sp else 0.0,
        })
        for k_, v_ in (("beta", beta), ("gamma", gamma), ("level", level0), ("trend", trend0)):
            if np.isfinite(v_):
                estimates[k_] = round(float(v_), 4)

        fc_df = pd.DataFrame({
            "step": list(range(1, h + 1)),
            "point": np.round(fc, 4),
            "lower": np.round(lower, 4),
            "upper": np.round(upper, 4),
        })
        fc_df.to_csv(d / "forecast.csv", index=False, encoding="utf-8")
        files.append("forecast.csv")

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(range(n), y, color="#333333", label="observed")
            ax.plot(range(n), fitted, color="#4C72B0", lw=1, alpha=0.8, label="fitted")
            fx = list(range(n, n + h))
            ax.plot(fx, fc, color="#C44E52", ls="--", label="forecast")
            ax.fill_between(fx, lower, upper, color="#C44E52", alpha=0.18, label="95% PI")
            ax.set_xlabel("period index")
            ax.set_ylabel(str(value_col))
            stitle = f"seasonal={sp}" if sp else "no seasonal"
            ax.set_title(f"Holt-Winters ({trend or 'no'} trend, {stitle}) — {value_col}")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "forecast.png", dpi=150)
            plt.close(fig)
            files.append("forecast.png")
        except Exception:
            pass

        trend_zh = {"add": "加性趋势", "mul": "乘性趋势", None: "无趋势"}.get(trend, str(trend))
        seas_zh = f"加性季节(周期={sp})" if sp else "无季节"
        src = "自动判定" if auto_trend else "config 指定"
        time_warn = "" if fp.time_col else "；⚠ 无时间列，按行序当作时间序处理（请确认行序即时序）"
        pi_zh = (
            "预测区间为**状态空间 ETS 模型一致区间**（statsmodels ETSModel.get_prediction，"
            "按拟合的趋势/季节结构正确随步长展宽，非手搓近似）"
            if pi_method == "state-space" else
            "⚠ 状态空间 ETS 区间不可用，已回退残差 sd×√h 近似区间（趋势/季节下可能偏窄）"
        )
        summary.append(
            f"{entry.method} 完成：对 {value_col}（n={n}）拟合 Holt-Winters（{trend_zh}、{seas_zh}，趋势{src}）；"
            f"α={alpha:.3f}" + (f"、β={beta:.3f}" if np.isfinite(beta) else "")
            + (f"、γ={gamma:.3f}" if np.isfinite(gamma) else "")
            + f"；AIC={aic:.2f}、SSE={sse:.3g}；未来 {h} 期预测见 forecast.csv（含 95% 预测区间），"
            f"下一期点预测={fc[0]:.4g}。{time_warn}"
            f" ⚠ 指数平滑假定平滑结构稳定、外推延续历史模式；{pi_zh}；"
            "季节周期需正确（自动取周期图主峰，可 config seasonal_periods 覆盖）；"
            "趋势/季节自动判定为启发式，可用 config trend/seasonal 强制。"
        )
        code += [
            "from statsmodels.tsa.holtwinters import ExponentialSmoothing  # 点预测/参数",
            "from statsmodels.tsa.exponential_smoothing.ets import ETSModel  # 状态空间区间",
            f"# ExponentialSmoothing(y, trend={trend!r}, seasonal={seasonal!r}, seasonal_periods={sp}).fit()",
            f"# ETSModel(y, error='add', trend={trend!r}, seasonal={seasonal!r}, seasonal_periods={sp}).fit()",
            f"# .get_prediction(start=n, end=n+{h}-1).summary_frame(alpha=0.05)  # mean + pi_lower/pi_upper",
        ]
    except Exception as err:
        summary.append(f"指数平滑失败：{err}")


# ─────────────────────────────────────────────────────────────────────────────
# B. theta_method — the Theta method (M3 winner)
# ─────────────────────────────────────────────────────────────────────────────
@register("theta_method")
def _branch_theta_method(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    value_col = _resolve_value_col(ctx)
    y, problem = _series_array(ctx, value_col, "Theta 方法")
    if problem is not None:
        summary.append(problem)
        return
    try:
        import numpy as np
        import pandas as pd

        n = len(y)
        try:
            h = int(cfg.get("h", 10))
        except (TypeError, ValueError):
            h = 10
        h = max(1, h)

        # ── optional classical multiplicative deseasonalization (ratio-to-MA) ──
        sp = _detect_period(ctx, y)
        if cfg.get("seasonal") in {"none", "no", "off"}:
            sp = None
        seas_idx = None
        work = y.astype(float).copy()
        if sp and n >= 2 * sp and np.all(y > 0):
            ma = pd.Series(y).rolling(sp, center=True).mean()
            if sp % 2 == 0:  # centered MA for even period
                ma = ma.rolling(2).mean().shift(-1)
            ratio = pd.Series(y) / ma
            pos = np.arange(n) % sp
            idx_means = np.array([np.nanmean(ratio[pos == k].to_numpy()) for k in range(sp)])
            if np.all(np.isfinite(idx_means)) and idx_means.sum() > 0:
                idx_means = idx_means * sp / idx_means.sum()  # normalize to mean 1
                seas_idx = idx_means
                work = y / seas_idx[pos]
            else:
                sp = None
        else:
            sp = None  # not enough data / non-positive -> no deseasonalization

        # ── theta=2 line: OLS linear trend (drift) on the (deseasonalized) series ──
        t = np.arange(n, dtype=float)
        slope, _intercept = np.polyfit(t, work, 1)
        drift = float(slope)

        # ── theta=0 line: SES; optimize alpha by minimizing one-step SSE ──
        def _sse_for(a):
            lvl, _ = _ses(work, a)
            return float(np.sum((work[1:] - lvl[1:]) ** 2))

        grid = np.linspace(0.05, 0.99, 40)
        ses_alpha = float(grid[int(np.argmin([_sse_for(a) for a in grid]))])
        _, ses_last = _ses(work, ses_alpha)

        # Theta combination (Hyndman & Billah 2003 operational rule): add half the
        # drift cumulatively beyond the SES level (the 50/50 of theta=0 & theta=2 lines).
        steps = np.arange(1, h + 1, dtype=float)
        fc_work = ses_last + 0.5 * drift * (steps - 1 + (1 - (1 - ses_alpha) ** n) / ses_alpha)

        # reseasonalize
        if seas_idx is not None:
            fpos = (np.arange(n, n + h)) % sp
            fc = fc_work * seas_idx[fpos]
        else:
            fc = fc_work

        estimates.update({
            "forecast_next": round(float(fc[0]), 4),
            "ses_alpha": round(ses_alpha, 4),
            "drift": round(drift, 4),
            "n": float(n),
            "h": float(h),
        })
        if sp:
            estimates["seasonal_periods"] = float(sp)

        fc_df = pd.DataFrame({"step": list(range(1, h + 1)), "point": np.round(fc, 4)})
        fc_df.to_csv(d / "forecast.csv", index=False, encoding="utf-8")
        files.append("forecast.csv")

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(range(n), y, color="#333333", label="observed")
            fx = list(range(n, n + h))
            ax.plot(fx, fc, color="#55A868", ls="--", marker="o", ms=3, label="Theta forecast")
            ax.set_xlabel("period index")
            ax.set_ylabel(str(value_col))
            stitle = f", deseasonalized (period={sp})" if sp else ""
            ax.set_title(f"Theta method — {value_col}{stitle}")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "forecast.png", dpi=150)
            plt.close(fig)
            files.append("forecast.png")
        except Exception:
            pass

        seas_zh = f"先做乘性去季节(周期={sp})再重构季节" if sp else "无季节调整"
        time_warn = "" if fp.time_col else "；⚠ 无时间列，按行序当作时间序处理（请确认行序即时序）"
        summary.append(
            f"{entry.method} 完成：对 {value_col}（n={n}）做 Theta 方法（{seas_zh}）；"
            f"SES α={ses_alpha:.3f}、线性漂移 drift={drift:.4g}；未来 {h} 期预测见 forecast.csv，"
            f"下一期点预测={fc[0]:.4g}。{time_warn}"
            " ⚠ Theta 方法 = θ=0(SES)线与 θ=2(线性趋势)线 50/50 重组（Assimakopoulos & Nikolopoulos 2000，M3 冠军）；"
            "外推假定漂移延续；本实现给点预测（无解析预测区间）；去季节为可选的经典乘性分解（需周期正确、序列为正），"
            "可 config seasonal='none' 关闭或 seasonal_periods 指定。"
        )
        code += [
            "# Theta method: deseasonalize (optional) -> SES(theta=0) + linear drift(theta=2) -> recombine 50/50",
            f"# ses_alpha≈{ses_alpha:.3f}, drift≈{drift:.4g}; forecast = SES_last + 0.5*drift*(step-1+...)",
        ]
    except Exception as err:
        summary.append(f"Theta 方法失败：{err}")


# ─────────────────────────────────────────────────────────────────────────────
# C. croston — Croston's method for intermittent demand + SBA variant
# ─────────────────────────────────────────────────────────────────────────────
@register("croston")
def _branch_croston(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    # Croston-specific materialization: allow sparse (mostly-zero) series, which the
    # near-constant guard in _series_array would wrongly reject. We do intermittent-
    # friendly checks here instead.
    import numpy as np
    import pandas as pd

    value_col = _resolve_value_col(ctx)
    if value_col is None:
        summary.append("Croston 跳过：未找到数值序列列（config['column'] 可指定）。")
        return
    d2 = df.sort_values(fp.time_col) if (fp.time_col and fp.time_col in df.columns) else df
    y = pd.to_numeric(d2[value_col], errors="coerce").dropna().reset_index(drop=True).to_numpy(dtype=float)
    if y.size < _MIN_N:
        summary.append(f"Croston 跳过：有效观测不足（n={y.size}<{_MIN_N}），序列太短无法预测。")
        return
    if np.any(y < 0):
        summary.append("Croston 跳过：序列含负值，Croston 仅适用于非负(间断需求)序列。")
        return
    nonzero = y[y > 0]
    if nonzero.size < 2:
        summary.append("Croston 跳过：非零需求点不足（<2），无法估计需求量/间隔。")
        return
    try:
        n = len(y)
        try:
            alpha = float(cfg.get("alpha", 0.1))
        except (TypeError, ValueError):
            alpha = 0.1
        if not (0 < alpha < 1):
            alpha = 0.1

        # positions of nonzero demand; inter-arrival intervals between events
        nz_pos = np.flatnonzero(y > 0)
        sizes = y[nz_pos]
        intervals = np.diff(np.concatenate(([-1], nz_pos))).astype(float)  # first = pos+1

        # SES on demand sizes and on intervals (Croston updates only at demand epochs)
        _, z_hat = _ses(sizes, alpha)        # smoothed demand size
        _, p_hat = _ses(intervals, alpha)    # smoothed inter-arrival interval

        forecast_rate = float(z_hat / p_hat) if p_hat > 0 else float("nan")
        # SBA (Syntetos-Boylan Approximation): bias-correct by (1 - alpha/2)
        sba_forecast = float((1 - alpha / 2.0) * z_hat / p_hat) if p_hat > 0 else float("nan")

        pct_zero = float(np.mean(y == 0) * 100.0)
        mean_interval = float(np.mean(intervals))

        estimates.update({
            "forecast_rate": round(forecast_rate, 4) if np.isfinite(forecast_rate) else float("nan"),
            "sba_forecast": round(sba_forecast, 4) if np.isfinite(sba_forecast) else float("nan"),
            "pct_zero": round(pct_zero, 2),
            "mean_interval": round(mean_interval, 4),
            "n": float(n),
        })

        try:
            h = int(cfg.get("h", 10))
        except (TypeError, ValueError):
            h = 10
        h = max(1, h)
        fc_df = pd.DataFrame({
            "step": list(range(1, h + 1)),
            "forecast_rate": np.round([forecast_rate] * h, 4),
            "sba_forecast": np.round([sba_forecast] * h, 4),
        })
        fc_df.to_csv(d / "forecast.csv", index=False, encoding="utf-8")
        files.append("forecast.csv")

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.stem(range(n), y, linefmt="#bbbbbb", markerfmt="o", basefmt=" ")
            ax.axhline(forecast_rate, color="#C44E52", ls="--", label=f"Croston rate={forecast_rate:.3g}")
            ax.axhline(sba_forecast, color="#4C72B0", ls=":", label=f"SBA rate={sba_forecast:.3g}")
            ax.set_xlabel("period index")
            ax.set_ylabel(str(value_col))
            ax.set_title(f"Croston intermittent demand — {value_col} (%zero={pct_zero:.0f}%)")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "forecast.png", dpi=150)
            plt.close(fig)
            files.append("forecast.png")
        except Exception:
            pass

        not_intermittent = pct_zero < 25.0
        intermit_note = (
            f"；⚠ 序列零值占比仅 {pct_zero:.0f}%（<25%），并非典型间断需求——Croston 优势在零值密集序列，"
            "对此类序列宜改用常规预测(指数平滑/Theta)"
            if not_intermittent else ""
        )
        time_warn = "" if fp.time_col else "；⚠ 无时间列，按行序当作时间序处理（请确认行序即时序）"
        summary.append(
            f"{entry.method} 完成：对 {value_col}（n={n}，零值占比 {pct_zero:.0f}%）做 Croston 方法（α={alpha:g}）；"
            f"平滑需求量 z={z_hat:.4g}、平滑间隔 p={p_hat:.4g}（均值间隔={mean_interval:.3g}）；"
            f"需求率预测={forecast_rate:.4g}/期，SBA 偏差校正预测={sba_forecast:.4g}/期（见 forecast.csv）。"
            f"{intermit_note}{time_warn}"
            " ⚠ Croston 专为间断/零值密集需求设计：分别对非零需求量与到达间隔做 SES 再相除；"
            "原始 Croston 有正偏，SBA(Syntetos-Boylan) 以 (1-α/2) 校正更可取；预测为恒定需求率(非逐期事件)；"
            "α 为平滑参数(config alpha，默认 0.1)。"
        )
        code += [
            "# Croston: SES on nonzero demand sizes z and inter-arrival intervals p; forecast = z/p",
            f"# alpha={alpha:g}; SBA = (1 - alpha/2) * z/p  (Syntetos-Boylan bias correction)",
        ]
    except Exception as err:
        summary.append(f"Croston 失败：{err}")
