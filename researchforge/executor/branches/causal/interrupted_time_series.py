"""Causal family branch handler: interrupted_time_series — segmented regression ITS.

A quasi-experimental design for a single series observed before and after a known
intervention: fit y_t = b0 + b1*t + b2*D_t + b3*(t-T0)*D_t + e, where D_t = 1 after the
intervention and (t-T0)*D_t is time-since-intervention. b2 is the IMMEDIATE LEVEL change
at the intervention and b3 is the SLOPE (trend) change (post-slope = b1 + b3); the
counterfactual is the extrapolated pre-intervention trend. Standard errors are
Newey-West (HAC) to handle the serial correlation typical of time series; Durbin-Watson
is reported. Pure Python (statsmodels OLS + HAC).
"""
from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


def _resolve_intervention(cfg, time_vals, n):
    """(T0 index, placeholder?) for the first POST-intervention position. Config
    intervention may be a value in the time column or an integer row index; absent →
    the series midpoint as a flagged placeholder."""
    import numpy as np

    spec = cfg.get("intervention")
    if spec is not None:
        try:
            val = float(spec)
            if time_vals is not None:
                ge = np.where(np.asarray(time_vals, dtype=float) >= val)[0]
                if len(ge) and 0 < ge[0] < n:
                    return int(ge[0]), False
            iv = int(round(val))
            if 0 < iv < n:
                return iv, False
        except (TypeError, ValueError):
            pass
    return n // 2, True


@register("interrupted_time_series")
def _branch_interrupted_time_series(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    excl = {fp.unit_col, fp.time_col}
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in excl]
    outcome = cfg.get("outcome") if cfg.get("outcome") in df.columns else (cont[0] if cont else None)
    if outcome is None:
        summary.append("中断时间序列跳过：未找到连续结果变量（outcome）。")
        return

    # time column: config time / profiler time_col / a column named time/date/year/t.
    tcol = cfg.get("time") if cfg.get("time") in df.columns else fp.time_col
    if tcol not in df.columns:
        tcol = next((c for c in df.columns
                     if any(k in str(c).lower() for k in ("time", "date", "year", "month", "week", "period"))
                     and c != outcome), None)

    work = df.copy()
    if tcol in work.columns:
        tnum = pd.to_numeric(work[tcol], errors="coerce")
        if tnum.notna().mean() > 0.8:
            work = work.assign(_t=tnum).sort_values("_t")
            time_vals = work["_t"].to_numpy(float)
        else:  # non-numeric time (e.g. date strings) → keep given order
            time_vals = None
    else:
        time_vals = None

    y = pd.to_numeric(work[outcome], errors="coerce")
    mask = y.notna()
    y = y[mask].to_numpy(float)
    if time_vals is not None:
        time_vals = time_vals[mask.to_numpy()]
    n = len(y)
    if n < 8:
        summary.append(f"中断时间序列跳过：有效观测过少（{n}<8；前后段各需足够点估计水平+趋势变化）。")
        return

    T0, placeholder = _resolve_intervention(cfg, time_vals, n)
    n_pre, n_post = T0, n - T0
    if n_pre < 3 or n_post < 3:
        summary.append(f"中断时间序列跳过：干预前/后观测过少（前 {n_pre} / 后 {n_post}，各需 ≥3）。")
        return

    try:
        t = np.arange(n, dtype=float)
        D = (t >= T0).astype(float)
        ts = (t - T0) * D
        X = sm.add_constant(np.column_stack([t, D, ts]))
        maxlags = max(1, int(round(n ** 0.25)))
        m = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
        from statsmodels.stats.stattools import durbin_watson

        b = np.asarray(m.params, float)         # [const, t(pre-slope), D(level), ts(slope change)]
        se = np.asarray(m.bse, float)
        p = np.asarray(m.pvalues, float)
        ci = np.asarray(m.conf_int(), float)
        dw = float(durbin_watson(m.resid))

        estimates["level_change"] = round(float(b[2]), 5)
        estimates["level_change_se"] = round(float(se[2]), 5)
        estimates["level_change_ci_low"] = round(float(ci[2, 0]), 5)
        estimates["level_change_ci_high"] = round(float(ci[2, 1]), 5)
        estimates["level_change_p"] = round(float(p[2]), 5)
        estimates["slope_change"] = round(float(b[3]), 5)
        estimates["slope_change_se"] = round(float(se[3]), 5)
        estimates["slope_change_ci_low"] = round(float(ci[3, 0]), 5)
        estimates["slope_change_ci_high"] = round(float(ci[3, 1]), 5)
        estimates["slope_change_p"] = round(float(p[3]), 5)
        estimates["pre_slope"] = round(float(b[1]), 5)
        estimates["post_slope"] = round(float(b[1] + b[3]), 5)
        estimates["baseline_intercept"] = round(float(b[0]), 5)
        estimates["durbin_watson"] = round(dw, 4)
        estimates["r_squared"] = round(float(m.rsquared), 4)
        estimates["n_pre"] = float(n_pre)
        estimates["n_post"] = float(n_post)
        estimates["hac_maxlags"] = float(maxlags)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            xaxis = time_vals if time_vals is not None else t
            fig, ax = plt.subplots(figsize=(6.2, 3.8))
            ax.scatter(xaxis, y, s=14, color="#4C72B0", label="observed")
            fit = X @ b
            ax.plot(xaxis[:T0], fit[:T0], color="#2f6f4f", lw=1.8, label="segmented fit")
            ax.plot(xaxis[T0:], fit[T0:], color="#2f6f4f", lw=1.8)
            cf = b[0] + b[1] * t                      # counterfactual = extrapolated pre-trend
            ax.plot(xaxis[T0:], cf[T0:], color="#C44E52", ls="--", lw=1.4, label="counterfactual")
            ax.axvline(xaxis[T0], color="grey", ls=":", lw=1)
            ax.set_xlabel(str(tcol) if tcol in df.columns else "time index")
            ax.set_ylabel(str(outcome))
            ax.set_title("Interrupted time series (segmented regression)")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "its_segmented.png", dpi=150)
            plt.close(fig)
            files.append("its_segmented.png")
        except Exception:
            pass

        lvl_sig = p[2] < 0.05
        slp_sig = p[3] < 0.05
        ph_note = (" ⚠ 未指定干预时点——已用序列中点占位，结果无意义，请用 config['intervention']"
                   "(时间值或行序号)指定真实干预时点。" if placeholder else "")
        dw_note = "（⚠ 残差自相关明显，已用 Newey-West HAC SE）" if (dw < 1.5 or dw > 2.5) else ""
        summary.append(
            f"{entry.method} 完成（分段回归，n={n}：干预前 {n_pre} / 后 {n_post}，HAC maxlags={maxlags}）："
            f"**即时水平变化**={b[2]:.4g}（95% CI [{ci[2, 0]:.3g}, {ci[2, 1]:.3g}]，p={p[2]:.3g}"
            f"{'，显著' if lvl_sig else '，不显著'}）；**斜率(趋势)变化**={b[3]:.4g}"
            f"（p={p[3]:.3g}{'，显著' if slp_sig else '，不显著'}）；"
            f"干预前斜率={b[1]:.3g} → 干预后斜率={b[1] + b[3]:.3g}。Durbin-Watson={dw:.2f}{dw_note}。"
            "分段拟合与反事实(延续干预前趋势)见 its_segmented.png。"
            " ⚠ ITS 的因果解读假设：干预时点除该干预外没有同时发生的其他变化(共干预/政策/测量方式变更)、"
            "且反事实=干预前趋势的合理外推；季节性需额外建模；前后段点数越多越可靠。"
            + ph_note
        )
        code += [
            "import statsmodels.api as sm  # 分段回归 ITS",
            "D=(t>=T0).astype(float); ts=(t-T0)*D",
            "m=sm.OLS(y, sm.add_constant(np.c_[t,D,ts])).fit(cov_type='HAC',cov_kwds={'maxlags':L})",
            "# D 系数=即时水平变化; ts 系数=斜率(趋势)变化; 干预后斜率 = t系数 + ts系数",
        ]
    except Exception as err:
        summary.append(f"中断时间序列失败：{err}")
