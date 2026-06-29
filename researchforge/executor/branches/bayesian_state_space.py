"""Branch handler for bayesian_state_space — Bayesian structural time series.

The Bayesian counterpart of the frequentist ``unobserved_components`` (family
time-series). Decomposes a univariate series into latent states with full posterior
uncertainty and a probabilistic forecast:

* ``local_level``         — random-walk level + observation noise (y_t = μ_t + ε,
                            μ_t = μ_{t-1} + η).
* ``local_linear_trend``  — level + a random-walk slope (DEFAULT): μ_t = μ_{t-1} +
                            δ_{t-1} + η, δ_t = δ_{t-1} + ζ (for t≥2; μ_0 is the
                            first-period level, the drift acts from the 2nd step).
                            Captures trend & forecasts it forward.

States are built with a NON-CENTERED cumulative-sum parameterization (innovations
z~N(0,1) scaled by their SD) to avoid the variance funnel; y is standardized so the
HalfNormal innovation-SD priors are scale-appropriate (states/variances/forecasts
back-transformed to the data scale). Reports the smoothed level (+ slope) with 94%
credible bands, the variance components (obs / level / slope), an h-step forecast with
predictive intervals, and convergence (R-hat / ESS / divergences).

PyMC optional + graceful degrade (no pymc/arviz -> honest skip pointing at
unobserved_components / markov_switching). English plot labels, Chinese summary with
⚠. Mutates ctx, never rebinds. See executor/_branch_api.py and CLAUDE.md.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.branches.bayesian_mcmc import (
    _conv_note,
    _convergence,
    _degrade,
    _have_pymc,
    _sampler_cfg,
)


def _resolve_series(ctx: Ctx):
    """(series values ordered in time, value-column name, problem). Value = config
    value/column or the first continuous column; ordered by the profiler's time_col
    when present, else row order. Returns (y, name, None) or (None, None, problem)."""
    import pandas as pd

    fp, cfg, df = ctx.fp, ctx.cfg, ctx.df
    name = cfg.get("value") or cfg.get("column")
    if name not in df.columns:
        cont = [c.name for c in fp.columns
                if c.kind == "continuous" and c.name not in {fp.time_col}]
        name = cont[0] if cont else None
    if name is None:
        return None, None, "贝叶斯状态空间 跳过：未找到连续的时间序列列（用 config value 指定）。"
    d = df
    if fp.time_col and fp.time_col in df.columns:
        d = df.sort_values(fp.time_col)
    y = pd.to_numeric(d[name], errors="coerce").dropna().to_numpy(float)
    if len(y) < 20:
        return None, None, "贝叶斯状态空间 跳过：有效观测 < 20（时间序列太短）。"
    return y, name, None


@register("bayesian_state_space")
def _branch_bayesian_state_space(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    method = entry.method
    if not _have_pymc():
        _degrade(summary, method, "unobserved_components / markov_switching")
        return

    y, name, problem = _resolve_series(ctx)
    if problem:
        summary.append(problem)
        return

    import numpy as np

    model_kind = str(cfg.get("model", "local_linear_trend")).lower()
    if model_kind not in {"local_level", "local_linear_trend"}:
        model_kind = "local_linear_trend"
    trend = model_kind == "local_linear_trend"
    T = len(y)
    horizon = int(cfg.get("horizon", min(12, max(3, T // 5))))
    ym, ysd = float(y.mean()), float(y.std() or 1.0)
    yz = (y - ym) / ysd

    sc = _sampler_cfg(cfg)
    target_accept = float(cfg.get("target_accept", 0.99))
    try:
        import pymc as pm
        import pytensor.tensor as pt

        with pm.Model() as model:
            mu0 = pm.Normal("mu0", yz[0], 1.0)
            s_obs = pm.HalfNormal("s_obs", 1.0)
            s_lvl = pm.HalfNormal("s_lvl", 0.3)
            z_lvl = pm.Normal("z_lvl", 0.0, 1.0, shape=T)
            if trend:
                d0 = pm.Normal("d0", 0.0, 1.0)
                s_slp = pm.HalfNormal("s_slp", 0.1)
                z_slp = pm.Normal("z_slp", 0.0, 1.0, shape=T)
                delta = d0 + s_slp * pt.cumsum(z_slp)               # slope RW (non-centered)
                dshift = pt.concatenate([[0.0], delta[:-1]])        # δ_{t-1}
                level = mu0 + pt.cumsum(dshift) + s_lvl * pt.cumsum(z_lvl)
                pm.Deterministic("slope", delta)
            else:
                level = mu0 + s_lvl * pt.cumsum(z_lvl)
            pm.Deterministic("level", level)
            pm.Normal("y_obs", level, s_obs, observed=yz)
            idata = pm.sample(draws=sc["draws"], tune=sc["tune"], chains=sc["chains"],
                              target_accept=target_accept, random_seed=sc["seed"],
                              progressbar=False)

        post = idata.posterior
        n_div = int(idata.sample_stats["diverging"].sum())
        conv_names = ["mu0", "s_obs", "s_lvl"] + (["d0", "s_slp"] if trend else [])
        max_rhat, min_ess = _convergence(idata, conv_names)

        def _flat(name_):
            v = post[name_].values
            return v.reshape(-1, *v.shape[2:])

        # smoothed states (response units)
        lvl_draws = _flat("level")                       # (S, T)
        lvl_mean = lvl_draws.mean(0) * ysd + ym
        lvl_lo, lvl_hi = np.percentile(lvl_draws, [3.0, 97.0], axis=0) * ysd + ym
        import pandas as pd

        states = pd.DataFrame({
            "t": np.arange(T), "observed": y,
            "level_mean": np.round(lvl_mean, 6),
            "level_lo": np.round(lvl_lo, 6), "level_hi": np.round(lvl_hi, 6),
        })
        if trend:
            slp_draws = _flat("slope")
            states["slope_mean"] = np.round(slp_draws.mean(0) * ysd, 6)
        states.to_csv(d / "state_space_states.csv", index=False, encoding="utf-8")
        files.append("state_space_states.csv")

        # forecast: simulate h steps forward per posterior draw (predictive intervals)
        S = lvl_draws.shape[0]
        rng = np.random.default_rng(sc["seed"])
        lev_p = lvl_draws[:, -1].copy()                  # standardized
        s_obs_d = _flat("s_obs"); s_lvl_d = _flat("s_lvl")
        if trend:
            del_p = _flat("slope")[:, -1].copy()
            s_slp_d = _flat("s_slp")
        fc = np.empty((S, horizon))
        for k in range(horizon):
            if trend:
                lev_p = lev_p + del_p + s_lvl_d * rng.standard_normal(S)
                del_p = del_p + s_slp_d * rng.standard_normal(S)
            else:
                lev_p = lev_p + s_lvl_d * rng.standard_normal(S)
            fc[:, k] = lev_p + s_obs_d * rng.standard_normal(S)
        fc = fc * ysd + ym
        fc_mean = fc.mean(0)
        fc_lo, fc_hi = np.percentile(fc, [3.0, 97.0], axis=0)
        forecast = pd.DataFrame({
            "h": np.arange(1, horizon + 1),
            "forecast": np.round(fc_mean, 6),
            "lo": np.round(fc_lo, 6), "hi": np.round(fc_hi, 6),
        })
        forecast.to_csv(d / "state_space_forecast.csv", index=False, encoding="utf-8")
        files.append("state_space_forecast.csv")

        # plot: observed + smoothed level (band) + forecast (band)
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(states["t"], states["observed"], ".", color="#666", ms=4, label="observed")
            ax.plot(states["t"], states["level_mean"], color="#4C72B0", lw=2, label="level (posterior mean)")
            ax.fill_between(states["t"], states["level_lo"], states["level_hi"],
                            color="#4C72B0", alpha=0.2, label="94% level band")
            ft = np.arange(T, T + horizon)
            ax.plot(ft, forecast["forecast"], color="#C44E52", lw=2, label="forecast")
            ax.fill_between(ft, forecast["lo"], forecast["hi"], color="#C44E52", alpha=0.2,
                            label="94% forecast band")
            ax.axvline(T - 0.5, ls="--", lw=1, color="grey", alpha=0.7)
            ax.set_xlabel("time index")
            ax.set_ylabel(str(name))
            ax.set_title(f"Bayesian structural time series ({model_kind}) — {name}")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "state_space.png", dpi=150)
            plt.close(fig)
            files.append("state_space.png")
        except Exception:
            pass

        sigma_obs = float(post["s_obs"].mean()) * ysd
        sigma_lvl = float(post["s_lvl"].mean()) * ysd
        final_level = float(lvl_mean[-1])
        estimates["n"] = float(T)
        estimates["local_linear_trend"] = 1.0 if trend else 0.0
        estimates["sigma_obs"] = round(sigma_obs, 5)
        estimates["sigma_level"] = round(sigma_lvl, 5)
        estimates["final_level"] = round(final_level, 5)
        estimates["forecast_next"] = round(float(fc_mean[0]), 5)
        estimates["forecast_horizon"] = float(horizon)
        estimates["forecast_last"] = round(float(fc_mean[-1]), 5)
        estimates["max_rhat"] = round(max_rhat, 4)
        estimates["min_ess"] = round(min_ess, 1)
        estimates["n_divergences"] = float(n_div)
        if trend:
            final_slope = float(states["slope_mean"].iloc[-1])
            estimates["sigma_slope"] = round(float(post["s_slp"].mean()) * ysd, 5)
            estimates["final_slope"] = round(final_slope, 5)
            trend_dir = "上升" if final_slope > 0 else "下降" if final_slope < 0 else "平稳"
        else:
            final_slope, trend_dir = None, "（局部水平模型，无斜率状态）"

        conv = _conv_note(max_rhat, min_ess, sc["chains"])
        div_note = f"；{n_div} 个发散（divergences）" if n_div else ""
        slope_line = (f"末期斜率（每步变化）≈{round(final_slope, 5)}（{trend_dir}），"
                      f"斜率新息 SD={round(float(post['s_slp'].mean()) * ysd, 5)}\n" if trend else "")
        (d / "state_space_summary.txt").write_text(
            f"贝叶斯结构时间序列（{model_kind}，PyMC NUTS）：序列 {name}\n"
            f"观测 {T}，预测 {horizon} 步\n"
            f"观测噪声 σ_obs={round(sigma_obs, 5)}，水平新息 σ_level={round(sigma_lvl, 5)}\n"
            + slope_line
            + f"末期水平={round(final_level, 5)}；下一步预测={round(float(fc_mean[0]), 5)} "
            f"[94%: {round(float(fc_lo[0]), 4)}, {round(float(fc_hi[0]), 4)}]；"
            f"第 {horizon} 步预测={round(float(fc_mean[-1]), 5)}\n"
            f"{conv}{div_note}\n"
            "注：状态用**非中心化累积和**参数化（新息 z~N(0,1)×SD，避免方差 funnel）；"
            "初始条件 mu0 为**首期水平**、初始漂移 d0 自第 2 期起作用（μ_t=μ_{t-1}+δ_{t-1}+η 对 t≥2）；"
            "y 已标准化使 HalfNormal 新息先验尺度合适（状态/方差/预测已回传原尺度）；"
            "局部线性趋势把当前斜率外推、预测区间随步长变宽；"
            "未含季节项（如需季节分解用频率派 unobserved_components）；σ_level/σ_slope 接近 0 表示该成分近似确定性。\n\n"
            "状态（前 10 行）：\n" + states.head(10).to_string(index=False)
            + "\n\n预测：\n" + forecast.to_string(index=False),
            encoding="utf-8",
        )
        files.append("state_space_summary.txt")

        summary.append(
            f"{method} 完成（PyMC, {model_kind}）：序列 {name}；{T} 观测；"
            f"σ_obs={round(sigma_obs, 4)}、σ_level={round(sigma_lvl, 4)}"
            + (f"、σ_slope={round(float(post['s_slp'].mean()) * ysd, 4)}；末期斜率≈{round(final_slope, 4)}（{trend_dir}）" if trend else "")
            + f"；下一步预测 {round(float(fc_mean[0]), 4)} [94%: {round(float(fc_lo[0]), 3)}, {round(float(fc_hi[0]), 3)}]、"
            f"{horizon} 步预测 {round(float(fc_mean[-1]), 4)}；{conv}{div_note}。"
            "⚠ 非中心化累积和状态（避 funnel）、y 标准化先验；预测区间随步长变宽；"
            "未含季节项（季节用 unobserved_components）；σ_level/σ_slope≈0 表近确定性成分。"
        )
        code += [
            "import pymc as pm, pytensor.tensor as pt",
            "# non-centered local linear trend: delta=d0+s_slp*cumsum(z_slp);",
            "# level=mu0+cumsum(shift(delta))+s_lvl*cumsum(z_lvl); y~N(level,s_obs)",
            "# forecast: simulate h steps forward per posterior draw -> predictive intervals",
        ]
    except Exception as err:  # pragma: no cover - sampler/runtime guard
        summary.append(f"{method} 失败：{err}")
