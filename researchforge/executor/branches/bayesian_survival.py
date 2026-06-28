"""Branch handler for Bayesian parametric survival — Weibull AFT with right-censoring (PyMC).

The frequentist parametric AFT models (Weibull / LogNormal / LogLogistic via lifelines)
live in ``survival.py`` (``parametric_survival``); this module adds the FULL Bayesian
Weibull accelerated-failure-time model fit by PyMC NUTS (no Stan/JAGS compiler needed).
The Bayesian fit yields full posteriors — per-covariate AFT effects with HDI credible
intervals, the Weibull shape with uncertainty, and honest convergence diagnostics —
where the frequentist fit gives only point + Wald CI.

Engine conventions (see CLAUDE.md「引擎约定」): a ``@register("<id>") def _branch_<id>(ctx)``
handler that unpacks ctx and MUTATES summary/estimates/files/code (never rebinds). Family
= ``survival``; auto-discovery registers it. Reuses the shared sampling / convergence /
HDI / forest / honest-degrade helpers from ``bayesian_mcmc.py`` (already inference-reviewed)
and the duration/event column resolver from ``survival.py``.

Column resolution (survival convention — duration IS a time column, so we do NOT exclude
fp.time_col): duration = config ``duration`` else fp.time_col else a positive
continuous/count/id column; event indicator = config ``event`` else a binary 0/1 column
(1=event/death, 0=right-censored). If no event column is found ALL rows are treated as
observed (no censoring) and this is disclosed. Needs ≥15 valid rows with ≥1 event.

THE LIKELIHOOD (the #1 thing to get right). We use ``pm.Censored``:

    obs   = pm.Weibull.dist(alpha=shape, beta=lam)         # latent un-censored survival time
    pm.Censored("y", obs, lower=None, upper=upper, observed=duration)

with ``upper = +inf`` for observed events and ``upper = duration`` for right-censored rows.
``pm.Censored`` evaluates, per observation:
  * observed < upper  → logpdf(observed)                          (an EVENT: full density)
  * observed >= upper → log(1 - cdf(upper)) = log-survival(upper) (CENSORED: P(T > c))
Because censored rows have ``upper == duration == observed``, they contribute the log
survival function logS(c); events have ``upper == +inf`` so ``observed < upper`` always and
they contribute the Weibull logpdf. That is EXACTLY the right-censored Weibull likelihood
(events use logpdf, censored use log-survival) — verified against the standard form below.

Optional + graceful degrade: if PyMC/ArviZ are unavailable the handler appends an honest
Chinese 跳过 pointing at the lifelines frequentist survival models and returns — it never
crashes. Sampling is seeded (reproducible) at target_accept=0.95, cores=1; convergence is
reported honestly (max R-hat > 1.05 raises a ⚠). Covariates are standardized for stable NUTS
geometry; AFT effects are reported on the standardized scale plus a back-transformed
acceleration factor exp(beta_std / sd_x) on the raw covariate scale.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.branches.bayesian_mcmc import (
    _conv_note,
    _convergence,
    _degrade,
    _forest,
    _have_pymc,
    _hdi_bounds,
    _sampler_cfg,
    _standardize,
)
from researchforge.executor.branches.survival import _resolve_duration_event


@register("bayesian_survival")
def _branch_bayesian_survival(ctx: Ctx) -> None:
    import numpy as np
    import pandas as pd

    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    method = entry.method

    if not _have_pymc():
        _degrade(summary, method, "lifelines 生存分析 survival_analysis / parametric_survival")
        return

    # ── resolve duration + (optional) event columns ───────────────────────────
    dur_col, event_col = _resolve_duration_event(df, fp, cfg)
    if dur_col is None:
        summary.append(
            f"{method} 跳过：未找到时长列（需 1 个正数时长列，如 time/duration/days）。"
            "用 config={\"duration\":..} 指定。"
        )
        return

    # covariates: config predictors else remaining numeric (continuous/count/binary),
    # excluding the duration, the event and the unit columns.
    excl = {fp.unit_col, dur_col} | ({event_col} if event_col else set())
    forced = [c for c in (cfg.get("predictors") or []) if c in df.columns and c not in excl]
    if forced:
        covars = forced[:15]
    else:
        covars = [
            c.name for c in fp.columns
            if c.kind in {"continuous", "count", "binary"} and c.name not in excl
        ][:15]
    covars = [c for c in covars if pd.to_numeric(df[c], errors="coerce").notna().any()]

    # ── clean + build duration / event / X ────────────────────────────────────
    keep = [dur_col] + ([event_col] if event_col else []) + covars
    sub = df[list(dict.fromkeys(keep))].copy()
    sub[dur_col] = pd.to_numeric(sub[dur_col], errors="coerce")
    if event_col:
        sub[event_col] = pd.to_numeric(sub[event_col], errors="coerce")
    for c in covars:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna()
    sub = sub[sub[dur_col].astype(float) > 0]          # Weibull support is t > 0

    if len(sub) < 15:
        summary.append(f"{method} 跳过：有效样本不足（去缺失/非正时长后 < 15 行）。")
        return

    dur = sub[dur_col].to_numpy(float)
    no_event_col = event_col is None
    if no_event_col:
        # no event indicator found → treat all as observed (no censoring) and disclose
        ev = np.ones(len(sub), dtype=int)
    else:
        ev_raw = sub[event_col].to_numpy(float)
        if set(np.unique(ev_raw)) - {0.0, 1.0}:
            summary.append(
                f"{method} 跳过：事件列 {event_col} 非 0/1 编码（应 1=事件/死亡, 0=右删失）。"
            )
            return
        ev = ev_raw.astype(int)
    n_events = int(ev.sum())
    n_censored = int((ev == 0).sum())
    if n_events < 1:
        summary.append(f"{method} 跳过：无观测事件（全部右删失），Weibull AFT 无法识别尺度。")
        return

    # design matrix (drop constant columns: zero variance kills the standardized slope)
    Xraw, used = [], []
    for c in covars:
        col = sub[c].to_numpy(float)
        if np.std(col) > 1e-12:
            Xraw.append(col)
            used.append(c)
    has_cov = len(used) > 0
    if has_cov:
        X = np.column_stack(Xraw)
        Xz, _, xsd = _standardize(X)
    covars = used

    import pymc as pm

    sc = _sampler_cfg(cfg)
    log_mean_dur = float(np.log(np.mean(dur)))
    # upper bound for pm.Censored: +inf for events (→ logpdf), the duration for censored
    # rows (observed == upper → log survival). This IS the right-censored likelihood.
    upper = np.where(ev == 1, np.inf, dur)

    with pm.Model() as model:
        alpha = pm.HalfNormal("alpha", 2.0)                              # Weibull shape k (>0)
        beta0 = pm.Normal("beta0", mu=log_mean_dur, sigma=2.0)          # AFT log-scale intercept
        if has_cov:
            betas = pm.Normal("betas", 0.0, 1.0, shape=Xz.shape[1])     # log-AFT effects (std scale)
            eta = beta0 + pm.math.dot(Xz, betas)
        else:
            eta = beta0
        lam = pm.Deterministic("lam", pm.math.exp(eta))                 # Weibull scale (AFT)
        obs = pm.Weibull.dist(alpha=alpha, beta=lam)
        pm.Censored("y", obs, lower=None, upper=upper, observed=dur)
        idata = pm.sample(draws=sc["draws"], tune=sc["tune"], chains=sc["chains"],
                          cores=1, random_seed=sc["seed"], progressbar=False,
                          target_accept=0.95)

    post = idata.posterior
    shape_mean = float(post["alpha"].values.mean())
    conv_vars = ["alpha"] + (["betas"] if has_cov else ["beta0"])
    max_rhat, min_ess = _convergence(idata, conv_vars)

    # median survival time at the average covariate profile (x = 0 on the std scale, so
    # the AFT scale is exp(beta0)). For Weibull(shape=k, scale=lam): median = lam·(ln2)^(1/k).
    beta0_mean = float(post["beta0"].values.mean())
    lam0 = float(np.exp(beta0_mean))
    median_surv = float(lam0 * (np.log(2.0)) ** (1.0 / shape_mean)) if shape_mean > 0 else float("nan")

    estimates["weibull_shape"] = round(shape_mean, 5)
    estimates["median_survival"] = round(median_surv, 5)
    estimates["n_events"] = float(n_events)
    estimates["n_censored"] = float(n_censored)
    estimates["n_obs"] = float(len(sub))
    estimates["max_rhat"] = round(max_rhat, 4)
    estimates["min_ess"] = round(min_ess, 1)

    lo_std, hi_std = [], []
    if has_cov:
        b_mean = post["betas"].values.reshape(-1, Xz.shape[1]).mean(axis=0)
        b_lo, b_hi = _hdi_bounds(idata, "betas", sc["hdi"])
        # back-transform the AFT log-effect to raw-covariate scale: beta_raw = beta_std / sd_x.
        # acceleration factor AF = exp(beta_raw); AF>1 → covariate LENGTHENS survival.
        raw_b = b_mean / xsd
        raw_lo, raw_hi = b_lo / xsd, b_hi / xsd
        af_mean, af_lo, af_hi = np.exp(raw_b), np.exp(raw_lo), np.exp(raw_hi)
        for j, name in enumerate(covars):
            estimates[f"aft_{name}"] = round(float(b_mean[j]), 5)            # std-scale AFT coef
            estimates[f"aft_{name}__hdi_low"] = round(float(b_lo[j]), 5)
            estimates[f"aft_{name}__hdi_high"] = round(float(b_hi[j]), 5)
            estimates[f"af_{name}"] = round(float(af_mean[j]), 5)            # raw-scale accel factor
            estimates[f"af_{name}__hdi_low"] = round(float(af_lo[j]), 5)
            estimates[f"af_{name}__hdi_high"] = round(float(af_hi[j]), 5)
        lo_std, hi_std = list(b_lo), list(b_hi)

    # ── per-covariate acceleration-factor CSV ─────────────────────────────────
    if has_cov:
        try:
            pd.DataFrame({
                "covariate": covars,
                "aft_coef_std": np.round(b_mean, 5),
                "accel_factor": np.round(af_mean, 5),
                f"af_hdi_{int(sc['hdi']*100)}_low": np.round(af_lo, 5),
                f"af_hdi_{int(sc['hdi']*100)}_high": np.round(af_hi, 5),
            }).to_csv(d / "bayesian_survival_aft.csv", index=False, encoding="utf-8")
            files.append("bayesian_survival_aft.csv")
        except Exception:
            pass
        _forest(idata, ["betas"], d / "bayesian_survival_forest.png",
                "Weibull AFT log-effects (standardized, 94% HDI)")
        if (d / "bayesian_survival_forest.png").exists():
            files.append("bayesian_survival_forest.png")

    # ── Chinese summary (headline first) ──────────────────────────────────────
    shape_txt = ("形状 α>1：风险随时间递增（老化/磨损型）" if shape_mean > 1.05
                 else "形状 α<1：风险随时间递减（早期淘汰型）" if shape_mean < 0.95
                 else "形状 α≈1：风险近似恒定（指数分布极限）")
    if has_cov:
        longer = [covars[j] for j in range(len(covars)) if lo_std[j] > 0]    # AF HDI 全 >1
        shorter = [covars[j] for j in range(len(covars)) if hi_std[j] < 0]   # AF HDI 全 <1
        eff_txt = (
            "延长存活（AF>1，HDI 不含 1）的协变量：" + ("、".join(longer) if longer else "无")
            + "；缩短存活（AF<1，HDI 不含 1）的：" + ("、".join(shorter) if shorter else "无")
        )
    else:
        eff_txt = "无可用协变量，仅估计基线 Weibull（截距-only AFT）"
    cens_note = (
        "未找到事件指示列，已把所有观测当作完整事件（无删失）处理 ⚠（如有右删失请用 config event=列名）"
        if no_event_col else f"事件 {n_events} 例、右删失 {n_censored} 例"
    )

    summary.append(
        f"{method} 完成：贝叶斯 Weibull 加速失效时间(AFT)生存模型（PyMC NUTS，"
        f"{sc['chains']}链×{sc['draws']}抽样，右删失似然），时长列={dur_col}"
        + (f"，事件列={event_col}" if event_col else "")
        + f"，{len(sub)} 例（{cens_note}）。"
        f"Weibull 形状 α≈{shape_mean:.3g}（{shape_txt}）；中位生存时间≈{median_surv:.3g}（平均协变量处）。"
        f"{eff_txt}。{_conv_note(max_rhat, min_ess, sc['chains'])}。"
        " ⚠ 这是 Weibull 参数 AFT 假定——请与非参 Kaplan-Meier 对照核验分布形状；"
        "AFT 效应在【时间尺度】上（加速因子 AF>1=存活更久/事件推迟），并非风险比 HR（除非显式换算）；"
        "右删失假定为随机非信息性；先验弱信息（α~HalfNormal(2)、AFT 系数~Normal(0,1) 于标准化尺度）；"
        "协变量已标准化采样，aft_<名> 为标准化尺度系数、af_<名>=exp(系数/sd) 已回传原尺度。"
    )
    code.append(
        "import pymc as pm, numpy as np\n"
        "# 右删失 Weibull AFT：events 用 logpdf、censored 用 log-survival（pm.Censored 自动切换）\n"
        "upper = np.where(event==1, np.inf, duration)  # +inf=事件→logpdf；duration=删失→logS\n"
        "with pm.Model():\n"
        "    alpha=pm.HalfNormal('alpha',2.0)               # Weibull 形状 k\n"
        "    beta0=pm.Normal('beta0',np.log(duration.mean()),2.0)\n"
        "    betas=pm.Normal('betas',0,1,shape=p)           # AFT 对数效应(标准化尺度)\n"
        "    lam=pm.Deterministic('lam',pm.math.exp(beta0+pm.math.dot(Xz,betas)))  # 尺度\n"
        "    obs=pm.Weibull.dist(alpha=alpha,beta=lam)\n"
        "    pm.Censored('y',obs,lower=None,upper=upper,observed=duration)\n"
        "    idata=pm.sample(1000,tune=1000,chains=2,target_accept=0.95,random_seed=42)\n"
        "# AF = exp(beta_std/sd_x)；AF>1 → 该协变量延长存活时间"
    )
