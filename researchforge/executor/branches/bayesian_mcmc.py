"""Branch handlers for the Bayesian (MCMC) methods — modern PyMC, pure Python.

The conjugate Bayesian methods (A/B, proportion, Poisson rate) live in
``bayesian.py``; this module adds the FULL Bayesian regression workhorses that used
to be blocked on a Stan/JAGS compiler. Modern PyMC (NUTS via PyTensor) needs no C
compiler — it samples fine on a bare Windows Python — so these are now feasible:

* ``bayesian_regression``            — Bayesian linear regression: coefficient
                                       posteriors + 94% HDI credible intervals,
                                       Bayesian R², convergence (R-hat / ESS).
* ``bayesian_logistic_regression``   — Bayesian logistic regression: posterior
                                       odds ratios + HDI, convergence.
* ``bayesian_hierarchical``          — varying-intercept multilevel model (partial
                                       pooling): population + per-group estimates,
                                       between/within variance, shrinkage.

Engine conventions (see CLAUDE.md「引擎约定」): each handler is
``@register("<id>") def _branch_<id>(ctx)``; unpack ctx, MUTATE summary/estimates/
files/code (never rebind). Family = ``bayesian`` (shared with the conjugate file;
auto-discovery registers both). Outcome = first continuous column, predictors = the
remaining numeric columns; both config-overridable (regression-family convention).

Optional + graceful degrade (mirrors the R bridge): if PyMC/ArviZ aren't installed,
each handler appends an honest Chinese "跳过" pointing at the frequentist equivalent
(ols_regression / logistic_regression / glmm) or the conjugate bayesian_* methods,
and returns — never crashes. Sampling is seeded (reproducible); predictors are
standardized for stable NUTS geometry and coefficients are back-transformed to the
raw scale (a constant rescaling, so HDIs map exactly). Convergence is reported
honestly: max R-hat > 1.05 raises a ⚠ non-convergence note. Plots (ArviZ forest,
matplotlib Agg, English labels) are best-effort in try/except.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.run import resolve_outcome, resolve_predictors

# default sampler settings — modest but enough for screening; config-overridable
_DRAWS, _TUNE, _CHAINS, _SEED, _HDI = 1000, 1000, 2, 42, 0.94


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────
def _have_pymc() -> bool:
    import importlib.util

    return (importlib.util.find_spec("pymc") is not None
            and importlib.util.find_spec("arviz") is not None)


def _degrade(summary, method: str, alt: str) -> None:
    summary.append(
        f"{method} 跳过：未安装 pymc/arviz（pip install pymc arviz）。"
        f"可改用频率派 {alt}，或共轭 bayesian_ab_test / bayesian_proportion / bayesian_poisson_rate。"
    )


def _sampler_cfg(cfg: dict) -> dict:
    def _int(key, default):
        try:
            return max(1, int(cfg.get(key)))
        except (TypeError, ValueError):
            return default

    hdi = cfg.get("hdi_prob")
    try:
        hdi = float(hdi)
        if not (0.5 <= hdi < 1.0):
            hdi = _HDI
    except (TypeError, ValueError):
        hdi = _HDI
    return {
        "draws": _int("draws", _DRAWS), "tune": _int("tune", _TUNE),
        "chains": _int("chains", _CHAINS), "seed": _int("seed", _SEED), "hdi": hdi,
    }


def _resolve_reg(ctx: Ctx, method: str, *, binary_outcome: bool):
    """Resolve (outcome, predictors) by the regression-family convention. When
    ``binary_outcome`` the outcome must be a binary / 2-value column; otherwise the
    first continuous column. predictors = remaining numeric (config-overridable).
    Returns (outcome, preds, problem) — problem set → caller appends + returns."""
    import pandas as pd

    fp, cfg, df = ctx.fp, ctx.cfg, ctx.df
    excl = {fp.unit_col, fp.time_col}
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in excl]

    def _is_binary(col):
        return df[col].dropna().nunique() == 2

    # shared resolver (config already honoured above via the wider any-column check):
    # high-confidence detected outcome > first non-treatment-named candidate > first.
    from researchforge.executor.run import resolve_outcome

    forced_y = cfg.get("outcome")
    if forced_y in df.columns:
        outcome = forced_y
    elif binary_outcome:
        bins = [c.name for c in fp.columns if c.kind == "binary" and c.name not in excl]
        bins += [c for c in df.columns if c not in bins and c not in excl and _is_binary(c)]
        if not bins:
            return None, [], f"{method} 跳过：未找到二值结果变量（需 1 个两类别列）。"
        outcome = resolve_outcome(fp, cfg, bins)
    elif cont:
        outcome = resolve_outcome(fp, cfg, cont)
    else:
        return None, [], f"{method} 跳过：未找到连续结果变量（需 1 个连续列）。"

    preds = resolve_predictors(
        fp, cfg, outcome, kinds=("continuous", "count", "binary"), cap=20, df=df
    )
    preds = [c for c in preds if pd.to_numeric(df[c], errors="coerce").notna().any()]
    if not preds:
        return None, [], f"{method} 跳过：未找到可用的数值预测变量（≥1 个）。"
    return outcome, preds, None


def _clean(df, outcome, preds, *, binary_outcome: bool):
    """Numeric-coerce, drop NaN rows. Returns (X, y, preds, problem)."""
    import pandas as pd

    X = df[preds].apply(lambda s: pd.to_numeric(s, errors="coerce"))
    y = pd.to_numeric(df[outcome], errors="coerce")
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X.loc[mask], y.loc[mask]
    if len(y) < 10:
        return None, None, None, "有效样本不足（去缺失后 < 10 行）。"
    if binary_outcome:
        vals = sorted(y.unique())
        if len(vals) != 2:
            return None, None, None, "二值结果变量去缺失后不是恰两类。"
        y = (y == vals[1]).astype(float)  # higher level = the modelled "success"
    return X.to_numpy(float), y.to_numpy(float), list(preds), None


def _standardize(X):
    """z-score each column; return (Xz, mean, sd) with sd guarded against 0."""
    import numpy as np

    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd, mu, sd


def _hdi_bounds(idata, name, prob):
    """(lo, hi) HDI arrays for a posterior variable at probability `prob`. The HDI
    kwarg name changed across ArviZ versions (`hdi_prob` → `prob`), so try both."""
    import arviz as az

    try:
        h = az.hdi(idata, var_names=[name], prob=prob)[name].values
    except TypeError:
        h = az.hdi(idata, var_names=[name], hdi_prob=prob)[name].values
    return h[..., 0], h[..., 1]


def _convergence(idata, names):
    """(max_rhat, min_ess) across the named variables; honest convergence read.

    ArviZ ≥1.2 default-formats r_hat as a STRING and ess as nullable Int64, so a
    plain ``.max()`` would be a lexicographic compare (silently hiding R-hat≥10) and
    ``float(pd.NA)`` could crash. ``round_to="none"`` returns raw floats — a true
    numeric max/min. Older ArviZ ignores the kwarg, so fall back gracefully."""
    import arviz as az
    import numpy as np

    try:
        summ = az.summary(idata, var_names=list(names), round_to="none")
    except TypeError:
        summ = az.summary(idata, var_names=list(names))
    max_rhat = float(np.nanmax(np.asarray(summ["r_hat"], dtype=float)))
    min_ess = float(np.nanmin(np.asarray(summ["ess_bulk"], dtype=float)))
    return max_rhat, min_ess


def _conv_note(max_rhat, min_ess, chains=_CHAINS):
    chain_warn = "；⚠ 仅 %d 链（推荐 ≥4 链以稳健估计 R-hat）" % chains if chains < 4 else ""
    if max_rhat > 1.05:
        return f"⚠ 未收敛（max R-hat={max_rhat:.3f}>1.05，加大 draws/tune 或检查模型）{chain_warn}"
    return f"收敛良好（max R-hat={max_rhat:.3f}, min ESS={min_ess:.0f}）{chain_warn}"


def _forest(idata, var_names, path, title):
    """Best-effort ArviZ forest plot of coefficient posteriors (English labels)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import arviz as az
        import matplotlib.pyplot as plt

        try:
            az.plot_forest(idata, var_names=var_names, combined=True, prob=0.94)
        except TypeError:
            az.plot_forest(idata, var_names=var_names, combined=True, hdi_prob=0.94)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path, dpi=110, bbox_inches="tight")
        plt.close("all")
    except Exception:
        pass


def _ppc(model, idata, y_obs, d, fname, estimates, *, seed=42):
    """Best-effort posterior predictive check. Samples y_rep from the fitted model,
    plots ``az.plot_ppc`` (observed density vs replicated densities, English labels),
    and computes Bayesian posterior-predictive p-values for two test statistics
    (mean & SD): p = P(T(y_rep) ≥ T(y_obs)). p ≈ 0.5 means the data are consistent
    with the model for that statistic; p near 0 or 1 means the model mis-captures it.

    Writes ``ppc_bayes_p_mean`` / ``ppc_bayes_p_sd`` into ``estimates`` and returns
    the plot filename (or None). NEVER raises — PPC is purely additive, so any failure
    (no matplotlib, ArviZ quirk, sampling hiccup) leaves the host method intact."""
    try:
        import arviz as az
        import matplotlib
        import numpy as np
        import pymc as pm

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with model:
            pp = pm.sample_posterior_predictive(idata, random_seed=seed, progressbar=False)
        # (chain, draw, obs) → (n_samples, n_obs); the likelihood var is named "y_obs"
        yrep = np.asarray(pp.posterior_predictive["y_obs"].values, dtype=float).reshape(
            -1, len(y_obs))
        yo = np.asarray(y_obs, dtype=float)
        # Bayesian p-value = P(T(y_rep) ≥ T(y_obs)); ~0.5 = good fit, near 0/1 = misfit
        p_mean = float(np.mean(yrep.mean(axis=1) >= yo.mean()))
        p_sd = float(np.mean(yrep.std(axis=1) >= yo.std()))
        estimates["ppc_bayes_p_mean"] = round(p_mean, 4)
        estimates["ppc_bayes_p_sd"] = round(p_sd, 4)
        try:
            idata.extend(pp)                       # attach posterior_predictive group
            az.plot_ppc(idata, num_pp_samples=100)
            plt.title("Posterior predictive check (observed vs replicated)")
            plt.tight_layout()
            plt.savefig(d / fname, dpi=110, bbox_inches="tight")
            plt.close("all")
            return fname if (d / fname).exists() else None
        except Exception:
            plt.close("all")
            return None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 1) Bayesian linear regression
# ─────────────────────────────────────────────────────────────────────────────
@register("bayesian_regression")
def _branch_bayesian_regression(ctx: Ctx) -> None:
    import numpy as np

    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    method = entry.method

    if not _have_pymc():
        _degrade(summary, method, "ols_regression")
        return
    outcome, preds, problem = _resolve_reg(ctx, method, binary_outcome=False)
    if problem:
        summary.append(problem)
        return
    X, y, preds, problem = _clean(df, outcome, preds, binary_outcome=False)
    if problem:
        summary.append(f"{method} 跳过：{problem}")
        return

    import pymc as pm

    sc = _sampler_cfg(cfg)
    Xz, xmu, xsd = _standardize(X)
    y_sd = float(np.std(y)) or 1.0
    y_mean = float(np.mean(y))

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=y_mean, sigma=2.5 * y_sd)
        beta = pm.Normal("beta", mu=0.0, sigma=2.5 * y_sd, shape=Xz.shape[1])
        sigma = pm.HalfNormal("sigma", sigma=2.5 * y_sd)
        mu = alpha + pm.math.dot(Xz, beta)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
        idata = pm.sample(draws=sc["draws"], tune=sc["tune"], chains=sc["chains"],
                          cores=1, random_seed=sc["seed"], progressbar=False)

    # posterior means + HDI on the standardized scale, then back-transform to raw:
    # β_raw_j = β_std_j / sd_j ; HDI scales by the same positive constant.
    b_mean = idata.posterior["beta"].values.reshape(-1, Xz.shape[1]).mean(axis=0)
    b_lo, b_hi = _hdi_bounds(idata, "beta", sc["hdi"])
    raw_b = b_mean / xsd
    raw_lo, raw_hi = b_lo / xsd, b_hi / xsd
    max_rhat, min_ess = _convergence(idata, ["alpha", "beta", "sigma"])

    # Bayesian R² (approximate, Gelman-style): var(fitted)/(var(fitted)+var(resid)) at the
    # posterior-MEAN fit (the exact Gelman et al. 2019 form averages a per-draw R²; this is a fast proxy)
    a_mean = float(idata.posterior["alpha"].values.mean())
    fitted = a_mean + Xz @ b_mean
    resid = y - fitted
    bayes_r2 = float(np.var(fitted) / (np.var(fitted) + np.var(resid))) if np.var(resid) > 0 else float("nan")

    for j, name in enumerate(preds):
        estimates[f"beta__{name}"] = round(float(raw_b[j]), 5)
        estimates[f"beta__{name}__hdi_low"] = round(float(raw_lo[j]), 5)
        estimates[f"beta__{name}__hdi_high"] = round(float(raw_hi[j]), 5)
    estimates["bayes_r2"] = round(bayes_r2, 4)
    estimates["max_rhat"] = round(max_rhat, 4)
    estimates["min_ess"] = round(min_ess, 1)

    try:
        import pandas as pd

        tbl = pd.DataFrame({
            "predictor": preds,
            "beta_mean": np.round(raw_b, 5),
            f"hdi_{int(sc['hdi']*100)}_low": np.round(raw_lo, 5),
            f"hdi_{int(sc['hdi']*100)}_high": np.round(raw_hi, 5),
        })
        tbl.to_csv(d / "bayesian_regression_coefs.csv", index=False, encoding="utf-8")
        files.append("bayesian_regression_coefs.csv")
    except Exception:
        pass
    _forest(idata, ["beta"], d / "bayesian_regression_forest.png", "Posterior coefficients (94% HDI)")
    if (d / "bayesian_regression_forest.png").exists():
        files.append("bayesian_regression_forest.png")

    ppc_png = _ppc(model, idata, y, d, "bayesian_regression_ppc.png", estimates, seed=sc["seed"])
    if ppc_png:
        files.append(ppc_png)

    sig = [preds[j] for j in range(len(preds)) if raw_lo[j] > 0 or raw_hi[j] < 0]
    ppc_note = (
        f" 后验预测检查：贝叶斯 p（均值）≈{estimates['ppc_bayes_p_mean']:.2f}、"
        f"p（标准差）≈{estimates['ppc_bayes_p_sd']:.2f}（≈0.5=数据与模型一致；"
        "接近 0 或 1=模型未能复现该统计量）。"
        if "ppc_bayes_p_mean" in estimates else ""
    )
    summary.append(
        f"{method} 完成：贝叶斯线性回归（PyMC NUTS，{sc['chains']}链×{sc['draws']}抽样），"
        f"结果={outcome}，预测变量 {len(preds)} 个。{int(sc['hdi']*100)}% HDI 不含 0 的："
        + ("、".join(sig) if sig else "无")
        + f"。贝叶斯 R²≈{bayes_r2:.3f}。{_conv_note(max_rhat, min_ess, sc['chains'])}。"
        + ppc_note
        + " ⚠ 系数为后验均值 + 最高密度可信区间（HDI，非频率派置信区间）；先验为弱信息 "
        "Normal/HalfNormal；预测变量已标准化采样后回传原尺度。"
    )
    code.append(
        "import pymc as pm, numpy as np\n"
        "# 标准化预测变量 → 弱信息先验 → NUTS 采样 → β 回传原尺度 β_raw=β_std/sd_x\n"
        "with pm.Model():\n"
        "    a=pm.Normal('alpha',mu=y.mean(),sigma=2.5*y.std())\n"
        "    b=pm.Normal('beta',0,2.5*y.std(),shape=X.shape[1])\n"
        "    s=pm.HalfNormal('sigma',2.5*y.std())\n"
        "    pm.Normal('y_obs',a+pm.math.dot(Xz,b),s,observed=y)\n"
        "    idata=pm.sample(1000,tune=1000,chains=2,random_seed=42)\n"
        "import arviz as az; az.summary(idata)  # mean/sd/HDI/r_hat/ess"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2) Bayesian logistic regression
# ─────────────────────────────────────────────────────────────────────────────
@register("bayesian_logistic_regression")
def _branch_bayesian_logistic_regression(ctx: Ctx) -> None:
    import numpy as np

    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    method = entry.method

    if not _have_pymc():
        _degrade(summary, method, "logistic_regression")
        return
    outcome, preds, problem = _resolve_reg(ctx, method, binary_outcome=True)
    if problem:
        summary.append(problem)
        return
    X, y, preds, problem = _clean(df, outcome, preds, binary_outcome=True)
    if problem:
        summary.append(f"{method} 跳过：{problem}")
        return

    import pymc as pm

    sc = _sampler_cfg(cfg)
    Xz, xmu, xsd = _standardize(X)

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0.0, sigma=2.5)
        beta = pm.Normal("beta", mu=0.0, sigma=2.5, shape=Xz.shape[1])  # weakly-informative (std-x log-odds)
        logit = alpha + pm.math.dot(Xz, beta)
        pm.Bernoulli("y_obs", logit_p=logit, observed=y)
        idata = pm.sample(draws=sc["draws"], tune=sc["tune"], chains=sc["chains"],
                          cores=1, random_seed=sc["seed"], progressbar=False)

    b_mean = idata.posterior["beta"].values.reshape(-1, Xz.shape[1]).mean(axis=0)
    b_lo, b_hi = _hdi_bounds(idata, "beta", sc["hdi"])
    raw_b = b_mean / xsd            # log-odds per 1 raw-unit of predictor
    raw_lo, raw_hi = b_lo / xsd, b_hi / xsd
    or_mean, or_lo, or_hi = np.exp(raw_b), np.exp(raw_lo), np.exp(raw_hi)
    max_rhat, min_ess = _convergence(idata, ["alpha", "beta"])

    for j, name in enumerate(preds):
        estimates[f"or__{name}"] = round(float(or_mean[j]), 5)
        estimates[f"or__{name}__hdi_low"] = round(float(or_lo[j]), 5)
        estimates[f"or__{name}__hdi_high"] = round(float(or_hi[j]), 5)
    estimates["max_rhat"] = round(max_rhat, 4)
    estimates["min_ess"] = round(min_ess, 1)

    try:
        import pandas as pd

        tbl = pd.DataFrame({
            "predictor": preds,
            "odds_ratio": np.round(or_mean, 5),
            f"hdi_{int(sc['hdi']*100)}_low": np.round(or_lo, 5),
            f"hdi_{int(sc['hdi']*100)}_high": np.round(or_hi, 5),
        })
        tbl.to_csv(d / "bayesian_logistic_or.csv", index=False, encoding="utf-8")
        files.append("bayesian_logistic_or.csv")
    except Exception:
        pass
    _forest(idata, ["beta"], d / "bayesian_logistic_forest.png", "Posterior log-odds (94% HDI)")
    if (d / "bayesian_logistic_forest.png").exists():
        files.append("bayesian_logistic_forest.png")

    sig = [preds[j] for j in range(len(preds)) if or_lo[j] > 1 or or_hi[j] < 1]
    summary.append(
        f"{method} 完成：贝叶斯逻辑回归（PyMC NUTS，{sc['chains']}链×{sc['draws']}抽样），"
        f"结果={outcome}（建模较高水平为“阳性”），预测变量 {len(preds)} 个。"
        f"优势比 OR {int(sc['hdi']*100)}% HDI 不含 1 的："
        + ("、".join(sig) if sig else "无")
        + f"。{_conv_note(max_rhat, min_ess, sc['chains'])}。"
        " ⚠ OR=exp(后验均值 log-odds)（HDI 可信区间）；先验为弱信息 Normal(0,2.5)"
        "（标准化预测变量的 log-odds 尺度；非 Gelman 默认的 Cauchy(0,2.5)/0.5-SD 缩放）。"
    )
    code.append(
        "import pymc as pm\n"
        "with pm.Model():\n"
        "    a=pm.Normal('alpha',0,2.5); b=pm.Normal('beta',0,2.5,shape=X.shape[1])\n"
        "    pm.Bernoulli('y_obs',logit_p=a+pm.math.dot(Xz,b),observed=y)\n"
        "    idata=pm.sample(1000,tune=1000,chains=2,random_seed=42)\n"
        "# OR = exp(beta_std / sd_x)  (back to raw predictor scale)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3) Bayesian hierarchical (varying-intercept partial pooling)
# ─────────────────────────────────────────────────────────────────────────────
@register("bayesian_hierarchical")
def _branch_bayesian_hierarchical(ctx: Ctx) -> None:
    import numpy as np
    import pandas as pd

    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    method = entry.method

    if not _have_pymc():
        _degrade(summary, method, "glmm（混合效应）")
        return

    excl = {fp.unit_col, fp.time_col}
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in excl]
    # grouping column: config 'group'/'unit', else unit_col, else a low-cardinality
    # categorical/count column with ≥2 (and not ~all-distinct) levels.
    forced_g = cfg.get("group") or cfg.get("unit")
    group_col = forced_g if forced_g in df.columns else fp.unit_col
    if group_col not in (df.columns if group_col else []):
        cands = [
            c.name for c in fp.columns
            if c.kind in {"categorical", "binary", "count", "id"} and c.name not in excl
            and 2 <= df[c.name].dropna().nunique() <= max(2, len(df) // 3)
        ]
        group_col = cands[0] if cands else None
    if not group_col:
        summary.append(f"{method} 跳过：未找到分组变量（需 1 个分组列，如个体/班级/地区）。")
        return

    outcome = cfg.get("outcome") if cfg.get("outcome") in df.columns else (resolve_outcome(fp, cfg, cont) if cont else None)
    if not outcome:
        summary.append(f"{method} 跳过：未找到连续结果变量。")
        return
    pred = cfg.get("predictor") if cfg.get("predictor") in df.columns else next(
        (c for c in cont if c != outcome), None)

    use_cols = [outcome, group_col] + ([pred] if pred else [])
    sub = df[use_cols].copy()
    sub[outcome] = pd.to_numeric(sub[outcome], errors="coerce")
    if pred:
        sub[pred] = pd.to_numeric(sub[pred], errors="coerce")
    sub = sub.dropna()
    if len(sub) < 12 or sub[group_col].nunique() < 3:
        summary.append(f"{method} 跳过：分组层级建模需 ≥3 组且 ≥12 个有效观测。")
        return

    codes, uniques = pd.factorize(sub[group_col])
    n_groups = len(uniques)
    y = sub[outcome].to_numpy(float)
    y_mean, y_sd = float(np.mean(y)), float(np.std(y)) or 1.0
    x = None
    if pred:
        xv = sub[pred].to_numpy(float)
        x = (xv - xv.mean()) / (xv.std(ddof=0) or 1.0)  # centered+scaled slope covariate

    import pymc as pm

    sc = _sampler_cfg(cfg)
    with pm.Model() as model:
        mu_a = pm.Normal("mu_a", mu=y_mean, sigma=2.5 * y_sd)        # population mean
        sigma_a = pm.HalfNormal("sigma_a", sigma=2.5 * y_sd)         # between-group SD
        z = pm.Normal("z", mu=0.0, sigma=1.0, shape=n_groups)        # non-centered
        a = pm.Deterministic("a", mu_a + sigma_a * z)               # group intercepts
        sigma = pm.HalfNormal("sigma", sigma=2.5 * y_sd)            # within-group SD
        mu = a[codes] + (pm.Normal("beta", 0.0, 2.5 * y_sd) * x if pred else 0.0)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
        idata = pm.sample(draws=sc["draws"], tune=sc["tune"], chains=sc["chains"],
                          cores=1, random_seed=sc["seed"], progressbar=False,
                          target_accept=0.95)

    conv_vars = ["mu_a", "sigma_a", "sigma"] + (["beta"] if pred else [])
    max_rhat, min_ess = _convergence(idata, conv_vars)
    post = idata.posterior
    mu_a_m = float(post["mu_a"].values.mean())
    sig_a_m = float(post["sigma_a"].values.mean())
    sig_m = float(post["sigma"].values.mean())
    icc = sig_a_m ** 2 / (sig_a_m ** 2 + sig_m ** 2) if (sig_a_m + sig_m) > 0 else float("nan")

    estimates["population_mean"] = round(mu_a_m, 5)
    estimates["between_group_sd"] = round(sig_a_m, 5)
    estimates["within_group_sd"] = round(sig_m, 5)
    estimates["icc"] = round(icc, 4)
    estimates["n_groups"] = float(n_groups)
    estimates["max_rhat"] = round(max_rhat, 4)
    estimates["min_ess"] = round(min_ess, 1)
    if pred:
        b_m = float(post["beta"].values.mean())
        b_lo, b_hi = _hdi_bounds(idata, "beta", sc["hdi"])
        estimates["beta_slope_std"] = round(b_m, 5)
        estimates["beta_slope_hdi_low"] = round(float(b_lo), 5)
        estimates["beta_slope_hdi_high"] = round(float(b_hi), 5)

    # per-group intercepts + observed group means (shows shrinkage toward population)
    try:
        a_mean = post["a"].values.reshape(-1, n_groups).mean(axis=0)
        obs_means = sub.groupby(group_col)[outcome].mean().reindex(uniques).to_numpy()
        gtbl = pd.DataFrame({
            "group": [str(u) for u in uniques],
            "n": sub.groupby(group_col)[outcome].size().reindex(uniques).to_numpy(),
            "observed_mean": np.round(obs_means, 5),
            "partial_pooled_intercept": np.round(a_mean, 5),
        })
        gtbl.to_csv(d / "bayesian_hierarchical_groups.csv", index=False, encoding="utf-8")
        files.append("bayesian_hierarchical_groups.csv")
    except Exception:
        pass
    _forest(idata, ["a"], d / "bayesian_hierarchical_forest.png", "Group intercepts (partial pooling, 94% HDI)")
    if (d / "bayesian_hierarchical_forest.png").exists():
        files.append("bayesian_hierarchical_forest.png")

    summary.append(
        f"{method} 完成：贝叶斯分层模型（变截距部分汇集，PyMC NUTS，{sc['chains']}链×{sc['draws']}抽样），"
        f"结果={outcome}，分组={group_col}（{n_groups} 组）"
        + (f"，组内预测变量={pred}" if pred else "")
        + f"。总体均值≈{mu_a_m:.3g}，组间 SD≈{sig_a_m:.3g}、组内 SD≈{sig_m:.3g}，"
        f"ICC≈{icc:.3f}（组间方差占比）。{_conv_note(max_rhat, min_ess, sc['chains'])}。"
        " ⚠ 各组截距向总体均值“收缩”（部分汇集，小组借力总体，比逐组单独估计更稳）；"
        "非中心化参数化采样；先验弱信息。"
    )
    code.append(
        "import pymc as pm\n"
        "# 变截距分层模型（非中心化部分汇集）\n"
        "with pm.Model():\n"
        "    mu_a=pm.Normal('mu_a',y.mean(),2.5*y.std()); sd_a=pm.HalfNormal('sigma_a',2.5*y.std())\n"
        "    z=pm.Normal('z',0,1,shape=n_groups); a=pm.Deterministic('a',mu_a+sd_a*z)\n"
        "    s=pm.HalfNormal('sigma',2.5*y.std())\n"
        "    pm.Normal('y_obs',a[group_idx],s,observed=y)\n"
        "    idata=pm.sample(1000,tune=1000,chains=2,target_accept=0.95,random_seed=42)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3b) Bayesian random slopes (varying intercept AND slope, correlated via LKJ)
# ─────────────────────────────────────────────────────────────────────────────
@register("bayesian_random_slopes")
def _branch_bayesian_random_slopes(ctx: Ctx) -> None:
    import numpy as np
    import pandas as pd

    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    method = entry.method

    if not _have_pymc():
        _degrade(summary, method, "glmm（随机斜率混合效应）")
        return

    excl = {fp.unit_col, fp.time_col}
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in excl]
    # grouping column: config 'group'/'unit', else unit_col, else a low-cardinality
    # categorical/count/id column with 2..(len/3) levels (same rule as hierarchical).
    forced_g = cfg.get("group") or cfg.get("unit")
    group_col = forced_g if forced_g in df.columns else fp.unit_col
    if group_col not in (df.columns if group_col else []):
        cands = [
            c.name for c in fp.columns
            if c.kind in {"categorical", "binary", "count", "id"} and c.name not in excl
            and 2 <= df[c.name].dropna().nunique() <= max(2, len(df) // 3)
        ]
        group_col = cands[0] if cands else None
    if not group_col:
        summary.append(f"{method} 跳过：未找到分组变量（需 1 个分组列，如个体/班级/地区）。")
        return

    outcome = cfg.get("outcome") if cfg.get("outcome") in df.columns else (resolve_outcome(fp, cfg, cont) if cont else None)
    if not outcome:
        summary.append(f"{method} 跳过：未找到连续结果变量。")
        return
    # random slopes REQUIRE a within-group covariate (the slope to vary). If none, skip
    # honestly and point at the varying-intercept-only model.
    pred = cfg.get("predictor") if cfg.get("predictor") in df.columns else next(
        (c for c in cont if c != outcome), None)
    if not pred:
        summary.append(
            f"{method} 跳过：随机斜率需 1 个组内连续预测变量（斜率才有可变对象）；"
            "仅 1 个连续列时请改用 bayesian_hierarchical（变截距）。"
        )
        return

    sub = df[[outcome, group_col, pred]].copy()
    sub[outcome] = pd.to_numeric(sub[outcome], errors="coerce")
    sub[pred] = pd.to_numeric(sub[pred], errors="coerce")
    sub = sub.dropna()
    if len(sub) < 12 or sub[group_col].nunique() < 3:
        summary.append(f"{method} 跳过：随机斜率建模需 ≥3 组且 ≥12 个有效观测。")
        return

    codes, uniques = pd.factorize(sub[group_col])
    n_groups = len(uniques)
    y = sub[outcome].to_numpy(float)
    y_mean, y_sd = float(np.mean(y)), float(np.std(y)) or 1.0
    xv = sub[pred].to_numpy(float)
    x_sd = float(xv.std(ddof=0)) or 1.0
    x = (xv - xv.mean()) / x_sd            # centered+scaled covariate (stable NUTS geometry)

    import pymc as pm
    import pytensor.tensor as pt

    sc = _sampler_cfg(cfg)
    with pm.Model() as model:
        mu_a = pm.Normal("mu_a", mu=y_mean, sigma=2.5 * y_sd)        # population intercept
        mu_b = pm.Normal("mu_b", mu=0.0, sigma=2.5 * y_sd)          # population slope (std-x)
        sd_dist = pm.HalfNormal.dist(2.5 * y_sd, shape=2)
        chol, corr, sigmas = pm.LKJCholeskyCov(
            "chol", n=2, eta=2.0, sd_dist=sd_dist, compute_corr=True)
        z = pm.Normal("z", 0.0, 1.0, shape=(2, n_groups))          # non-centered
        ab = pm.Deterministic("ab", pt.dot(chol, z))               # 2 x n_groups offsets
        a = pm.Deterministic("a", mu_a + ab[0])                    # group intercepts
        b = pm.Deterministic("b", mu_b + ab[1])                    # group slopes (std-x)
        sigma = pm.HalfNormal("sigma", sigma=2.5 * y_sd)          # within-group SD
        mu = a[codes] + b[codes] * x
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
        idata = pm.sample(draws=sc["draws"], tune=sc["tune"], chains=sc["chains"],
                          cores=1, random_seed=sc["seed"], progressbar=False,
                          target_accept=0.95)

    conv_vars = ["mu_a", "mu_b", "sigma", "chol_corr", "chol_stds"]
    max_rhat, min_ess = _convergence(idata, conv_vars)
    post = idata.posterior
    mu_a_m = float(post["mu_a"].values.mean())
    mu_b_m = float(post["mu_b"].values.mean())
    b_lo, b_hi = _hdi_bounds(idata, "mu_b", sc["hdi"])
    sig_m = float(post["sigma"].values.mean())
    # LKJCholeskyCov(compute_corr=True) exposes <name>_stds (the 2 SDs: [intercept, slope])
    # and <name>_corr (2x2 correlation matrix). sigmas[0]=intercept SD, sigmas[1]=slope SD.
    stds = post["chol_stds"].values.reshape(-1, 2).mean(axis=0)
    intercept_sd, slope_sd = float(stds[0]), float(stds[1])
    corr_vals = post["chol_corr"].values.reshape(-1, 2, 2)
    corr_is = float(corr_vals[:, 0, 1].mean())                     # intercept-slope correlation
    # Intercept ICC, evaluated at x = mean(x) (x is centered before scaling, so the
    # intercept is at mean-x). Under random slopes the total between-group variance is
    # var_a + 2·x·cov_ab + x²·var_b — it depends on x — so a single ICC is an
    # approximation (the intercept-ICC at mean-x), disclosed in the summary/biases.
    icc = (intercept_sd ** 2 / (intercept_sd ** 2 + sig_m ** 2)
           if (intercept_sd + sig_m) > 0 else float("nan"))

    estimates["population_slope_std"] = round(mu_b_m, 5)           # headline FIRST
    estimates["population_slope_hdi_low"] = round(float(b_lo), 5)
    estimates["population_slope_hdi_high"] = round(float(b_hi), 5)
    estimates["population_slope_raw"] = round(mu_b_m / x_sd, 5)    # back-transform to raw-x scale
    estimates["population_intercept"] = round(mu_a_m, 5)
    estimates["slope_sd"] = round(slope_sd, 5)                     # between-group slope SD (std-x)
    estimates["intercept_sd"] = round(intercept_sd, 5)
    estimates["intercept_slope_corr"] = round(corr_is, 4)
    estimates["within_sd"] = round(sig_m, 5)
    estimates["icc"] = round(icc, 4)
    estimates["n_groups"] = float(n_groups)
    estimates["max_rhat"] = round(max_rhat, 4)
    estimates["min_ess"] = round(min_ess, 1)

    # per-group (intercept, slope) table — both on the model (std-x) scale plus raw-x slope
    try:
        a_mean = post["a"].values.reshape(-1, n_groups).mean(axis=0)
        bg_mean = post["b"].values.reshape(-1, n_groups).mean(axis=0)
        gtbl = pd.DataFrame({
            "group": [str(u) for u in uniques],
            "n": sub.groupby(group_col)[outcome].size().reindex(uniques).to_numpy(),
            "intercept": np.round(a_mean, 5),
            "slope_std": np.round(bg_mean, 5),
            "slope_raw": np.round(bg_mean / x_sd, 5),
        })
        gtbl.to_csv(d / "bayesian_random_slopes_groups.csv", index=False, encoding="utf-8")
        files.append("bayesian_random_slopes_groups.csv")
    except Exception:
        pass
    _forest(idata, ["b"], d / "bayesian_random_slopes_forest.png",
            "Group slopes (partial pooling, 94% HDI)")
    if (d / "bayesian_random_slopes_forest.png").exists():
        files.append("bayesian_random_slopes_forest.png")

    ppc_png = _ppc(model, idata, y, d, "bayesian_random_slopes_ppc.png", estimates, seed=sc["seed"])
    if ppc_png:
        files.append(ppc_png)

    slope_excludes0 = b_lo > 0 or b_hi < 0
    corr_txt = ("正相关（截距高的组斜率也偏高）" if corr_is > 0.1
                else "负相关（截距高的组斜率偏低）" if corr_is < -0.1
                else "近乎无关")
    ppc_note = (
        f" 后验预测检查：贝叶斯 p（均值）≈{estimates['ppc_bayes_p_mean']:.2f}、"
        f"p（标准差）≈{estimates['ppc_bayes_p_sd']:.2f}（≈0.5=数据与模型一致；接近 0 或 1=模型未能复现该统计量）。"
        if "ppc_bayes_p_mean" in estimates else ""
    )
    summary.append(
        f"{method} 完成：贝叶斯随机斜率模型（变截距+变斜率，相关随机效应经 LKJ，"
        f"PyMC NUTS，{sc['chains']}链×{sc['draws']}抽样），结果={outcome}，分组={group_col}"
        f"（{n_groups} 组），组内预测变量={pred}。"
        f"总体斜率（标准化 x）≈{mu_b_m:.3g}，{int(sc['hdi']*100)}% HDI=[{float(b_lo):.3g}, {float(b_hi):.3g}]"
        f"（{'不含 0，效应方向稳健' if slope_excludes0 else '含 0，方向不确定'}）。"
        f"斜率组间 SD≈{slope_sd:.3g}（衡量各组剂量-反应差异），"
        f"截距-斜率相关≈{corr_is:.2f}（{corr_txt}）。组内 SD≈{sig_m:.3g}，"
        f"ICC≈{icc:.3f}（截距 ICC、于 x 均值处）。"
        f"{_conv_note(max_rhat, min_ess, sc['chains'])}。{ppc_note}"
        " ⚠ 随机斜率让每组拥有自己的剂量-反应斜率；部分汇集把各组斜率向总体收缩（小组借力总体）；"
        "LKJ(η=2) 轻度偏好低相关；非中心化参数化采样；斜率为标准化 x 尺度"
        "（population_slope_raw 已回传原始 x 尺度）；x 已中心化，故 population_intercept "
        "是 x 取均值处的截距（≈y 的均值）、非 x=0 处；随机斜率下总组间方差随 x 变化，"
        "单一 ICC 为截距处的近似。"
    )
    code.append(
        "import pymc as pm, pytensor.tensor as pt\n"
        "# 随机斜率：相关的 (截距,斜率) 随机效应，LKJ 先验 + 非中心化\n"
        "with pm.Model():\n"
        "    mu_a=pm.Normal('mu_a',y.mean(),2.5*y.std()); mu_b=pm.Normal('mu_b',0,2.5*y.std())\n"
        "    sd_dist=pm.HalfNormal.dist(2.5*y.std(),shape=2)\n"
        "    chol,corr,sds=pm.LKJCholeskyCov('chol',n=2,eta=2.0,sd_dist=sd_dist,compute_corr=True)\n"
        "    z=pm.Normal('z',0,1,shape=(2,n_groups)); ab=pt.dot(chol,z)\n"
        "    a=mu_a+ab[0]; b=mu_b+ab[1]; s=pm.HalfNormal('sigma',2.5*y.std())\n"
        "    pm.Normal('y_obs',a[g]+b[g]*x,s,observed=y)\n"
        "    idata=pm.sample(1000,tune=1000,chains=2,target_accept=0.95,random_seed=42)\n"
        "# 斜率为标准化 x 尺度；原尺度 slope_raw = mu_b / sd_x"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared: count-outcome resolver (Poisson/NB) — outcome = first count column
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_count(ctx: Ctx, method: str):
    import pandas as pd

    fp, cfg, df = ctx.fp, ctx.cfg, ctx.df
    excl = {fp.unit_col, fp.time_col}
    counts = [c.name for c in fp.columns if c.kind == "count" and c.name not in excl]
    forced_y = cfg.get("outcome")
    if forced_y in df.columns:
        outcome = forced_y
    elif counts:
        outcome = counts[0]
    else:
        return None, [], f"{method} 跳过：未找到计数型结果变量（非负整数列）。"
    forced = [c for c in (cfg.get("predictors") or []) if c in df.columns and c != outcome]
    if forced:
        preds = forced[:20]
    else:
        preds = [
            c.name for c in fp.columns
            if c.kind in {"continuous", "count", "binary"}
            and c.name not in {outcome, fp.unit_col, fp.time_col}
        ][:20]
    preds = [c for c in preds if pd.to_numeric(df[c], errors="coerce").notna().any()]
    if not preds:
        return None, [], f"{method} 跳过：未找到可用的数值预测变量（≥1 个）。"
    return outcome, preds, None


# ─────────────────────────────────────────────────────────────────────────────
# 4) Bayesian count regression (Poisson / Negative-Binomial, auto by overdispersion)
# ─────────────────────────────────────────────────────────────────────────────
@register("bayesian_poisson_regression")
def _branch_bayesian_poisson_regression(ctx: Ctx) -> None:
    import numpy as np

    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    method = entry.method

    if not _have_pymc():
        _degrade(summary, method, "poisson_regression / negative_binomial_regression")
        return
    outcome, preds, problem = _resolve_count(ctx, method)
    if problem:
        summary.append(problem)
        return
    X, y, preds, problem = _clean(df, outcome, preds, binary_outcome=False)
    if problem:
        summary.append(f"{method} 跳过：{problem}")
        return
    if (y < 0).any() or not np.allclose(y, np.round(y)):
        summary.append(f"{method} 跳过：结果变量 {outcome} 不是非负整数计数。")
        return
    y = np.round(y).astype(int)

    import pymc as pm

    sc = _sampler_cfg(cfg)
    Xz, xmu, xsd = _standardize(X)
    mean, var = float(np.mean(y)), float(np.var(y, ddof=1))
    dispersion = var / mean if mean > 0 else 1.0
    fam = str(cfg.get("family") or "auto").strip().lower()
    if fam not in {"poisson", "negbinomial", "negative_binomial", "nb"}:
        fam = "negbinomial" if dispersion > 1.5 else "poisson"
    use_nb = fam in {"negbinomial", "negative_binomial", "nb"}

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=np.log(mean + 1e-6), sigma=5.0)
        beta = pm.Normal("beta", mu=0.0, sigma=2.5, shape=Xz.shape[1])  # log-rate scale
        lam = pm.math.exp(alpha + pm.math.dot(Xz, beta))
        if use_nb:
            # PyMC NegativeBinomial(mu, alpha): Var = mu + mu²/alpha, so LARGE alpha → Poisson
            # limit. Gamma(2, 0.1) (mean 20) is weakly-informative and does NOT penalize the
            # Poisson limit the way Exponential(1) does — better geometry for mild overdispersion.
            disp = pm.Gamma("alpha_nb", alpha=2.0, beta=0.1)
            pm.NegativeBinomial("y_obs", mu=lam, alpha=disp, observed=y)
        else:
            pm.Poisson("y_obs", mu=lam, observed=y)
        idata = pm.sample(draws=sc["draws"], tune=sc["tune"], chains=sc["chains"],
                          cores=1, random_seed=sc["seed"], progressbar=False)

    b_mean = idata.posterior["beta"].values.reshape(-1, Xz.shape[1]).mean(axis=0)
    b_lo, b_hi = _hdi_bounds(idata, "beta", sc["hdi"])
    raw_b = b_mean / xsd                 # log-rate per 1 raw-unit of predictor
    rr_mean, rr_lo, rr_hi = np.exp(raw_b), np.exp(b_lo / xsd), np.exp(b_hi / xsd)
    max_rhat, min_ess = _convergence(idata, ["alpha", "beta"])

    for j, name in enumerate(preds):
        estimates[f"rr__{name}"] = round(float(rr_mean[j]), 5)
        estimates[f"rr__{name}__hdi_low"] = round(float(rr_lo[j]), 5)
        estimates[f"rr__{name}__hdi_high"] = round(float(rr_hi[j]), 5)
    estimates["dispersion"] = round(dispersion, 4)
    estimates["max_rhat"] = round(max_rhat, 4)
    estimates["min_ess"] = round(min_ess, 1)

    try:
        import pandas as pd

        pd.DataFrame({
            "predictor": preds, "rate_ratio": np.round(rr_mean, 5),
            f"hdi_{int(sc['hdi']*100)}_low": np.round(rr_lo, 5),
            f"hdi_{int(sc['hdi']*100)}_high": np.round(rr_hi, 5),
        }).to_csv(d / "bayesian_count_rr.csv", index=False, encoding="utf-8")
        files.append("bayesian_count_rr.csv")
    except Exception:
        pass
    _forest(idata, ["beta"], d / "bayesian_count_forest.png", "Posterior log-rate (94% HDI)")
    if (d / "bayesian_count_forest.png").exists():
        files.append("bayesian_count_forest.png")

    fam_txt = "负二项（过离散）" if use_nb else "Poisson"
    sig = [preds[j] for j in range(len(preds)) if rr_lo[j] > 1 or rr_hi[j] < 1]
    summary.append(
        f"{method} 完成：贝叶斯计数回归（{fam_txt}，PyMC NUTS，{sc['chains']}链×{sc['draws']}抽样），"
        f"结果={outcome}（离散系数 var/mean={dispersion:.2f}），预测变量 {len(preds)} 个。"
        f"率比 RR {int(sc['hdi']*100)}% HDI 不含 1 的：" + ("、".join(sig) if sig else "无")
        + f"。{_conv_note(max_rhat, min_ess, sc['chains'])}。"
        " ⚠ RR=exp(后验均值 log-rate)（HDI 可信区间）；"
        + ("过离散已用负二项吸收（var≫mean）；" if use_nb else "若 var≫mean 应改负二项（config family=negbinomial）；")
        + "先验弱信息，预测变量标准化采样后回传原尺度。"
    )
    code.append(
        "import pymc as pm\n"
        f"# {'NegativeBinomial' if use_nb else 'Poisson'} 计数回归（log 链接）\n"
        "with pm.Model():\n"
        "    a=pm.Normal('alpha',np.log(y.mean()),5); b=pm.Normal('beta',0,2.5,shape=X.shape[1])\n"
        "    lam=pm.math.exp(a+pm.math.dot(Xz,b))\n"
        + ("    disp=pm.Gamma('alpha_nb',2,0.1); pm.NegativeBinomial('y_obs',mu=lam,alpha=disp,observed=y)\n"
           if use_nb else "    pm.Poisson('y_obs',mu=lam,observed=y)\n")
        + "    idata=pm.sample(1000,tune=1000,chains=2,random_seed=42)\n"
        "# RR = exp(beta_std / sd_x)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5) Bayesian robust regression (Student-t likelihood, heavy tails → outlier-robust)
# ─────────────────────────────────────────────────────────────────────────────
@register("bayesian_robust_regression")
def _branch_bayesian_robust_regression(ctx: Ctx) -> None:
    import numpy as np

    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    method = entry.method

    if not _have_pymc():
        _degrade(summary, method, "robust_regression（Huber/Theil-Sen）")
        return
    outcome, preds, problem = _resolve_reg(ctx, method, binary_outcome=False)
    if problem:
        summary.append(problem)
        return
    X, y, preds, problem = _clean(df, outcome, preds, binary_outcome=False)
    if problem:
        summary.append(f"{method} 跳过：{problem}")
        return

    import pymc as pm

    sc = _sampler_cfg(cfg)
    Xz, xmu, xsd = _standardize(X)
    y_sd = float(np.std(y)) or 1.0
    y_mean = float(np.mean(y))

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=y_mean, sigma=2.5 * y_sd)
        beta = pm.Normal("beta", mu=0.0, sigma=2.5 * y_sd, shape=Xz.shape[1])
        sigma = pm.HalfNormal("sigma", sigma=2.5 * y_sd)
        nu = pm.Gamma("nu", alpha=2.0, beta=0.1)   # df > 0, mean 20 — small ν = heavy tails
        mu = alpha + pm.math.dot(Xz, beta)
        pm.StudentT("y_obs", nu=nu, mu=mu, sigma=sigma, observed=y)
        idata = pm.sample(draws=sc["draws"], tune=sc["tune"], chains=sc["chains"],
                          cores=1, random_seed=sc["seed"], progressbar=False)

    b_mean = idata.posterior["beta"].values.reshape(-1, Xz.shape[1]).mean(axis=0)
    b_lo, b_hi = _hdi_bounds(idata, "beta", sc["hdi"])
    raw_b, raw_lo, raw_hi = b_mean / xsd, b_lo / xsd, b_hi / xsd
    nu_mean = float(idata.posterior["nu"].values.mean())
    max_rhat, min_ess = _convergence(idata, ["alpha", "beta", "sigma", "nu"])

    for j, name in enumerate(preds):
        estimates[f"beta__{name}"] = round(float(raw_b[j]), 5)
        estimates[f"beta__{name}__hdi_low"] = round(float(raw_lo[j]), 5)
        estimates[f"beta__{name}__hdi_high"] = round(float(raw_hi[j]), 5)
    estimates["nu"] = round(nu_mean, 3)
    estimates["max_rhat"] = round(max_rhat, 4)
    estimates["min_ess"] = round(min_ess, 1)

    try:
        import pandas as pd

        pd.DataFrame({
            "predictor": preds, "beta_mean": np.round(raw_b, 5),
            f"hdi_{int(sc['hdi']*100)}_low": np.round(raw_lo, 5),
            f"hdi_{int(sc['hdi']*100)}_high": np.round(raw_hi, 5),
        }).to_csv(d / "bayesian_robust_coefs.csv", index=False, encoding="utf-8")
        files.append("bayesian_robust_coefs.csv")
    except Exception:
        pass
    _forest(idata, ["beta"], d / "bayesian_robust_forest.png", "Robust posterior coefficients (94% HDI)")
    if (d / "bayesian_robust_forest.png").exists():
        files.append("bayesian_robust_forest.png")

    tail = ("ν 小（重尾）——数据有离群/厚尾，稳健回归比正态回归更可信" if nu_mean < 10
            else "ν 较大——残差接近正态，与普通回归结果相近")
    sig = [preds[j] for j in range(len(preds)) if raw_lo[j] > 0 or raw_hi[j] < 0]
    summary.append(
        f"{method} 完成：贝叶斯稳健回归（Student-t 似然，PyMC NUTS，{sc['chains']}链×{sc['draws']}抽样），"
        f"结果={outcome}，预测变量 {len(preds)} 个。自由度 ν≈{nu_mean:.1f}（{tail}）。"
        f"{int(sc['hdi']*100)}% HDI 不含 0 的：" + ("、".join(sig) if sig else "无")
        + f"。{_conv_note(max_rhat, min_ess, sc['chains'])}。"
        " ⚠ Student-t 重尾似然自动降权离群点（无需手动剔除）；系数为后验均值+HDI，标准化采样后回传原尺度。"
    )
    code.append(
        "import pymc as pm\n"
        "# Student-t 似然 → 重尾自动降权离群点\n"
        "with pm.Model():\n"
        "    a=pm.Normal('alpha',y.mean(),2.5*y.std()); b=pm.Normal('beta',0,2.5*y.std(),shape=X.shape[1])\n"
        "    s=pm.HalfNormal('sigma',2.5*y.std()); nu=pm.Gamma('nu',2,0.1)\n"
        "    pm.StudentT('y_obs',nu=nu,mu=a+pm.math.dot(Xz,b),sigma=s,observed=y)\n"
        "    idata=pm.sample(1000,tune=1000,chains=2,random_seed=42)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6) Bayesian model comparison (PSIS-LOO over candidate models)
# ─────────────────────────────────────────────────────────────────────────────
@register("bayesian_model_comparison")
def _branch_bayesian_model_comparison(ctx: Ctx) -> None:
    import numpy as np

    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    method = entry.method

    if not _have_pymc():
        _degrade(summary, method, "AIC/BIC 模型比较（频率派）")
        return
    outcome, preds, problem = _resolve_reg(ctx, method, binary_outcome=False)
    if problem:
        summary.append(problem)
        return
    X, y, preds, problem = _clean(df, outcome, preds, binary_outcome=False)
    if problem:
        summary.append(f"{method} 跳过：{problem}")
        return

    import arviz as az
    import pymc as pm

    sc = _sampler_cfg(cfg)
    Xz, _, _ = _standardize(X)
    y_sd = float(np.std(y)) or 1.0
    y_mean = float(np.mean(y))

    def _fit(kind):
        with pm.Model() as m:
            a = pm.Normal("alpha", mu=y_mean, sigma=2.5 * y_sd)
            s = pm.HalfNormal("sigma", sigma=2.5 * y_sd)
            mu = a
            if kind != "null":
                b = pm.Normal("beta", 0.0, 2.5 * y_sd, shape=Xz.shape[1])
                mu = a + pm.math.dot(Xz, b)
            if kind == "robust":
                nu = pm.Gamma("nu", 2.0, 0.1)
                pm.StudentT("y_obs", nu=nu, mu=mu, sigma=s, observed=y)
            else:
                pm.Normal("y_obs", mu=mu, sigma=s, observed=y)
            idata = pm.sample(draws=sc["draws"], tune=sc["tune"], chains=sc["chains"],
                              cores=1, random_seed=sc["seed"], progressbar=False)
            pm.compute_log_likelihood(idata, progressbar=False)
        return idata

    labels = {"null": "仅截距", "linear": "线性(正态)", "robust": "稳健(Student-t)"}
    idatas = {}
    for kind in ("null", "linear", "robust"):
        try:
            idatas[kind] = _fit(kind)
        except Exception:
            continue
    if len(idatas) < 2:
        summary.append(f"{method} 跳过：可比较的模型不足 2 个（采样失败）。")
        return

    cmp = az.compare(idatas)            # PSIS-LOO; columns: rank, elpd, se, elpd_diff, dse, weight, p
    best = cmp.index[cmp["rank"] == 0][0]
    for name in cmp.index:
        estimates[f"elpd__{name}"] = round(float(cmp.loc[name, "elpd"]), 3)
        estimates[f"weight__{name}"] = round(float(cmp.loc[name, "weight"]), 4)
    estimates["n_models"] = float(len(idatas))
    estimates["best_rank0_weight"] = round(float(cmp.loc[best, "weight"]), 4)

    try:
        cmp_out = cmp.copy()
        cmp_out.insert(0, "model", [labels.get(i, i) for i in cmp_out.index])
        cmp_out.to_csv(d / "bayesian_model_comparison.csv", index=False, encoding="utf-8")
        files.append("bayesian_model_comparison.csv")
    except Exception:
        pass

    rank_txt = "、".join(
        f"{labels.get(i, i)}(ELPD={cmp.loc[i, 'elpd']:.1f}, w={cmp.loc[i, 'weight']:.2f})"
        for i in cmp.sort_values("rank").index
    )
    summary.append(
        f"{method} 完成：贝叶斯模型比较（PSIS-LOO 留一交叉验证，PyMC，{len(idatas)} 个候选），"
        f"结果={outcome}。最优={labels.get(best, best)}（rank 0，LOO 权重 {cmp.loc[best, 'weight']:.2f}）。"
        f"排名：{rank_txt}。"
        " ⚠ ELPD 越高越好；ΔELPD 应对照其标准误判断是否显著优于次优；LOO 权重是模型平均的相对支持度，"
        "非后验概率。候选含『仅截距/线性/稳健』三模型。"
    )
    code.append(
        "import pymc as pm, arviz as az\n"
        "# 拟合候选模型并各自 compute_log_likelihood → az.compare 做 PSIS-LOO\n"
        "az.compare({'null':idata_null,'linear':idata_linear,'robust':idata_robust})"
    )

