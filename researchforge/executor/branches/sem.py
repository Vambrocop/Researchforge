"""Branch handlers for the sem family (migrated from the run.py monolith).

Each handler unpacks ctx into the same local names run_analysis used and runs the
original branch body verbatim. See executor/_branch_api.py.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.run import (
    _sem_latents,
    _sem_via_lavaan,
    _sem_via_semopy,
)


def _parse_measurement(spec, columns):
    """Parse a lavaan-style measurement spec ('F1 =~ a + b + c' lines) into
    ([(factor_name, [indicators]), ...], dropped). Keeps only existing columns,
    preserves order, MERGES repeated factor names, strips lavaan fixed-loading
    modifiers ('0.7*x' / 'NA*x' -> 'x'), and assigns each indicator to its FIRST
    factor (no cross-loadings). ``dropped`` lists (indicator, factor) pairs that were
    dropped because the indicator was already claimed by an earlier factor — surfaced
    honestly by the caller so a silently-dropped cross-loading isn't hidden."""
    import re

    factors_map: dict[str, list] = {}
    order: list[str] = []
    assigned: set[str] = set()
    dropped: list[tuple[str, str]] = []
    for line in re.split(r"[\n;]+", str(spec)):
        if "=~" not in line:
            continue
        lhs, rhs = line.split("=~", 1)
        fname = lhs.strip()
        if not fname:
            continue
        if fname not in factors_map:
            factors_map[fname] = []
            order.append(fname)
        for tok in rhs.split("+"):
            c = tok.strip()
            if "*" in c:  # lavaan fixed/free-loading modifier: 0.7*x / NA*x -> x
                c = c.split("*", 1)[1].strip()
            if c not in columns:
                continue
            if c in assigned:
                dropped.append((c, fname))
            else:
                factors_map[fname].append(c)
                assigned.add(c)
    factors = [(name, factors_map[name]) for name in order if factors_map[name]]
    return factors, dropped


def _parse_structural(spec, factor_names):
    """Parse structural regressions among FACTORS ('F_out ~ F_p1 + F_p2' lines — '~'
    but not '=~') into [(outcome_factor, predictor_factor), ...] over known factor names."""
    import re

    fset = set(factor_names)
    paths, seen = [], set()
    for line in re.split(r"[\n;]+", str(spec)):
        if "=~" in line or "~" not in line:
            continue
        lhs, rhs = line.split("~", 1)
        out = lhs.strip()
        if out not in fset:
            continue
        for tok in rhs.split("+"):
            pr = tok.strip()
            if "*" in pr:
                pr = pr.split("*", 1)[1].strip()
            if pr in fset and pr != out and (out, pr) not in seen:
                paths.append((out, pr))
                seen.add((out, pr))
    return paths


def _topo_order(names, paths):
    """Topological order so every path goes predictor->outcome (recursive SEM); None if
    the structural graph has a cycle (non-recursive models are not supported here)."""
    from collections import deque

    preds = {n: set() for n in names}
    succ = {n: [] for n in names}
    for out, pr in paths:
        preds[out].add(pr)
    for out, pr in paths:
        succ[pr].append(out)
    indeg = {n: len(preds[n]) for n in names}
    q = deque([n for n in names if indeg[n] == 0])
    order = []
    while q:
        n = q.popleft()
        order.append(n)
        for s in succ[n]:
            indeg[s] -= 1
            if indeg[s] == 0:
                q.append(s)
    return order if len(order) == len(names) else None


def _run_bayesian_structural(ctx, factors, paths, dropped=()) -> None:
    """Recursive structural Bayesian SEM: a measurement model (Λ) PLUS directed paths
    between latents (η = Bη + ζ ⇒ Cov(η) = (I−B)⁻¹ Σ_ζ (I−B)⁻ᵀ, B strictly lower-
    triangular under a topological order). Marker-variable identification (each factor's
    first loading fixed = 1) pins both scale AND sign (so paths are sign-identified and
    NOT bimodal). Reports the STANDARDIZED path coefficients (the directed structural
    effects — mediation/path model) + endogenous-factor R²."""
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    method = entry.method

    import numpy as np
    import pandas as pd
    import pymc as pm
    import pytensor.tensor as pt

    from researchforge.executor.branches.bayesian_mcmc import (
        _conv_note, _convergence, _hdi_bounds, _sampler_cfg,
    )

    fdict = dict(factors)
    order = _topo_order([f for f, _ in factors], paths)
    if order is None:
        # cyclic / non-recursive → fall back to the correlated multi-factor CFA honestly.
        summary.append("⚠ 结构路径含回路（非递归），本引擎仅支持递归路径模型——已退回相关多因子 CFA。")
        _run_bayesian_multifactor(ctx, factors, dropped)
        return
    fac_names = order                                  # factors in topological order
    fidx = {nm: i for i, nm in enumerate(fac_names)}
    ind_list, fac_of = [], []
    for nm in fac_names:
        for c in fdict[nm]:
            ind_list.append(c)
            fac_of.append(fidx[nm])
    p, k = len(ind_list), len(fac_names)

    Y = df[ind_list].apply(lambda s: pd.to_numeric(s, errors="coerce")).dropna()
    if len(Y) < 24:
        summary.append("贝叶斯 SEM 跳过：有效样本不足（去缺失后 < 24 行）。")
        return
    n = len(Y)
    Yv = Y.to_numpy(float)
    sdv = Yv.std(axis=0, ddof=0)
    sdv = np.where(sdv < 1e-12, 1.0, sdv)
    Z = (Yv - Yv.mean(axis=0)) / sdv
    fac_idx = np.asarray(fac_of)
    anchor = np.array([int(np.where(fac_idx == f)[0][0]) for f in range(k)])
    is_anchor = np.zeros(p, dtype=bool)
    is_anchor[anchor] = True
    path_ij = [(fidx[o], fidx[pr]) for o, pr in paths]   # (outcome_idx, predictor_idx), idx_o>idx_pr

    sc = _sampler_cfg(cfg)
    with pm.Model() as model:
        lam_free = pm.Normal("lam_free", 0.0, 1.0, shape=int((~is_anchor).sum()))
        lam_t = pt.ones(p)                              # marker loadings fixed = 1 (scale+sign id)
        lam_t = pt.set_subtensor(lam_t[~is_anchor], lam_free)
        lam = pm.Deterministic("lam", lam_t)
        psi = pm.HalfNormal("psi", 1.0, shape=p)
        beta = pm.Normal("beta", 0.0, 1.0, shape=len(path_ij))
        B = pt.zeros((k, k))
        for bi, (i, j) in enumerate(path_ij):
            B = pt.set_subtensor(B[i, j], beta[bi])
        zvar = pm.HalfNormal("zvar", 1.0, shape=k)      # factor disturbance/exogenous variances
        Iinv = pt.linalg.inv(pt.eye(k) - B)
        Ceta = pm.Deterministic("Ceta", Iinv @ pt.diag(zvar) @ Iinv.T)
        Lam = pt.zeros((p, k))
        for j in range(p):
            Lam = pt.set_subtensor(Lam[j, int(fac_idx[j])], lam[j])
        pm.MvNormal("y_obs", mu=pt.zeros(p), cov=Lam @ Ceta @ Lam.T + pt.diag(psi ** 2), observed=Z)
        idata = pm.sample(draws=sc["draws"], tune=sc["tune"], chains=sc["chains"],
                          cores=1, random_seed=sc["seed"], progressbar=False,
                          target_accept=0.95)

    max_rhat, min_ess = _convergence(idata, ["beta", "psi", "zvar"])
    post = idata.posterior
    beta_d = post["beta"].values.reshape(-1, len(path_ij))
    Ceta_d = post["Ceta"].values.reshape(-1, k, k)
    zvar_d = post["zvar"].values.reshape(-1, k)
    fac_sd = np.sqrt(np.clip(np.diagonal(Ceta_d, axis1=1, axis2=2), 1e-12, None))  # (draws,k)
    fac_var = np.clip(np.diagonal(Ceta_d, axis1=1, axis2=2), 1e-12, None)

    # standardized path = beta * SD(predictor)/SD(outcome); headline structural effects.
    endo = sorted({i for i, _ in path_ij})
    for bi, (i, j) in enumerate(path_ij):
        bstd = beta_d[:, bi] * fac_sd[:, j] / fac_sd[:, i]
        lo, hi = float(np.percentile(bstd, 3)), float(np.percentile(bstd, 97))
        key = f"path_{fac_names[i]}_on_{fac_names[j]}_std"   # outcome <- predictor
        estimates[key] = round(float(bstd.mean()), 4)
        estimates[f"{key}_hdi_low"] = round(lo, 4)
        estimates[f"{key}_hdi_high"] = round(hi, 4)
        estimates[f"path_{fac_names[i]}_on_{fac_names[j]}_raw"] = round(float(beta_d[:, bi].mean()), 4)
    # endogenous-factor R² = 1 - disturbance variance / total factor variance
    for f in endo:
        r2 = 1.0 - (zvar_d[:, f] / fac_var[:, f])
        estimates[f"r2_{fac_names[f]}"] = round(float(np.mean(r2)), 4)
    estimates["n_factors"] = float(k)
    estimates["n_paths"] = float(len(path_ij))
    estimates["n_indicators"] = float(p)
    estimates["n_obs"] = float(n)
    estimates["max_rhat"] = round(max_rhat, 4)
    estimates["min_ess"] = round(min_ess, 1)

    try:
        rows = []
        for bi, (i, j) in enumerate(path_ij):
            bstd = beta_d[:, bi] * fac_sd[:, j] / fac_sd[:, i]
            rows.append({"outcome": fac_names[i], "predictor": fac_names[j],
                         "std_coef": round(float(bstd.mean()), 5),
                         "raw_coef": round(float(beta_d[:, bi].mean()), 5),
                         "hdi_low": round(float(np.percentile(bstd, 3)), 5),
                         "hdi_high": round(float(np.percentile(bstd, 97)), 5)})
        pd.DataFrame(rows).to_csv(d / "bayesian_sem_paths.csv", index=False, encoding="utf-8")
        files.append("bayesian_sem_paths.csv")
    except Exception:
        pass

    path_txt = "、".join(
        f"{fac_names[j]}→{fac_names[i]} β*≈{estimates[f'path_{fac_names[i]}_on_{fac_names[j]}_std']:.2f}"
        for (i, j) in path_ij)
    r2_txt = "、".join(f"{fac_names[f]} R²≈{estimates[f'r2_{fac_names[f]}']:.2f}" for f in endo)
    drop_note = ""
    if dropped:
        _dl = "、".join(f"{c}(被 {f} 重复声明)" for c, f in dropped)
        drop_note = f" ⚠ 不支持交叉载荷：指标 {_dl} 已归属更早因子、被丢弃。"
    summary.append(
        f"{method} 完成：递归结构贝叶斯 SEM（{k} 因子 / {len(path_ij)} 条有向路径 / {p} 指标，"
        f"PyMC NUTS，{sc['chains']}链×{sc['draws']}抽样，标记变量识别 marker=1）。"
        f"标准化路径系数（潜变量间有向结构效应）：{path_txt}。内生因子被解释方差：{r2_txt}。"
        f"路径见 bayesian_sem_paths.csv。{_conv_note(max_rhat, min_ess, sc['chains'])}。"
        " ⚠ 这是**递归（无回路）**路径模型；每因子首指标载荷固定为 1 以识别尺度与符号；"
        "标准化路径=原始路径×SD(自变量因子)/SD(因变量因子)。路径为模型设定下的有向关联，"
        "因果解读仍需设计与假设支撑（无未观测混杂等）；弱信息先验、指标已标准化。"
        + drop_note
    )
    code += [
        "import pymc as pm, pytensor.tensor as pt  # 递归结构贝叶斯 SEM（marker=1 识别）",
        "with pm.Model():",
        "    # 每因子首指标载荷=1; B=路径矩阵(拓扑序严格下三角)",
        "    Iinv=pt.linalg.inv(pt.eye(k)-B); Ceta=Iinv@pt.diag(zvar)@Iinv.T  # Cov(η)",
        "    cov=Lam@Ceta@Lam.T+pt.diag(psi**2); pm.MvNormal('y_obs',mu=0,cov=cov,observed=Z)",
        "# 标准化路径 = beta * SD(自变量因子)/SD(因变量因子)",
    ]


def _run_bayesian_multifactor(ctx, factors, dropped=()) -> None:
    """Correlated multi-factor Bayesian CFA (marginalized): Σ = Λ Φ Λᵀ + Ψ, Φ the factor
    CORRELATION matrix (LKJ, unit-variance factors for identification). Reports per-factor
    standardized loadings, indicator residuals, factor reliabilities (omega), and the
    inter-factor correlations — the standardized structural associations between latents.
    A positive anchor (marker) loading per factor breaks the per-factor sign indeterminacy
    (so the correlations are consistently signed and chains don't split into sign modes)."""
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    method = entry.method

    import numpy as np
    import pandas as pd
    import pymc as pm
    import pytensor.tensor as pt

    from researchforge.executor.branches.bayesian_mcmc import (
        _conv_note, _convergence, _hdi_bounds, _sampler_cfg,
    )

    fac_names = [f for f, _ in factors]
    ind_list, fac_of = [], []
    for fi, (_, inds) in enumerate(factors):
        for c in inds:
            ind_list.append(c)
            fac_of.append(fi)
    p, k = len(ind_list), len(factors)

    Y = df[ind_list].apply(lambda s: pd.to_numeric(s, errors="coerce")).dropna()
    if len(Y) < 24:
        summary.append("贝叶斯 SEM 跳过：有效样本不足（去缺失后 < 24 行）。")
        return
    n = len(Y)
    Yv = Y.to_numpy(float)
    sd = Yv.std(axis=0, ddof=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Z = (Yv - Yv.mean(axis=0)) / sd
    fac_idx = np.asarray(fac_of)
    # anchor = first indicator (global position) of each factor — its loading is forced
    # positive to identify that factor's sign.
    anchor = np.array([int(np.where(fac_idx == f)[0][0]) for f in range(k)])
    is_anchor = np.zeros(p, dtype=bool)
    is_anchor[anchor] = True

    sc = _sampler_cfg(cfg)
    # Robustness guard: this path needs pm.LKJCorr(n=k) to return a k×k Cholesky factor
    # (modern PyMC). Older PyMC returned a flat off-diagonal vector — degrade honestly
    # to the single-factor path rather than build a broadcast-garbage covariance.
    _probe = pm.LKJCorr.dist(n=k, eta=2.0)
    if getattr(_probe, "ndim", 2) != 2:
        summary.append(
            "贝叶斯 SEM 多因子跳过：本环境的 PyMC LKJCorr 返回格式不兼容（需返回 k×k Cholesky）。"
            "已退回单因子 CFA——请升级 pymc，或用 config indicators 指定单因子指标。"
        )
        return
    with pm.Model() as model:
        lam_pos = pm.HalfNormal("lam_pos", 1.0, shape=k)              # positive anchor per factor
        lam_free = pm.Normal("lam_free", 0.0, 1.0, shape=int((~is_anchor).sum()))
        lam_t = pt.zeros(p)
        lam_t = pt.set_subtensor(lam_t[anchor], lam_pos)
        lam_t = pt.set_subtensor(lam_t[~is_anchor], lam_free)
        lam = pm.Deterministic("lam", lam_t)
        psi = pm.HalfNormal("psi", 1.0, shape=p)
        Lchol = pm.LKJCorr("L", n=k, eta=2.0)                        # Cholesky of factor corr matrix
        Phi = pm.Deterministic("Phi", pt.dot(Lchol, Lchol.T))        # factor correlation matrix
        Lam = pt.zeros((p, k))
        for j in range(p):
            Lam = pt.set_subtensor(Lam[j, int(fac_idx[j])], lam[j])
        cov = Lam @ Phi @ Lam.T + pt.diag(psi ** 2)
        pm.MvNormal("y_obs", mu=pt.zeros(p), cov=cov, observed=Z)
        idata = pm.sample(draws=sc["draws"], tune=sc["tune"], chains=sc["chains"],
                          cores=1, random_seed=sc["seed"], progressbar=False,
                          target_accept=0.95)

    max_rhat, min_ess = _convergence(idata, ["lam", "psi", "Phi"])
    post = idata.posterior
    lam_mean = post["lam"].values.reshape(-1, p).mean(axis=0)
    psi_mean = post["psi"].values.reshape(-1, p).mean(axis=0)
    lam_lo, lam_hi = _hdi_bounds(idata, "lam", sc["hdi"])
    Phi_mean = post["Phi"].values.reshape(-1, k, k).mean(axis=0)

    # per-factor reliability (omega) on the standardized scale
    omega = {}
    for f in range(k):
        idxs = [j for j in range(p) if fac_idx[j] == f]
        sl = float(np.sum(lam_mean[idxs]))
        sp = float(np.sum(psi_mean[idxs] ** 2))
        omega[f] = (sl ** 2) / ((sl ** 2) + sp) if (sl ** 2 + sp) > 0 else float("nan")

    # estimates — factor correlations (the structural associations) first, then loadings.
    for i in range(k):
        for j in range(i + 1, k):
            estimates[f"corr_{fac_names[i]}_{fac_names[j]}"] = round(float(Phi_mean[i, j]), 4)
    for f in range(k):
        estimates[f"omega_{fac_names[f]}"] = round(float(omega[f]), 4)
    for j, c in enumerate(ind_list):
        estimates[f"lam_{c}"] = round(float(lam_mean[j]), 4)
        estimates[f"lam_{c}_hdi_low"] = round(float(lam_lo[j]), 4)
        estimates[f"lam_{c}_hdi_high"] = round(float(lam_hi[j]), 4)
    estimates["n_factors"] = float(k)
    estimates["n_indicators"] = float(p)
    estimates["n_obs"] = float(n)
    estimates["max_rhat"] = round(max_rhat, 4)
    estimates["min_ess"] = round(min_ess, 1)

    try:
        pd.DataFrame({
            "indicator": ind_list,
            "factor": [fac_names[f] for f in fac_idx],
            "loading_mean": np.round(lam_mean, 5),
            "hdi_low": np.round(lam_lo, 5),
            "hdi_high": np.round(lam_hi, 5),
            "residual_sd": np.round(psi_mean, 5),
        }).to_csv(d / "bayesian_sem_loadings.csv", index=False, encoding="utf-8")
        files.append("bayesian_sem_loadings.csv")
        pd.DataFrame(np.round(Phi_mean, 5), index=fac_names, columns=fac_names).to_csv(
            d / "bayesian_sem_factor_corr.csv", encoding="utf-8")
        files.append("bayesian_sem_factor_corr.csv")
    except Exception:
        pass
    from researchforge.executor.branches.bayesian_mcmc import _forest
    _forest(idata, ["lam"], d / "bayesian_sem_loadings.png",
            "Bayesian multi-factor CFA loadings (94% HDI)")
    if (d / "bayesian_sem_loadings.png").exists():
        files.append("bayesian_sem_loadings.png")

    corr_txt = "、".join(
        f"{fac_names[i]}↔{fac_names[j]} r≈{Phi_mean[i, j]:.2f}"
        for i in range(k) for j in range(i + 1, k))
    omega_txt = "、".join(f"{fac_names[f]} ω≈{omega[f]:.2f}" for f in range(k))
    drop_note = ""
    if dropped:
        _dl = "、".join(f"{c}(被 {f} 重复声明)" for c, f in dropped)
        drop_note = (f" ⚠ 不支持交叉载荷：指标 {_dl} 已归属更早的因子、在后续因子中被丢弃；"
                     "如需交叉载荷请改用频率派 sem 指定完整 lavaan 模型。")
    summary.append(
        f"{method} 完成：相关多因子贝叶斯 CFA（{k} 因子 / {p} 指标，PyMC NUTS，"
        f"{sc['chains']}链×{sc['draws']}抽样，边际化 Σ=ΛΦΛᵀ+Ψ）。"
        f"因子相关（潜变量间的标准化结构关联）：{corr_txt}。构念信度：{omega_txt}。"
        f"载荷+HDI 见 bayesian_sem_loadings.csv、因子相关阵见 bayesian_sem_factor_corr.csv。"
        f"{_conv_note(max_rhat, min_ess, sc['chains'])}。"
        " ⚠ 因子方差固定为 1（Φ 为相关阵）以识别尺度；每因子首指标载荷锚定为正以定符号（约定）；"
        "因子相关即标准化的潜变量间结构关联（两因子时等于标准化回归系数）；"
        "这是**相关因子测量模型**，有向结构路径（中介/通径）仍属后续扩展；弱信息先验、指标已标准化。"
        + drop_note
    )
    code += [
        "import pymc as pm, pytensor.tensor as pt  # 相关多因子贝叶斯 CFA（边际化）",
        "with pm.Model():",
        "    # 每因子首指标载荷正锚定(定符号); 其余 Normal",
        "    L=pm.LKJCorr('L',n=k,eta=2.0); Phi=pt.dot(L,L.T)   # 因子相关阵",
        "    cov=Lam@Phi@Lam.T+pt.diag(psi**2)                   # Σ=ΛΦΛᵀ+Ψ",
        "    pm.MvNormal('y_obs', mu=pt.zeros(p), cov=cov, observed=Z)",
    ]


@register("bayesian_sem")
def _branch_bayesian_sem(ctx: Ctx) -> None:
    """Bayesian confirmatory factor analysis — the auto-runnable core of Bayesian SEM (a
    measurement model). Modern PyMC NUTS, so no R blavaan / JAGS / Stan compiler needed.
    Default = single-factor CFA on the continuous columns; a lavaan-style ``model_spec``
    with ≥2 factors ('F1 =~ a+b+c \\n F2 =~ d+e+f') routes to a CORRELATED multi-factor
    CFA that also reports the inter-factor correlations (the standardized structural
    associations between latents). Directed structural regressions remain future work."""
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    method = entry.method
    import re

    import numpy as np
    import pandas as pd

    from researchforge.executor.branches.bayesian_mcmc import (
        _conv_note,
        _convergence,
        _degrade,
        _forest,
        _have_pymc,
        _hdi_bounds,
        _sampler_cfg,
    )

    if not _have_pymc():
        _degrade(summary, method, "频率派 sem（CB-SEM, lavaan/semopy）/ efa（探索因子）")
        return

    _excl = {fp.unit_col, fp.time_col}
    # multi-factor route: a model_spec defining ≥2 factors (each ≥2 indicators) → a
    # correlated multi-factor CFA (reports inter-factor correlations).
    _user_spec0 = cfg.get("model_spec")
    if _user_spec0:
        _factors, _dropped = _parse_measurement(_user_spec0, set(df.columns))
        if len(_factors) >= 2 and all(len(i) >= 2 for _, i in _factors):
            # directed structural paths among factors ('F2 ~ F1') → recursive structural
            # SEM; else (only '=~' measurement) → correlated multi-factor CFA.
            _paths = _parse_structural(_user_spec0, [f for f, _ in _factors])
            if _paths:
                _run_bayesian_structural(ctx, _factors, _paths, _dropped)
            else:
                _run_bayesian_multifactor(ctx, _factors, _dropped)
            return
    # resolve indicators: config indicators, else columns named in a lavaan-style
    # model_spec, else the continuous columns (single-factor CFA on all of them).
    user_spec = cfg.get("model_spec")
    cfg_inds = cfg.get("indicators")
    if cfg_inds and isinstance(cfg_inds, (list, tuple)):
        inds = [c for c in cfg_inds if c in df.columns]
    elif user_spec:
        inds = [c for c in df.columns
                if re.search(rf"(?<![\w.]){re.escape(str(c))}(?![\w.])", str(user_spec))]
    else:
        inds = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
    inds = inds[:8]
    if len(inds) < 3:
        summary.append("贝叶斯 SEM 跳过：需要 ≥3 个连续指标变量（单因子 CFA 识别要求）。")
        return

    Y = df[inds].apply(lambda s: pd.to_numeric(s, errors="coerce")).dropna()
    if len(Y) < 24:
        summary.append("贝叶斯 SEM 跳过：有效样本不足（去缺失后 < 24 行）。")
        return
    n, p = Y.shape
    Yv = Y.to_numpy(float)
    sd_y = Yv.std(axis=0, ddof=0)
    sd_y = np.where(sd_y < 1e-12, 1.0, sd_y)
    Z = (Yv - Yv.mean(axis=0)) / sd_y  # standardized indicators (loadings on std scale)

    sc = _sampler_cfg(cfg)
    import pymc as pm
    import pytensor.tensor as pt

    # Marginalize out the latent factor: with a standard-normal factor (var fixed = 1
    # for scale identification) and diagonal residuals, the indicator covariance is
    # Σ = λλᵀ + Ψ (Ψ = diag(ψ²)). Marginalizing avoids the n per-observation factor
    # scores (a multiplicative funnel that mixes terribly) — far better-conditioned for
    # NUTS, recovering loadings reliably with modest draws.
    with pm.Model() as model:
        lam = pm.Normal("lam", 0.0, 1.0, shape=p)        # loadings (standardized-indicator scale)
        psi = pm.HalfNormal("psi", 1.0, shape=p)         # indicator residual SDs
        cov = pt.outer(lam, lam) + pt.diag(psi ** 2)     # Σ = λλᵀ + Ψ  (always PD)
        pm.MvNormal("y_obs", mu=pt.zeros(p), cov=cov, observed=Z)  # Z standardized → mean 0
        idata = pm.sample(draws=sc["draws"], tune=sc["tune"], chains=sc["chains"],
                          cores=1, random_seed=sc["seed"], progressbar=False,
                          target_accept=0.95)

    # The loadings enter the likelihood only through Σ=λλᵀ+Ψ, so the posterior is
    # bimodal across the GLOBAL sign (λ and −λ fit identically). If two chains land in
    # opposite sign modes, R-hat on lam would inflate spuriously and trip a false
    # "未收敛" warning. De-alias the sign PER CHAIN (flip a chain whose mean loading-sum
    # is negative) BEFORE computing convergence, so R-hat reflects genuine mixing, not
    # the sign-mode split. (psi is sign-invariant; the global report-flip happens below.)
    post = idata.posterior
    _lam_v = post["lam"].values
    _chain_sgn = np.where(_lam_v.reshape(_lam_v.shape[0], -1).sum(axis=1) < 0.0, -1.0, 1.0)
    post["lam"].values[...] = _lam_v * _chain_sgn[:, None, None]

    # convergence keys off the structural params (lam/psi); eta has n noisy params.
    max_rhat, min_ess = _convergence(idata, ["lam", "psi"])
    lam_mean0 = post["lam"].values.reshape(-1, p).mean(axis=0)
    psi_mean = post["psi"].values.reshape(-1, p).mean(axis=0)
    # sign indeterminacy (factor sign + all loadings can flip): fix loadings to be
    # predominantly positive (a single global flip; disclosed).
    flip = bool(lam_mean0.sum() < 0)
    lam_mean = -lam_mean0 if flip else lam_mean0
    lam_lo0, lam_hi0 = _hdi_bounds(idata, "lam", sc["hdi"])
    lam_lo, lam_hi = ((-np.asarray(lam_hi0), -np.asarray(lam_lo0)) if flip
                      else (np.asarray(lam_lo0), np.asarray(lam_hi0)))

    # McDonald's omega (single-factor construct reliability) + AVE on the std scale.
    s_lam = float(np.sum(lam_mean))
    sum_psi2 = float(np.sum(psi_mean ** 2))
    omega = (s_lam ** 2) / ((s_lam ** 2) + sum_psi2) if (s_lam ** 2 + sum_psi2) > 0 else float("nan")
    sum_lam2 = float(np.sum(lam_mean ** 2))
    ave = sum_lam2 / (sum_lam2 + sum_psi2) if (sum_lam2 + sum_psi2) > 0 else float("nan")

    # estimates — headline reliability + per-indicator standardized loadings first.
    estimates["omega_reliability"] = round(omega, 4)
    estimates["avg_variance_extracted"] = round(ave, 4)
    for j, ind in enumerate(inds):
        estimates[f"lam_{ind}"] = round(float(lam_mean[j]), 4)
        estimates[f"lam_{ind}_hdi_low"] = round(float(lam_lo[j]), 4)
        estimates[f"lam_{ind}_hdi_high"] = round(float(lam_hi[j]), 4)
    estimates["n_indicators"] = float(p)
    estimates["n_obs"] = float(n)
    estimates["max_rhat"] = round(max_rhat, 4)
    estimates["min_ess"] = round(min_ess, 1)

    try:
        ltbl = pd.DataFrame({
            "indicator": inds,
            "loading_mean": np.round(lam_mean, 5),
            "hdi_low": np.round(lam_lo, 5),
            "hdi_high": np.round(lam_hi, 5),
            "residual_sd": np.round(psi_mean, 5),
        })
        ltbl.to_csv(d / "bayesian_sem_loadings.csv", index=False, encoding="utf-8")
        files.append("bayesian_sem_loadings.csv")
    except Exception:
        pass
    _forest(idata, ["lam"], d / "bayesian_sem_loadings.png",
            "Bayesian CFA standardized loadings (94% HDI)")
    if (d / "bayesian_sem_loadings.png").exists():
        files.append("bayesian_sem_loadings.png")

    n_strong = int(np.sum(np.abs(lam_mean) > 0.5))
    summary.append(
        f"{method} 完成：单因子贝叶斯验证性因子分析（CFA，PyMC NUTS，{sc['chains']}链×{sc['draws']}抽样），"
        f"指标={inds}（{p} 个，n={n}）。构念信度 McDonald's ω≈{omega:.3f}"
        f"（>0.7 通常可接受），AVE≈{ave:.3f}；标准化载荷范围 "
        f"[{float(np.min(lam_mean)):.2f}, {float(np.max(lam_mean)):.2f}]，"
        f"其中 {n_strong}/{p} 个载荷 |λ|>0.5（强载荷）。载荷+HDI 见 bayesian_sem_loadings.csv。"
        f"{_conv_note(max_rhat, min_ess, sc['chains'])}。"
        " ⚠ 这是**单因子测量模型**（贝叶斯 CFA），非含潜变量间结构路径的完整 SEM"
        "（后者需理论驱动的多因子设定，属后续扩展）；因子方差固定为 1 以识别尺度；"
        "存在符号不定性，已将载荷统一翻转为以正为主（约定）；先验为弱信息、指标已标准化。"
    )
    code += [
        "import pymc as pm, pytensor.tensor as pt  # 单因子贝叶斯 CFA（边际化潜因子）",
        "with pm.Model():",
        "    lam=pm.Normal('lam',0,1,shape=p); psi=pm.HalfNormal('psi',1,shape=p)",
        "    cov=pt.outer(lam,lam)+pt.diag(psi**2)        # Σ=λλᵀ+Ψ（因子方差固定=1）",
        "    pm.MvNormal('y_obs', mu=pt.zeros(p), cov=cov, observed=Z)  # Z=标准化指标",
        "    idata=pm.sample(1000,tune=1000,chains=2,target_accept=0.95,random_seed=42)",
        "# omega=(Σλ)^2/((Σλ)^2+Σψ^2)  # 单因子构念信度（McDonald's omega）",
    ]



@register("pls_sem")
def _branch_pls_sem(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    summary.append(
        "PLS-SEM（偏最小二乘结构方程）需要你指定**测量模型**（哪些指标→哪个潜变量）与结构路径；"
        "引擎无法自动推断（随意分组会产出无意义结果，故不自动跑）。请指定测量模型后用 plspm / SmartPLS 运行；"
        "或先用 **SEM**（CB-SEM，自动单因子 CFA，经 lavaan/semopy）/ **EFA**（探索因子结构）作可自动执行的替代。"
    )



@register("sem")
def _branch_sem(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import re

    _excl = {fp.unit_col, fp.time_col}
    indicators = [
        c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl
    ]
    # config={"model_spec": "<lavaan/semopy syntax>"} lets the user supply their
    # theoretical structure (multi-factor CFA / paths) instead of the auto
    # single-factor template. Columns are taken from those named in the spec.
    user_spec = cfg.get("model_spec") or cfg.get("sem_spec")
    if user_spec:
        used = [
            c for c in df.columns
            if re.search(rf"(?<![\w.]){re.escape(str(c))}(?![\w.])", user_spec)
        ]
        spec = user_spec
    else:
        used = indicators[:8]
        spec = "F =~ " + " + ".join(used)
    if not user_spec and len(indicators) < 3:
        summary.append("SEM 失败：需要 ≥3 个连续指标变量（单因子模型识别要求）。")
    elif user_spec and len(used) < 2:
        summary.append("SEM 失败：config model_spec 中未匹配到 ≥2 个数据列名。")
    else:
        import pandas as pd

        from researchforge.executor import rbridge

        inds = used
        sub = df[inds].dropna()
        # prefer lavaan (R, gold standard — also gives SRMR) when available;
        # fall back to pure-Python semopy so the analysis runs anywhere.
        # Only use the R backend with identifier-safe column names: names go
        # into the R model string, so a name with quotes/commas could break
        # parsing or inject R — semopy takes the names as data, no eval.
        names_safe = all(re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", str(c)) for c in inds)
        # the spec is interpolated into cfa("...") as an R string literal; a stray
        # double-quote/backslash would break out, so a custom spec with those is
        # routed to semopy instead (takes the spec as a Python string, no R eval).
        spec_safe = '"' not in spec and "\\" not in spec
        result = None
        if names_safe and spec_safe and rbridge.r_available() and rbridge.r_package_available("lavaan"):
            csv = d / "_sem_input.csv"
            sub.to_csv(csv, index=False)
            try:
                result = _sem_via_lavaan(csv, spec)
            except Exception:
                result = None
            finally:
                try:
                    csv.unlink()
                except OSError:
                    pass
        if result is None:
            try:
                result = _sem_via_semopy(sub, spec)
            except Exception as err:
                summary.append(f"SEM 拟合失败：{err}")
        if result is not None:
            load = result["loadings"]
            fit = result["fit"]
            (d / "summary.txt").write_text(result["summary"], encoding="utf-8")
            files.append("summary.txt")
            load.to_csv(d / "loadings.csv", index=False, encoding="utf-8")
            files.append("loadings.csv")
            pd.DataFrame([fit]).to_csv(d / "fit_indices.csv", index=False, encoding="utf-8")
            files.append("fit_indices.csv")
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(5, 3.2))
                _ylab = (
                    load["indicator"].astype(str) + " ← " + load["factor"].astype(str)
                    if "factor" in load.columns and load["factor"].nunique() > 1
                    else load["indicator"].astype(str)
                )
                ax.barh(_ylab, load["std_loading"], color="#4C72B0")
                ax.set_xlabel("standardised loading")
                ax.set_title("SEM measurement loadings")
                fig.tight_layout()
                fig.savefig(d / "loadings.png", dpi=150)
                plt.close(fig)
                files.append("loadings.png")
            except Exception:
                pass
            cfi, tli, rmsea = fit["cfi"], fit["tli"], fit["rmsea"]
            chi2, dof, srmr = fit["chi2"], fit["dof"], fit.get("srmr", float("nan"))
            for kk, vv in (("cfi", cfi), ("tli", tli), ("rmsea", rmsea), ("chi2", chi2), ("dof", dof)):
                estimates[kk] = round(vv, 4)
            if dof <= 0:
                # 3 indicators -> just-identified (df=0): CFI/RMSEA perfect by
                # construction, say nothing about fit (Opus double-review catch).
                verdict = "恰好识别(df=0)，拟合指数无意义(CFI/RMSEA 必完美)；需 ≥4 指标才能评估拟合"
            elif cfi >= 0.95 and rmsea <= 0.06:
                verdict = "拟合良好"
            else:
                verdict = "拟合一般/欠佳"
            srmr_txt = f" SRMR={srmr:.3f}" if srmr == srmr else ""  # NaN-safe
            _n_factors = len(set(_sem_latents(spec))) or 1
            _model_desc = (
                f"自定义模型（{_n_factors} 因子，按 config model_spec）"
                if user_spec
                else "单因子 CFA"
            )
            _tail = (
                "" if user_spec
                else "（此为探索性模板；可用 config={\"model_spec\": \"lavaan语法\"} 按理论结构改写后重跑）"
            )
            summary.append(
                f"{entry.method} 完成（后端：{result['backend']}）：{_model_desc} over "
                f"{len(inds)} 个指标（df={dof:.0f}）；CFI={cfi:.3f} TLI={tli:.3f} "
                f"RMSEA={rmsea:.3f}{srmr_txt} → {verdict}" + _tail
            )
            code += [
                "# SEM single-factor CFA — prefers R/lavaan, falls back to semopy",
                f'spec = "{spec}"',
                "# lavaan: cfa(spec, data=df, std.lv=TRUE); semopy: semopy.Model(spec).fit(df)",
            ]

