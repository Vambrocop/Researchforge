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


@register("bayesian_sem")
def _branch_bayesian_sem(ctx: Ctx) -> None:
    """Bayesian confirmatory factor analysis (single latent factor) — the auto-runnable
    core of Bayesian SEM (a measurement model). Re-implemented with modern PyMC NUTS,
    so it no longer needs R blavaan + a JAGS/Stan compiler (same unblocking as the
    Bayesian regression family). Full structural paths between multiple latents are out
    of scope here (need a theory-driven multi-factor spec) — disclosed."""
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

    # convergence keys off the structural params (lam/psi); eta has n noisy params.
    max_rhat, min_ess = _convergence(idata, ["lam", "psi"])
    post = idata.posterior
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

