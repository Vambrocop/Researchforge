"""Branch handler for bayesian_gam — Bayesian generalized additive model.

The Bayesian counterpart of the frequentist ``gam``/``gamm`` (family statistics).
Each continuous predictor gets a smooth f_j(x_j) from a truncated-cubic spline basis
with an iid Gaussian (ridge) prior on the knot coefficients — the mixed-model
representation of a penalized spline (Ruppert-Wand-Carroll, *Semiparametric
Regression*): the smoothing parameter is the prior SD su_j (estimated, NON-CENTERED
u=su·z to avoid the variance funnel), so wiggliness is shrunk toward a straight line
and learned from the data. Reports each smooth + 94% credible band, the per-predictor
smoothing SD and linear slope, residual sigma, Bayesian R², and convergence.

PyMC optional + graceful degrade (mirrors bayesian_mcmc): no pymc/arviz -> honest
Chinese skip pointing at gam/gamm. y and predictors are standardized for stable NUTS
geometry; smooths are back-transformed to response units. Plots use matplotlib Agg
with ENGLISH labels; summary is Chinese with ⚠ disclosures. Mutates ctx, never rebinds.
See executor/_branch_api.py and CLAUDE.md「引擎约定」.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.branches.bayesian_mcmc import (
    _clean,
    _conv_note,
    _convergence,
    _degrade,
    _have_pymc,
    _resolve_reg,
    _sampler_cfg,
)

# additive smooths get expensive fast; cap the number of smooth terms
_MAX_SMOOTHS = 4


def _trunc_cubic(xz, knots):
    """Centered truncated-cubic basis (n×K): (x − knot_k)_+³, columns mean-centered so
    the curvature block sits ⊥ to the intercept/linear term. Returns (Z_centered, colmeans)."""
    import numpy as np

    Z = np.maximum(xz[:, None] - knots[None, :], 0.0) ** 3
    cm = Z.mean(axis=0)
    return Z - cm, cm


@register("bayesian_gam")
def _branch_bayesian_gam(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    method = entry.method
    if not _have_pymc():
        _degrade(summary, method, "gam / gamm")
        return

    outcome, preds, problem = _resolve_reg(ctx, method, binary_outcome=False)
    if problem:
        summary.append(problem)
        return
    X, y, preds, problem = _clean(df, outcome, preds, binary_outcome=False)
    if problem:
        summary.append(f"{method} 跳过：{problem}")
        return

    import numpy as np

    preds = preds[:_MAX_SMOOTHS]
    X = X[:, : len(preds)]
    p = len(preds)
    n = len(y)

    # standardize y and each predictor; build a centered truncated-cubic basis per smooth
    ym, ys = float(y.mean()), float(y.std() or 1.0)
    yz = (y - ym) / ys
    K = int(cfg.get("n_knots", min(15, max(5, n // 8))))
    xinfo, knots_l, Z_l = [], [], []
    for j in range(p):
        xj = X[:, j]
        mu, sd = float(xj.mean()), float(xj.std() or 1.0)
        xjz = (xj - mu) / sd
        kn = np.unique(np.quantile(xjz, np.linspace(0, 1, K + 2)[1:-1]))
        Zc, _cm = _trunc_cubic(xjz, kn)
        xinfo.append((xjz, mu, sd))
        knots_l.append(kn)
        Z_l.append(Zc)

    sc = _sampler_cfg(cfg)
    target_accept = float(cfg.get("target_accept", 0.95))
    try:
        import pymc as pm
        import pytensor.tensor as pt

        with pm.Model() as model:
            b0 = pm.Normal("b0", 0.0, 2.0)
            mu_t = b0 + 0.0 * yz  # broadcast to length n
            for j in range(p):
                xjz = xinfo[j][0]
                Kj = Z_l[j].shape[1]
                b1 = pm.Normal(f"b1_{j}", 0.0, 2.0)
                su = pm.HalfNormal(f"su_{j}", 1.0)           # smoothing parameter
                z = pm.Normal(f"z_{j}", 0.0, 1.0, shape=Kj)  # non-centered curvature coefs
                mu_t = mu_t + b1 * xjz + pt.dot(Z_l[j], su * z)
            sigma = pm.HalfNormal("sigma", 1.0)
            pm.Normal("y_obs", mu_t, sigma, observed=yz)
            pm.Deterministic("f", mu_t)
            idata = pm.sample(draws=sc["draws"], tune=sc["tune"], chains=sc["chains"],
                              target_accept=target_accept, random_seed=sc["seed"],
                              progressbar=False)

        post = idata.posterior
        n_div = int(idata.sample_stats["diverging"].sum())
        conv_names = ["b0", "sigma"] + [f"b1_{j}" for j in range(p)] + [f"su_{j}" for j in range(p)]
        max_rhat, min_ess = _convergence(idata, conv_names)

        # posterior-mean fit (response units) + Bayesian R² (Gelman: var(fit)/(var(fit)+σ²))
        f_resp = post["f"].mean(("chain", "draw")).values * ys + ym
        sigma_resp = float(post["sigma"].mean()) * ys
        var_fit = float(np.var(f_resp))
        bayes_r2 = var_fit / (var_fit + sigma_resp ** 2) if (var_fit + sigma_resp ** 2) > 0 else float("nan")

        # flatten draws once for reconstructing each smooth on a grid
        def _flat(name):
            return post[name].values.reshape(-1, *post[name].values.shape[2:])

        b0d = _flat("b0")  # (S,)
        smooth_rows = []
        grid_n = 60
        per_pred = {}
        for j in range(p):
            xjz, mu, sd = xinfo[j]
            kn = knots_l[j]
            b1d = _flat(f"b1_{j}")          # (S,)
            sud = _flat(f"su_{j}")          # (S,)
            zd = _flat(f"z_{j}")            # (S, Kj)
            ud = sud[:, None] * zd          # (S, Kj)
            gz = np.linspace(xjz.min(), xjz.max(), grid_n)
            Zg = np.maximum(gz[:, None] - kn[None, :], 0.0) ** 3
            Zg = Zg - Zg.mean(axis=0)
            # f_j(grid) per draw, centered to mean 0 over the grid (partial effect), response units
            fj = b1d[:, None] * gz[None, :] + ud @ Zg.T      # (S, grid_n)
            fj = fj - fj.mean(axis=1, keepdims=True)
            fj = fj * ys
            fm = fj.mean(axis=0)
            lo, hi = np.percentile(fj, [3.0, 97.0], axis=0)
            gx = gz * sd + mu
            for k in range(grid_n):
                smooth_rows.append({"predictor": preds[j], "x": round(float(gx[k]), 6),
                                    "smooth": round(float(fm[k]), 6),
                                    "lo": round(float(lo[k]), 6), "hi": round(float(hi[k]), 6)})
            # linear slope in response units per predictor unit; smoothing SD; curvature magnitude
            per_pred[preds[j]] = {
                "linear_slope": float(b1d.mean()) * ys / sd,
                "smoothing_sd": float(sud.mean()),
                "smooth_range": float(fm.max() - fm.min()),
            }

        import pandas as pd

        sm_df = pd.DataFrame(smooth_rows)
        sm_df.to_csv(d / "gam_smooths.csv", index=False, encoding="utf-8")
        files.append("gam_smooths.csv")

        # plot the FIRST smooth (partial effect + 94% band + partial residuals)
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            j0 = 0
            sub = sm_df[sm_df["predictor"] == preds[j0]].sort_values("x")
            # partial residuals aligned to the PLOTTED smooth (identical centering) by
            # interpolating the displayed grid smooth at the observed x — so the point
            # cloud sits on the line rather than offset by a different centering constant.
            fj_obs = np.interp(X[:, j0], sub["x"].to_numpy(), sub["smooth"].to_numpy())
            partial = (y - f_resp) + fj_obs        # partial residuals for predictor 0
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.scatter(X[:, j0], partial, s=14, alpha=0.4, color="#4C72B0", label="partial residuals")
            ax.plot(sub["x"], sub["smooth"], color="#C44E52", lw=2, label="posterior mean smooth")
            ax.fill_between(sub["x"], sub["lo"], sub["hi"], color="#C44E52", alpha=0.2, label="94% credible band")
            ax.set_xlabel(str(preds[j0]))
            ax.set_ylabel(f"partial effect on {outcome}")
            ax.set_title(f"Bayesian GAM smooth: f({preds[j0]})")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "gam_smooth.png", dpi=150)
            plt.close(fig)
            files.append("gam_smooth.png")
        except Exception:
            pass

        estimates["n"] = float(n)
        estimates["n_predictors"] = float(p)
        estimates["sigma"] = round(sigma_resp, 5)
        estimates["bayesian_r2"] = round(bayes_r2, 4) if bayes_r2 == bayes_r2 else float("nan")
        estimates["max_rhat"] = round(max_rhat, 4)
        estimates["min_ess"] = round(min_ess, 1)
        estimates["n_divergences"] = float(n_div)
        for name, dd in per_pred.items():
            estimates[f"linear_slope_{name}"] = round(dd["linear_slope"], 5)
            estimates[f"smoothing_sd_{name}"] = round(dd["smoothing_sd"], 5)
            estimates[f"smooth_range_{name}"] = round(dd["smooth_range"], 5)

        conv = _conv_note(max_rhat, min_ess, sc["chains"])
        div_note = f"；{n_div} 个发散（divergences）" if n_div else ""
        pred_lines = "\n".join(
            f"  {name}: 线性斜率≈{round(dd['linear_slope'], 4)}（响应/预测变量单位），"
            f"平滑强度 su={round(dd['smoothing_sd'], 4)}，曲线幅度={round(dd['smooth_range'], 4)}"
            for name, dd in per_pred.items()
        )
        (d / "gam_summary.txt").write_text(
            f"贝叶斯广义可加模型（Bayesian GAM，PyMC NUTS）：结果 {outcome} ~ "
            + " + ".join(f"s({c})" for c in preds) + "\n"
            f"样本 {n}，平滑项 {p} 个（每个 {K} 节点截断三次样条 + 岭先验 = 惩罚样条的混合模型表示）\n"
            + pred_lines + "\n"
            f"残差 σ={round(sigma_resp, 4)}，贝叶斯 R²≈{round(bayes_r2, 4) if bayes_r2 == bayes_r2 else 'NaN'}\n"
            f"{conv}{div_note}\n"
            "注：每个平滑项 f_j 由截断三次样条基 + **非中心化岭先验**（平滑参数=先验 SD su_j，"
            "数据估计、向直线收缩）拟合——惩罚样条的贝叶斯/混合模型表示（Ruppert-Wand-Carroll）；"
            "曲线带为 94% 后验可信带（同时量化拟合不确定性，频率派 GAM 的置信带更近似）；"
            "贝叶斯 R² 为后验均值 plug-in（近似）；平滑项已中心化为偏效应（绝对水平归入截距）；"
            "linear_slope 为**显式线性项**系数（非整条平滑的平均斜率，截断样条基也可携带趋势）；"
            f"最多拟合 {_MAX_SMOOTHS} 个平滑项（其余数值列略去）。\n\n"
            "平滑曲线（前 12 行）：\n" + sm_df.head(12).to_string(index=False),
            encoding="utf-8",
        )
        files.append("gam_summary.txt")

        summary.append(
            f"{method} 完成（PyMC）：{outcome} ~ " + " + ".join(f"s({c})" for c in preds) + "；"
            f"{n} 样本、{p} 平滑项；残差 σ={round(sigma_resp, 3)}、贝叶斯 R²≈"
            f"{round(bayes_r2, 3) if bayes_r2 == bayes_r2 else 'NaN'}；{conv}{div_note}。"
            "⚠ 平滑=截断三次样条+非中心化岭先验（惩罚样条混合模型表示，平滑参数数据估计）；"
            "曲线带为 94% 可信带；R² 为 plug-in 近似；平滑项中心化为偏效应；"
            f"最多 {_MAX_SMOOTHS} 个平滑项。"
        )
        code += [
            "import pymc as pm, pytensor.tensor as pt, numpy as np",
            "# per predictor: truncated-cubic basis Z=(x-knot)_+**3 (centered) + ridge prior",
            "# su=pm.HalfNormal(...); z=pm.Normal(0,1,K); f = b1*x + Z @ (su*z)  # non-centered P-spline",
            "# y ~ N(b0 + sum_j f_j, sigma); smoothing param su learned from data",
        ]
    except Exception as err:  # pragma: no cover - sampler/runtime guard
        summary.append(f"{method} 失败：{err}")
