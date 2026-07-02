"""Branch handlers for the limited-dependent-variable (LDV) econometrics family.

Fills a real catalog gap: there is OLS / logit / IV / quantile / panel but NO
censored / truncated / sample-selection models. All three are PURE PYTHON
(numpy + scipy.optimize + statsmodels for the Heckman probit — all core deps,
NO R, NO new dependencies, NO network):

  * tobit_regression     — Type-I Tobit (censored normal) by MLE. Uncensored obs
                           contribute log φ((y-Xβ)/σ)/σ; censored obs contribute
                           log Φ((c-Xβ)/σ) (left) or log Φ((Xβ-c)/σ) (right).
                           Maximised over (β, log σ). β is the LATENT-index effect.
  * truncated_regression — truncated-normal regression by MLE. The sample is
                           truncated (obs beyond the threshold are ABSENT). LL (lower
                           trunc at a) = Σ[ log φ((y-Xβ)/σ)/σ − log(1−Φ((a−Xβ)/σ)) ].
                           Contrast with naive OLS (attenuated under truncation).
  * heckman_selection    — Heckman (1979) two-step heckit. Step 1: probit of the
                           selection indicator on [predictors+exclusion] over ALL rows;
                           inverse Mills ratio λ=φ(zγ)/Φ(zγ). Step 2: OLS of outcome on
                           [predictors+λ] over SELECTED rows; the λ coefficient = ρ·σ_ε
                           and its t-test is the selection-bias test (H0: ρ=0).

Each handler resolves column roles like the regression family (outcome = first
continuous; predictors = remaining numeric), reads config overrides, degrades
honestly (Chinese "跳过/失败：<原因>" + return, never crashes), writes a
coefficients CSV (always) + a matplotlib PNG (best-effort, Agg, ENGLISH labels),
fills float `estimates` (CIs/SEs as SEPARATE scalar keys, headline FIRST), appends
a Chinese `summary` ending with ⚠ disclosures, and MUTATES ctx (never rebinds).
See executor/_branch_api.py, branches/panel_extra.py, and CLAUDE.md.

Math self-checked against Wooldridge ch.17 and Greene (Tobit/truncated MLE,
Heckman two-step & inverse Mills ratio).
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


# ─────────────────────────────────────────────────────────────────────────────
# Shared cross-sectional role-resolution (regression-family convention):
#   outcome   = cfg["outcome"] if a column, else FIRST continuous col (not unit/time)
#   predictors= cfg["predictors"] (valid list) else remaining numeric cols (capped)
# Returns (outcome, predictors, problem_msg). When problem_msg is not None the
# caller appends it to summary and returns.
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_xy(ctx: Ctx, label: str, max_pred: int = 20):
    fp, cfg, df = ctx.fp, ctx.cfg, ctx.df
    _exc = {fp.unit_col, fp.time_col}

    outcome = cfg.get("outcome")
    if not (outcome and outcome in df.columns):
        outcome = next(
            (c.name for c in fp.columns
             if c.kind == "continuous" and c.name not in _exc),
            None,
        )
    if outcome is None:
        return None, [], f"{label}跳过：需要 1 个连续结果变量（outcome）。"

    cfg_pred = cfg.get("predictors")
    if cfg_pred and isinstance(cfg_pred, (list, tuple)):
        predictors = [c for c in cfg_pred if c in df.columns and c != outcome]
    else:
        predictors = [
            c.name for c in fp.columns
            if c.kind in {"continuous", "count", "binary"}
            and c.name not in _exc and c.name != outcome
        ]
    predictors = predictors[:max_pred]
    if not predictors:
        return outcome, [], f"{label}跳过：需要 ≥1 个预测变量。"
    return outcome, predictors, None


def _numeric_matrix(df, outcome, predictors):
    """Coerce to numeric, drop NaN rows over [outcome, *predictors], drop predictors
    that became constant/all-NaN. Returns (y, X_no_const, kept_predictors) with X as a
    float ndarray WITHOUT an intercept column (callers prepend the const)."""
    import pandas as pd

    cols = [outcome, *predictors]
    sub = df[cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
    sub = sub.dropna()
    if sub.empty:
        return None, None, []
    kept = [p for p in predictors if float(sub[p].std(ddof=0)) > 1e-12]
    if not kept:
        return sub[outcome].to_numpy(float), None, []
    y = sub[outcome].to_numpy(float)
    X = sub[kept].to_numpy(float)
    return y, X, kept


def _ols_fit(y, X_with_const):
    """Plain OLS via lstsq; returns (beta, resid_std). X already has a const column."""
    import numpy as np

    beta, *_ = np.linalg.lstsq(X_with_const, y, rcond=None)
    resid = y - X_with_const @ beta
    dof = max(1, len(y) - X_with_const.shape[1])
    sigma = float(np.sqrt(float(resid @ resid) / dof))
    return beta, max(sigma, 1e-6)


def _num_hessian(f, x, eps: float = 1e-4):
    """Central-difference numerical Hessian of scalar f at x."""
    import numpy as np

    x = np.asarray(x, float)
    n = len(x)
    H = np.zeros((n, n))
    h = np.maximum(np.abs(x), 1.0) * eps
    for i in range(n):
        for j in range(i, n):
            xpp = x.copy(); xpp[i] += h[i]; xpp[j] += h[j]
            xpm = x.copy(); xpm[i] += h[i]; xpm[j] -= h[j]
            xmp = x.copy(); xmp[i] -= h[i]; xmp[j] += h[j]
            xmm = x.copy(); xmm[i] -= h[i]; xmm[j] -= h[j]
            val = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4.0 * h[i] * h[j])
            H[i, j] = H[j, i] = val
    return H


def _se_from_hessian(neg_ll, theta_hat):
    """Standard errors from the inverse numerical Hessian of the NEGATIVE log-lik at
    the optimum (the observed information). Returns array of SEs (NaN where the Hessian
    is not invertible / a variance is non-positive)."""
    import numpy as np

    k = len(theta_hat)
    se = np.full(k, np.nan)
    try:
        H = _num_hessian(neg_ll, theta_hat)
        cov = np.linalg.inv(H)
        diag = np.diag(cov)
        se = np.where(diag > 0, np.sqrt(np.abs(diag)), np.nan)
    except Exception:
        pass
    return se


def _coef_plot_ci(d, files, terms, coefs, ses, xlabel, title, fname="coefficients.png"):
    """Best-effort coefficient plot with 95% CIs (Agg, ENGLISH labels)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        ses_safe = [s if (s is not None and np.isfinite(s)) else 0.0 for s in ses]
        fig, ax = plt.subplots(figsize=(5.8, 3.4))
        ax.errorbar(coefs, range(len(terms)), xerr=[1.96 * s for s in ses_safe], fmt="o")
        ax.axvline(0, color="grey", ls="--")
        ax.set_yticks(range(len(terms)))
        ax.set_yticklabels(terms)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(d / fname, dpi=150)
        plt.close(fig)
        files.append(fname)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# (A) tobit_regression — Type-I Tobit (censored normal regression) by MLE
# ─────────────────────────────────────────────────────────────────────────────
@register("tobit_regression")
def _branch_tobit_regression(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    outcome, predictors, problem = _resolve_xy(ctx, "Tobit 回归")
    if problem is not None:
        summary.append(problem)
        return
    try:
        import numpy as np
        from scipy.optimize import minimize
        from scipy.stats import norm

        y, Xraw, kept = _numeric_matrix(df, outcome, predictors)
        if y is None or Xraw is None or not kept:
            summary.append("Tobit 回归跳过：数据清洗后无有效预测变量（全缺失或常量）。")
            return
        n = len(y)
        if n < 20:
            summary.append(f"Tobit 回归跳过：有效样本太少（n={n}，需 ≥20）。")
            return

        side = str(cfg.get("censoring") or "left").lower()
        if side not in {"left", "right"}:
            side = "left"
        # default censor point = observed min (left) / max (right)
        if cfg.get("censor_value") is not None:
            censor = float(cfg.get("censor_value"))
        else:
            censor = float(y.min()) if side == "left" else float(y.max())

        tol = 1e-9 * max(1.0, abs(censor))
        if side == "left":
            cens_mask = y <= censor + tol
        else:
            cens_mask = y >= censor - tol
        n_cens = int(cens_mask.sum())
        n_unc = int((~cens_mask).sum())

        # Require a real censoring mass; otherwise Tobit reduces to OLS — say so.
        if n_cens < 3 or (n_cens / n) < 0.05:
            summary.append(
                f"Tobit 回归跳过：未检测到足够的删失质量（在{('下' if side=='left' else '上')}限 "
                f"{censor:g} 处仅 {n_cens}/{n} 个观测，<5%）。无删失时 Tobit 退化为 OLS，"
                "请改用 ols_regression。可用 config censoring/censor_value 指定删失方式与门限。"
            )
            return
        if n_unc < len(kept) + 2:
            summary.append(
                f"Tobit 回归跳过：未删失观测太少（{n_unc} 个，少于参数数），无法识别。"
            )
            return

        X = np.column_stack([np.ones(n), Xraw])  # const + predictors
        k = X.shape[1]
        terms = ["const", *kept]

        # OLS start (β from all data; σ from resid std)
        beta0, sigma0 = _ols_fit(y, X)
        theta0 = np.concatenate([beta0, [np.log(sigma0)]])

        def neg_ll(theta):
            b = theta[:k]
            sig = np.exp(theta[k])  # logσ parameterisation -> σ>0
            mu = X @ b
            ll = 0.0
            # uncensored: log φ((y-μ)/σ)/σ
            zu = (y[~cens_mask] - mu[~cens_mask]) / sig
            ll += np.sum(norm.logpdf(zu) - np.log(sig))
            # censored: left -> log Φ((c-μ)/σ); right -> log Φ((μ-c)/σ)
            mu_c = mu[cens_mask]
            if side == "left":
                zc = (censor - mu_c) / sig
            else:
                zc = (mu_c - censor) / sig
            ll += np.sum(norm.logcdf(zc))
            return -ll

        res = minimize(neg_ll, theta0, method="BFGS",
                       options={"maxiter": 2000, "gtol": 1e-6})
        theta = res.x
        converged = bool(res.success)
        beta = theta[:k]
        sigma = float(np.exp(theta[k]))
        loglik = -float(res.fun)

        se_full = _se_from_hessian(neg_ll, theta)
        se_beta = se_full[:k]
        # SE of σ via delta method (σ = exp(logσ) -> se_σ = σ * se_logσ)
        se_sigma = float(sigma * se_full[k]) if np.isfinite(se_full[k]) else float("nan")

        # marginal effect on observed E[y] scale: Φ((x̄β−c)/σ) left-censored, Φ((c−x̄β)/σ) right.
        # (dropping the censor point c — only valid at c=0 — understated/overstated this factor.)
        mu_bar = float(np.mean(X @ beta))
        if sigma > 0:
            z_scale = (mu_bar - censor) / sigma if side == "left" else (censor - mu_bar) / sigma
            scale = float(norm.cdf(z_scale))
        else:
            scale = float("nan")

        # naive OLS on the (censored) y — biased toward 0; for the contrast
        ols_beta, _ = _ols_fit(y, X)

        import pandas as pd

        rows = []
        for i, t in enumerate(terms):
            b = float(beta[i])
            s = float(se_beta[i])
            if np.isfinite(s) and s > 0:
                z = b / s
                p = float(2 * norm.sf(abs(z)))
                lo, hi = b - 1.96 * s, b + 1.96 * s
            else:
                z = p = lo = hi = float("nan")
            rows.append({
                "term": t,
                "tobit_coef": round(b, 6),
                "se": round(s, 6) if np.isfinite(s) else float("nan"),
                "z": round(z, 4) if np.isfinite(z) else float("nan"),
                "p": round(p, 6) if np.isfinite(p) else float("nan"),
                "ci_low": round(lo, 6) if np.isfinite(lo) else float("nan"),
                "ci_high": round(hi, 6) if np.isfinite(hi) else float("nan"),
                "naive_ols_coef": round(float(ols_beta[i]), 6),
                "marginal_on_Ey": round(b * scale, 6) if np.isfinite(scale) else float("nan"),
            })
        tab = pd.DataFrame(rows)
        tab.to_csv(d / "tobit_coefficients.csv", index=False, encoding="utf-8")
        files.append("tobit_coefficients.csv")

        _coef_plot_ci(
            d, files, terms,
            [float(beta[i]) for i in range(k)],
            [float(se_beta[i]) for i in range(k)],
            "Tobit latent-index coefficient (95% CI)",
            f"Tobit regression (censored {side}) - {outcome}",
        )

        # estimates: headline = slopes first (latent-index β), then diagnostics
        for i, t in enumerate(terms):
            if t == "const":
                continue
            estimates[t] = round(float(beta[i]), 6)
            estimates[f"{t}_se"] = (round(float(se_beta[i]), 6)
                                    if np.isfinite(se_beta[i]) else float("nan"))
            if np.isfinite(se_beta[i]) and se_beta[i] > 0:
                estimates[f"{t}_ci_low"] = round(float(beta[i] - 1.96 * se_beta[i]), 6)
                estimates[f"{t}_ci_high"] = round(float(beta[i] + 1.96 * se_beta[i]), 6)
            else:
                estimates[f"{t}_ci_low"] = float("nan")
                estimates[f"{t}_ci_high"] = float("nan")
            estimates[f"{t}_naive_ols"] = round(float(ols_beta[i]), 6)  # OLS-on-censored contrast
        estimates["const"] = round(float(beta[0]), 6)
        estimates["sigma"] = round(sigma, 6)
        estimates["sigma_se"] = round(se_sigma, 6) if np.isfinite(se_sigma) else float("nan")
        estimates["scale_factor_Ey"] = round(scale, 6) if np.isfinite(scale) else float("nan")
        estimates["loglik"] = round(loglik, 4)
        estimates["n_censored"] = float(n_cens)
        estimates["n_uncensored"] = float(n_unc)
        estimates["n_obs"] = float(n)
        estimates["converged"] = 1.0 if converged else 0.0

        conv_note = "" if converged else "（⚠ 优化器未完全收敛，结果谨慎使用）"
        summary.append(
            f"{entry.method} 完成{conv_note}：n={n}（删失 {n_cens}、未删失 {n_unc}，"
            f"{('下' if side=='left' else '上')}删失于 {censor:g}），{len(kept)} 个预测变量。"
            f"σ̂={sigma:.4f}，对数似然={loglik:.2f}。系数（潜变量指数 β + SE + 95%CI + z/p，"
            f"并列 naive OLS 与对 E[y] 的边际近似）见 tobit_coefficients.csv。"
            "\n⚠ 披露：(1) Tobit 的 β 是对**潜变量指数 y\\* 的效应，不是**对观测 E[y] 的边际效应"
            f"——后者需乘以尺度因子 Φ((x̄β−c)/σ)≈{scale:.3f}（c=删失点，已在 CSV 的 marginal_on_Ey 列给出近似）。"
            "(2) 在删失数据上直接做 OLS 会把系数**向 0 偏**（CSV 的 naive_ols_coef 列可对照）。"
            "(3) Tobit 假定潜变量误差**正态且同方差**——异方差/非正态会使 MLE 不一致。"
            "(4) SE 来自数值 Hessian（观测信息阵）的逆；若 Hessian 近奇异则相应 SE 标 NaN。"
        )
        code += [
            "import numpy as np; from scipy.optimize import minimize; from scipy.stats import norm",
            "# Tobit MLE (logσ param): uncensored -> norm.logpdf(z)-log σ; censored -> norm.logcdf(z)",
            "#   left:  z=(censor-Xβ)/σ ;  right: z=(Xβ-censor)/σ ;  maximise over (β, logσ)",
            "# β is the LATENT-index effect; marginal on E[y] = β * Φ((x̄β-c)/σ) [left; right: Φ((c-x̄β)/σ)]",
        ]
    except Exception as err:
        summary.append(f"Tobit 回归跳过：{err}")


# ─────────────────────────────────────────────────────────────────────────────
# (B) truncated_regression — truncated-normal regression by MLE
# ─────────────────────────────────────────────────────────────────────────────
@register("truncated_regression")
def _branch_truncated_regression(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    outcome, predictors, problem = _resolve_xy(ctx, "截断回归")
    if problem is not None:
        summary.append(problem)
        return
    try:
        import numpy as np
        from scipy.optimize import minimize
        from scipy.stats import norm

        y, Xraw, kept = _numeric_matrix(df, outcome, predictors)
        if y is None or Xraw is None or not kept:
            summary.append("截断回归跳过：数据清洗后无有效预测变量（全缺失或常量）。")
            return
        n = len(y)
        if n < 20:
            summary.append(f"截断回归跳过：有效样本太少（n={n}，需 ≥20）。")
            return

        side = str(cfg.get("truncation") or "lower").lower()
        if side not in {"lower", "upper"}:
            side = "lower"
        if cfg.get("trunc_value") is not None:
            a = float(cfg.get("trunc_value"))
        else:
            a = float(y.min()) if side == "lower" else float(y.max())

        # Sanity: under lower truncation at a, all y should be >= a (and <= a for upper).
        # If the data clearly violate it the truncation point is mis-specified; warn but
        # proceed (the LL still well-defined; we honestly disclose).
        if side == "lower":
            viol = int(np.sum(y < a - 1e-9))
        else:
            viol = int(np.sum(y > a + 1e-9))

        X = np.column_stack([np.ones(n), Xraw])
        k = X.shape[1]
        terms = ["const", *kept]

        beta0, sigma0 = _ols_fit(y, X)
        theta0 = np.concatenate([beta0, [np.log(sigma0)]])

        def neg_ll(theta):
            b = theta[:k]
            sig = np.exp(theta[k])
            mu = X @ b
            z = (y - mu) / sig
            dens = norm.logpdf(z) - np.log(sig)
            # truncation correction: divide density by P(in observed region)
            alpha = (a - mu) / sig
            if side == "lower":
                # P(Y > a) = 1 - Φ(α) = Φ(-α) -> use logsf for stability
                logp = norm.logsf(alpha)
            else:
                # upper truncation at a: P(Y < a) = Φ(α)
                logp = norm.logcdf(alpha)
            return -np.sum(dens - logp)

        res = minimize(neg_ll, theta0, method="BFGS",
                       options={"maxiter": 2000, "gtol": 1e-6})
        theta = res.x
        converged = bool(res.success)
        beta = theta[:k]
        sigma = float(np.exp(theta[k]))
        loglik = -float(res.fun)

        se_full = _se_from_hessian(neg_ll, theta)
        se_beta = se_full[:k]
        se_sigma = float(sigma * se_full[k]) if np.isfinite(se_full[k]) else float("nan")

        ols_beta, _ = _ols_fit(y, X)

        import pandas as pd

        rows = []
        for i, t in enumerate(terms):
            b = float(beta[i]); s = float(se_beta[i])
            if np.isfinite(s) and s > 0:
                z = b / s
                p = float(2 * norm.sf(abs(z)))
                lo, hi = b - 1.96 * s, b + 1.96 * s
            else:
                z = p = lo = hi = float("nan")
            rows.append({
                "term": t,
                "trunc_coef": round(b, 6),
                "se": round(s, 6) if np.isfinite(s) else float("nan"),
                "z": round(z, 4) if np.isfinite(z) else float("nan"),
                "p": round(p, 6) if np.isfinite(p) else float("nan"),
                "ci_low": round(lo, 6) if np.isfinite(lo) else float("nan"),
                "ci_high": round(hi, 6) if np.isfinite(hi) else float("nan"),
                "naive_ols_coef": round(float(ols_beta[i]), 6),
            })
        tab = pd.DataFrame(rows)
        tab.to_csv(d / "truncated_coefficients.csv", index=False, encoding="utf-8")
        files.append("truncated_coefficients.csv")

        _coef_plot_ci(
            d, files, terms,
            [float(beta[i]) for i in range(k)],
            [float(se_beta[i]) for i in range(k)],
            "Truncated-regression coefficient (95% CI)",
            f"Truncated regression ({side} at {a:g}) - {outcome}",
        )

        for i, t in enumerate(terms):
            if t == "const":
                continue
            estimates[t] = round(float(beta[i]), 6)
            estimates[f"{t}_se"] = (round(float(se_beta[i]), 6)
                                    if np.isfinite(se_beta[i]) else float("nan"))
            if np.isfinite(se_beta[i]) and se_beta[i] > 0:
                estimates[f"{t}_ci_low"] = round(float(beta[i] - 1.96 * se_beta[i]), 6)
                estimates[f"{t}_ci_high"] = round(float(beta[i] + 1.96 * se_beta[i]), 6)
            else:
                estimates[f"{t}_ci_low"] = float("nan")
                estimates[f"{t}_ci_high"] = float("nan")
            estimates[f"{t}_naive_ols"] = round(float(ols_beta[i]), 6)
        estimates["const"] = round(float(beta[0]), 6)
        estimates["sigma"] = round(sigma, 6)
        estimates["sigma_se"] = round(se_sigma, 6) if np.isfinite(se_sigma) else float("nan")
        estimates["loglik"] = round(loglik, 4)
        estimates["trunc_value"] = round(a, 6)
        estimates["n_obs"] = float(n)
        estimates["converged"] = 1.0 if converged else 0.0

        conv_note = "" if converged else "（⚠ 优化器未完全收敛，结果谨慎使用）"
        viol_note = (
            f" ⚠ 有 {viol} 个观测落在指定{('下' if side=='lower' else '上')}截断点 {a:g} 的"
            "禁区外，截断点可能设定有误。" if viol else ""
        )
        summary.append(
            f"{entry.method} 完成{conv_note}：n={n}，"
            f"{('下' if side=='lower' else '上')}截断于 {a:g}，{len(kept)} 个预测变量。"
            f"σ̂={sigma:.4f}，对数似然={loglik:.2f}。系数（β + SE + 95%CI + z/p，"
            f"并列 naive OLS 对照）见 truncated_coefficients.csv。{viol_note}"
            "\n⚠ 披露：(1) **截断 ≠ 删失**——截断样本里超出门限的观测**根本不存在**（Tobit 的删失观测"
            "则在样本内、只是值被压在门限）；本方法按观测落在区域内的条件密度做 MLE。"
            "(2) 在截断样本上直接做 OLS 系数会**有偏/衰减**（CSV 的 naive_ols_coef 列可对照）。"
            "(3) 估计依赖**正确的截断点与误差正态**假定；门限设错或非正态会使估计不一致。"
            "(4) SE 来自数值 Hessian（观测信息阵）的逆；近奇异时相应 SE 标 NaN。"
        )
        code += [
            "import numpy as np; from scipy.optimize import minimize; from scipy.stats import norm",
            "# Truncated-normal MLE (lower trunc at a): ",
            "#   LL = Σ[ norm.logpdf((y-Xβ)/σ)-log σ  −  log P(in region) ]",
            "#   lower: log P = norm.logsf((a-Xβ)/σ);  upper: log P = norm.logcdf((a-Xβ)/σ)",
        ]
    except Exception as err:
        summary.append(f"截断回归跳过：{err}")


# ─────────────────────────────────────────────────────────────────────────────
# (C) heckman_selection — Heckman (1979) two-step sample-selection correction
# ─────────────────────────────────────────────────────────────────────────────
@register("heckman_selection")
def _branch_heckman_selection(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    try:
        import numpy as np
        import pandas as pd
        import statsmodels.api as sm
        from scipy.stats import norm

        # Work on a 0-based positional index so the step-2 frame (which mixes a
        # positionally-assigned IMR with label-reindexed outcome/predictors) stays
        # aligned even if a caller passes a non-RangeIndex / pre-filtered frame.
        df = df.reset_index(drop=True)
        fpcols = fp.columns
        _exc = {fp.unit_col, fp.time_col}

        # outcome = first continuous (not unit/time); predictors = remaining numeric.
        outcome = cfg.get("outcome")
        if not (outcome and outcome in df.columns):
            outcome = next(
                (c.name for c in fpcols
                 if c.kind == "continuous" and c.name not in _exc), None)
        if outcome is None:
            summary.append("Heckman 选择模型跳过：需要 1 个连续结果变量（outcome）。")
            return

        # selection indicator: config, else first binary col, else derive from the
        # outcome's missingness (selected = outcome observed).
        sel_col = cfg.get("selection")
        derived_sel = False
        if sel_col and sel_col in df.columns:
            sel_raw = pd.to_numeric(df[sel_col], errors="coerce")
            selected = (sel_raw > 0.5).to_numpy()
        else:
            bincol = next(
                (c.name for c in fpcols
                 if c.kind == "binary" and c.name not in _exc and c.name != outcome),
                None,
            )
            if bincol is not None:
                sel_col = bincol
                sel_raw = pd.to_numeric(df[bincol], errors="coerce")
                selected = (sel_raw > 0.5).to_numpy()
            else:
                # derive from outcome missingness
                sel_col = f"({outcome} observed)"
                selected = pd.to_numeric(df[outcome], errors="coerce").notna().to_numpy()
                derived_sel = True

        # predictors (outcome-equation regressors)
        cfg_pred = cfg.get("predictors")
        if cfg_pred and isinstance(cfg_pred, (list, tuple)):
            predictors = [c for c in cfg_pred
                          if c in df.columns and c != outcome and c != sel_col]
        else:
            predictors = [
                c.name for c in fpcols
                if c.kind in {"continuous", "count", "binary"}
                and c.name not in _exc and c.name != outcome and c.name != sel_col
            ]
        predictors = predictors[:20]
        if not predictors:
            summary.append("Heckman 选择模型跳过：需要 ≥1 个预测变量。")
            return

        # exclusion restriction: vars entering ONLY the selection equation
        excl = cfg.get("exclusion") or []
        if not isinstance(excl, (list, tuple)):
            excl = []
        exclusion = [c for c in excl
                     if c in df.columns and c not in predictors
                     and c != outcome and c != sel_col]
        sel_vars = predictors + exclusion  # selection-equation regressors

        # Build numeric frames. Selection equation uses ALL rows where selection &
        # selection-regressors are observed. Outcome equation uses SELECTED rows where
        # outcome & predictors are observed.
        sel_df = df[sel_vars].apply(lambda s: pd.to_numeric(s, errors="coerce"))
        base = pd.DataFrame({"_sel": selected.astype(float)})
        base = pd.concat([base.reset_index(drop=True), sel_df.reset_index(drop=True)], axis=1)
        base = base.dropna(subset=["_sel", *sel_vars])
        if base.empty or base["_sel"].nunique() < 2:
            summary.append(
                "Heckman 选择模型跳过：选择指示变量无变异（全选中或全未选中），无法识别概率模型。"
            )
            return

        n_total = int(len(base))
        n_selected = int(base["_sel"].sum())
        if n_selected < len(predictors) + 3 or (n_total - n_selected) < 3:
            summary.append(
                f"Heckman 选择模型跳过：选中样本太少（{n_selected}）或未选中样本太少"
                f"（{n_total - n_selected}），无法稳定估计两步法。"
            )
            return

        # ── Step 1: probit of selection on [predictors + exclusion] over ALL rows ──
        Z = sm.add_constant(base[sel_vars].to_numpy(float), has_constant="add")
        try:
            probit = sm.Probit(base["_sel"].to_numpy(float), Z).fit(disp=0, maxiter=200)
        except Exception as perr:
            summary.append(f"Heckman 选择模型跳过：probit 选择方程未收敛（{perr}）。")
            return
        zb = Z @ probit.params  # linear index zγ̂ for all rows
        # inverse Mills ratio λ = φ(zγ)/Φ(zγ); guard tiny Φ
        Phi = norm.cdf(zb)
        phi = norm.pdf(zb)
        Phi_safe = np.clip(Phi, 1e-12, 1.0)
        imr = phi / Phi_safe

        # ── Step 2: OLS of outcome on [predictors + λ] over SELECTED rows ──
        sel_idx = base["_sel"].to_numpy() > 0.5
        base_pos = base.index  # positions into the ORIGINAL df
        y_full = pd.to_numeric(df[outcome], errors="coerce")
        out_block = pd.DataFrame({
            "_y": y_full.reindex(base_pos).to_numpy(float),
        })
        for p in predictors:
            out_block[p] = pd.to_numeric(df[p], errors="coerce").reindex(base_pos).to_numpy(float)
        out_block["_imr"] = imr
        out_block["_zb"] = np.asarray(zb, float)
        out_block["_selflag"] = sel_idx
        sel_block = out_block[out_block["_selflag"]].drop(columns="_selflag")
        sel_block = sel_block.dropna(subset=["_y", *predictors])
        if len(sel_block) < len(predictors) + 3:
            summary.append(
                "Heckman 选择模型跳过：选中样本在结果/预测变量上缺失过多，第二步无法估计。"
            )
            return

        Xo = sm.add_constant(sel_block[[*predictors, "_imr"]].to_numpy(float),
                             has_constant="add")
        ols2 = sm.OLS(sel_block["_y"].to_numpy(float), Xo).fit()
        out_terms = ["const", *predictors, "lambda_IMR"]
        params2 = np.asarray(ols2.params, float)
        se2 = np.asarray(ols2.bse, float)
        p2 = np.asarray(ols2.pvalues, float)
        ci2 = np.asarray(ols2.conf_int(), float)  # rows = terms

        # λ coefficient = ρ·σ_ε ; recover ρ̂ and σ̂_ε (Heckman 1979).
        # σ_ε^2 ≈ MSE_OLS + (β_λ^2 / n) Σ δ_i, where δ_i = λ_i(λ_i + zγ_i) for selected.
        lam_coef = float(params2[-1])
        lam_se = float(se2[-1])
        lam_p = float(p2[-1])
        resid2 = sel_block["_y"].to_numpy(float) - Xo @ params2
        mse = float(resid2 @ resid2) / max(1, len(resid2))
        # δ_i = λ_i(λ_i + zγ_i), averaged over EXACTLY the rows that entered step-2
        # OLS (post-dropna sel_block), so σ_ε² uses the same sample as MSE/β_λ.
        imr_sel = sel_block["_imr"].to_numpy(float)
        zb_sel = sel_block["_zb"].to_numpy(float)
        delta = imr_sel * (imr_sel + zb_sel)
        sigma2_eps = mse + (lam_coef ** 2) * float(np.mean(delta))
        sigma_eps = float(np.sqrt(sigma2_eps)) if sigma2_eps > 0 else float("nan")
        rho = float(lam_coef / sigma_eps) if (sigma_eps and np.isfinite(sigma_eps)
                                              and sigma_eps > 0) else float("nan")
        if np.isfinite(rho):
            rho = float(np.clip(rho, -0.999, 0.999))

        rows = []
        for i, t in enumerate(out_terms):
            rows.append({
                "term": t,
                "coef": round(float(params2[i]), 6),
                "se": round(float(se2[i]), 6),
                "p": round(float(p2[i]), 6),
                "ci_low": round(float(ci2[i, 0]), 6),
                "ci_high": round(float(ci2[i, 1]), 6),
            })
        tab = pd.DataFrame(rows)
        tab.to_csv(d / "heckman_coefficients.csv", index=False, encoding="utf-8")
        files.append("heckman_coefficients.csv")

        # naive OLS (selected rows, NO λ) for the bias contrast
        Xn = sm.add_constant(sel_block[predictors].to_numpy(float), has_constant="add")
        naive = sm.OLS(sel_block["_y"].to_numpy(float), Xn).fit()
        naive_terms = ["const", *predictors]
        naive_map = {t: float(np.asarray(naive.params)[i]) for i, t in enumerate(naive_terms)}

        _coef_plot_ci(
            d, files, out_terms,
            [float(params2[i]) for i in range(len(out_terms))],
            [float(se2[i]) for i in range(len(out_terms))],
            "Heckman outcome-eq coefficient (95% CI)",
            f"Heckman two-step - {outcome}",
        )

        # estimates: headline = outcome-eq slopes first, then selection diagnostics
        for i, t in enumerate(out_terms):
            if t == "const":
                continue
            estimates[t] = round(float(params2[i]), 6)
            estimates[f"{t}_se"] = round(float(se2[i]), 6)
            estimates[f"{t}_ci_low"] = round(float(ci2[i, 0]), 6)
            estimates[f"{t}_ci_high"] = round(float(ci2[i, 1]), 6)
        estimates["const"] = round(float(params2[0]), 6)
        estimates["lambda_coef"] = round(lam_coef, 6)
        estimates["lambda_se"] = round(lam_se, 6)
        estimates["lambda_p"] = round(lam_p, 6)
        estimates["rho"] = round(rho, 6) if np.isfinite(rho) else float("nan")
        estimates["sigma_eps"] = round(sigma_eps, 6) if np.isfinite(sigma_eps) else float("nan")
        estimates["n_selected"] = float(n_selected)
        estimates["n_total"] = float(n_total)
        estimates["selection_bias_p"] = round(lam_p, 6)
        for p in predictors:
            estimates[f"{p}_naive_ols"] = round(naive_map.get(p, float("nan")), 6)

        bias_sig = lam_p < 0.05
        sel_note = (
            f"选择指示由结果变量缺失推断（{sel_col}）" if derived_sel
            else f"选择指示列 = {sel_col}"
        )
        excl_note = (
            f"排除性约束变量（仅入选择方程）：{exclusion}" if exclusion
            else "⚠ 未提供排除性约束（仅入选择方程的工具）——识别仅靠 probit 非线性，较弱"
        )
        summary.append(
            f"{entry.method} 完成：{sel_note}；总样本 {n_total}、选中 {n_selected}。"
            f"{excl_note}。第二步（结果方程）系数 + 逆米尔斯比 λ 见 heckman_coefficients.csv。"
            f"λ(IMR) 系数={lam_coef:.4f}（SE={lam_se:.4f}，p={lam_p:.3g}）"
            f"——这是**选择偏差检验**（H0: ρ=0，无选择偏差）："
            f"{'拒绝 H0，存在显著选择偏差，需要校正' if bias_sig else '不拒绝 H0，未见显著选择偏差'}。"
            f"ρ̂={rho:.3f}，σ̂_ε={sigma_eps:.4f}。"
            "\n⚠ 披露：(1) 识别**主要依赖排除性约束**（一个进入选择方程但不进入结果方程的工具）；"
            "无此工具时仅靠 probit 的非线性识别，**很弱**、易共线。"
            "(2) 两步法的标准误是**未校正**的 naive OLS SE（第二步把 λ 当作已知回归元，"
            "忽略了它是估计量带来的额外方差与异方差）——精确推断请用 bootstrap 或 Heckman MLE，"
            "本实现的 SE 仅供参考。"
            "(3) ρ̂、σ̂_ε 由 λ 系数与残差方差按 Heckman(1979) 公式近似反解（σ_ε² ≈ MSE + β_λ²·mean(δ)）。"
            "(4) 选中样本上的 naive OLS（CSV 对照/estimates 的 *_naive_ols）在 ρ≠0 时有偏。"
        )
        code += [
            "import statsmodels.api as sm; from scipy.stats import norm  # Heckman (1979) two-step",
            "# Step 1: probit(selection ~ predictors+exclusion) over ALL rows; "
            "imr = norm.pdf(zγ)/norm.cdf(zγ)",
            "# Step 2: OLS(outcome ~ predictors + imr) over SELECTED rows; "
            "coef on imr = ρ·σ_ε (t-test = selection-bias test)",
        ]
    except Exception as err:
        summary.append(f"Heckman 选择模型跳过：{err}")
