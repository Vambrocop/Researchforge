"""Branch handler for BAYESIAN MODEL AVERAGING (BMA) over a linear-regression model space.

`bayesian_model_averaging` — instead of SELECTING one regression model, average over the
whole space of predictor subsets, weighted by each model's posterior probability. The
honest answer to variable-selection uncertainty: "which predictors matter, accounting for
the fact that we don't know the right model?"

Methodology (Raftery 1995; Hoeting et al. 1999 — the standard BIC approximation):
  * model space = all 2^p subsets of the p predictors (each model = an intercept + a subset),
    enumerated for p ≤ 12 (predictors truncated + disclosed beyond that);
  * each model M_k is fit by OLS; its evidence is approximated from the BIC,
    BIC_k = n·ln(RSS_k/n) + k·ln(n), so the marginal likelihood ≈ exp(-½·BIC_k)
    (a unit-information-prior approximation);
  * posterior model weight (equal model prior): w_k ∝ exp(-½·(BIC_k - BIC_min)), normalized;
  * Posterior Inclusion Probability of predictor j: PIP_j = Σ_{k: j∈M_k} w_k — near 1 means
    strong evidence the variable belongs; near 0 means it does not;
  * model-averaged coefficient β̄_j = Σ_k w_k·β_{j,k} (0 when j∉M_k), with the BMA posterior
    SD from the law of total variance: Var(β_j) = Σ_k w_k·(β_{j,k}² + se_{j,k}²) − β̄_j²
    (within-model variance + between-model variance).

Continuous outcome only (OLS-based). Deterministic; no new deps (numpy). This is an
INFERENCE-bearing method — cold-reviewed. Auto-registered by branches/__init__.py.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register

_MAX_PREDICTORS = 12  # 2^12 = 4096 models enumerated; beyond this, truncate + disclose


def _fit_ols(Xs, y, np):
    """OLS on design matrix Xs (with intercept col): returns (beta, rss, se_vector)."""
    n, k = Xs.shape
    beta, *_ = np.linalg.lstsq(Xs, y, rcond=None)
    resid = y - Xs @ beta
    rss = float(resid @ resid)
    dof = n - k
    if dof > 0 and rss > 0:
        sigma2 = rss / dof
        try:
            xtx_inv = np.linalg.inv(Xs.T @ Xs)
            se = np.sqrt(np.maximum(sigma2 * np.diag(xtx_inv), 0.0))
        except np.linalg.LinAlgError:
            se = np.full(k, np.nan)
    else:
        se = np.full(k, np.nan)
    return beta, rss, se


def _bma_linear(y, X, np):
    """BMA over all 2^p predictor subsets. Returns a dict of arrays + the model list."""
    from itertools import combinations

    n, p = X.shape
    tss = float(((y - y.mean()) ** 2).sum())
    models = []  # (subset_tuple, k_params, rss, bic, beta, se)
    for r in range(0, p + 1):
        for subset in combinations(range(p), r):
            cols = list(subset)
            Xs = (np.column_stack([np.ones(n)] + [X[:, c] for c in cols])
                  if cols else np.ones((n, 1)))
            beta, rss, se = _fit_ols(Xs, y, np)
            kparams = Xs.shape[1]
            # clamp rss so a (near-)perfect fit yields a very low BIC (very high weight)
            # rather than -inf, which would poison the exp-normalized weights with NaN.
            bic = n * np.log(max(rss, 1e-300) / n) + kparams * np.log(n)
            models.append((subset, kparams, rss, bic, beta, se))

    bics = np.array([m[3] for m in models], dtype=float)
    bmin = float(np.nanmin(bics))
    w = np.exp(-0.5 * (bics - bmin))
    w = w / w.sum()

    pip = np.zeros(p)
    ma_coef = np.zeros(p)
    ma_second = np.zeros(p)  # Σ w_k (β² + se²) over models containing j
    for (subset, _k, _rss, _bic, beta, se), wk in zip(models, w):
        for local_idx, j in enumerate(subset):
            coef_j = float(beta[1 + local_idx])
            se_j = float(se[1 + local_idx]) if np.isfinite(se[1 + local_idx]) else 0.0
            pip[j] += wk
            ma_coef[j] += wk * coef_j
            ma_second[j] += wk * (coef_j ** 2 + se_j ** 2)
    ma_sd = np.sqrt(np.maximum(ma_second - ma_coef ** 2, 0.0))
    return {
        "pip": pip, "ma_coef": ma_coef, "ma_sd": ma_sd,
        "models": models, "weights": w, "tss": tss,
    }


@register("bayesian_model_averaging")
def _branch_bayesian_model_averaging(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    import numpy as np
    import pandas as pd

    from researchforge.executor.run import resolve_outcome, resolve_predictors

    cont = [c.name for c in fp.columns if c.kind == "continuous"
            and c.name not in {fp.unit_col, fp.time_col}]
    if not cont:
        summary.append(
            "贝叶斯模型平均跳过：需要一个连续结果变量（当前无连续列）——本法基于线性回归模型空间，"
            "config outcome 指定连续结果。"
        )
        return
    outcome = resolve_outcome(fp, cfg, cont)
    # resolve WITHOUT the cap so we can SEE how many were eligible and disclose the
    # truncation honestly — resolve_predictors caps silently, which would hide it.
    eligible = resolve_predictors(
        fp, cfg, outcome, kinds=("continuous", "count", "binary"), cap=10**6, df=df
    )
    if len(eligible) < 2:
        summary.append(
            f"贝叶斯模型平均跳过：需要 ≥2 个预测变量（现有 {len(eligible)}）——BMA 对单变量无意义。"
        )
        return

    over_cap = len(eligible) > _MAX_PREDICTORS      # too many candidates for 2^p enumeration
    predictors = eligible[:_MAX_PREDICTORS]
    shrunk_for_n = False

    try:
        cols = [outcome, *predictors]
        data = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
        n = int(len(data))
        # OLS in the FULL model needs n > p+1; keep a couple of residual dof.
        if n < len(predictors) + 3:
            keep = max(2, n - 3)
            if keep < len(predictors):
                predictors = predictors[:keep]
                shrunk_for_n = True
                data = df[[outcome, *predictors]].apply(pd.to_numeric, errors="coerce").dropna()
                n = int(len(data))
        if len(predictors) < 2 or n < 15:
            summary.append(
                f"贝叶斯模型平均跳过：有效样本/预测变量太少（n={n}, p={len(predictors)}）。"
            )
            return

        y = data[outcome].to_numpy(dtype=float)
        X = data[predictors].to_numpy(dtype=float)
        res = _bma_linear(y, X, np)
        pip, ma_coef, ma_sd = res["pip"], res["ma_coef"], res["ma_sd"]
        models, w, tss = res["models"], res["weights"], res["tss"]

        # --- PIP + model-averaged coefficient table --------------------------
        order = np.argsort(pip)[::-1]
        pip_df = pd.DataFrame({
            "predictor": [predictors[i] for i in order],
            "pip": np.round(pip[order], 6),
            "ma_coef": np.round(ma_coef[order], 6),
            "ma_sd": np.round(ma_sd[order], 6),
            "sign": ["+" if ma_coef[i] > 0 else "-" if ma_coef[i] < 0 else "0" for i in order],
        })
        pip_df.to_csv(d / "bma_pip.csv", index=False, encoding="utf-8")
        files.append("bma_pip.csv")

        # --- top models by posterior weight ----------------------------------
        top_idx = np.argsort(w)[::-1][:10]
        rows = []
        for rank, k in enumerate(top_idx, start=1):
            subset, kparams, rss, bic, beta, se = models[k]
            incl = [predictors[j] for j in subset] or ["(intercept only)"]
            rows.append({
                "rank": rank,
                "predictors": " + ".join(incl),
                "r_squared": round(1.0 - rss / tss, 4) if tss > 0 else float("nan"),
                "bic": round(float(bic), 3),
                "posterior_weight": round(float(w[k]), 6),
            })
        top_df = pd.DataFrame(rows)
        top_df.to_csv(d / "bma_top_models.csv", index=False, encoding="utf-8")
        files.append("bma_top_models.csv")

        # --- PNG: PIP bar ----------------------------------------------------
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            k = len(predictors)
            oo = np.argsort(pip)
            fig, ax = plt.subplots(figsize=(6.5, 0.4 * k + 1.0))
            colors = ["#4C72B0" if pip[i] >= 0.5 else "#B0B0B0" for i in oo]
            ax.barh(range(k), pip[oo], color=colors)
            ax.set_yticks(range(k))
            ax.set_yticklabels([predictors[i] for i in oo], fontsize=8)
            ax.set_xlim(0, 1)
            ax.axvline(0.5, color="0.5", linestyle="--", linewidth=1)
            ax.set_xlabel("posterior inclusion probability (PIP)")
            ax.set_title(f"BMA — variable importance for {outcome}", fontsize=10)
            fig.tight_layout()
            fig.savefig(d / "bma_pip.png", dpi=150)
            plt.close(fig)
            files.append("bma_pip.png")
        except Exception:
            pass

        n_models = len(models)
        top_w = float(w[np.argmax(w)])
        estimates["n_predictors"] = float(len(predictors))
        estimates["n_models"] = float(n_models)
        estimates["n"] = float(n)
        estimates["top_model_weight"] = round(top_w, 6)
        estimates["max_pip"] = round(float(pip.max()), 6)
        # per-predictor PIP into estimates (named)
        for j, name in enumerate(predictors):
            estimates[f"pip_{name}"] = round(float(pip[j]), 6)

        weak = [predictors[i] for i in order if pip[i] < 0.5]
        strong_txt = "、".join(f"{predictors[i]}(PIP={pip[i]:.2f})" for i in order if pip[i] >= 0.5) or "（无变量 PIP≥0.5）"
        # disclose truncation honestly, distinguishing the two causes (>12 enumeration cap vs
        # small-n dof shrink); dropped candidates are named so nothing is silently absent.
        dropped = [c for c in eligible if c not in predictors]
        trunc_parts = []
        if over_cap:
            trunc_parts.append(f"候选预测变量 {len(eligible)} 个超枚举上限、取前 {_MAX_PREDICTORS} 个")
        if shrunk_for_n:
            trunc_parts.append(f"样本量偏小、进一步缩至前 {len(predictors)} 个(保 OLS 自由度)")
        trunc_txt = (
            f"⚠ {'；'.join(trunc_parts)}（未纳入：{', '.join(dropped[:8])}{'…' if len(dropped) > 8 else ''}；"
            f"config predictors 可控）；" if trunc_parts else ""
        )
        summary.append(
            f"{entry.method} 完成（线性回归模型空间 BMA，BIC 近似，结果={outcome}，"
            f"{len(predictors)} 个预测变量 → {n_models} 个模型，n={n}）：{trunc_txt}"
            f"后验纳入概率 PIP 高(≥0.5)的变量：{strong_txt}"
            f"（{len(weak)} 个 PIP<0.5，证据弱）；最优模型后验权重={top_w:.2f}"
            f"（详见 bma_pip.csv 的模型平均系数±SD、bma_top_models.csv 的 top 模型）。"
            f"⚠ BMA 处理的是**变量选择的不确定性**（不预先选定一个模型，而按证据权重平均）——"
            f"PIP 是「该变量属于真模型」的后验概率、**非因果**；模型平均系数是**无条件**均值"
            f"（被 1−PIP 向 0 收缩；条件于纳入的系数≈ma_coef/PIP）；用 BIC 近似边际似然（隐含单位信息先验、"
            f"等模型先验），小样本下为近似；结果依赖所考虑的预测变量集与模型先验。"
            f"可用 config outcome/predictors 覆盖。"
        )
        code += [
            "import numpy as np; from itertools import combinations",
            f"y = df['{outcome}'].to_numpy(float); X = df[{predictors!r}].to_numpy(float)",
            "# 枚举 2^p 子集,每模型 OLS→BIC→权重 w∝exp(-0.5*BIC); PIP_j = Σ_{{j∈M}} w",
            "# BIC_k = n*log(RSS_k/n) + k*log(n); ma_coef_j = Σ w_k β_{{j,k}}",
        ]
    except Exception as err:
        summary.append(f"贝叶斯模型平均失败：{type(err).__name__}: {err}")
