"""Causal family branch handler: aipw (augmented inverse-propensity weighting).

Doubly-robust ATE estimator. Auto-registered via the causal package
(branches/__init__.py pkgutil.walk_packages). Drop-in sibling of ipw.py.

Estimator (doubly robust / AIPW):
    ê(X)   = P(T=1|X)      via sklearn LogisticRegression on standardized covariates
    μ0(X)  = E[Y|T=0,X]    via sklearn regression fit on the control arm only
    μ1(X)  = E[Y|T=1,X]    via sklearn regression fit on the treated arm only
    score_i = (μ1_i − μ0_i)
              + T_i·(Y_i − μ1_i)/ê_i
              − (1−T_i)·(Y_i − μ0_i)/(1−ê_i)
    ATE     = mean_i(score_i)
    SE      = sd(score_i) / sqrt(n)              (influence-function / plug-in)
    95% CI  = ATE ± 1.96·SE

Doubly robust: consistent if EITHER the propensity model OR the outcome models are
correctly specified. Propensities are trimmed to [0.01, 0.99] and the trimmed
fraction is disclosed (positivity). No cross-fitting (disclosed); the SE is the
plug-in influence-function variance.
"""
from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.profiler.semantics import (
    survival_event_column,
    treatment_never_named,
)
from researchforge.executor.run import resolve_outcome, resolve_treatment


@register("aipw")
def _branch_aipw(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import importlib.util

    if importlib.util.find_spec("sklearn") is None:
        summary.append("AIPW 跳过：需要 scikit-learn（未检测到）。安装：pip install scikit-learn。")
        return

    import numpy as np
    import pandas as pd

    _excl = {fp.unit_col, fp.time_col}
    bins = [c.name for c in fp.columns if c.kind == "binary" and c.name not in _excl]
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]

    # --- roles: treatment (binary), outcome (continuous), covariates (numeric) ---
    # H4d: fp.binary_columns is EVERY binary column, so [0] meant "first
    # binary in file order" — on [.., event, treatment, ..] that picked the
    # survival EVENT as the intervention. resolve_treatment applies the name
    # signal that roles.py already computed (and records the binding).
    treatment = resolve_treatment(fp, cfg, fp.binary_columns or bins, df=df)
    # H4c: bind the DETECTED outcome among the non-treatment continuous columns
    # (config > high-confidence outcome name > first non-treatment-named) instead of
    # plain column order — and record it, so the report can name what was modeled.
    outcome = cfg.get("outcome") if cfg.get("outcome") in df.columns else (
        resolve_outcome(fp, cfg, [c for c in cont if c != treatment])
        if [c for c in cont if c != treatment] else None)
    if cfg.get("covariates"):
        covs = [c for c in cfg["covariates"] if c in df.columns and c not in {outcome, treatment}]
    else:
        # 处理后协变量: in a survival-shaped frame the event indicator is a descendant of
        # the duration (event = 1{T<=C}) and hence of the treatment, so adjusting for it is
        # post-treatment adjustment. Measured 33-73% attenuation of the ATT, and an outright
        # failure once the event rate saturates. AUTO set only — an explicit config
        # covariates list stays the user's call — and disclosed in the summary.
        _post = survival_event_column(fp)
        # A control/negated flag is a deterministic function of the treatment (control_arm
        # = 1 - treated), so it is never a confounder — and putting it in the propensity
        # model separates it perfectly, which surfaced as a naked "Singular matrix".
        _mirror = {c.name for c in fp.columns
                   if c.name != treatment and treatment_never_named(c.name)}
        covs = [c.name for c in fp.columns if c.kind in {"continuous", "binary", "count"}
                and c.name not in (_excl | {outcome, treatment} | _mirror
                                   | ({_post} if _post else set()))]
        if _mirror:
            summary.append(
                f"⚠ 已把 {'、'.join(sorted(_mirror))} 排除在自动协变量之外："
                "它标记的是未处理/对照，是处理变量的确定性函数（而非混杂因子），"
                "放进倾向模型会造成完全分离（此前表现为一句裸的 Singular matrix）。"
                "若确需纳入，用 config covariates 显式指定。"
            )
        # Cold review A MUST-FIX 5: the AUTO covariate set is "every numeric column that is
        # not the outcome/treatment" — it cannot tell a confounder from a MEDIATOR, and a
        # mediator is post-treatment adjustment. Measured on a randomised trial whose true
        # effect is -8, with a post-treatment mediator `adherence` carrying the full effect:
        #     IPW -3.207   PSM +1.197 (SIGN FLIPPED)   AIPW -1.358, 95% CI [-2.04, -0.68]
        # i.e. the AIPW interval does not cover the truth at all. Auto-DROPPING mediators is
        # not the fix (there is no name test that separates a mediator from a real
        # confounder, and dropping a confounder is the opposite bias) — but the set was never
        # even shown. Name it, and say what would make it wrong.
        if covs:
            summary.append(
                f"⚠ 自动选取的协变量：{'、'.join(covs)}。它们是按「非结果、非处理的数值列」"
                "挑的，**引擎无法分辨混杂因子与中介变量**。若其中任何一列是在处理之后"
                "测得的（依从性、实际剂量、随访时长、中间结局），那是处理后调整，"
                "会把效应吸走甚至翻号（实测中介占比 100% 时 PSM 由 −8 变成 +1.197、"
                "AIPW 的 95% CI 完全不覆盖真值）。请核对这份名单，必要时"
                '用 config={"covariates":[..]} 指定基线前（处理前）变量。'
            )
        if _post:
            summary.append(
                f"⚠ 已把 '{_post}' 排除在自动协变量之外：它是生存数据的事件指示列"
                "（event = 1{时长 ≤ 删失时间}），是时长、进而是处理变量的后代，"
                "放进倾向得分属于处理后调整（实测会把 ATT 衰减 33-73%，"
                "事件率饱和时甚至直接失败）。若确需纳入，用 config covariates 显式指定。"
            )

    # 常数协变量：zero variance cannot inform a propensity model and makes the design
    # matrix singular (surfaced by a saturated event rate, where the indicator collapses to
    # one value and the profiler types it `count`, so the survival-event exclusion above
    # cannot see it). Drop with disclosure instead of failing on "Singular matrix".
    _const = [c for c in covs if df[c].dropna().nunique() < 2]
    if _const:
        covs = [c for c in covs if c not in set(_const)]
        summary.append(
            f"⚠ 已剔除常数协变量 {_const}：取值唯一，无法为倾向得分提供信息，"
            "保留会让设计矩阵奇异（报 Singular matrix）。"
        )
    if treatment is None or outcome is None or not covs:
        summary.append('AIPW 跳过：需要 二值处理 + 连续结果 + ≥1 协变量。'
                       'config={"treatment":..,"outcome":..,"covariates":[..]}。')
        return

    sub = df[[outcome, treatment, *covs]].dropna().copy()
    if len(sub) < 30:
        summary.append(f"AIPW 跳过：有效样本太少（n={len(sub)} < 30）。")
        return

    # --- coerce treatment to 0/1; disclose which level is "treated" ---
    tvals = set(pd.unique(sub[treatment].dropna()))
    treated_level = None
    if not (tvals <= {0, 1}):
        if len(tvals) == 2:
            treated_level = sorted(tvals)[1]  # higher value = treated; disclose so ATE sign is clear
            sub[treatment] = (sub[treatment] == treated_level).astype(int)
        else:
            summary.append("AIPW 跳过：处理变量必须是二值（0/1 或恰两类）。")
            return
    sub[treatment] = sub[treatment].astype(int)

    n_t, n_c = int((sub[treatment] == 1).sum()), int((sub[treatment] == 0).sum())
    if n_t < 10 or n_c < 10:
        summary.append(f"AIPW 跳过：处理组 {n_t}、对照组 {n_c}，每臂需 ≥10。")
        return

    # constant covariates carry no information and break standardization.
    nonconst = [c for c in covs if float(sub[c].std(ddof=0)) > 0]
    if not nonconst:
        summary.append("AIPW 跳过：协变量全为常量，无法建模。")
        return
    covs = nonconst

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        # Outcome learner: GradientBoostingRegressor when each arm is large enough for a
        # tree ensemble, else linear regression (disclose which). Boosting needs enough
        # rows per arm to be sensible; below that, a linear outcome model is more stable.
        use_boost = min(n_t, n_c) >= 50
        if use_boost:
            from sklearn.ensemble import GradientBoostingRegressor

            def _make_outcome():
                return GradientBoostingRegressor(random_state=42)
            outcome_model_name = "GradientBoostingRegressor"
        else:
            from sklearn.linear_model import LinearRegression

            def _make_outcome():
                return LinearRegression()
            outcome_model_name = "LinearRegression"

        Xraw = sub[covs].to_numpy(dtype=float)
        t = sub[treatment].to_numpy(dtype=int)
        y = sub[outcome].to_numpy(dtype=float)
        n = len(sub)

        # --- propensity ê(X) = P(T=1|X) via standardized logistic regression ---
        scaler = StandardScaler()
        Xs = scaler.fit_transform(Xraw)
        ps_model = LogisticRegression(max_iter=1000)
        ps_model.fit(Xs, t)
        e = ps_model.predict_proba(Xs)[:, 1]

        # positivity: trim extreme propensities, disclose the trimmed fraction.
        trim_lo, trim_hi = 0.01, 0.99
        trimmed_frac = float(((e < trim_lo) | (e > trim_hi)).mean())
        e = np.clip(e, trim_lo, trim_hi)

        # --- outcome models μ0(X), μ1(X) fit on each arm, predicted for everyone ---
        # Fit μ1 on treated rows, μ0 on control rows (the standard T-learner), then
        # predict the counterfactual for ALL units (raw covariates — tree/linear learners
        # do not need scaling; scaling only matters for the logistic propensity model).
        m1 = _make_outcome()
        m1.fit(Xraw[t == 1], y[t == 1])
        m0 = _make_outcome()
        m0.fit(Xraw[t == 0], y[t == 0])
        mu1 = m1.predict(Xraw)
        mu0 = m0.predict(Xraw)

        # --- AIPW / doubly-robust score and ATE ---
        tf = t.astype(float)
        score = (mu1 - mu0) + tf * (y - mu1) / e - (1.0 - tf) * (y - mu0) / (1.0 - e)
        ate = float(np.mean(score))
        se = float(np.std(score, ddof=1) / np.sqrt(n))  # influence-function plug-in SE
        ci_lo, ci_hi = ate - 1.96 * se, ate + 1.96 * se
        # two-sided z p-value
        from math import erfc, sqrt as _sqrt

        z = ate / se if se > 0 else float("nan")
        pval = float(erfc(abs(z) / _sqrt(2.0))) if z == z else float("nan")

        # --- naive difference in means (for contrast) ---
        naive = float(np.mean(y[t == 1]) - np.mean(y[t == 0]))

        extreme = float((((e <= trim_lo) | (e >= trim_hi)).mean()))

        estimates.update({
            "ate": round(ate, 6), "se": round(se, 6),
            "pvalue": round(pval, 6) if pval == pval else float("nan"),
            "ci_low": round(ci_lo, 6), "ci_high": round(ci_hi, 6),
            "naive_diff": round(naive, 6),
            "trimmed_frac": round(trimmed_frac, 6),
            "extreme_ps_frac": round(extreme, 6),
            "n": float(n), "n_treated": float(n_t), "n_control": float(n_c),
        })

        # --- propensity overlap plot (positivity diagnostic) ---
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(e[t == 1], bins=20, alpha=0.5, label="treated", color="#C44E52")
            ax.hist(e[t == 0], bins=20, alpha=0.5, label="control", color="#4C72B0")
            ax.set_xlabel("estimated propensity e(X)")
            ax.set_ylabel("count")
            ax.set_title("Propensity overlap (AIPW positivity)")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "aipw_propensity_overlap.png", dpi=150)
            plt.close(fig)
            files.append("aipw_propensity_overlap.png")
        except Exception:
            pass

        # --- per-unit AIPW score CSV (for inspection / influence diagnostics) ---
        try:
            out_df = pd.DataFrame({
                treatment: t, outcome: y, "propensity": e,
                "mu0": mu0, "mu1": mu1, "aipw_score": score,
            }, index=sub.index)
            out_df.to_csv(d / "aipw_scores.csv", index=True, encoding="utf-8")
            files.append("aipw_scores.csv")
        except Exception:
            pass

        overlap = "良好" if extreme < 0.05 else f"⚠ 重叠/正性存疑（极端倾向 {extreme:.0%}，已截尾 {trimmed_frac:.0%}）"
        sig = "显著" if (pval == pval and pval < 0.05) else "不显著"
        trt_note = f"（处理组 = {treatment}='{treated_level}'）" if treated_level is not None else ""
        summary.append(
            f"{entry.method} 完成：ATE={ate:.4f}（影响函数 SE={se:.4f}, "
            f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}], p={pval:.3g}，{sig}）{trt_note}；"
            f"对照：朴素均值差={naive:.4f}（不调整混杂）；"
            f"倾向得分截尾至 [0.01,0.99]（截尾 {trimmed_frac:.0%}），重叠 {overlap}；"
            f"结果模型={outcome_model_name}，倾向模型=逻辑回归。"
            " ⚠ AIPW 为**双重稳健**估计 ATE：倾向模型或结果模型**任一**正确即一致——但**不是因果保证**，"
            "因果有效仍需 **可忽略性/无未观测混杂 + 正性/重叠**（倾向不近 0/1，已报截尾占比）；"
            "极端倾向放大方差/偏差；SE 来自影响函数（样本得分标准差/√n），**未做交叉拟合**(cross-fitting)——"
            "若结果模型过拟合，未交叉拟合的 SE 可能偏乐观，重模型建议交叉拟合（如 DML/AIPW-CF）。"
        )
        code += [
            "from sklearn.linear_model import LogisticRegression  # AIPW 双重稳健 ATE",
            "from sklearn.preprocessing import StandardScaler",
            f"# e=P(T=1|X) 逻辑回归; mu0/mu1 由 {outcome_model_name} 分臂拟合",
            "# score = (mu1-mu0) + T*(Y-mu1)/e - (1-T)*(Y-mu0)/(1-e); ATE=mean(score)",
            "# SE = std(score, ddof=1)/sqrt(n)  # 影响函数/插件方差，未交叉拟合",
        ]
    except Exception as err:
        summary.append(f"AIPW 失败：{err}")
