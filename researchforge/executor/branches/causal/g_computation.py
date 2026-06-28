"""Causal family branch handler: g_computation — parametric g-formula / standardization.

Estimates the average treatment effect (ATE) of a binary treatment by the parametric
g-formula: fit an outcome model E[Y | T, W] (with T×W interactions so the effect may
vary with covariates), then predict each unit's outcome under T=1 and under T=0 and
average the difference — standardizing over the observed covariate distribution. Unlike
IPW/AIPW this is purely outcome-model based (a complementary, non-doubly-robust route).
Continuous outcome → OLS (ATE = mean difference); binary outcome → logistic (ATE = risk
difference). Confidence intervals by nonparametric bootstrap. Pure Python (statsmodels).
"""
from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


@register("g_computation")
def _branch_g_computation(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    excl = {fp.unit_col, fp.time_col}

    # treatment: config else first binary column (2-valued).
    def _is_binary(col):
        return pd.to_numeric(df[col], errors="coerce").dropna().nunique() == 2

    treat = cfg.get("treatment")
    if treat not in df.columns:
        treat = next((c.name for c in fp.columns
                      if c.kind == "binary" and c.name not in excl), None)
        if treat is None:
            treat = next((c for c in df.columns if c not in excl and _is_binary(c)), None)
    if treat is None:
        summary.append("g-计算跳过：未找到二值处理变量（treatment，需 0/1 两类）。")
        return

    # outcome: config else first continuous (not the treatment).
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in excl and c.name != treat]
    outcome = cfg.get("outcome") if cfg.get("outcome") in df.columns else None
    if outcome is None:
        outcome = cont[0] if cont else next((c for c in df.columns if c not in excl and c != treat and _is_binary(c)), None)
    if outcome is None or outcome == treat:
        summary.append("g-计算跳过：未找到结果变量（outcome）。")
        return

    # covariates: config else remaining numeric (continuous/count/binary).
    forced = [c for c in (cfg.get("covariates") or []) if c in df.columns and c not in (treat, outcome)]
    if forced:
        covs = forced[:15]
    else:
        covs = [c.name for c in fp.columns
                if c.kind in {"continuous", "count", "binary"}
                and c.name not in {treat, outcome, fp.unit_col, fp.time_col}][:15]
    if not covs:
        summary.append("g-计算跳过：未找到协变量（≥1 个用于标准化/混杂调整）。")
        return

    use = [outcome, treat, *covs]
    sub = df[use].apply(lambda s: pd.to_numeric(s, errors="coerce")).dropna()
    if len(sub) < 20:
        summary.append("g-计算跳过：有效样本不足（去缺失后 < 20 行）。")
        return
    # binarize treatment (higher level = treated=1)
    tv_vals = sorted(sub[treat].unique())
    if len(tv_vals) != 2:
        summary.append("g-计算跳过：处理变量去缺失后不是恰两类。")
        return
    T = (sub[treat] == tv_vals[1]).to_numpy(float)
    n = len(sub)
    n_treat = int(T.sum())
    if n_treat < 5 or (n - n_treat) < 5:
        summary.append(f"g-计算跳过：处理/对照组样本过少（处理 {n_treat} / 对照 {n - n_treat}，各需 ≥5）。")
        return

    Y = sub[outcome].to_numpy(float)
    W = sub[covs].to_numpy(float)
    y_vals = np.unique(Y)
    binary_y = len(y_vals) == 2
    if binary_y:
        Y = (Y == y_vals[1]).astype(float)

    def _design(tvec):
        # [const, T, W..., (T×W)...] — interactions let the effect vary with covariates.
        inter = tvec[:, None] * W
        return sm.add_constant(np.column_stack([tvec, W, inter]), has_constant="add")

    def _ate(Tt, Yt, Wt):
        """g-formula ATE on one (possibly bootstrap) sample."""
        Xo = sm.add_constant(np.column_stack([Tt, Wt, Tt[:, None] * Wt]), has_constant="add")
        nn = len(Tt)
        if binary_y:
            mod = sm.Logit(Yt, Xo).fit(disp=0, maxiter=100)
        else:
            mod = sm.OLS(Yt, Xo).fit()
        d1 = sm.add_constant(np.column_stack([np.ones(nn), Wt, np.ones(nn)[:, None] * Wt]), has_constant="add")
        d0 = sm.add_constant(np.column_stack([np.zeros(nn), Wt, np.zeros(nn)[:, None] * Wt]), has_constant="add")
        p1 = mod.predict(d1)
        p0 = mod.predict(d0)
        return float(p1.mean()), float(p0.mean())

    try:
        # point estimate on the full sample (also used for the design/plot)
        Xo = _design(T)
        model = sm.Logit(Y, Xo).fit(disp=0, maxiter=100) if binary_y else sm.OLS(Y, Xo).fit()
        nn = n
        p1 = model.predict(sm.add_constant(np.column_stack([np.ones(nn), W, np.ones(nn)[:, None] * W]), has_constant="add"))
        p0 = model.predict(sm.add_constant(np.column_stack([np.zeros(nn), W, np.zeros(nn)[:, None] * W]), has_constant="add"))
        ey1, ey0 = float(p1.mean()), float(p0.mean())
        ate = ey1 - ey0
        naive = float(Y[T == 1].mean() - Y[T == 0].mean())

        # bootstrap CI
        rng = np.random.default_rng(int(cfg.get("seed", 42)) if str(cfg.get("seed", 42)).isdigit() else 42)
        B = int(cfg.get("n_boot", 500)) if str(cfg.get("n_boot", 500)).isdigit() else 500
        B = min(max(B, 100), 2000)
        boot = []
        for _ in range(B):
            idx = rng.integers(0, n, n)
            try:
                b1, b0 = _ate(T[idx], Y[idx], W[idx])
                boot.append(b1 - b0)
            except Exception:
                continue
        boot = np.asarray(boot, float)
        if boot.size >= 50:
            ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
            se = float(np.std(boot, ddof=1))
        else:
            ci_lo = ci_hi = se = float("nan")

        estimates["ate"] = round(ate, 5)
        estimates["ate_se"] = round(se, 5)
        estimates["ate_ci_low"] = round(ci_lo, 5)
        estimates["ate_ci_high"] = round(ci_hi, 5)
        estimates["e_y1"] = round(ey1, 5)
        estimates["e_y0"] = round(ey0, 5)
        estimates["naive_diff"] = round(naive, 5)
        estimates["n"] = float(n)
        estimates["n_treated"] = float(n_treat)
        estimates["n_covariates"] = float(len(covs))
        estimates["outcome_binary"] = 1.0 if binary_y else 0.0

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(4.8, 3.6))
            labels = ["E[Y|do(T=0)]", "E[Y|do(T=1)]"]
            ax.bar(labels, [ey0, ey1], color=["#bbbbbb", "#2f6f4f"], width=0.6)
            ax.set_ylabel(("risk P(Y=1)" if binary_y else str(outcome)))
            ax.set_title(f"g-computation: ATE = {ate:.3g}")
            if np.isfinite(ci_lo):
                ax.errorbar(1, ey1, yerr=[[ey1 - (ey0 + ci_lo)], [(ey0 + ci_hi) - ey1]],
                            fmt="none", ecolor="#333", capsize=4)
            fig.tight_layout()
            fig.savefig(d / "g_computation_ate.png", dpi=150)
            plt.close(fig)
            files.append("g_computation_ate.png")
        except Exception:
            pass

        scale = "风险差(RD)" if binary_y else "均值差"
        sig = (np.isfinite(ci_lo) and (ci_lo > 0 or ci_hi < 0))
        summary.append(
            f"{entry.method} 完成（参数 g-公式/标准化，n={n}，处理 {n_treat}/对照 {n - n_treat}，"
            f"协变量 {len(covs)} 个，结果模型={'logistic' if binary_y else 'OLS'}）："
            f"**ATE（{scale}）={ate:.4g}**"
            + (f"（95% bootstrap CI [{ci_lo:.3g}, {ci_hi:.3g}]{'，排除 0、显著' if sig else '，含 0'}）" if np.isfinite(ci_lo) else "")
            + f"；E[Y|do(1)]={ey1:.4g}、E[Y|do(0)]={ey0:.4g}；朴素差={naive:.4g}"
            "（朴素差与 ATE 之差体现混杂调整）。对比图见 g_computation_ate.png。"
            " ⚠ g-公式假设：①无未观测混杂（给定协变量的条件可交换性）②正性（各协变量层处理/对照都有样本）"
            "③结果模型设定正确——它是**纯结果模型驱动**，对设定误差敏感（不像 aipw 双稳健，可与之相互印证）；"
            "标准化在观测协变量分布上、含 T×协变量交互以允许异质效应；ATE 在结果原尺度。"
        )
        code += [
            "import statsmodels.api as sm  # 参数 g-公式 / 标准化",
            "m = sm.OLS(Y, sm.add_constant(np.c_[T, W, T[:,None]*W])).fit()  # 结果模型(含 T×W)",
            "p1 = m.predict(design(T=1)); p0 = m.predict(design(T=0))        # 对全样本预测两种处理",
            "ate = p1.mean() - p0.mean()                                     # 标准化 ATE (bootstrap CI)",
        ]
    except Exception as err:
        summary.append(f"g-计算失败：{err}")
