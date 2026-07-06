"""Count / limited-dependent-variable models (zero-inflation + Tweedie).

Extends the count-regression family beyond Poisson / negative-binomial with
methods for the two pathologies real count data shows: an *excess* of zeros
(more than a single count process predicts) and semi-continuous nonnegative
data (a point mass at zero plus a continuous positive part).

All three branches are pure statsmodels (no R needed):
- ``zero_inflated_poisson``  → ZeroInflatedPoisson (structural-zero logistic + Poisson)
- ``zero_inflated_negbin``   → ZeroInflatedNegativeBinomialP (+ overdispersion alpha)
- ``tweedie_glm``            → GLM with the Tweedie family (compound Poisson-Gamma, 1<p<2)

Conventions follow the regression family (see CLAUDE.md「引擎约定」):
outcome = a count column (config ``outcome`` override); predictors =
continuous/binary columns (config ``predictors``). Products: CSV + PNG
(Agg, English labels, best-effort) + estimates (floats) + a Chinese summary
ending with ⚠ disclosures. Failures degrade honestly ("<method>失败：<reason>").
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.run import resolve_outcome


# ---------------------------------------------------------------------------
# shared column resolution (count outcome + continuous/binary predictors)
# ---------------------------------------------------------------------------
def _resolve_count_outcome(fp, df, cfg):
    """Return (outcome, count_cols) honouring a config['outcome'] override.

    A count column = integer-valued, non-negative, with repeats (profiler tags
    all-unique integers as 'id', not 'count'). config['outcome'] may force any
    existing column (the user knows better than the profiler) — deliberately WIDER
    than resolve_outcome's candidate check, so keep this branch first. The unforced
    path then delegates to the shared resolver (high-confidence detected outcome >
    first non-treatment-named count > first count).
    """
    _excl = {fp.unit_col, fp.time_col}
    count_cols = [c.name for c in fp.columns if c.kind == "count" and c.name not in _excl]
    forced = cfg.get("outcome")
    if forced is not None and forced in df.columns:
        return forced, count_cols
    if not count_cols:
        return None, count_cols
    return resolve_outcome(fp, cfg, count_cols), count_cols


def _resolve_predictors(fp, df, cfg, outcome):
    """Continuous/binary predictors, with a config['predictors'] override."""
    exclude = {outcome, fp.unit_col, fp.time_col}
    auto = [
        c.name
        for c in fp.columns
        if c.kind in {"continuous", "binary"} and c.name not in exclude
    ]
    forced = cfg.get("predictors")
    if forced:
        preds = [c for c in forced if c in df.columns and c != outcome]
    else:
        preds = auto
    return preds[:5]


def _ambiguity_note(count_cols, outcome):
    return (
        f"（数据有 {len(count_cols)} 个计数列，已取 {outcome}；若它实为 ID/编码而非计数结果，请改选）"
        if len(count_cols) > 1
        else ""
    )


def _zero_diagnostics(y):
    """Observed zero-fraction vs the Poisson-expected zero-fraction at ybar."""
    import numpy as np

    n = len(y)
    obs_zero = float((y == 0).sum()) / n if n else float("nan")
    mean = float(np.mean(y)) if n else float("nan")
    exp_zero = float(np.exp(-mean)) if n else float("nan")  # P(Y=0)=e^-λ for Poisson(λ=mean)
    return obs_zero, exp_zero, mean


def _zi_coef_plot(count_params, count_errs, labels, path):
    """Forest plot of the COUNT-model slopes (95% CI), English labels."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        if not labels:
            return
        coefs = np.asarray(count_params, dtype=float)
        errs = np.asarray(count_errs, dtype=float)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.errorbar(coefs, range(len(labels)), xerr=1.96 * errs, fmt="o")
        ax.axvline(0, color="grey", ls="--")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("count-model coefficient (95% CI)")
        ax.set_title("Count component (log-rate)")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception:
        pass


# ===========================================================================
# 1) Zero-inflated Poisson
# ===========================================================================
@register("zero_inflated_poisson")
def _branch_zero_inflated_poisson(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.discrete.count_model import ZeroInflatedPoisson

    outcome, count_cols = _resolve_count_outcome(fp, df, cfg)
    if outcome is None:
        summary.append("零膨胀泊松失败：未找到计数型结果变量。")
        return

    predictors = _resolve_predictors(fp, df, cfg, outcome)
    # inflation (structural-zero) covariates default to a constant; users may
    # name separate inflation predictors via config['inflation'].
    infl_cfg = cfg.get("inflation")
    infl_vars = (
        [c for c in infl_cfg if c in df.columns and c != outcome] if infl_cfg else []
    )

    sub_cols = [outcome] + predictors + [c for c in infl_vars if c not in predictors]
    data = df[sub_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < 20:
        summary.append(f"零膨胀泊松失败：有效行数不足（{len(data)} < 20）。")
        return

    y = data[outcome].to_numpy()
    if not np.all(y >= 0) or not np.allclose(y, np.round(y)):
        summary.append("零膨胀泊松失败：结果变量不是非负整数计数。")
        return

    X = sm.add_constant(data[predictors], has_constant="add") if predictors else \
        pd.DataFrame({"const": np.ones(len(data))}, index=data.index)
    # exog_infl: a constant (+ optional separate inflation covariates).
    Z = sm.add_constant(data[infl_vars], has_constant="add") if infl_vars else \
        pd.DataFrame({"const": np.ones(len(data))}, index=data.index)

    amb = _ambiguity_note(count_cols, outcome)
    recipe = (
        "import statsmodels.api as sm\n"
        "from statsmodels.discrete.count_model import ZeroInflatedPoisson\n"
        f"X = sm.add_constant(df[{predictors!r}])\n"
        "Z = sm.add_constant(df[[]])  # inflation = constant by default\n"
        f"m = ZeroInflatedPoisson(df['{outcome}'], X, exog_infl=Z).fit(disp=False)\n"
        "print(m.summary())"
    )
    try:
        model = ZeroInflatedPoisson(y, X, exog_infl=Z, inflation="logit").fit(
            method="bfgs", maxiter=200, disp=False
        )
        # plain Poisson for an AIC comparison (does excess-zero modelling pay?).
        pois = sm.GLM(y, X, family=sm.families.Poisson()).fit()

        (d / "summary.txt").write_text(str(model.summary()), encoding="utf-8")
        files.append("summary.txt")

        # --- separate count vs inflation parameters --------------------------
        # statsmodels prefixes the zero-inflation params with "inflate_"; the
        # remaining params are the COUNT (Poisson log-rate) coefficients.
        params = model.params
        bse = model.bse
        idx = list(params.index)
        infl_idx = [i for i in idx if str(i).startswith("inflate_")]
        count_idx = [i for i in idx if not str(i).startswith("inflate_")]

        count_tab = pd.DataFrame(
            {
                "coef": params[count_idx],
                "std_err": bse[count_idx],
                "z": model.tvalues[count_idx],
                "P>|z|": model.pvalues[count_idx],
                "IRR": np.exp(params[count_idx].to_numpy()),  # incidence-rate ratio
            }
        )
        infl_tab = pd.DataFrame(
            {
                "coef": params[infl_idx],
                "std_err": bse[infl_idx],
                "z": model.tvalues[infl_idx],
                "P>|z|": model.pvalues[infl_idx],
            }
        )
        count_tab.to_csv(d / "count_coefficients.csv", encoding="utf-8")
        files.append("count_coefficients.csv")
        infl_tab.to_csv(d / "inflation_coefficients.csv", encoding="utf-8")
        files.append("inflation_coefficients.csv")

        # forest plot of the count-model slopes (drop the intercept)
        slope_names = [c for c in predictors if c in count_idx]
        _zi_coef_plot(
            params[slope_names].to_numpy(),
            bse[slope_names].to_numpy(),
            slope_names,
            d / "count_coefficients.png",
        )
        if (d / "count_coefficients.png").exists():
            files.append("count_coefficients.png")

        for v in predictors:
            if v in params.index:
                estimates[v] = float(params[v])
        estimates["aic_zip"] = float(model.aic)
        estimates["aic_poisson"] = float(pois.aic)

        obs_zero, exp_zero, ybar = _zero_diagnostics(y)
        estimates["pct_zeros_observed"] = obs_zero
        estimates["pct_zeros_poisson_expected"] = exp_zero

        delta = pois.aic - model.aic  # >0 → ZIP preferred
        verdict = "更优（建议）" if delta > 0 else "并不更优（零膨胀可能不必要）"
        summary.append(
            f"{entry.method} 完成：计数结果 {outcome}，{len(predictors)} 个预测变量{amb}。"
            f"观测零占比 {obs_zero:.1%}，纯泊松预期零占比 {exp_zero:.1%}"
            f"（差越大、零膨胀越可能为真）。"
            f"AIC：ZIP={model.aic:.1f} vs 泊松={pois.aic:.1f}，ZIP {verdict}。"
        )
        summary.append(
            "⚠ ZIP 同时建模两个过程——一个「结构性零」logistic（永不计数的个体）"
            "与一个泊松计数；仅当过量零确为真时才用。膨胀协变量默认仅含常数"
            "（用 config['inflation'] 指定单独的膨胀预测变量）。"
            "若仍存在过度离散（方差远大于均值），应改用 ZINB。"
            "IRR=exp(计数系数)；膨胀系数预测「属于结构零类」的对数几率，不可与计数系数混读。"
        )
        code += [recipe]
    except Exception as err:
        summary.append(f"零膨胀泊松失败/未收敛：{err}")


# ===========================================================================
# 2) Zero-inflated negative binomial
# ===========================================================================
@register("zero_inflated_negbin")
def _branch_zero_inflated_negbin(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.discrete.count_model import (
        ZeroInflatedNegativeBinomialP,
        ZeroInflatedPoisson,
    )

    outcome, count_cols = _resolve_count_outcome(fp, df, cfg)
    if outcome is None:
        summary.append("零膨胀负二项失败：未找到计数型结果变量。")
        return

    predictors = _resolve_predictors(fp, df, cfg, outcome)
    infl_cfg = cfg.get("inflation")
    infl_vars = (
        [c for c in infl_cfg if c in df.columns and c != outcome] if infl_cfg else []
    )

    sub_cols = [outcome] + predictors + [c for c in infl_vars if c not in predictors]
    data = df[sub_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < 30:
        summary.append(f"零膨胀负二项失败：有效行数不足（{len(data)} < 30）。")
        return

    y = data[outcome].to_numpy()
    if not np.all(y >= 0) or not np.allclose(y, np.round(y)):
        summary.append("零膨胀负二项失败：结果变量不是非负整数计数。")
        return

    X = sm.add_constant(data[predictors], has_constant="add") if predictors else \
        pd.DataFrame({"const": np.ones(len(data))}, index=data.index)
    Z = sm.add_constant(data[infl_vars], has_constant="add") if infl_vars else \
        pd.DataFrame({"const": np.ones(len(data))}, index=data.index)

    amb = _ambiguity_note(count_cols, outcome)
    recipe = (
        "import statsmodels.api as sm\n"
        "from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP\n"
        f"X = sm.add_constant(df[{predictors!r}])\n"
        "Z = sm.add_constant(df[[]])  # inflation = constant by default\n"
        f"m = ZeroInflatedNegativeBinomialP(df['{outcome}'], X, exog_infl=Z).fit(disp=False)\n"
        "print(m.summary())"
    )
    try:
        model = ZeroInflatedNegativeBinomialP(y, X, exog_infl=Z, inflation="logit").fit(
            method="bfgs", maxiter=300, disp=False
        )
        # AIC baselines: ZIP and plain Poisson.
        try:
            zip_m = ZeroInflatedPoisson(y, X, exog_infl=Z, inflation="logit").fit(
                method="bfgs", maxiter=200, disp=False
            )
            aic_zip = float(zip_m.aic)
        except Exception:
            aic_zip = float("nan")
        pois = sm.GLM(y, X, family=sm.families.Poisson()).fit()

        (d / "summary.txt").write_text(str(model.summary()), encoding="utf-8")
        files.append("summary.txt")

        params = model.params
        bse = model.bse
        idx = list(params.index)
        infl_idx = [i for i in idx if str(i).startswith("inflate_")]
        # the dispersion is reported as 'alpha' (last param); it is neither
        # an inflation coef nor a count slope — pull it out separately.
        alpha_idx = [i for i in idx if str(i) == "alpha"]
        count_idx = [
            i for i in idx if not str(i).startswith("inflate_") and str(i) != "alpha"
        ]

        count_tab = pd.DataFrame(
            {
                "coef": params[count_idx],
                "std_err": bse[count_idx],
                "z": model.tvalues[count_idx],
                "P>|z|": model.pvalues[count_idx],
                "IRR": np.exp(params[count_idx].to_numpy()),
            }
        )
        infl_tab = pd.DataFrame(
            {
                "coef": params[infl_idx],
                "std_err": bse[infl_idx],
                "z": model.tvalues[infl_idx],
                "P>|z|": model.pvalues[infl_idx],
            }
        )
        count_tab.to_csv(d / "count_coefficients.csv", encoding="utf-8")
        files.append("count_coefficients.csv")
        infl_tab.to_csv(d / "inflation_coefficients.csv", encoding="utf-8")
        files.append("inflation_coefficients.csv")

        slope_names = [c for c in predictors if c in count_idx]
        _zi_coef_plot(
            params[slope_names].to_numpy(),
            bse[slope_names].to_numpy(),
            slope_names,
            d / "count_coefficients.png",
        )
        if (d / "count_coefficients.png").exists():
            files.append("count_coefficients.png")

        for v in predictors:
            if v in params.index:
                estimates[v] = float(params[v])
        alpha = float(params[alpha_idx[0]]) if alpha_idx else float("nan")
        estimates["alpha"] = alpha
        estimates["aic_zinb"] = float(model.aic)
        estimates["aic_zip"] = aic_zip
        estimates["aic_poisson"] = float(pois.aic)

        obs_zero, exp_zero, ybar = _zero_diagnostics(y)
        estimates["pct_zeros_observed"] = obs_zero
        estimates["pct_zeros_poisson_expected"] = exp_zero

        disp_note = (
            f"α={alpha:.3f}>0，确认存在过度离散" if alpha == alpha and alpha > 0
            else f"α={alpha:.3f}（≤0 或缺失，过度离散证据弱）"
        )
        aic_zip_str = f"{aic_zip:.1f}" if aic_zip == aic_zip else "NA"
        summary.append(
            f"{entry.method} 完成：计数结果 {outcome}，{len(predictors)} 个预测变量{amb}。"
            f"{disp_note}。观测零占比 {obs_zero:.1%}（泊松预期 {exp_zero:.1%}）。"
            f"AIC：ZINB={model.aic:.1f}、ZIP={aic_zip_str}、泊松={pois.aic:.1f}（越小越好）。"
        )
        summary.append(
            "⚠ ZINB 同时处理过量零与过度离散；α>0 证实过度离散（α→0 时退化为 ZIP）。"
            "收敛可能不稳——若未收敛会如实报告。"
            "IRR=exp(计数系数)；膨胀系数为「结构零」logistic 的对数几率，"
            "与计数系数、与离散参数 α 分属三类，不可混读。"
        )
        code += [recipe]
    except Exception as err:
        summary.append(f"零膨胀负二项失败/未收敛：{err}")


# ===========================================================================
# 3) Tweedie GLM (semi-continuous / compound Poisson-Gamma)
# ===========================================================================
@register("tweedie_glm")
def _branch_tweedie_glm(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    # Tweedie targets SEMI-CONTINUOUS nonnegative data (a mass at zero + a
    # continuous positive part), which usually profiles as 'continuous' rather
    # than 'count'. Accept a count column too, and honour config['outcome'].
    _excl = {fp.unit_col, fp.time_col}
    cont_cols = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
    count_cols = [c.name for c in fp.columns if c.kind == "count" and c.name not in _excl]
    forced = cfg.get("outcome")
    if forced is not None and forced in df.columns:
        outcome = forced
    elif cont_cols:
        outcome = resolve_outcome(fp, cfg, cont_cols)
    elif count_cols:
        outcome = resolve_outcome(fp, cfg, count_cols)
    else:
        outcome = None

    if outcome is None:
        summary.append("Tweedie GLM 失败：未找到非负数值结果变量。")
        return

    exclude = {outcome, fp.unit_col, fp.time_col}
    auto = [
        c.name
        for c in fp.columns
        if c.kind in {"continuous", "binary", "count"} and c.name not in exclude
    ]
    forced_p = cfg.get("predictors")
    predictors = (
        [c for c in forced_p if c in df.columns and c != outcome] if forced_p else auto
    )[:5]

    sub_cols = [outcome] + predictors
    data = df[sub_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < 20:
        summary.append(f"Tweedie GLM 失败：有效行数不足（{len(data)} < 20）。")
        return

    y = data[outcome].to_numpy(dtype=float)
    if not np.all(y >= 0):
        summary.append("Tweedie GLM 失败：结果变量含负值（Tweedie 要求非负）。")
        return

    # var_power p: 1<p<2 = compound Poisson-Gamma (zeros + positive continuous).
    var_power = float(cfg.get("var_power", 1.5))
    if not (1.0 < var_power < 2.0):
        summary.append(
            f"⚠ var_power={var_power} 不在 (1,2) 区间；1<p<2 才是零膨胀连续数据的"
            "复合泊松-伽马情形，已按所给值拟合，请确认这是有意为之。"
        )

    X = sm.add_constant(data[predictors], has_constant="add") if predictors else \
        pd.DataFrame({"const": np.ones(len(data))}, index=data.index)

    recipe = (
        "import statsmodels.api as sm\n"
        f"X = sm.add_constant(df[{predictors!r}])\n"
        f"fam = sm.families.Tweedie(var_power={var_power}, link=sm.families.links.Log())\n"
        f"m = sm.GLM(df['{outcome}'], X, family=fam).fit()\n"
        "print(m.summary())"
    )
    try:
        try:
            link = sm.families.links.Log()
        except Exception:  # older statsmodels capitalises it
            link = sm.families.links.log()
        fam = sm.families.Tweedie(var_power=var_power, link=link)
        model = sm.GLM(y, X, family=fam).fit()

        (d / "summary.txt").write_text(str(model.summary()), encoding="utf-8")
        files.append("summary.txt")

        params = model.params
        bse = model.bse
        tab = pd.DataFrame(
            {
                "coef": params,
                "std_err": bse,
                "z": model.tvalues,
                "P>|z|": model.pvalues,
                "exp_coef": np.exp(params.to_numpy()),  # multiplicative on the mean (log link)
            }
        )
        tab.to_csv(d / "coefficients.csv", encoding="utf-8")
        files.append("coefficients.csv")

        slope_names = [c for c in predictors if c in list(params.index)]
        _zi_coef_plot(
            params[slope_names].to_numpy(),
            bse[slope_names].to_numpy(),
            slope_names,
            d / "coefficients.png",
        )
        if (d / "coefficients.png").exists():
            files.append("coefficients.png")

        for v in predictors:
            if v in params.index:
                estimates[v] = float(params[v])
        estimates["deviance"] = float(model.deviance)
        estimates["var_power"] = var_power
        obs_zero = float((y == 0).sum()) / len(y)
        estimates["pct_zeros_observed"] = obs_zero

        summary.append(
            f"{entry.method} 完成：非负结果 {outcome}，{len(predictors)} 个预测变量。"
            f"var_power={var_power}（1<p<2 复合泊松-伽马），观测零占比 {obs_zero:.1%}，"
            f"偏差(deviance)={model.deviance:.1f}。对数链接下 exp(系数) 为对均值的乘性效应。"
        )
        summary.append(
            "⚠ Tweedie 在 1<p<2 时刻画「零膨胀连续」数据（保费/降雨/生物量等：零质点+正连续部分）。"
            "var_power 是建模选择（此处由 config['var_power'] 给定、默认 1.5；"
            "理想做法是按数据剖面化 profile p，本分支未自动 profile）。"
            "模型假定均值-方差幂律 Var(Y)=φ·μ^p。"
        )
        code += [recipe]
    except Exception as err:
        summary.append(f"Tweedie GLM 失败/未收敛：{err}")
