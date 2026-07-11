"""Branch handlers: negative_binomial_regression, poisson_regression (statistics family).

Each handler unpacks ctx into the same local names run_analysis used and runs the
original branch body verbatim. See executor/_branch_api.py.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.run import _coef_plot, resolve_outcome


@register("negative_binomial_regression")
def _branch_negative_binomial_regression(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import statsmodels.formula.api as smf
    import numpy as np

    _excl = {fp.unit_col, fp.time_col}
    count_cols = [
        c.name for c in fp.columns if c.kind == "count" and c.name not in _excl
    ]
    # config may force ANY existing column (id-trap: an all-unique-int count profiles as
    # 'id' — the user knows better), matching count_models._resolve_count_outcome; else the
    # shared resolver (high-confidence outcome > first non-treatment-named count > first).
    forced = cfg.get("outcome")
    if forced is not None and forced in df.columns:
        outcome = forced
    elif count_cols:
        outcome = resolve_outcome(fp, cfg, count_cols)
    else:
        outcome = None

    if outcome is None:
        summary.append("负二项回归失败：未找到计数型结果变量。")
    else:
        amb = (
            f"（数据有 {len(count_cols)} 个计数列，已取 {outcome}；若它实为 ID/编码而非计数结果，请改选）"
            if len(count_cols) > 1
            else ""
        )
        exclude = {outcome, fp.unit_col, fp.time_col}
        predictors = [
            c.name
            for c in fp.columns
            if c.kind in {"continuous", "binary"} and c.name not in exclude
        ][:5]
        rhs = [f"Q('{v}')" for v in predictors]
        formula = f"Q('{outcome}') ~ " + (" + ".join(rhs) if rhs else "1")
        recipe = (
            "import statsmodels.formula.api as smf\n"
            f'model = smf.negativebinomial("{formula}", data=df).fit(disp=False)\n'
            "print(model.summary())"
        )
        try:
            model = smf.negativebinomial(formula, data=df).fit(disp=False)
            (d / "summary.txt").write_text(str(model.summary()), encoding="utf-8")
            files.append("summary.txt")
            tab = model.summary2().tables[1].copy()
            # model.params includes an 'alpha' (dispersion) row at the end;
            # summary2().tables[1] also includes it — lengths always match,
            # so exp() of all rows is safe (exp(alpha) is a positive scalar,
            # harmless alongside the log-rate coefficients).
            tab["rate_ratio"] = np.exp(model.params.values)
            tab.to_csv(d / "coefficients.csv", encoding="utf-8")
            files.append("coefficients.csv")
            _coef_plot(model, predictors, d / "coefficients.png")
            files.append("coefficients.png")
            for v in predictors:
                kn = f"Q('{v}')"
                if kn in model.params.index:
                    estimates[v] = float(model.params[kn])
            summary.append(
                f"{entry.method} 完成：计数结果 {outcome}，{len(predictors)} 个预测变量{amb}"
            )
            code += [recipe]
        except Exception as err:
            summary.append(f"负二项回归失败：{err}")



@register("poisson_regression")
def _branch_poisson_regression(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    import numpy as np

    _excl = {fp.unit_col, fp.time_col}
    count_cols = [
        c.name for c in fp.columns if c.kind == "count" and c.name not in _excl
    ]
    # config may force ANY existing column (id-trap: an all-unique-int count profiles as
    # 'id' — the user knows better), matching count_models._resolve_count_outcome; else the
    # shared resolver (high-confidence outcome > first non-treatment-named count > first).
    forced = cfg.get("outcome")
    if forced is not None and forced in df.columns:
        outcome = forced
    elif count_cols:
        outcome = resolve_outcome(fp, cfg, count_cols)
    else:
        outcome = None

    if outcome is None:
        summary.append("泊松回归失败：未找到计数型结果变量。")
    else:
        amb = (
            f"（数据有 {len(count_cols)} 个计数列，已取 {outcome}；若它实为 ID/编码而非计数结果，请改选）"
            if len(count_cols) > 1
            else ""
        )
        exclude = {outcome, fp.unit_col, fp.time_col}
        predictors = [
            c.name
            for c in fp.columns
            if c.kind in {"continuous", "binary"} and c.name not in exclude
        ][:5]
        rhs = [f"Q('{v}')" for v in predictors]
        formula = f"Q('{outcome}') ~ " + (" + ".join(rhs) if rhs else "1")
        recipe = (
            "import statsmodels.formula.api as smf\n"
            "import statsmodels.api as sm\n"
            f'model = smf.glm("{formula}", data=df, family=sm.families.Poisson()).fit()\n'
            "print(model.summary())"
        )
        try:
            model = smf.glm(formula, data=df, family=sm.families.Poisson()).fit()
            (d / "summary.txt").write_text(str(model.summary()), encoding="utf-8")
            files.append("summary.txt")
            tab = model.summary2().tables[1].copy()
            tab["rate_ratio"] = np.exp(model.params.values)
            tab.to_csv(d / "coefficients.csv", encoding="utf-8")
            files.append("coefficients.csv")
            _coef_plot(model, predictors, d / "coefficients.png")
            files.append("coefficients.png")
            for v in predictors:
                kn = f"Q('{v}')"
                if kn in model.params.index:
                    estimates[v] = float(model.params[kn])
            summary.append(
                f"{entry.method} 完成：计数结果 {outcome}，{len(predictors)} 个预测变量{amb}"
            )
            code += [recipe]
        except Exception as err:
            summary.append(f"泊松回归失败：{err}")
