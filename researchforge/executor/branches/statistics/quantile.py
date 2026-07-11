"""Branch handler: quantile_regression (statistics family).

Each handler unpacks ctx into the same local names run_analysis used and runs the
original branch body verbatim. See executor/_branch_api.py.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.run import _coef_plot, _quantile_process_plot


@register("quantile_regression")
def _branch_quantile_regression(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import statsmodels.formula.api as smf
    import pandas as pd

    _excl = {fp.unit_col, fp.time_col}
    outcome = next(
        (c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl),
        None,
    )
    if outcome is None:
        summary.append("分位数回归失败：未找到连续型结果变量。")
    else:
        exclude = {outcome, fp.unit_col, fp.time_col}
        predictors = [
            c.name
            for c in fp.columns
            if c.kind in {"continuous", "binary", "count"} and c.name not in exclude
        ][:5]
        rhs = [f"Q('{v}')" for v in predictors]
        formula = f"Q('{outcome}') ~ " + (" + ".join(rhs) if rhs else "1")
        taus = [0.25, 0.50, 0.75]
        recipe = (
            "import statsmodels.formula.api as smf\n"
            f'qr = smf.quantreg("{formula}", data=df)\n'
            "for tau in (0.25, 0.5, 0.75):\n"
            "    print(tau, qr.fit(q=tau).params)\n"
        )
        try:
            qr = smf.quantreg(formula, data=df)
            fits = {tau: qr.fit(q=tau) for tau in taus}
            med = fits[0.50]
            (d / "summary.txt").write_text(str(med.summary()), encoding="utf-8")
            files.append("summary.txt")
            # coefficients side by side across quantiles — the whole point of
            # quantile regression is seeing how effects differ down the
            # outcome distribution (τ=0.25 lower tail … 0.75 upper tail).
            tab = pd.DataFrame({f"tau={tau}": fits[tau].params for tau in taus})
            tab.to_csv(d / "coefficients.csv", encoding="utf-8")
            files.append("coefficients.csv")
            _coef_plot(med, predictors, d / "coefficients.png")
            files.append("coefficients.png")
            _quantile_process_plot(qr, predictors, d / "quantile_process.png")
            if (d / "quantile_process.png").exists():
                files.append("quantile_process.png")
            for v in predictors:
                kn = f"Q('{v}')"
                if kn in med.params.index:
                    estimates[v] = float(med.params[kn])
            summary.append(
                f"{entry.method} 完成：结果 {outcome}，{len(predictors)} 个预测变量，"
                "τ=0.25/0.50/0.75（中位数与尾部效应对比见 coefficients.csv）"
            )
            code += [recipe]
        except Exception as err:
            summary.append(f"分位数回归失败：{err}")
