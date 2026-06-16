"""Branch handler for the OLS-formula regression family — ols_regression /
panel_fixed_effects / did — dispatched by membership in ``run._REGRESSION`` (they
share one statsmodels OLS-with-HC1 body). Migrated from the run.py monolith;
registered via ``*_REGRESSION`` so the id set has a single source of truth.
See executor/_branch_api.py.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.run import _REGRESSION, _coef_plot, _regression, _resid_plot


@register(*_REGRESSION)
def _branch_regression(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    y, rhs_vars, formula, model = _regression(df, fp, entry, cfg)
    (d / "summary.txt").write_text(str(model.summary()), encoding="utf-8")
    files.append("summary.txt")
    model.summary2().tables[1].to_csv(d / "coefficients.csv", encoding="utf-8")
    files.append("coefficients.csv")
    _coef_plot(model, rhs_vars, d / "coefficients.png")
    files.append("coefficients.png")
    _resid_plot(model, d / "residuals_vs_fitted.png")
    if (d / "residuals_vs_fitted.png").exists():
        files.append("residuals_vs_fitted.png")
    for v in rhs_vars:
        kn = f"Q('{v}')"
        if kn in model.params.index:
            estimates[v] = float(model.params[kn])
    key = ""
    if rhs_vars:
        kname = f"Q('{rhs_vars[0]}')"
        if kname in model.params.index:
            key = f"，关键系数 {rhs_vars[0]} = {model.params[kname]:.4f} (p={model.pvalues[kname]:.3g})"
    n_cont = sum(1 for c in fp.columns if c.kind == "continuous")
    dv_note = f"（数据有 {n_cont} 个连续列，默认取 {y} 为因变量）" if n_cont > 1 else ""
    summary.append(f"{entry.method} 完成：因变量 {y}{key}{dv_note}")
    if not rhs_vars:
        summary.append("⚠️ 无可用解释变量，仅拟合了截距模型，结果无解释意义。")
    if entry.id == "did" and rhs_vars and fp.unit_col:
        if int(df.groupby(fp.unit_col)[rhs_vars[0]].nunique().max()) <= 1:
            summary.append(
                f"⚠️ 处理变量 {rhs_vars[0]} 在每个单位内不随时间变化，可能不是有效的 DID 处理。"
            )
    code += [
        "import statsmodels.formula.api as smf",
        f'model = smf.ols("{formula}", data=df).fit(cov_type="HC1")',
        "print(model.summary())",
    ]
