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
    clustered = bool(fp.is_panel and fp.unit_col)
    if clustered:
        _n_clusters = int(df[fp.unit_col].nunique()) if fp.unit_col in df.columns else 0
        _cl = (f"标准误按 {fp.unit_col} 聚类（共 {_n_clusters} 个单位；面板数据默认用聚类稳健 SE，"
               "而非普通 HC1，避免同一单位内序列相关导致 SE 偏小、p 值虚低）。")
        if 0 < _n_clusters < 30:  # D1 冷审 SHOULD：少簇下 statsmodels 正态参照偏乐观
            _cl += (f"⚠️ 聚类单位偏少（{_n_clusters}），聚类稳健推断用正态参照、少簇时偏乐观（p 偏小），"
                    "宜按 t(G−1) 或 wild cluster bootstrap 审慎解读。")
        summary.append(_cl)
    if not rhs_vars:
        summary.append("⚠️ 无可用解释变量，仅拟合了截距模型，结果无解释意义。")
    if entry.id == "did" and rhs_vars and fp.unit_col:
        if int(df.groupby(fp.unit_col)[rhs_vars[0]].nunique().max()) <= 1:
            summary.append(
                f"⚠️ 处理变量 {rhs_vars[0]} 在每个单位内不随时间变化，可能不是有效的 DID 处理。"
            )
    if entry.id == "did" and clustered:  # D1 冷审 SHOULD：聚类维度诚实披露
        summary.append(
            f"⚠️ DID 标准误按 {fp.unit_col} 聚类（处理按单位层赋值时正确）；若处理实为更粗层级赋值"
            "（如政策打到多个单位），应按该更粗层聚类——引擎按可得的最细单位聚类。"
        )
    if entry.id == "ols_regression" and clustered:
        summary.append(
            f"⚠️ 数据疑似面板结构（单位列 {fp.unit_col}），当前是 pooled OLS，忽略了个体固定效应，"
            "估计可能有偏——如需控制个体异质性，考虑改用 panel_fixed_effects。"
        )
    if clustered:
        code += [
            "import statsmodels.formula.api as smf",
            f"fit = df.dropna(subset={[y, *rhs_vars, fp.unit_col]!r})  # listwise 删缺失，对齐 groups 与拟合样本",
            f'model = smf.ols("{formula}", data=fit).fit(cov_type="cluster", cov_kwds={{"groups": fit["{fp.unit_col}"]}})',
            "print(model.summary())",
        ]
    else:
        code += [
            "import statsmodels.formula.api as smf",
            f'model = smf.ols("{formula}", data=df).fit(cov_type="HC1")',
            "print(model.summary())",
        ]
