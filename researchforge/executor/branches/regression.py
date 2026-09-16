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
    # dogfood: a treatment that never varies within unit is perfectly collinear with the unit
    # fixed effects. statsmodels still "fits" — it reported a coefficient with SE ≈ 1e-13 on a
    # repeated-measures RCT where `arm` is constant per subject — and the branch reported 完成.
    # An SE orders of magnitude below the coefficient is an artefact of a rank-deficient design,
    # not an estimate. (The experimental-design family has refused this shape since its own
    # cold review; the regression family never checked.)
    _degenerate = None
    try:
        import numpy as _np

        _exog = _np.asarray(model.model.exog, dtype=float)
        if _exog.ndim == 2 and _exog.shape[1] and _exog.size <= 20_000_000:
            _rank = int(_np.linalg.matrix_rank(_exog))
            if _rank < _exog.shape[1]:
                _degenerate = (f"设计矩阵秩亏（秩 {_rank} < 列数 {_exog.shape[1]}）")
        if _degenerate is None:
            # near-collinearity the rank test rounds away: a KEY term whose SE is orders of
            # magnitude below its own coefficient. Check the key terms themselves — a max over
            # all params never sees it, because rank deficiency kills one DIRECTION, not the
            # whole fit (measured: max|bse|=0.80 while the treatment term's bse was 1.09e-13).
            for _v in rhs_vars:
                _b = f"Q('{_v}')"
                for _k in model.params.index:
                    if _k == _b or _k.startswith(_b + "[") or _k.startswith(f"C({_b})["):
                        _c = abs(float(model.params[_k]))
                        _se = float(model.bse[_k])
                        if _c > 0 and (not _np.isfinite(_se) or _se < 1e-8 * _c):
                            _degenerate = (f"关键系数 {_k} 的标准误≈0（{_se:.3g}）")
                            break
                if _degenerate:
                    break
    except Exception:  # noqa: BLE001 — a guard must never break the run
        _degenerate = None
    if _degenerate:
        summary.append(
            f"{entry.method} 失败：{_degenerate}——说明某个预测变量被其它项完全解释"
            "（最常见：处理变量在每个单位内不随时间变化，被单位固定效应吸收）。"
            "此时该系数与其 p 值无意义。"
            "若为重复测量设计，请改用 repeated_measures_anova / mixed_effects；"
            '或用 config={"predictors":[..]} 换一组预测变量。'
        )
        return
    (d / "summary.txt").write_text(str(model.summary()), encoding="utf-8")
    files.append("summary.txt")
    model.summary2().tables[1].to_csv(d / "coefficients.csv", encoding="utf-8")
    files.append("coefficients.csv")
    _coef_plot(model, rhs_vars, d / "coefficients.png")
    files.append("coefficients.png")
    _resid_plot(model, d / "residuals_vs_fitted.png")
    if (d / "residuals_vs_fitted.png").exists():
        files.append("residuals_vs_fitted.png")
    # dogfood: a STRING-coded binary/categorical predictor is treatment-coded by patsy, so
    # its real parameter name is `Q('arm')[T.placebo]`, not `Q('arm')`. Matching only the bare
    # key dropped the estimate silently and left the summary as a naked "完成" — while
    # coefficients.csv had the number all along. Match the level terms too, and when a factor
    # expands to several levels report each.
    def _terms_for(v):
        base = f"Q('{v}')"
        exact = [k for k in model.params.index if k == base]
        if exact:
            return [(v, base)]
        return [(f"{v}{k[len(base):]}", k) for k in model.params.index
                if k.startswith(base + "[") or k.startswith(f"C({base})[")]

    _key_terms = []
    for v in rhs_vars:
        for label, kn in _terms_for(v):
            estimates[label] = float(model.params[kn])
            if v == rhs_vars[0]:
                _key_terms.append((label, kn))
    key = ""
    if _key_terms:
        _lbl, _kn = _key_terms[0]
        key = f"，关键系数 {_lbl} = {model.params[_kn]:.4f} (p={model.pvalues[_kn]:.3g})"
    # Columns the model never saw. The predictor set is continuous/count/binary, so a
    # multi-level categorical is dropped — measured: `grp` explaining ~95% of the
    # variance vanished and the report still read "关键系数 x = -0.057 (p=0.724)".
    # mixed_effects fixed this by dummy-coding (Wave K-B3); doing that here changes the
    # model spec across the whole family, so for now say it out loud.
    _dropped = [c.name for c in fp.columns
                if c.kind == "categorical" and c.name not in set(rhs_vars)
                and c.name not in {y, fp.unit_col, fp.time_col}
                and 1 < int(df[c.name].nunique(dropna=True)) <= 20]
    n_cont = sum(1 for c in fp.columns if c.kind == "continuous")
    dv_note = f"（数据有 {n_cont} 个连续列，默认取 {y} 为因变量）" if n_cont > 1 else ""
    summary.append(f"{entry.method} 完成：因变量 {y}{key}{dv_note}")
    if _dropped:
        summary.append(
            f"⚠ 分类预测变量 {_dropped} 未进入模型（本族只取 连续/计数/二值 预测变量）——"
            "若它们与结果有关，上面的系数是在未控制它们的情况下估计的，R² 也会被低估。"
            '需要纳入可改用 mixed_effects（分类固定效应已哑变量化）或 factorial_anova。'
        )
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
