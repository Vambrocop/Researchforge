"""Branch handlers: logistic_regression, multinomial_logit, ordered_logit (statistics family).

Each handler unpacks ctx into the same local names run_analysis used and runs the
original branch body verbatim. See executor/_branch_api.py.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor._helpers.diagnostics import suspicious_fit_warnings
from researchforge.executor._helpers.formula import safe_formula_terms
from researchforge.executor.run import _coef_plot, _ordinal_prob_plot, resolve_outcome


@register("logistic_regression")
def _branch_logistic_regression(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import statsmodels.formula.api as smf

    # outcome: config override > high-confidence detected binary outcome (e.g. 'died',
    # 'approved') > first non-treatment-named binary > first binary. Closes the
    # selection→execution loop for the binary family — a leading treatment flag
    # ({treated, died, …}) is no longer mistaken for the dependent variable.
    binary_cols = [
        c.name
        for c in fp.columns
        if c.kind == "binary" and c.name not in {fp.unit_col, fp.time_col}
    ]
    outcome = resolve_outcome(fp, cfg, binary_cols) if binary_cols else None
    exclude = {outcome, fp.unit_col, fp.time_col}
    kind_by_name = {c.name: c.kind for c in fp.columns}
    # Wave K-E3: predictors now also pull binary + categorical columns (dummy-coded
    # below), not just continuous/count — a binary risk factor like smoking/sex used
    # to be silently excluded from its own logistic model.
    _auto_predictor_cols = [
        c for c in fp.columns
        if c.kind in {"continuous", "count", "binary", "categorical"}
        and c.name not in exclude
        # a constant (single-level) categorical dummy-codes to zero columns -> silent
        # no-op predictor; skip it (mirrors mixed_effects's B3 guard).
        and not (c.kind == "categorical" and df[c.name].nunique(dropna=True) < 2)
    ]
    predictors = [c for c in (cfg.get("predictors") or []) if c in df.columns and c != outcome] or [
        c.name for c in _auto_predictor_cols
    ][:5]

    if outcome is None:
        summary.append("逻辑回归失败：未找到二值结果变量。")
    else:
        # Wave K-B4b: formula identifiers now go through the shared
        # safe_formula_terms() helper (non-identifier-safe names, e.g. Chinese
        # columns, get aliased to v1/v2/…) instead of ad-hoc Q('col') string
        # splicing — a single collection point. _term_pairs (安全项, 原名) is used
        # below to restore the original names in coefficients.csv/estimates/summary.
        terms, _ = safe_formula_terms([outcome, *predictors])
        term_outcome, *term_predictors = terms
        col_map = dict(zip([outcome, *predictors], terms))  # 原名 -> 安全项
        _term_pairs = list(zip(term_predictors, predictors))  # (安全项, 原名)
        rhs = [
            f"C({t})" if kind_by_name.get(v) == "categorical" else t
            for t, v in _term_pairs
        ]
        formula = f"{term_outcome} ~ " + (" + ".join(rhs) if rhs else "1")
        sub = df[[outcome, *predictors]].rename(columns=col_map)
        try:
            model = smf.logit(formula, data=sub).fit(disp=False)
            (d / "summary.txt").write_text(str(model.summary()), encoding="utf-8")
            files.append("summary.txt")

            def _delabel(idx: str) -> str:
                """安全项(含 C(term)[T.level]/term[T.level] 哑变量后缀)还原成原始列名。"""
                for t, o in _term_pairs:
                    if idx == t:
                        return o
                    if idx.startswith(f"C({t})["):
                        return f"{o}{idx[len(f'C({t})'):]}"
                    if idx.startswith(f"{t}["):
                        return f"{o}{idx[len(t):]}"
                return idx

            # Wave K-E1: 出 OR = exp(β) + 95%CI = exp(conf_int()) 端点
            # （必须是 exp(conf_int) 端点，绝不是 exp(点估±se)）。
            ci = model.conf_int()
            coefs_table = model.summary2().tables[1].copy()
            coefs_table["OR"] = np.exp(model.params)
            coefs_table["OR_CI_low"] = np.exp(ci[0])
            coefs_table["OR_CI_high"] = np.exp(ci[1])
            coefs_table.index = [_delabel(i) for i in coefs_table.index]
            coefs_table.to_csv(d / "coefficients.csv", encoding="utf-8")
            files.append("coefficients.csv")
            _coef_plot(model, term_predictors, d / "coefficients.png")
            files.append("coefficients.png")

            for t, v in _term_pairs:
                if t in model.params.index:
                    estimates[v] = float(model.params[t])
                    estimates[f"{v}_OR"] = float(np.exp(model.params[t]))
                    estimates[f"{v}_OR_CI_low"] = float(np.exp(ci.loc[t, 0]))
                    estimates[f"{v}_OR_CI_high"] = float(np.exp(ci.loc[t, 1]))
                else:
                    # 分类/字符串二值列被 patsy 哑变量化成 term[T.level] 或
                    # C(term)[T.level]，精确键落空 -> 回退前缀扫描逐水平取值。
                    for idx in model.params.index:
                        if idx.startswith(f"C({t})[") or idx.startswith(f"{t}["):
                            lbl = _delabel(idx)
                            estimates[lbl] = float(model.params[idx])
                            estimates[f"{lbl}_OR"] = float(np.exp(model.params[idx]))
                            estimates[f"{lbl}_OR_CI_low"] = float(np.exp(ci.loc[idx, 0]))
                            estimates[f"{lbl}_OR_CI_high"] = float(np.exp(ci.loc[idx, 1]))

            key = ""
            if predictors:
                t0 = term_predictors[0]
                if t0 in model.params.index:
                    or0 = float(np.exp(model.params[t0]))
                    key = (
                        f"，关键系数 {predictors[0]} = {model.params[t0]:.4f} "
                        f"(OR={or0:.3f}, p={model.pvalues[t0]:.3g})"
                    )
            amb = (
                f"（数据有 {len(binary_cols)} 个二值列，已取 {outcome}；若它实为处理/标志变量请改选）"
                if len(binary_cols) > 1
                else ""
            )
            summary.append(f"{entry.method} 完成：结果变量 {outcome}{key}{amb}")
            try:  # Wave K-F3: 完美分离检测（分离时 bse 爆大/p≈1，系数与 OR 不可解读）
                for _w in suspicious_fit_warnings(
                    coefs=model.params.to_numpy(), ses=model.bse.to_numpy(),
                    pvalues=model.pvalues.to_numpy(),
                ):
                    summary.append(_w)
            except Exception:
                pass
            code += [
                "import statsmodels.formula.api as smf",
                f"sub = df[{[outcome, *predictors]!r}].rename(columns={col_map!r})",
                f'model = smf.logit("{formula}", data=sub).fit(disp=False)',
                "print(model.summary())",
            ]
        except Exception as err:
            summary.append(f"逻辑回归未收敛/失败：{err}")



@register("multinomial_logit")
def _branch_multinomial_logit(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    _excl = {fp.unit_col, fp.time_col}
    out_cands = [
        c
        for c in fp.columns
        if c.kind in {"count", "categorical"} and 3 <= c.n_unique <= 10 and c.name not in _excl
    ]
    out_cands.sort(key=lambda c: 0 if c.kind == "categorical" else 1)  # prefer nominal
    outcome = out_cands[0].name if out_cands else None
    predictors = [
        c.name
        for c in fp.columns
        if c.kind in {"continuous", "binary"} and c.name not in _excl | {outcome}
    ][:6]
    if outcome is None or not predictors:
        summary.append("多项 logit 失败：需要 3–10 类名义结果变量 + ≥1 个连续/二值预测变量。")
    else:
        try:
            sub = df[[outcome, *predictors]].dropna()
            codes, cats = pd.factorize(sub[outcome])
            X = sm.add_constant(sub[predictors].astype(float))
            model = sm.MNLogit(codes, X).fit(disp=False)
            (d / "summary.txt").write_text(str(model.summary()), encoding="utf-8")
            files.append("summary.txt")
            params, pvals = model.params, model.pvalues
            rrr = np.exp(params)
            rows = []
            for ci in params.columns:  # ci = 0..K-2 -> class cats[ci+1] vs baseline cats[0]
                cls = str(cats[ci + 1])
                for term in params.index:
                    rows.append(
                        (
                            cls,
                            str(term),
                            round(float(params.loc[term, ci]), 4),
                            round(float(rrr.loc[term, ci]), 4),
                            round(float(pvals.loc[term, ci]), 4),
                        )
                    )
            pd.DataFrame(
                rows, columns=["class_vs_baseline", "term", "coef", "RRR", "p_value"]
            ).to_csv(d / "coefficients.csv", index=False, encoding="utf-8")
            files.append("coefficients.csv")
            pred = np.asarray(model.predict(X))
            acc = float((pred.argmax(axis=1) == codes).mean())
            estimates["accuracy"] = round(acc, 4)
            estimates["n_classes"] = float(len(cats))
            estimates["pseudo_r2"] = round(float(model.prsquared), 4)
            summary.append(
                f"{entry.method} 完成：名义结果 {outcome}（{len(cats)} 类，基准={cats[0]}），"
                f"{len(predictors)} 个预测变量；类内准确率={acc:.1%}，"
                f"McFadden pseudo-R²={model.prsquared:.3f}；相对风险比(RRR)见 coefficients.csv。"
                "⚠ 假定结果无序（名义）——若类别有序请用 ordered_logit；并假定 IIA（无关方案独立性）。"
            )
            code += [
                "import statsmodels.api as sm  # 多项 logit",
                f"# codes,_=pd.factorize(df['{outcome}']); sm.MNLogit(codes, sm.add_constant(X)).fit()",
            ]
        except Exception as err:
            summary.append(f"多项 logit 失败：{err}")



@register("ordered_logit")
def _branch_ordered_logit(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import pandas as pd
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    _excl = {fp.unit_col, fp.time_col}
    # ordinal outcome: a small ordered scale (3–10 levels). Prefer numeric
    # (count) where the level order is unambiguous; fall back to categorical.
    ord_cols = [
        c
        for c in fp.columns
        if c.kind in {"count", "categorical"}
        and 3 <= c.n_unique <= 10
        and c.name not in _excl
    ]
    ord_cols.sort(key=lambda c: 0 if c.kind == "count" else 1)
    outcome = ord_cols[0].name if ord_cols else None

    if outcome is None:
        summary.append("有序 Logit 失败：未找到有序结果变量（3–10 个等级）。")
    else:
        exclude = {outcome, fp.unit_col, fp.time_col}
        predictors = [
            c.name
            for c in fp.columns
            if c.kind in {"continuous", "binary", "count"} and c.name not in exclude
        ][:5]
        try:
            if not predictors:
                raise ValueError("没有可用预测变量")
            yc = pd.Categorical(df[outcome], ordered=True)
            levels = list(yc.categories)
            model = OrderedModel(yc, df[predictors], distr="logit").fit(
                method="bfgs", disp=False
            )
            (d / "summary.txt").write_text(str(model.summary()), encoding="utf-8")
            files.append("summary.txt")
            # OrderedResults lacks summary2(); build the table from arrays.
            # Rows include predictor slopes plus threshold cutpoints. statsmodels
            # OrderedModel stores thresholds in an unconstrained space ([c1, log(c2-c1), …]);
            # transform the threshold rows to the TRUE ordered cutpoints (only the first
            # raw value is a cutpoint, the rest are log-increments). Their library SE/z/p
            # are increment-scale (not cutpoint-scale), so blank them out.
            import numpy as np

            _coef = model.params.astype(float).copy()
            _se = model.bse.astype(float).copy()
            _z = model.tvalues.astype(float).copy()
            _p = model.pvalues.astype(float).copy()
            _thr_keys = [i for i in _coef.index if i not in set(predictors)]
            try:
                _raw = np.asarray(model.params[_thr_keys], dtype=float)
                _cuts = [c for c in np.asarray(
                    model.model.transform_threshold_params(_raw), dtype=float) if np.isfinite(c)]
                if len(_cuts) == len(_thr_keys):
                    for _k, _c in zip(_thr_keys, _cuts):
                        _coef[_k] = _c
                        _se[_k] = np.nan
                        _z[_k] = np.nan
                        _p[_k] = np.nan
            except Exception:
                pass
            pd.DataFrame(
                {"coef": _coef, "std_err": _se, "z": _z, "P>|z|": _p}
            ).to_csv(d / "coefficients.csv", encoding="utf-8")
            files.append("coefficients.csv")
            _coef_plot(model, predictors, d / "coefficients.png")
            files.append("coefficients.png")
            _ordinal_prob_plot(model, df, predictors, levels, d / "predicted_probabilities.png")
            if (d / "predicted_probabilities.png").exists():
                files.append("predicted_probabilities.png")
            for v in predictors:
                if v in model.params.index:
                    estimates[v] = float(model.params[v])
            is_text = df[outcome].dtype == object or str(df[outcome].dtype) == "string"
            note = f"（等级顺序假定为 {levels}；若不符请重新编码）" if is_text else ""
            summary.append(
                f"{entry.method} 完成：有序结果 {outcome}（{len(levels)} 级），"
                f"{len(predictors)} 个预测变量{note}"
            )
            code += [
                "from statsmodels.miscmodels.ordinal_model import OrderedModel",
                f"yc = pd.Categorical(df['{outcome}'], ordered=True)",
                f"model = OrderedModel(yc, df[{predictors!r}], distr='logit')"
                ".fit(method='bfgs', disp=False)",
                "print(model.summary())",
            ]
        except Exception as err:
            summary.append(f"有序 Logit 未收敛/失败：{err}")
