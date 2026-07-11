"""Branch handler: mixed_effects (statistics family).

Each handler unpacks ctx into the same local names run_analysis used and runs the
original branch body verbatim. See executor/_branch_api.py.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor._helpers.formula import safe_formula_terms


@register("mixed_effects")
def _branch_mixed_effects(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import statsmodels.formula.api as smf

    # outcome: first continuous column
    outcome = next((c.name for c in fp.columns if c.kind == "continuous"), None)
    if outcome is None:
        summary.append("混合模型失败：未找到连续结果变量。")
    else:
        # group_col: prefer unit_col; else first categorical/binary that is not outcome
        if fp.unit_col:
            group_col = fp.unit_col
        else:
            group_col = next(
                (
                    c.name
                    for c in fp.columns
                    if c.kind in {"categorical", "binary"} and c.name != outcome
                ),
                None,
            )
        if group_col is None:
            summary.append("混合模型失败：未找到分组变量(随机效应)。")
        else:
            # Wave K-B3: categorical fixed effects used to be silently dropped
            # (only continuous/count/binary were collected) — a treatment factor
            # like "variety"/"处理" with >2 levels vanished from the model without
            # any disclosure. Now categorical predictors are kept and dummy-coded
            # in the formula via `C(...)`.
            # Wave K-B4b: formula identifiers now go through the shared
            # safe_formula_terms() helper (aliasing non-identifier-safe names like
            # Chinese columns to v1/v2/…) instead of the earlier ad-hoc
            # `Q('col')`/`C(Q('col'))` quoting — a single collection point; the
            # alias->original mapping restores the original names in
            # coefficients.csv/estimates below. group_col itself is passed
            # straight through via `groups=sub[group_col]` (not formula-parsed),
            # so it needs no aliasing.
            _excl_names = {outcome, group_col, fp.unit_col, fp.time_col}
            predictor_cols = [
                c for c in fp.columns
                if c.kind in {"continuous", "count", "binary", "categorical"}
                and c.name not in _excl_names
                # skip a single-level categorical: C() yields zero dummies → a silent
                # "完成 but no estimate" pseudo-success (B3 冷审 NICE)
                and not (c.kind == "categorical" and df[c.name].nunique(dropna=True) < 2)
            ][:5]
            predictors = [c.name for c in predictor_cols]
            _time_in_formula = bool(fp.time_col) and fp.time_col != group_col
            _formula_cols = [outcome, *predictors] + ([fp.time_col] if _time_in_formula else [])
            terms, _ = safe_formula_terms(_formula_cols)
            term_outcome = terms[0]
            term_predictors = terms[1: 1 + len(predictors)]
            term_time = terms[1 + len(predictors)] if _time_in_formula else None
            col_map = dict(zip(_formula_cols, terms))  # 原名 -> 安全项（不含 group_col）

            rhs = [
                f"C({t})" if c.kind == "categorical" else t
                for c, t in zip(predictor_cols, term_predictors)
            ]
            # Control for time on panel data — otherwise a staggered treatment is
            # confounded with the time trend (mirrors _regression's FE handling).
            if _time_in_formula:
                rhs.append(f"C({term_time})")

            if not rhs:
                # Degenerate to an intercept-only "model" (no fixed-effect
                # predictors survived filtering, and no time control either) —
                # this used to still report "完成" (looked like a fitted model
                # with nothing to say). Report failure instead of a misleading
                # success.
                summary.append("mixed_effects 失败：无可用固定效应预测变量")
            else:
                formula = f"{term_outcome} ~ " + " + ".join(rhs)
                # Wave K-B3 冷审(SHOULD)：mixedlm.from_formula 默认 missing='none'，patsy 丢 NaN
                # 行后 endog 变短、groups 数组仍全长 → IndexError，被下方 except 谎报"未收敛"。
                # 分类预测变量更常带缺失（B3 放大此隐患），故拟合前手动 listwise 删除对齐。
                _acols = [outcome, group_col, *predictors]
                if _time_in_formula:
                    _acols.append(fp.time_col)
                sub_orig = df[[c for c in dict.fromkeys(_acols) if c in df.columns]].dropna().reset_index(drop=True)
                if len(sub_orig) < 20 or sub_orig[group_col].nunique() < 2:
                    summary.append("mixed_effects 失败：删除缺失后有效样本(<20)或分组(<2)不足")
                else:
                    try:
                        sub = sub_orig.rename(columns=col_map)
                        model = smf.mixedlm(formula, data=sub, groups=sub[group_col]).fit()
                        (d / "summary.txt").write_text(str(model.summary()), encoding="utf-8")
                        files.append("summary.txt")

                        # (安全项, 原名) 配对 + 还原函数：把 C(term)[T.level] / term[T.level]
                        # 哑变量后缀形式的行标签还原成原始列名，供 coefficients.csv/estimates 用。
                        _term_pairs = list(zip(term_predictors, predictors))
                        if _time_in_formula:
                            _term_pairs.append((term_time, fp.time_col))

                        def _delabel(idx: str) -> str:
                            for t, o in _term_pairs:
                                if idx == t:
                                    return o
                                if idx.startswith(f"C({t})["):
                                    return f"{o}{idx[len(f'C({t})'):]}"
                                if idx.startswith(f"{t}["):
                                    return f"{o}{idx[len(t):]}"
                            return idx

                        try:
                            import pandas as pd
                            coefs_df = pd.DataFrame(model.summary().tables[1])
                            coefs_df.index = [_delabel(i) for i in coefs_df.index]
                            coefs_df.to_csv(d / "coefficients.csv", encoding="utf-8")
                        except Exception:
                            import pandas as pd
                            fallback = model.params.to_frame(name="coef")
                            fallback.index = [_delabel(i) for i in fallback.index]
                            fallback.to_csv(d / "coefficients.csv", encoding="utf-8")
                        files.append("coefficients.csv")
                        _n_cat = 0
                        for c, t in zip(predictor_cols, term_predictors):
                            v = c.name
                            if c.kind == "categorical":
                                prefix = f"C({t})["
                                hit = False
                                for idx in model.params.index:
                                    if idx.startswith(prefix):
                                        estimates[_delabel(idx)] = float(model.params[idx])
                                        hit = True
                                if hit:
                                    _n_cat += 1
                            else:
                                # 数值预测变量用精确键；但**字符串编码的二值列**被 patsy 自动
                                # 哑变量化成 term[T.level]，精确键落空 → 回退前缀扫描，免得系数
                                # 静默漏出 estimates（B3 冷审 SHOULD）。
                                if t in model.params.index:
                                    estimates[v] = float(model.params[t])
                                else:
                                    for idx in model.params.index:
                                        if idx.startswith(f"{t}["):
                                            estimates[_delabel(idx)] = float(model.params[idx])
                        _cat_note = (
                            f"（含 {_n_cat} 个分类固定效应已哑变量化，以各自首水平为参照）" if _n_cat else ""
                        )
                        summary.append(
                            f"{entry.method} 完成：结果变量 {outcome}，随机效应分组 {group_col}，"
                            f"固定效应 {len(predictors)} 个{_cat_note}"
                        )
                        code += [
                            "import statsmodels.formula.api as smf",
                            f"sub = df.dropna(subset={[outcome, group_col, *predictors]!r})"
                            f".rename(columns={col_map!r})  # 对齐 groups/endog + formula 标识符别名化",
                            f'model = smf.mixedlm("{formula}", data=sub, groups=sub["{group_col}"]).fit()',
                            "print(model.summary())",
                        ]
                    except Exception as err:
                        summary.append(f"混合模型未收敛/失败：{err}")
