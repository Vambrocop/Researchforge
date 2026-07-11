"""Branch handlers for the statistics family (migrated from the run.py monolith).

Each handler unpacks ctx into the same local names run_analysis used and runs the
original branch body verbatim. See executor/_branch_api.py.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor._helpers.diagnostics import suspicious_fit_warnings
from researchforge.executor.run import (
    _coef_plot,
    _gam_via_r,
    _gamm_via_r,
    _glmm_via_r,
    _heatmap,
    _ordinal_prob_plot,
    _plotly_corr_heatmap,
    _quantile_process_plot,
    _varimax,
    resolve_outcome,
)


@register("correlation")
def _branch_correlation(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    num = df.select_dtypes(include="number")
    corr = num.corr()
    corr.to_csv(d / "correlation.csv", encoding="utf-8")
    files.append("correlation.csv")
    _heatmap(corr, d / "correlation_heatmap.png")
    files.append("correlation_heatmap.png")
    _plotly_corr_heatmap(corr, d / "correlation_heatmap.html")
    if (d / "correlation_heatmap.html").exists():
        files.append("correlation_heatmap.html")
    summary.append(f"相关分析完成：{num.shape[1]} 个数值变量")
    code += ["num = df.select_dtypes(include='number')", "num.corr().to_csv('correlation.csv')"]



@register("descriptive_stats")
def _branch_descriptive_stats(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    df.describe(include="all").transpose().to_csv(d / "table_describe.csv", encoding="utf-8")
    files.append("table_describe.csv")
    summary.append(f"描述统计完成：{df.shape[0]} 行 × {df.shape[1]} 列")
    high_card = [c.name for c in fp.columns if c.kind in {"id", "categorical"} and c.n_unique > 50]
    if high_card:
        summary.append(f"注意：{len(high_card)} 个高基数列（如 {high_card[0]}）描述统计意义有限。")
    code.append("df.describe(include='all').transpose().to_csv('table_describe.csv')")



@register("efa")
def _branch_efa(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    _excl = {fp.unit_col, fp.time_col}
    items = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
    if len(items) < 3:
        summary.append("EFA 失败：需要 ≥3 个连续变量（题项）。")
    else:
        try:
            from sklearn.decomposition import FactorAnalysis
            from sklearn.preprocessing import StandardScaler

            sub = df[items].dropna()
            z = StandardScaler().fit_transform(sub.to_numpy(dtype=float))
            # sampling adequacy (factor_analyzer's KMO/Bartlett work despite its
            # FactorAnalyzer.fit sklearn-compat bug; extraction via sklearn instead).
            kmo = bartlett_p = float("nan")
            try:
                from factor_analyzer.factor_analyzer import (
                    calculate_bartlett_sphericity,
                    calculate_kmo,
                )

                _, bartlett_p = calculate_bartlett_sphericity(sub)
                _, kmo = calculate_kmo(sub)
            except Exception:
                pass
            ev = np.sort(np.linalg.eigvalsh(np.corrcoef(z, rowvar=False)))[::-1]
            n_factors = max(1, int((ev > 1).sum()))  # Kaiser criterion
            fa = FactorAnalysis(n_components=n_factors, random_state=0).fit(z)
            load = _varimax(fa.components_.T)  # (items, factors), varimax-rotated
            ssl = (load**2).sum(axis=0)  # SS loadings per factor
            prop_var = ssl / len(items)
            load_df = pd.DataFrame(
                np.round(load, 4),
                index=items,
                columns=[f"F{i + 1}" for i in range(n_factors)],
            )
            load_df.to_csv(d / "loadings.csv", encoding="utf-8")
            files.append("loadings.csv")
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(range(1, len(ev) + 1), ev, "-o", color="#4C72B0")
                ax.axhline(1.0, color="#C44E52", ls="--", lw=0.8, label="Kaiser (eigenvalue=1)")
                ax.set_xlabel("factor")
                ax.set_ylabel("eigenvalue")
                ax.set_title(f"Scree plot (retained {n_factors} factors)")
                ax.legend(fontsize=8)
                fig.tight_layout()
                fig.savefig(d / "scree.png", dpi=150)
                plt.close(fig)
                files.append("scree.png")
            except Exception:
                pass
            estimates["n_factors"] = float(n_factors)
            estimates["kmo"] = round(float(kmo), 4) if kmo == kmo else -1.0
            estimates["cumulative_variance"] = round(float(prop_var.sum()), 4)
            kmo_txt = (
                f"KMO={kmo:.3f}（{'适合因子分析' if kmo >= 0.6 else '⚠ <0.6 数据不太适合'}）"
                if kmo == kmo
                else "KMO 不可用"
            )
            bart_txt = f"，Bartlett p={bartlett_p:.2g}" if bartlett_p == bartlett_p else ""
            summary.append(
                f"{entry.method} 完成：{len(items)} 个变量 → Kaiser 准则保留 {n_factors} 个因子"
                f"（累计解释方差 {prop_var.sum():.1%}）；{kmo_txt}{bart_txt}；varimax 旋转载荷见 loadings.csv。"
                "⚠ 因子数(特征值>1)是启发式,碎石/平行分析可能不同；EFA 为探索性,确证用 SEM/CFA；"
                "提取用 sklearn FactorAnalysis(factor_analyzer 提取器与本机 sklearn 不兼容)。"
            )
            code += [
                "from sklearn.decomposition import FactorAnalysis  # EFA",
                "# 标准化 -> 相关阵特征值定 Kaiser n -> FactorAnalysis -> 手写 varimax 旋转",
            ]
        except Exception as err:
            summary.append(f"EFA 失败：{err}")



@register("gam")
def _branch_gam(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import re

    from researchforge.executor import rbridge

    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in {fp.unit_col, fp.time_col}]
    y = cfg["outcome"] if cfg.get("outcome") in cont else (cont[0] if cont else None)
    forced = [c for c in (cfg.get("predictors") or []) if c in df.columns and c != y]
    if forced:
        preds = forced[:8]
    else:
        preds = [
            c.name for c in fp.columns
            if c.kind in {"continuous", "count", "binary"} and c.name not in {y, fp.unit_col, fp.time_col}
        ][:6]
    # smooth continuous predictors with enough distinct values (mgcv s() needs
    # ~>=10 unique points); binary / low-cardinality enter as parametric terms.
    smooth = [p for p in preds if p in cont and df[p].dropna().nunique() >= 10]
    linear = [p for p in preds if p not in smooth]
    names_safe = y is not None and all(
        re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", str(c)) for c in [y, *preds]
    )
    if y is None or not preds:
        summary.append("GAM 失败：需要 1 个连续结果变量 + ≥1 个预测变量。")
    elif not smooth:
        summary.append(
            "GAM 跳过：没有可平滑的连续预测变量（需 ≥10 个不同取值）。"
            "用 ols_regression（线性）或确认预测变量为连续型。"
        )
    elif not (rbridge.r_available() and rbridge.r_package_available("mgcv")):
        summary.append("GAM 需要 R 的 mgcv 包（未检测到）。安装：install.packages('mgcv')；或用 ols_regression。")
    elif not names_safe:
        summary.append("GAM 失败：列名需为标识符式（字母/数字/. _），R 公式要求。")
    else:

        sub = df[[y, *preds]].dropna()
        csv = d / "_gam_input.csv"
        sub.to_csv(csv, index=False)
        try:
            smooth_df, param_df, fit = _gam_via_r(csv, y, smooth, linear, d / "gam_smooths.png")
            if len(smooth_df):
                smooth_df.to_csv(d / "smooth_terms.csv", index=False, encoding="utf-8")
                files.append("smooth_terms.csv")
            if len(param_df):
                param_df.to_csv(d / "parametric_terms.csv", index=False, encoding="utf-8")
                files.append("parametric_terms.csv")
            if (d / "gam_smooths.png").exists():
                files.append("gam_smooths.png")
            dev, r2, n = fit["dev_expl"], fit["r_sq"], int(fit["n"])
            estimates["deviance_explained"] = round(dev, 4)
            estimates["adj_r_squared"] = round(r2, 4)
            estimates["n"] = float(n)
            # edf>~1 means the term bends away from a straight line (nonlinear)
            nonlin = []
            for _, r in smooth_df.iterrows():
                estimates[f"edf_{r['term']}"] = round(float(r["edf"]), 3)
                if float(r["edf"]) > 1.5 and float(r["p_value"]) < 0.05:
                    nonlin.append(str(r["term"]))
            nl_txt = (
                f"显著非线性项：{nonlin}（edf>1.5 且 p<0.05，提示曲线关系）"
                if nonlin
                else "各平滑项 edf≈1 或不显著（近线性，可考虑普通回归）"
            )
            cc_txt = ""
            if "concurvity" in fit:
                estimates["worst_concurvity"] = round(fit["concurvity"], 3)
                if fit["concurvity"] > 0.8:
                    cc_txt = (
                        f"；⚠ 最差 concurvity={fit['concurvity']:.2f}>0.8（平滑项间强非线性共线），"
                        "单项偏效应/显著性不稳"
                    )
            sig_s = [str(r["term"]) for _, r in smooth_df.iterrows() if float(r["p_value"]) < 0.05]
            (d / "gam_summary.txt").write_text(
                f"广义可加模型 GAM（mgcv，REML）：{y} ~ "
                + " + ".join([f"s({t})" for t in smooth] + linear) + "\n"
                f"偏差解释 {dev:.1%}，调整 R² {r2:.3f}，n={n}\n"
                f"显著平滑项（p<0.05）：{sig_s}\n{nl_txt}\n"
                "edf=有效自由度（1=直线，越大越弯）；平滑项 p 检验该项整体是否≠0（非"
                "「是否非线性」的正式检验）——edf>1.5 仅为非线性的描述性标记。\n"
                "默认高斯族+identity link，假定结果连续无界、近似同方差正态。\n\n"
                "平滑项：\n" + (smooth_df.to_string(index=False) if len(smooth_df) else "（无）")
                + "\n\n参数项：\n" + (param_df.to_string(index=False) if len(param_df) else "（无）"),
                encoding="utf-8",
            )
            files.append("gam_summary.txt")
            summary.append(
                f"{entry.method} 完成（R/mgcv，REML）：{y} ~ {len(smooth)} 个平滑项 + "
                f"{len(linear)} 个线性项；偏差解释 {dev:.1%}，调整 R²={r2:.3f}（n={n}）；{nl_txt}{cc_txt}。"
                "⚠ 高斯族(连续无界结果)；平滑项默认薄板样条 k=10，边界/稀疏区外推不可靠；偏效应图 gam_smooths.png。"
            )
            _rhs_preview = " + ".join([f"s({t})" for t in smooth] + linear)
            code += [
                "library(mgcv)  # 广义可加模型 GAM",
                f"# gam({y} ~ {_rhs_preview}, method='REML'); summary()/plot(shade=TRUE)",
            ]
        except Exception as err:
            summary.append(f"GAM 拟合失败：{err}")
        finally:
            try:
                csv.unlink()
            except OSError:
                pass



@register("gamm")
def _branch_gamm(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import re

    from researchforge.executor import rbridge

    # grouping for the random intercept (config, else panel unit, else a
    # categorical/id column with real clustering — >=5 groups, like GLMM)
    group = cfg.get("group") or fp.unit_col
    if not group:
        group = next(
            (c.name for c in fp.columns if c.kind in {"categorical", "id"}
             and c.name not in {fp.unit_col, fp.time_col} and 5 <= c.n_unique < fp.n_rows),
            None,
        )
    _excl = {fp.unit_col, fp.time_col, group}
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
    # outcome may be continuous (gaussian), binary (binomial), or count (poisson). Default picks a
    # continuous outcome (gaussian); a non-Gaussian response is opt-in via config outcome/family.
    ykind = {c.name: c.kind for c in fp.columns if c.name not in _excl and c.kind in {"continuous", "count", "binary"}}
    y = cfg["outcome"] if cfg.get("outcome") in ykind else (cont[0] if cont else None)
    fam = cfg.get("family")
    if fam not in {"gaussian", "binomial", "poisson"}:
        fam = {"binary": "binomial", "count": "poisson"}.get(ykind.get(y), "gaussian")
    forced = [c for c in (cfg.get("predictors") or []) if c in df.columns and c not in {y, group}]
    if forced:
        preds = forced[:8]
    else:
        preds = [
            c.name for c in fp.columns
            if c.kind in {"continuous", "count", "binary"} and c.name not in {y, group, fp.unit_col, fp.time_col}
        ][:6]
    smooth = [p for p in preds if p in cont and df[p].dropna().nunique() >= 10]
    linear = [p for p in preds if p not in smooth]
    names_safe = y is not None and group is not None and all(
        re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", str(c)) for c in [y, group, *preds]
    )
    if y is None or not preds:
        summary.append("GAMM 失败：需要 1 个结果变量（连续/二值/计数）+ ≥1 个预测变量。")
    elif group is None:
        summary.append("GAMM 失败：需要一个分组变量做随机截距（面板单位列或重复出现的类别列，≥5 组）。")
    elif not smooth:
        summary.append(
            "GAMM 跳过：没有可平滑的连续预测变量（需 ≥10 个不同取值）。"
            "若只需随机效应用 mixed_effects；若只需平滑用 gam。"
        )
    elif not (rbridge.r_available() and rbridge.r_package_available("mgcv")):
        summary.append("GAMM 需要 R 的 mgcv 包（未检测到）。安装：install.packages('mgcv')。")
    elif not names_safe:
        summary.append("GAMM 失败：列名需为标识符式（字母/数字/. _），R 公式要求。")
    else:

        sub = df[[y, group, *preds]].dropna()
        csv = d / "_gamm_input.csv"
        sub.to_csv(csv, index=False)
        try:
            smooth_df, param_df, re_d, fit = _gamm_via_r(csv, y, smooth, linear, group, d / "gamm_smooths.png", family=fam)
            if len(smooth_df):
                smooth_df.to_csv(d / "smooth_terms.csv", index=False, encoding="utf-8")
                files.append("smooth_terms.csv")
            if len(param_df):  # surface the linear/parametric coefficients (Opus: were computed but unsaved)
                param_df.to_csv(d / "parametric_terms.csv", index=False, encoding="utf-8")
                files.append("parametric_terms.csv")
            if (d / "gamm_smooths.png").exists():
                files.append("gamm_smooths.png")
            # few-groups variance component is unreliable; warn when the group (config/unit) has <5 levels
            n_grp = int(sub[group].nunique())
            fewg_txt = f"；⚠ 仅 {n_grp} 组随机效应，方差分量不稳，RE-SD/显著性仅供参考" if n_grp < 5 else ""
            dev, r2, n = fit["dev_expl"], fit["r_sq"], int(fit["n"])
            re_sd = re_d.get("re_sd", float("nan"))
            re_p = re_d.get("re_p", float("nan"))
            fam_label = {"gaussian": "高斯族 + identity link（连续结果）",
                         "binomial": "二项族 + logit link（0/1 结果，逻辑可加模型）",
                         "poisson": "泊松族 + log link（计数结果）"}[fam]
            fam_short = {"gaussian": "高斯族", "binomial": "二项族/logit", "poisson": "泊松族/log"}[fam]
            estimates["deviance_explained"] = round(dev, 4)
            estimates["adj_r_squared"] = round(r2, 4)
            estimates["random_intercept_sd"] = round(re_sd, 4) if re_sd == re_sd else float("nan")
            estimates["n"] = float(n)
            nonlin = []
            for _, r in smooth_df.iterrows():
                estimates[f"edf_{r['term']}"] = round(float(r["edf"]), 3)
                if float(r["edf"]) > 1.5 and float(r["p_value"]) < 0.05:
                    nonlin.append(str(r["term"]))
            nl_txt = (
                f"显著非线性项：{nonlin}（edf>1.5 且 p<0.05）"
                if nonlin else "各平滑项近线性或不显著"
            )
            re_txt = f"随机截距 SD={re_sd:.3f}（按 {group} 聚类）" if re_sd == re_sd else f"随机截距（按 {group}）"
            (d / "gamm_summary.txt").write_text(
                f"广义可加混合模型 GAMM（mgcv，REML）：{y} ~ "
                + " + ".join([f"s({t})" for t in smooth] + linear) + f" + s({group}, bs='re')\n"
                f"偏差解释 {dev:.1%}，调整 R² {r2:.3f}，n={n}；{re_txt}，RE 显著性 p={re_p:.3g}\n"
                f"{nl_txt}{fewg_txt}\n"
                "edf=有效自由度（1=直线）；随机截距吸收组间基线差异。平滑项 p 检验项≠0"
                "（edf>1.5 仅为非线性的描述性标记，非正式检验）；RE 的 p/edf 为近似自由度检验。\n"
                + f"分布族：{fam_label}"
                + ("。\n\n" if fam == "gaussian"
                   else "；非高斯族下偏差解释为主、调整 R² 仅供参考，平滑项效应在 link 尺度上。\n\n")
                + "平滑项：\n" + (smooth_df.to_string(index=False) if len(smooth_df) else "（无）"),
                encoding="utf-8",
            )
            files.append("gamm_summary.txt")
            summary.append(
                f"{entry.method} 完成（R/mgcv，REML）：{y} ~ {len(smooth)} 个平滑项 + {len(linear)} 个线性项"
                f" + (1|{group})；{fam_short}；偏差解释 {dev:.1%}，调整 R²={r2:.3f}（n={n}）；{re_txt}；{nl_txt}。"
                f"⚠ {fam_short}"
                + ("（连续结果）" if fam == "gaussian" else "（平滑项在 link 尺度，偏差解释为主）")
                + "；平滑项默认薄板样条 k=10，边界外推不可靠；随机截距假定组效应~正态。" + fewg_txt
            )
            code += [
                "library(mgcv)  # 广义可加混合模型 GAMM",
                f"# gam({y} ~ " + " + ".join([f's({t})' for t in smooth] + linear) + f" + s({group}, bs='re'), family={fam}, method='REML')",
            ]
        except Exception as err:
            summary.append(f"GAMM 拟合失败：{err}")
        finally:
            try:
                csv.unlink()
            except OSError:
                pass



@register("glmm")
def _branch_glmm(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import re

    from researchforge.executor import rbridge

    _excl = {fp.unit_col, fp.time_col}
    binary = [c.name for c in fp.columns if c.kind == "binary" and c.name not in _excl]
    counts = [c.name for c in fp.columns if c.kind == "count" and c.name not in _excl]
    # outcome + family: config outcome (kind decides family), else binary->binomial,
    # else count->poisson. GLMM is the non-Gaussian complement of mixed_effects.
    cfg_out = cfg.get("outcome")
    if cfg_out in binary:
        outcome, family = cfg_out, "binomial"
    elif cfg_out in counts:
        outcome, family = cfg_out, "poisson"
    elif binary:
        outcome, family = binary[0], "binomial"
    elif counts:
        outcome, family = counts[0], "poisson"
    else:
        outcome, family = None, None
    # grouping for the random intercept: config group, else panel unit, else a
    # categorical/id column with real clustering (2..n-1 distinct values).
    group = cfg.get("group") or fp.unit_col
    if not group:
        group = next(
            (
                c.name for c in fp.columns
                if c.kind in {"categorical", "id"} and c.name not in _excl
                # >=5 groups for a stable variance component (Opus); config can override
                and 5 <= c.n_unique < fp.n_rows
            ),
            None,
        )
    forced = [c for c in (cfg.get("predictors") or []) if c in df.columns and c not in {outcome, group}]
    if forced:
        preds = forced[:6]
    else:
        preds = [
            c.name for c in fp.columns
            if c.kind in {"continuous", "count", "binary"}
            and c.name not in {outcome, group, fp.unit_col, fp.time_col}
        ][:5]
    names_safe = outcome and group and all(
        re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", str(c)) for c in [outcome, group, *preds]
    )
    if outcome is None:
        summary.append(
            "GLMM 失败：需要二值或计数型结果变量（高斯/连续结果用 mixed_effects 线性混合模型）。"
        )
    elif group is None:
        summary.append("GLMM 失败：需要一个分组变量做随机截距（面板单位列或重复出现的类别列）。")
    elif not preds:
        summary.append("GLMM 失败：需要 ≥1 个固定效应预测变量。")
    elif not (rbridge.r_available() and rbridge.r_package_available("lme4")):
        summary.append("GLMM 需要 R 的 lme4 包（未检测到）。安装：install.packages('lme4')；二值结果可先用 logistic_regression。")
    elif not names_safe:
        summary.append("GLMM 失败：列名需为标识符式（字母/数字/. _），R 公式要求。")
    else:
        import math


        sub = df[[outcome, group, *preds]].dropna()
        csv = d / "_glmm_input.csv"
        sub.to_csv(csv, index=False)
        try:
            fixed, re_d = _glmm_via_r(csv, outcome, preds, group, family)
            link = "logit（二值, 系数 exp→OR）" if family == "binomial" else "log（计数, 系数 exp→IRR）"
            fixed["exp_coef"] = fixed["estimate"].map(lambda b: round(math.exp(b), 4))
            fixed.to_csv(d / "fixed_effects.csv", index=False, encoding="utf-8")
            files.append("fixed_effects.csv")
            gv, gsd = re_d["group_var"], re_d["group_sd"]
            ng, nobs = int(re_d["n_groups"]), int(re_d["n_obs"])
            aic, od = re_d["aic"], re_d.get("overdispersion", float("nan"))
            estimates["group_intercept_var"] = round(gv, 4)
            estimates["n_groups"] = float(ng)
            estimates["aic"] = round(aic, 2)
            for _, r in fixed.iterrows():
                if "Intercept" not in str(r["term"]):
                    estimates[str(r["term"])] = round(float(r["estimate"]), 4)
            icc_txt = ""
            if "icc" in re_d:
                estimates["icc"] = round(re_d["icc"], 4)
                icc_txt = (
                    f"，组内相关 ICC={re_d['icc']:.3f}（潜变量 logit 尺度，"
                    f"{re_d['icc']:.0%} 方差在组间；观测 0/1 尺度通常更低）"
                )
            od_txt = ""
            if family == "poisson" and od == od:  # NaN-safe
                estimates["overdispersion"] = round(od, 3)
                # ratio & residual df are approximate for a mixed model (Bolker GLMM-FAQ)
                od_txt = (
                    f"；⚠ 过离散≈{od:.2f}>1.5（近似），泊松假定可能被违反，考虑负二项 GLMM"
                    if od > 1.5 else f"；过离散≈{od:.2f}（近似，≈1 可接受）"
                )
            sig = [str(r["term"]) for _, r in fixed.iterrows()
                   if "Intercept" not in str(r["term"]) and float(r["p_value"]) < 0.05]
            (d / "glmm_summary.txt").write_text(
                f"广义线性混合模型 GLMM（lme4 glmer，{family}，{link}）\n"
                f"{outcome} ~ {' + '.join(preds)} + (1|{group})\n"
                f"随机截距方差={gv:.4f}（SD={gsd:.4f}），分组数={ng}，n={nobs}，AIC={aic:.1f}{icc_txt}{od_txt}\n"
                f"显著固定效应（p<0.05）：{sig}\n"
                "exp(系数)=OR(二值)/IRR(计数)；随机截距吸收组间基线差异。\n\n"
                + fixed.to_string(index=False),
                encoding="utf-8",
            )
            files.append("glmm_summary.txt")
            summary.append(
                f"{entry.method} 完成（R/lme4，{family}）：{outcome} ~ {len(preds)} 个固定效应 "
                f"+ (1|{group})；随机截距方差={gv:.3f}（{ng} 组，n={nobs}）{icc_txt}；"
                f"显著项 {sig}{od_txt}。⚠ exp(系数)=OR/IRR；随机截距假定组效应~正态。"
            )
            code += [
                "library(lme4)  # 广义线性混合模型 GLMM",
                f"# glmer({outcome} ~ {' + '.join(preds)} + (1|{group}), family={family})",
            ]
        except Exception as err:
            summary.append(f"GLMM 拟合失败（可能不收敛）：{err}")
        finally:
            try:
                csv.unlink()
            except OSError:
                pass



def _welch_anova(groups: list) -> tuple[float, float, float, float] | None:
    """Welch's F-test for one-way ANOVA with unequal variances (Satterthwaite-
    corrected df; Welch 1951). Groups i=1..k with size n_i, mean m_i, sample
    variance v_i (ddof=1). Returns (F, p, df1, df2), or None if any group has
    zero within-group variance (a constant group makes the weight w_i = n_i/v_i
    infinite — degenerate, so callers should skip honestly instead)."""
    import numpy as np
    from scipy import stats as _stats

    k = len(groups)
    ns = np.array([len(g) for g in groups], dtype=float)
    means = np.array([np.mean(g) for g in groups], dtype=float)
    variances = np.array([np.var(g, ddof=1) for g in groups], dtype=float)

    if np.any(variances == 0):
        return None

    w = ns / variances
    W = w.sum()
    m_bar = (w * means).sum() / W
    numer = (w * (means - m_bar) ** 2).sum() / (k - 1)
    A = ((1 - w / W) ** 2 / (ns - 1)).sum()
    denom = 1 + (2 * (k - 2) / (k**2 - 1)) * A
    F = numer / denom
    df1 = float(k - 1)
    df2 = (k**2 - 1) / (3 * A)
    p = float(_stats.f.sf(F, df1, df2))
    return float(F), p, df1, float(df2)


@register("group_comparison")
def _branch_group_comparison(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    from scipy import stats

    _excl = {fp.unit_col, fp.time_col}
    bin_cols = [c.name for c in fp.columns if c.kind == "binary" and c.name not in _excl]
    cat_cols = [c.name for c in fp.columns if c.kind == "categorical" and c.name not in _excl]
    # prefer a binary group; otherwise the lowest-cardinality categorical, so a
    # high-cardinality unit/id column is never picked as the grouping variable.
    cat_cols.sort(key=lambda name: int(df[name].nunique()))
    group_candidates = bin_cols + cat_cols
    cont_cols = [c.name for c in fp.columns if c.kind == "continuous"]
    group_col = group_candidates[0] if group_candidates else None
    outcome = cont_cols[0] if cont_cols else None

    if group_col is None or outcome is None:
        summary.append("组间比较失败：未找到分组变量或连续结果变量。")
    else:
        # Per-group means/counts
        group_means = df.groupby(group_col)[outcome].agg(["mean", "count", "std"])
        group_means.to_csv(d / "group_means.csv", encoding="utf-8")
        files.append("group_means.csv")

        # Split outcome by group levels, drop NaN
        levels = df[group_col].dropna().unique().tolist()
        groups = [df.loc[df[group_col] == lv, outcome].dropna().values for lv in levels]
        n_groups = len(groups)

        # Guard: a group with <2 non-null observations makes var(ddof=1) NaN —
        # skip honestly instead of reporting "统计量=nan, p=nan" as a result.
        _too_small = [str(lv) for lv, g in zip(levels, groups) if len(g) < 2]
        if _too_small:
            summary.append(
                f"组间比较失败：分组变量 {group_col} 下的组 {', '.join(_too_small)} "
                "样本量 <2，组内方差不可估计，已跳过该比较（不报告 NaN 统计量）。"
            )
        else:
            var_note = ""
            welch_degenerate = False
            if n_groups == 2:
                stat, p = stats.ttest_ind(groups[0], groups[1], equal_var=False)
                test_name = "Welch t-test"
            else:
                # k>=3: Welch's ANOVA is used unconditionally (variance-robust by
                # default — a two-stage "test Levene then pick a test" procedure
                # has poor error control; Delacre, Leys & Lakens 2019). Levene's
                # test is still computed and disclosed, but purely as a diagnostic
                # explaining *why* Welch is the default, not as a gate.
                welch_result = _welch_anova(groups)
                try:
                    lstat, lp = stats.levene(*groups)
                    levene_note = f"Levene 方差齐性检验：统计量={lstat:.4f}，p={lp:.3g}"
                except Exception as err:
                    levene_note = f"方差齐性检验(Levene)未能完成：{err}"

                if welch_result is None:
                    welch_degenerate = True
                else:
                    stat, p, welch_df1, welch_df2 = welch_result
                    test_name = "Welch 稳健单因素方差分析（不假定方差齐性）"
                    estimates["welch_df1"] = welch_df1
                    estimates["welch_df2"] = welch_df2
                    var_note = (
                        "⚠ 已默认用 Welch 稳健单因素方差分析（不假定方差齐性）；"
                        f"{levene_note}"
                        "（<0.05 时方差不齐，正是默认用 Welch 的原因；≥0.05 亦不改变默认）。"
                    )

            if welch_degenerate:
                summary.append(
                    f"组间比较失败：分组变量 {group_col} 下至少一组结果变量 {outcome} 组内方差为 0"
                    "（该组所有观测值相同），Welch 方差分析的组权重不可定义，"
                    "已跳过该比较（不报告 inf/NaN 统计量）。"
                )
            else:
                estimates["statistic"] = float(stat)
                estimates["pvalue"] = float(p)

                # Boxplot
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(5, 4))
                    plot_data = [df.loc[df[group_col] == lv, outcome].dropna().values for lv in levels]
                    ax.boxplot(plot_data, tick_labels=[str(lv) for lv in levels])
                    ax.set_xlabel(group_col)
                    ax.set_ylabel(outcome)
                    ax.set_title(f"{outcome} by {group_col}")
                    fig.tight_layout()
                    fig.savefig(d / "boxplot.png", dpi=150)
                    plt.close(fig)
                    files.append("boxplot.png")
                except Exception:
                    pass

                summary.append(
                    f"{entry.method} 完成：{outcome} 按 {group_col} 分 {n_groups} 组，"
                    f"统计量={stat:.4f}，p={p:.3g}"
                    + (f"。{var_note}" if var_note else "")
                )
                code += [
                    "from scipy import stats",
                    f"groups = [df.loc[df['{group_col}'] == lv, '{outcome}'].dropna().values",
                    f"         for lv in df['{group_col}'].dropna().unique()]",
                    "stat, p = stats.ttest_ind(*groups[:2], equal_var=False)  # or Welch's ANOVA for k>=3",
                    "print(f'statistic={stat:.4f}, p={p:.3g}')",
                ]



@register("logistic_regression")
def _branch_logistic_regression(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
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
    predictors = [c for c in (cfg.get("predictors") or []) if c in df.columns and c != outcome] or [
        c.name
        for c in fp.columns
        if c.kind in {"continuous", "count"} and c.name not in exclude
    ][:5]

    if outcome is None:
        summary.append("逻辑回归失败：未找到二值结果变量。")
    else:
        rhs = [f"Q('{v}')" for v in predictors]
        formula = f"Q('{outcome}') ~ " + (" + ".join(rhs) if rhs else "1")
        try:
            model = smf.logit(formula, data=df).fit(disp=False)
            (d / "summary.txt").write_text(str(model.summary()), encoding="utf-8")
            files.append("summary.txt")
            model.summary2().tables[1].to_csv(d / "coefficients.csv", encoding="utf-8")
            files.append("coefficients.csv")
            _coef_plot(model, predictors, d / "coefficients.png")
            files.append("coefficients.png")
            for v in predictors:
                kn = f"Q('{v}')"
                if kn in model.params.index:
                    estimates[v] = float(model.params[kn])
            key = ""
            if predictors:
                kname = f"Q('{predictors[0]}')"
                if kname in model.params.index:
                    key = f"，关键系数 {predictors[0]} = {model.params[kname]:.4f} (p={model.pvalues[kname]:.3g})"
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
                f'model = smf.logit("{formula}", data=df).fit(disp=False)',
                "print(model.summary())",
            ]
        except Exception as err:
            summary.append(f"逻辑回归未收敛/失败：{err}")



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
            predictors = [
                c.name
                for c in fp.columns
                if c.kind in {"continuous", "count", "binary"}
                and c.name not in {outcome, group_col, fp.unit_col, fp.time_col}
            ][:5]
            rhs = [f"Q('{v}')" for v in predictors]
            # Control for time on panel data — otherwise a staggered treatment is
            # confounded with the time trend (mirrors _regression's FE handling).
            if fp.time_col and fp.time_col != group_col:
                rhs.append(f"C(Q('{fp.time_col}'))")
            formula = f"Q('{outcome}') ~ " + (" + ".join(rhs) if rhs else "1")
            try:
                model = smf.mixedlm(formula, data=df, groups=df[group_col]).fit()
                (d / "summary.txt").write_text(str(model.summary()), encoding="utf-8")
                files.append("summary.txt")
                try:
                    import pandas as pd
                    pd.DataFrame(model.summary().tables[1]).to_csv(
                        d / "coefficients.csv", encoding="utf-8"
                    )
                except Exception:
                    import pandas as pd
                    model.params.to_frame(name="coef").to_csv(
                        d / "coefficients.csv", encoding="utf-8"
                    )
                files.append("coefficients.csv")
                for v in predictors:
                    kn = f"Q('{v}')"
                    if kn in model.params.index:
                        estimates[v] = float(model.params[kn])
                summary.append(
                    f"{entry.method} 完成：结果变量 {outcome}，随机效应分组 {group_col}，"
                    f"固定效应 {len(predictors)} 个"
                )
                code += [
                    "import statsmodels.formula.api as smf",
                    f'model = smf.mixedlm("{formula}", data=df, groups=df["{group_col}"]).fit()',
                    "print(model.summary())",
                ]
            except Exception as err:
                summary.append(f"混合模型未收敛/失败：{err}")



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

