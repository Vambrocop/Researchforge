"""Branch handlers: gam, gamm, glmm (statistics family).

Each handler unpacks ctx into the same local names run_analysis used and runs the
original branch body verbatim. See executor/_branch_api.py.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.run import _gam_via_r, _gamm_via_r, _glmm_via_r


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
