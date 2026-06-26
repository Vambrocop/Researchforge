"""Causal family branch handler: iv_regression — instrumental variables (2SLS).

Runs real two-stage least squares via linearmodels IV2SLS WHEN the user supplies an
excluded instrument through config (the engine cannot auto-pick a valid instrument —
that requires domain knowledge of the exclusion restriction — so without one it stays
honest guidance). With a config instrument it estimates the causal coefficient of the
endogenous regressor, reports first-stage strength (weak-instrument F), the Wu-Hausman
endogeneity test, the Sargan overidentification test (when overidentified), and the
naive-OLS contrast. Graceful degrade if linearmodels is missing.
"""
from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


def _as_list(v):
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


@register("iv_regression")
def _branch_iv_regression(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    import numpy as np
    import pandas as pd

    excl = {fp.unit_col, fp.time_col}
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in excl]

    # instruments are REQUIRED to actually run (no valid auto-pick) — config
    # instrument / instruments. Without them, stay honest guidance.
    instruments = [c for c in (_as_list(cfg.get("instruments")) or _as_list(cfg.get("instrument")))
                   if c in df.columns]
    if not instruments:
        summary.append(
            "工具变量回归（2SLS）需要你指定外生工具变量——引擎无法自动识别有效工具"
            "（需满足排除约束的领域知识）。请用 config={\"instrument\":\"<工具列>\","
            "\"endogenous\":\"<内生回归元>\",\"outcome\":\"<结果>\"} 指定后即自动运行 2SLS"
            "（可多工具：\"instruments\":[...]）；或先用 panel_fixed_effects / did 作可自动执行的替代。"
        )
        return

    try:
        from linearmodels.iv import IV2SLS
    except Exception:
        summary.append(
            "工具变量回归（2SLS）跳过：未安装 linearmodels（pip install linearmodels）。"
            "config 已给工具变量，装包后即可运行；或用 panel_fixed_effects / did 替代。"
        )
        return

    # outcome = config else first continuous (not an instrument).
    outcome = cfg.get("outcome")
    if outcome not in df.columns:
        outcome = next((c for c in cont if c not in instruments), None)
    if outcome is None:
        summary.append("工具变量回归（2SLS）失败：未找到连续结果变量（outcome）。")
        return

    # endogenous regressor(s) = config else first continuous not outcome/instrument.
    endog = [c for c in _as_list(cfg.get("endogenous")) if c in df.columns and c != outcome]
    if not endog:
        endog = [c for c in cont if c not in instruments and c != outcome][:1]
    if not endog:
        summary.append("工具变量回归（2SLS）失败：未找到内生回归元（endogenous）。")
        return

    # order condition: need >= as many instruments as endogenous regressors.
    if len(instruments) < len(endog):
        summary.append(
            f"工具变量回归（2SLS）失败：欠识别——工具变量（{len(instruments)} 个）少于"
            f"内生回归元（{len(endog)} 个）。请至少提供与内生回归元等量的工具。"
        )
        return

    # exogenous controls = config controls/exogenous, else remaining numeric covariates.
    forced_ctrl = _as_list(cfg.get("controls")) or _as_list(cfg.get("exogenous"))
    used = set(endog) | set(instruments) | {outcome, fp.unit_col, fp.time_col}
    if forced_ctrl:
        controls = [c for c in forced_ctrl if c in df.columns and c not in used]
    else:
        controls = [c.name for c in fp.columns
                    if c.kind in {"continuous", "count", "binary"} and c.name not in used][:10]

    allcols = [outcome, *endog, *instruments, *controls]
    sub = df[allcols].apply(lambda s: pd.to_numeric(s, errors="coerce")).dropna()
    if len(sub) < 30:
        summary.append("工具变量回归（2SLS）跳过：有效样本不足（去缺失后 < 30 行）。")
        return
    sub = sub.copy()
    sub["const"] = 1.0

    try:
        exog_cols = ["const", *controls]
        res = IV2SLS(dependent=sub[outcome], exog=sub[exog_cols],
                     endog=sub[endog], instruments=sub[instruments]).fit(cov_type="robust")

        params, se, pvals = res.params, res.std_errors, res.pvalues
        ci = res.conf_int()

        # naive OLS (treats the endogenous regressor as exogenous) — the biased baseline.
        try:
            import statsmodels.api as sm
            ols = sm.OLS(sub[outcome].to_numpy(float),
                         sm.add_constant(sub[[*controls, *endog]].to_numpy(float))).fit()
            naive = {c: float(ols.params[i + 1 + len(controls)]) for i, c in enumerate(endog)}
        except Exception:
            naive = {c: float("nan") for c in endog}

        # coefficient table
        rows = []
        for term in params.index:
            rows.append({
                "term": term,
                "coef": round(float(params[term]), 6),
                "se": round(float(se[term]), 6),
                "ci_low": round(float(ci.loc[term, "lower"]), 6),
                "ci_high": round(float(ci.loc[term, "upper"]), 6),
                "p_value": round(float(pvals[term]), 6),
                "endogenous": term in endog,
                "naive_ols_coef": round(naive.get(term, float("nan")), 6) if term in endog else "",
            })
        pd.DataFrame(rows).to_csv(d / "iv_2sls_coefficients.csv", index=False, encoding="utf-8")
        files.append("iv_2sls_coefficients.csv")

        # estimates — headline causal coefficient(s) on the endogenous regressor(s) first.
        for c in endog:
            estimates[f"iv_coef_{c}"] = round(float(params[c]), 6)
            estimates[f"iv_coef_{c}_se"] = round(float(se[c]), 6)
            estimates[f"iv_coef_{c}_ci_low"] = round(float(ci.loc[c, "lower"]), 6)
            estimates[f"iv_coef_{c}_ci_high"] = round(float(ci.loc[c, "upper"]), 6)
            estimates[f"iv_coef_{c}_p"] = round(float(pvals[c]), 6)
            estimates[f"naive_ols_coef_{c}"] = round(naive.get(c, float("nan")), 6)

        # first-stage weak-instrument diagnostics (partial F per endogenous regressor).
        min_first_F = float("nan")
        try:
            fs = res.first_stage.diagnostics
            fcol = "f.stat" if "f.stat" in fs.columns else fs.columns[-3]
            for c in endog:
                if c in fs.index:
                    estimates[f"first_stage_F_{c}"] = round(float(fs.loc[c, fcol]), 4)
            min_first_F = float(np.nanmin([float(fs.loc[c, fcol]) for c in endog if c in fs.index]))
        except Exception:
            pass

        # Wu-Hausman endogeneity test (H0: regressor is exogenous → OLS fine).
        wh_p = float("nan")
        try:
            wh = res.wu_hausman()
            wh_p = float(wh.pval)
            estimates["wu_hausman_p"] = round(wh_p, 4)
        except Exception:
            pass

        # Sargan overidentification test (only when overidentified).
        sargan_p = float("nan")
        if len(instruments) > len(endog):
            try:
                sg = res.sargan
                sargan_p = float(sg.pval)
                estimates["sargan_p"] = round(sargan_p, 4)
            except Exception:
                pass

        estimates["n_obs"] = float(len(sub))
        estimates["n_instruments"] = float(len(instruments))
        estimates["n_endogenous"] = float(len(endog))

        # coefficient plot (English labels)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plot = [r for r in rows if r["term"] != "const"]
            fig, ax = plt.subplots(figsize=(5.4, 3.2))
            ys = range(len(plot))
            ax.errorbar([r["coef"] for r in plot], ys,
                        xerr=[[r["coef"] - r["ci_low"] for r in plot],
                              [r["ci_high"] - r["coef"] for r in plot]],
                        fmt="o", ecolor="grey", linestyle="none")
            for i, r in enumerate(plot):
                ax.plot(r["coef"], i, "o", color="#C44E52" if r["endogenous"] else "#4C72B0")
            ax.axvline(0, color="grey", ls="--")
            ax.set_yticks(list(ys))
            ax.set_yticklabels([r["term"] for r in plot])
            ax.set_xlabel("2SLS coefficient (95% CI; red = endogenous, IV-identified)")
            ax.set_title("Instrumental variables (2SLS)")
            fig.tight_layout()
            fig.savefig(d / "iv_2sls_coefficients.png", dpi=150)
            plt.close(fig)
            files.append("iv_2sls_coefficients.png")
        except Exception:
            pass

        head = endog[0]
        b_iv = float(params[head])
        b_ols = naive.get(head, float("nan"))
        has_F = (min_first_F == min_first_F)
        weak = (has_F and min_first_F < 10.0)
        f_txt = (f"第一阶段最弱工具 F={min_first_F:.4g}"
                 f"（{'⚠ <10 弱工具，2SLS 偏向 OLS、SE 不可靠' if weak else '≥10 工具够强'}）。"
                 if has_F else "第一阶段 F 未能提取（无法判断工具强度）。")
        wh_txt = ("内生性显著（Wu-Hausman 拒绝外生→IV 优于 OLS）" if (wh_p == wh_p and wh_p < 0.05)
                  else "未检出显著内生性（Wu-Hausman 不拒→OLS 可能已足够）" if wh_p == wh_p else "")
        sg_txt = ("、Sargan 过度识别不拒（工具有效）" if (sargan_p == sargan_p and sargan_p > 0.05)
                  else f"、⚠ Sargan 被拒(p={sargan_p:.3g}, 工具有效性存疑)" if sargan_p == sargan_p else "")
        weak_ci_note = (" ⚠ 弱工具下报告的稳健 CI 仍不可靠（cov_type='robust' 仅异方差稳健、非弱工具稳健）"
                        "——应改看 Anderson-Rubin/CLR 弱工具稳健区间。" if weak else "")
        summary.append(
            f"{entry.method} 完成（linearmodels 2SLS，稳健 SE，n={len(sub)}）：内生回归元={endog}，"
            f"工具={instruments}" + (f"，控制={controls}" if controls else "") + "。"
            f"因果系数 {head}：2SLS={b_iv:.4g}（朴素 OLS={b_ols:.4g}，差异即内生性偏误的纠正）。"
            f"{f_txt}{wh_txt}{sg_txt}。系数见 iv_2sls_coefficients.csv。"
            " ⚠ 2SLS 一致性依赖**排除约束**（工具只通过内生回归元影响结果、与误差无关）——此为不可检验的"
            "识别假设；弱工具会放大偏误与 SE；异质效应下 2SLS 估计的是工具相关子总体的 LATE（需单调性）。"
            + weak_ci_note
        )
        code += [
            "from linearmodels.iv import IV2SLS",
            f"# IV2SLS(dependent=y, exog=[const,{controls}], endog={endog}, instruments={instruments})",
            "res = IV2SLS(...).fit(cov_type='robust')  # res.first_stage / res.wu_hausman() / res.sargan",
        ]
    except Exception as err:
        summary.append(f"工具变量回归（2SLS）失败：{err}")
