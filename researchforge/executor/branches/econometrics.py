"""Branch handlers for the econometrics family (migrated from the run.py monolith).

Each handler unpacks ctx into the same local names run_analysis used and runs the
original branch body verbatim. See executor/_branch_api.py.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.run import (
    _dynamic_gmm_via_r,
    _gmm_lags,
)


@register("dynamic_panel_gmm")
def _branch_dynamic_panel_gmm(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import re

    from researchforge.executor import rbridge

    if not (fp.unit_col and fp.time_col):
        summary.append("动态面板 GMM 失败：需要面板数据（单位列 + 时间列）。")
    else:
        y = next(
            (c.name for c in fp.columns if c.kind == "continuous" and c.name not in {fp.unit_col, fp.time_col}),
            None,
        )
        preds = [
            c.name
            for c in fp.columns
            if c.kind in {"continuous", "binary"} and c.name not in {y, fp.unit_col, fp.time_col}
        ][:5]
        names_safe = y is not None and all(
            re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", str(c))
            for c in [y, *preds, fp.unit_col, fp.time_col]
        )
        n_periods = int(df[fp.time_col].nunique())
        if y is None or not preds:
            summary.append("动态面板 GMM 失败：需要连续结果变量 + ≥1 个预测变量。")
        elif n_periods < 3:
            summary.append("动态面板 GMM 失败：需要 ≥3 个时间期（差分 GMM 与 AR 检验要求）。")
        elif not (rbridge.r_available() and rbridge.r_package_available("plm")):
            summary.append(
                "动态面板 GMM 需要 R 的 plm 包（未检测到）。安装：install.packages('plm')；"
                "或用 panel_fixed_effects / random_effects。"
            )
        elif not names_safe:
            summary.append("动态面板 GMM 失败：列名需为标识符式（字母/数字/. _）。")
        else:
            sub = df[[fp.unit_col, fp.time_col, y, *preds]].dropna()
            csv = d / "_gmm_input.csv"
            sub.to_csv(csv, index=False)
            try:
                import numpy as np

                _endo = [p for p in (cfg.get("endogenous") or []) if p in preds]
                _lo, _hi = _gmm_lags(cfg)
                coef, diag = _dynamic_gmm_via_r(
                    csv, fp.unit_col, fp.time_col, y, preds,
                    endogenous=_endo, gmm_lags=(_lo, _hi),
                )
                coef["term"] = [
                    f"lag_{y}" if str(t).startswith("lag(") else str(t) for t in coef["term"]
                ]
                coef.to_csv(d / "gmm_coefficients.csv", index=False, encoding="utf-8")
                files.append("gmm_coefficients.csv")
                sargan_p = diag.get("sargan_p", float("nan"))
                ar1_p = diag.get("ar1_p", float("nan"))
                ar2_p = diag.get("ar2_p", float("nan"))
                sargan_ok = sargan_p > 0.05
                ar2_ok = ar2_p > 0.05
                _endo_note = (
                    f"内生变量（用 lag {_lo}:{_hi} 工具）：{_endo}"
                    if _endo
                    else "全部协变量设为严格外生（可用 config endogenous 标出内生变量）"
                )
                if _lo < 2:
                    _endo_note += (
                        "；⚠ gmm_lags 起始<2 仅对前定(predetermined)变量有效，"
                        "滞后被解释变量已强制 lag≥2（差分方程中 lag1 为无效工具）"
                    )
                (d / "diagnostics.txt").write_text(
                    "动态面板 GMM (Arellano-Bond 差分 GMM, twosteps, Windmeijer 稳健 SE)\n"
                    f"工具集限 lag {_lo}-{_hi}（抑制工具过度增殖；多工具会抬高/弱化 Sargan）。\n"
                    f"{_endo_note}。\n"
                    f"Sargan 过度识别检验 p = {sargan_p:.4g}（注：Sargan 非稳健,非 Hansen J）"
                    f"（{'工具有效（不拒）' if sargan_ok else '被拒 → 工具集可疑'}）\n"
                    f"AR(1) p = {ar1_p:.4g}（差分后通常显著，正常）\n"
                    f"AR(2) p = {ar2_p:.4g} "
                    f"（{'无二阶自相关 → GMM 一致' if ar2_ok else '⚠ 有二阶自相关 → GMM 不一致'}）\n\n"
                    + coef.to_string(index=False),
                    encoding="utf-8",
                )
                files.append("diagnostics.txt")
                try:
                    import matplotlib

                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.errorbar(
                        coef["estimate"], range(len(coef)),
                        xerr=1.96 * coef["std_err"], fmt="o",
                    )
                    ax.axvline(0, color="grey", ls="--")
                    ax.set_yticks(range(len(coef)))
                    ax.set_yticklabels(coef["term"])
                    ax.set_xlabel("GMM coefficient (95% CI)")
                    ax.set_title("Dynamic panel GMM (Arellano-Bond)")
                    fig.tight_layout()
                    fig.savefig(d / "coefficients.png", dpi=150)
                    plt.close(fig)
                    files.append("coefficients.png")
                except Exception:
                    pass
                lag_rows = coef[coef["term"].str.startswith("lag_")]
                persistence = float(lag_rows.iloc[0]["estimate"]) if len(lag_rows) else float("nan")
                estimates["persistence_lag_coef"] = round(persistence, 4)
                estimates["sargan_p"] = round(sargan_p, 4)
                estimates["ar2_p"] = round(ar2_p, 4)
                for _, r in coef.iterrows():
                    if not str(r["term"]).startswith("lag_"):
                        estimates[str(r["term"])] = round(float(r["estimate"]), 4)
                valid = sargan_ok and ar2_ok
                summary.append(
                    f"{entry.method} 完成（R/plm，{n_periods} 期）：滞后被解释变量系数（持续性）"
                    f"={persistence:.3f}；Sargan p={sargan_p:.3g}"
                    f"（{'工具有效' if sargan_ok else '工具可疑'}），AR(2) p={ar2_p:.3g}"
                    f"（{'无二阶自相关' if ar2_ok else '⚠有二阶自相关、GMM不一致'}）"
                    f"{'' if valid else ' —— ⚠ 诊断未全通过,结果存疑'}。系数见 gmm_coefficients.csv。"
                    + (f"（内生变量 {_endo}、工具滞后 {_lo}:{_hi} 按 config 指定）" if (_endo or cfg.get("gmm_lags")) else "")
                )
                code += [
                    "library(plm)  # Arellano-Bond 差分 GMM",
                    f"# pgmm({y} ~ lag({y},1) + ... | lag({y},{_lo}:{_hi})[+ lag(endo)], model='twosteps')",
                ]
            except Exception as err:
                summary.append(f"动态面板 GMM 失败：{err}")
            finally:
                try:
                    csv.unlink()
                except OSError:
                    pass



@register("random_effects")
def _branch_random_effects(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    if not (fp.unit_col and fp.time_col):
        summary.append("随机效应模型失败：需要面板数据（单位列 + 时间列）。")
    else:
        import numpy as np
        import pandas as pd
        from scipy.stats import chi2

        y = next(
            (c.name for c in fp.columns if c.kind == "continuous" and c.name not in {fp.unit_col, fp.time_col}),
            None,
        )
        preds = [
            c.name
            for c in fp.columns
            if c.kind in {"continuous", "binary"} and c.name not in {y, fp.unit_col, fp.time_col}
        ][:6]
        if y is None or not preds:
            summary.append("随机效应模型失败：需要连续结果变量 + ≥1 个预测变量。")
        else:
            try:
                from linearmodels.panel import PanelOLS, RandomEffects

                dd = (
                    df[[fp.unit_col, fp.time_col, y, *preds]]
                    .dropna()
                    .drop_duplicates([fp.unit_col, fp.time_col])
                    .set_index([fp.unit_col, fp.time_col])
                )
                fe = PanelOLS(dd[y], dd[preds], entity_effects=True).fit(
                    cov_type="clustered", cluster_entity=True
                )
                re = RandomEffects(dd[y], dd[preds].assign(const=1.0)).fit(
                    cov_type="clustered", cluster_entity=True
                )
                # classic Hausman needs the EFFICIENT (unadjusted) covariances so
                # V_FE - V_RE is PSD; clustered/robust covs break the χ² basis (Opus
                # catch). Use unadjusted fits for the test; keep clustered for SEs.
                fe_u = PanelOLS(dd[y], dd[preds], entity_effects=True).fit()
                re_u = RandomEffects(dd[y], dd[preds].assign(const=1.0)).fit()
                common = [p for p in preds if p in fe.params.index and p in re.params.index]
                diff = (fe_u.params[common] - re_u.params[common]).values
                vdiff = (fe_u.cov.loc[common, common] - re_u.cov.loc[common, common]).values
                h_stat = max(0.0, float(diff @ np.linalg.pinv(vdiff) @ diff))
                h_p = float(chi2.sf(h_stat, len(common)))
                use_fe = h_p < 0.05
                rec = "FE（固定效应）" if use_fe else "RE（随机效应）"
                tab = pd.DataFrame(
                    {
                        "term": common,
                        "FE_coef": [round(float(fe.params[c]), 4) for c in common],
                        "FE_p": [round(float(fe.pvalues[c]), 4) for c in common],
                        "RE_coef": [round(float(re.params[c]), 4) for c in common],
                        "RE_p": [round(float(re.pvalues[c]), 4) for c in common],
                    }
                )
                tab.to_csv(d / "fe_re_coefficients.csv", index=False, encoding="utf-8")
                files.append("fe_re_coefficients.csv")
                chosen = fe if use_fe else re
                try:
                    import matplotlib

                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    se = chosen.std_errors[common]
                    co = chosen.params[common]
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.errorbar(co.values, range(len(common)), xerr=1.96 * se.values, fmt="o")
                    ax.axvline(0, color="grey", ls="--")
                    ax.set_yticks(range(len(common)))
                    ax.set_yticklabels(common)
                    ax.set_xlabel(f"{rec[:2]} coefficient (95% CI)")
                    fig.tight_layout()
                    fig.savefig(d / "coefficients.png", dpi=150)
                    plt.close(fig)
                    files.append("coefficients.png")
                except Exception:
                    pass
                estimates["hausman_stat"] = round(h_stat, 4)
                estimates["hausman_p"] = round(h_p, 4)
                for c in common:
                    estimates[c] = round(float(chosen.params[c]), 4)
                summary.append(
                    f"{entry.method} 完成：面板 {dd.index.get_level_values(0).nunique()} 单位 × "
                    f"{dd.index.get_level_values(1).nunique()} 期；结果 {y}，{len(common)} 个预测变量。"
                    f"Hausman H={h_stat:.3f}, p={h_p:.3g} → 推荐 {rec}"
                    f"（p<0.05 表示随机效应与回归元相关、RE 不一致，应用 FE）。系数对比见 fe_re_coefficients.csv（聚类稳健 SE）。"
                )
                code += [
                    "from linearmodels.panel import PanelOLS, RandomEffects  # 面板 RE + Hausman",
                    "# FE=PanelOLS(entity_effects); RE=RandomEffects; H=(b_fe-b_re)'pinv(Vfe-Vre)(b_fe-b_re)",
                ]
            except Exception as err:
                summary.append(f"随机效应模型失败：{err}")

