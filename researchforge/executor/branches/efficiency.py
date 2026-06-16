"""Branch handlers for the efficiency family (migrated from the run.py monolith).

Each handler unpacks ctx into the same local names run_analysis used and runs the
original branch body verbatim. See executor/_branch_api.py.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.run import (
    _dea_cross,
    _dea_efficiency,
    _dea_io,
    _io_names,
    _mcda_inputs,
    _mcda_rank_plot,
    _sfa_via_r,
)


@register("dea")
def _branch_dea(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np

    try:
        X, crit, labels = _mcda_inputs(df, fp)
    except ValueError as err:
        summary.append(f"DEA 失败：{err}")
    else:
        # engine default: first numeric column = output, the rest = inputs;
        # config={"inputs":[...],"outputs":[...]} overrides the i/o roles.
        inputs, outputs, in_names, out_names = _dea_io(X, crit, cfg)
        if inputs.shape[1] < 1:
            summary.append("DEA 失败：需要 ≥1 个投入 + 1 个产出（≥2 个数值列）。")
        elif (inputs <= 0).any() or (outputs <= 0).any():
            summary.append(
                "DEA 失败：投入/产出需为正值（DEA 假定正数据）。请确保投入产出列均为正，"
                "或移除含 0/负值的列。"
            )
        else:
            import pandas as pd

            ccr = _dea_efficiency(inputs, outputs, vrs=False)
            bcc = _dea_efficiency(inputs, outputs, vrs=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                scale = np.where(bcc > 0, ccr / bcc, np.nan)
            res = pd.DataFrame(
                {
                    "DMU": labels,
                    "CCR_efficiency": np.round(ccr, 4),
                    "BCC_efficiency": np.round(bcc, 4),
                    "scale_efficiency": np.round(scale, 4),
                }
            )
            res["rank"] = res["CCR_efficiency"].rank(ascending=False, method="min").astype(int)
            res = res.sort_values("rank").reset_index(drop=True)
            res.to_csv(d / "dea_efficiency.csv", index=False, encoding="utf-8")
            files.append("dea_efficiency.csv")
            rplot = res.rename(columns={"DMU": "alternative"})
            _mcda_rank_plot(
                rplot, "CCR_efficiency", "DEA CCR efficiency (top 20)", d / "dea_efficiency.png"
            )
            if (d / "dea_efficiency.png").exists():
                files.append("dea_efficiency.png")
            n_eff = int(np.sum(np.isclose(ccr, 1.0, atol=1e-4)))
            estimates["n_ccr_efficient"] = float(n_eff)
            estimates["mean_ccr_efficiency"] = round(float(np.nanmean(ccr)), 4)
            estimates["n_dmu"] = float(len(labels))
            _io_note = (
                "（按 config 指定）"
                if (cfg.get("inputs") and cfg.get("outputs"))
                else "⚠ 默认首列为产出、其余为投入——可用 config={\"inputs\":[...],\"outputs\":[...]} 指定。"
            )
            summary.append(
                f"{entry.method} 完成：{len(labels)} 个 DMU，产出 {out_names}，"
                f"投入 {in_names}; CCR 技术有效 {n_eff} 个（θ=1），平均效率 "
                f"{np.nanmean(ccr):.3f}；规模效率=CCR/BCC。" + _io_note
            )
            code += [
                "from scipy.optimize import linprog  # 投入导向 DEA(CCR+BCC)",
                f'# 产出={out_names}, 投入={in_names}; min θ s.t. Σλx≤θx_o, Σλy≥y_o, λ≥0',
            ]



@register("malmquist")
def _branch_malmquist(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np

    _excl = {fp.unit_col, fp.time_col}
    crit = [
        c.name for c in fp.columns if c.kind in {"continuous", "count"} and c.name not in _excl
    ]
    if not (fp.unit_col and fp.time_col):
        summary.append("Malmquist 失败：需要面板数据（单位列 + 时间列）。")
    elif len(crit) < 2:
        summary.append("Malmquist 失败：需要 ≥1 投入 + 1 产出（≥2 个数值列）。")
    else:
        import pandas as pd

        # config={"inputs":[...],"outputs":[...]} overrides i/o; Malmquist DEA
        # supports multiple outputs. out_col kept as a single label for the title.
        in_names, out_names = _io_names(crit, cfg)
        in_cols, out_cols = in_names, out_names
        out_col = ", ".join(out_cols)
        periods = sorted(df[fp.time_col].dropna().unique())
        if len(periods) < 2:
            summary.append("Malmquist 失败：需要 ≥2 个时间期。")
        else:
            # default: first vs last period; config={"periods":[start,end]} picks
            # a specific base/end pair (both must exist in the data).
            t0, t1 = periods[0], periods[-1]
            want = cfg.get("periods")
            if isinstance(want, (list, tuple)) and len(want) == 2:
                pset = set(periods)
                if want[0] in pset and want[1] in pset:
                    t0, t1 = want[0], want[1]
            d0 = df[df[fp.time_col] == t0].drop_duplicates(fp.unit_col).set_index(fp.unit_col)
            d1 = df[df[fp.time_col] == t1].drop_duplicates(fp.unit_col).set_index(fp.unit_col)
            common = [u for u in d0.index if u in d1.index]
            d0, d1 = d0.loc[common], d1.loc[common]
            xi0, yo0 = d0[in_cols].to_numpy(float), d0[out_cols].to_numpy(float)
            xi1, yo1 = d1[in_cols].to_numpy(float), d1[out_cols].to_numpy(float)
            if len(common) < 3:
                summary.append("Malmquist 失败：两期共同单位不足（<3）。")
            elif (xi0 <= 0).any() or (yo0 <= 0).any() or (xi1 <= 0).any() or (yo1 <= 0).any():
                summary.append("Malmquist 失败：投入/产出需为正值。")
            else:
                # CRS distance functions (4 cross-period DEA scores per DMU)
                e_tt = _dea_cross(xi0, yo0, xi0, yo0)  # t  obs vs t  frontier
                e_11 = _dea_cross(xi1, yo1, xi1, yo1)  # t1 obs vs t1 frontier
                e_t_1 = _dea_cross(xi1, yo1, xi0, yo0)  # t1 obs vs t  frontier
                e_1_t = _dea_cross(xi0, yo0, xi1, yo1)  # t  obs vs t1 frontier
                ec = e_11 / e_tt  # efficiency change (catch-up)
                tc = np.sqrt((e_t_1 / e_11) * (e_tt / e_1_t))  # technical change (frontier shift)
                m = ec * tc  # Malmquist TFP change (>1 = growth)
                res = pd.DataFrame(
                    {
                        str(fp.unit_col): common,
                        "malmquist_tfp": np.round(m, 4),
                        "efficiency_change": np.round(ec, 4),
                        "technical_change": np.round(tc, 4),
                    }
                )
                res.to_csv(d / "malmquist.csv", index=False, encoding="utf-8")
                files.append("malmquist.csv")

                def _gmean(a):
                    a = a[np.isfinite(a) & (a > 0)]
                    return float(np.exp(np.mean(np.log(a)))) if len(a) else float("nan")

                gm_m, gm_ec, gm_tc = _gmean(m), _gmean(ec), _gmean(tc)
                try:
                    import matplotlib

                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(6, max(3, len(common) * 0.3)))
                    ax.barh([str(u) for u in common][::-1], m[::-1], color="#55A868")
                    ax.axvline(1.0, color="grey", ls="--", lw=0.8)
                    ax.set_xlabel("Malmquist TFP index (>1 = growth)")
                    ax.set_title(f"Malmquist productivity change {t0}→{t1}")
                    fig.tight_layout()
                    fig.savefig(d / "malmquist.png", dpi=150)
                    plt.close(fig)
                    files.append("malmquist.png")
                except Exception:
                    pass
                estimates["mean_malmquist_tfp"] = round(gm_m, 4)
                estimates["mean_efficiency_change"] = round(gm_ec, 4)
                estimates["mean_technical_change"] = round(gm_tc, 4)
                estimates["n_dmu"] = float(len(common))
                verdict = "TFP 上升" if gm_m > 1.01 else ("TFP 下降" if gm_m < 0.99 else "TFP 基本不变")
                summary.append(
                    f"{entry.method} 完成：{len(common)} 个单位 {t0}→{t1}；产出 {out_cols}，"
                    f"投入 {in_cols}；总体 Malmquist TFP={gm_m:.3f}（{verdict}）"
                    f"= 效率变化 {gm_ec:.3f} × 技术变化 {gm_tc:.3f}（>1 为增长）。"
                    + (
                        "（投入产出/期间按 config 指定）"
                        if (cfg.get("inputs") and cfg.get("outputs")) or cfg.get("periods")
                        else "⚠ 默认首数值列为产出、其余投入、首末两期；CRS 距离函数（可配 inputs/outputs/periods）"
                    )
                )
                code += [
                    "from scipy.optimize import linprog  # Malmquist(Färe1994), CRS 距离函数",
                    "# M = (E11/Ett)·sqrt((E[t1|t]/E11)·(Ett/E[t|t1])); 分解 EC×TC",
                ]



@register("sfa")
def _branch_sfa(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import re

    import numpy as np

    from researchforge.executor import rbridge

    _excl = {fp.unit_col, fp.time_col}
    crit = [
        c.name for c in fp.columns if c.kind in {"continuous", "count"} and c.name not in _excl
    ]
    names_safe = all(re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", str(c)) for c in crit)
    if len(crit) < 2:
        summary.append("SFA 失败：需要 ≥1 投入 + 1 产出（≥2 个数值列）。")
    elif not (rbridge.r_available() and rbridge.r_package_available("frontier")):
        summary.append(
            "SFA 需要 R 的 frontier 包（未检测到）。安装：install.packages('frontier')；"
            "或用 DEA（确定性前沿，纯 Python，无需 R）。"
        )
    elif not names_safe:
        summary.append("SFA 失败：列名需为标识符式（字母/数字/. _）。")
    else:
        # config={"inputs":[...],"outputs":[...]} overrides i/o roles; SFA is
        # single-output (Cobb-Douglas), so only the first output is used.
        in_names, out_names = _io_names(crit, cfg)
        output_col, in_cols = out_names[0], in_names
        sfa_multi_out = (
            "；⚠ SFA 为单产出模型，仅用首个产出 " + output_col if len(out_names) > 1 else ""
        )
        label_col = next(
            (c.name for c in fp.columns if c.kind in {"id", "categorical"} and c.name not in _excl),
            None,
        )
        sub = df[crit + ([label_col] if label_col else [])].dropna()
        if (sub[crit].to_numpy(dtype=float) <= 0).any():
            summary.append("SFA 失败：投入/产出需为正值（Cobb-Douglas 取对数）。")
        else:
            labels = (
                sub[label_col].astype(str).tolist()
                if label_col
                else [f"row{i + 1}" for i in range(len(sub))]
            )
            csv = d / "_sfa_input.csv"
            sub[crit].to_csv(csv, index=False)
            try:
                import pandas as pd

                coef, te = _sfa_via_r(csv, output_col, in_cols)
                elastic = {
                    k: v
                    for k, v in coef.items()
                    if k not in ("sigmaSq", "gamma", "lr_stat", "lr_pvalue")
                }
                pd.DataFrame(
                    {"term": list(elastic), "elasticity": [round(v, 4) for v in elastic.values()]}
                ).to_csv(d / "frontier_coefficients.csv", index=False, encoding="utf-8")
                files.append("frontier_coefficients.csv")
                teres = pd.DataFrame({"unit": labels, "technical_efficiency": np.round(te, 4)})
                teres = teres.sort_values("technical_efficiency", ascending=False).reset_index(
                    drop=True
                )
                teres.to_csv(d / "technical_efficiency.csv", index=False, encoding="utf-8")
                files.append("technical_efficiency.csv")
                try:
                    import matplotlib

                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.hist(te, bins=min(20, max(5, len(te) // 4)), color="#4C72B0", edgecolor="white")
                    ax.axvline(float(np.mean(te)), color="#C44E52", ls="--", label=f"mean={np.mean(te):.3f}")
                    ax.set_xlabel("technical efficiency")
                    ax.set_ylabel("count")
                    ax.set_title("SFA technical-efficiency distribution")
                    ax.legend(fontsize=8)
                    fig.tight_layout()
                    fig.savefig(d / "efficiency_distribution.png", dpi=150)
                    plt.close(fig)
                    files.append("efficiency_distribution.png")
                except Exception:
                    pass
                mean_te = float(np.mean(te))
                gamma = coef.get("gamma", float("nan"))
                lr_p = coef.get("lr_pvalue", float("nan"))
                estimates["mean_technical_efficiency"] = round(mean_te, 4)
                estimates["gamma"] = round(gamma, 4)
                estimates["lr_inefficiency_pvalue"] = round(lr_p, 4)
                estimates["n_dmu"] = float(len(labels))
                for k, v in elastic.items():
                    if "Intercept" not in k:
                        estimates[k] = round(v, 4)
                # if the one-sided LR test can't reject γ=0, the model is ~OLS and
                # the technical-efficiency scores are not trustworthy (Opus catch).
                ineff_sig = lr_p < 0.05
                te_warn = (
                    ""
                    if ineff_sig
                    else "；⚠ 低效 LR 检验不显著（无统计学上的无效率），模型接近 OLS，技术效率值不可靠"
                )
                summary.append(
                    f"{entry.method} 完成（R/frontier）：Cobb-Douglas 前沿 [{output_col}] ~ {in_cols}；"
                    f"平均技术效率 {mean_te:.3f}（最优=1）；γ={gamma:.3f}"
                    "（=σ_u²/(σ_u²+σ_v²) 比值，越近 1 越说明偏离前沿主要是低效而非噪声）；"
                    f"低效存在性 LR 检验 p={lr_p:.3g}（{'显著存在低效' if ineff_sig else '不显著'}）"
                    f"{te_warn}{sfa_multi_out}。弹性见 frontier_coefficients.csv。"
                    + (
                        "（投入产出按 config 指定）"
                        if (cfg.get("inputs") and cfg.get("outputs"))
                        else "⚠ 默认首列为产出、其余投入（可用 config inputs/outputs 指定）"
                    )
                    + "；假定 Cobb-Douglas + 半正态低效。"
                )
                code += [
                    "library(frontier)  # 随机前沿(Cobb-Douglas, ML)",
                    f"# sfa(log({output_col}) ~ {' + '.join(f'log({c})' for c in in_cols)}); efficiencies()",
                ]
            except Exception as err:
                summary.append(f"SFA 拟合失败：{err}")
            finally:
                try:
                    csv.unlink()
                except OSError:
                    pass

