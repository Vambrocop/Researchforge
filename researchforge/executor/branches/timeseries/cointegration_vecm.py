"""Johansen cointegration and the vector error-correction model.

One analysis, one module (see branches/__init__.py — walk_packages auto-registers it).
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


@register("cointegration_vecm")
def _branch_cointegration_vecm(ctx: Ctx) -> None:
    # Cointegration (Engle-Granger + Johansen) and, if cointegrated, a VECM:
    # long-run equilibrium relation among I(1) series + short-run adjustment speeds.
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    _excl = {fp.unit_col, fp.time_col}
    forced = [c for c in (cfg.get("series") or cfg.get("predictors") or []) if c in df.columns and c not in _excl]
    series = (forced if len(forced) >= 2 else
              [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl])[:6]
    if len(series) < 2:
        summary.append("协整/VECM 失败：需要 ≥2 个连续时间序列变量（config['series'] 可指定）。")
        return
    try:
        from statsmodels.tsa.stattools import adfuller, coint
        from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen, select_coint_rank, select_order

        d2 = df.sort_values(fp.time_col) if (fp.time_col and fp.time_col in df.columns) else df
        data = d2[series].dropna().reset_index(drop=True).astype(float)
        n = len(data)
        if n < 25:
            summary.append("协整/VECM 失败：观测不足（<25），无法稳健做 Johansen/VECM。")
            return
        # I(1) precondition: levels should be non-stationary, first differences stationary
        lvl_nonstat = sum(adfuller(data[s].to_numpy(), autolag="AIC")[1] > 0.05 for s in series)
        diff_stat = sum(adfuller(data[s].diff().dropna().to_numpy(), autolag="AIC")[1] <= 0.05 for s in series)
        # lag order (in differences) by AIC; fall back to 1
        try:
            kmax = max(1, min(8, n // (len(series) + 1) - 1))
            k = max(1, int(select_order(data, maxlags=kmax, deterministic="ci").aic))
        except Exception:
            k = 1
        joh = coint_johansen(data, det_order=0, k_ar_diff=k)
        trace, cv95 = joh.lr1, joh.cvt[:, 1]
        # Johansen trace is SEQUENTIAL (test rank<=0, rank<=1, …; STOP at the first non-rejection).
        # Summing exceedances over-counts (~2.5% of cases): the trace stats AND their critical values
        # both shrink across steps and can re-cross. Use the canonical sequential routine.
        r = int(select_coint_rank(data, det_order=0, k_ar_diff=k, signif=0.05).rank)
        eg_p = float(coint(data[series[0]], data[series[1]])[1])  # Engle-Granger (first pair)

        estimates.update({
            "n_coint_relations": float(r), "johansen_trace_r0": float(trace[0]),
            "johansen_cv95_r0": float(cv95[0]), "eg_pvalue_pair": round(eg_p, 4),
            "levels_nonstationary": float(lvl_nonstat), "diffs_stationary": float(diff_stat),
            "k_ar_diff": float(k), "n_obs": float(n),
        })
        # NOTE: reject_pointwise is an element-wise trace>cv95 flag per row; it is NOT the
        # authoritative rank (sequential re-crossing can make it disagree with select_coint_rank).
        # The authoritative rank is `r` above (from the sequential select_coint_rank routine).
        pd.DataFrame({"r_le": list(range(len(trace))), "trace_stat": np.round(trace, 3),
                      "crit_95": np.round(cv95, 3), "reject_pointwise_(coint>r)_not_authoritative": trace > cv95}
                     ).to_csv(d / "johansen_trace.csv", index=False, encoding="utf-8")
        files.append("johansen_trace.csv")

        longrun = ""
        full_rank = r >= len(series)  # r == #vars -> levels stationary (I(0)), not a cointegrated I(1) system
        if 1 <= r < len(series):
            # deterministic="co": matches the det_order=0 Johansen rank test above. statsmodels'
            # coint_johansen only tabulates nc/co/lo critical values (no "ci" table exists), so fitting
            # the VECM with the restricted-constant "ci" would test and fit under DIFFERENT deterministic
            # assumptions (co's lenient 95% critical values vs ci's stricter ones) -> spurious cointegration
            # in the gap between them. "co" keeps the constant unrestricted and OUTSIDE the cointegrating
            # relation, consistent with the rank test that was actually used to pick r.
            vecm = VECM(data, k_ar_diff=k, coint_rank=r, deterministic="co").fit()
            beta = np.asarray(vecm.beta)[:, 0].astype(float)
            alpha = np.asarray(vecm.alpha)[:, 0].astype(float)
            beta_n = beta / beta[0] if abs(beta[0]) > 1e-12 else beta
            terms = " ".join(f"{'+' if b >= 0 else '-'}{abs(b):.3f}·{s}" for b, s in zip(beta_n, series))
            longrun = f"长期均衡关系（标准化 {series[0]}=1）：{terms} ≈ 0；"
            estimates["adjustment_speed_eq1"] = round(float(alpha[0]), 4)
            # deterministic="co": the constant is unrestricted and lives OUTSIDE the cointegrating
            # relation (each equation's own intercept), so no constant term is added into the ECT here
            # (a nonzero mean of the equilibrium error shows up via the axhline in the plot instead).
            ect = data.to_numpy() @ beta
            try:
                ect_p = float(adfuller(ect, autolag="AIC")[1])
                estimates["ect_adf_pvalue"] = round(ect_p, 4)
            except Exception:
                pass
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(8, 3.6))
                ax.plot(ect, color="#4C72B0")
                ax.axhline(float(np.mean(ect)), color="grey", ls="--", lw=1)
                ax.set_title("Cointegrating residual (ECT) — mean-reverting if cointegrated")
                ax.set_xlabel("period index")
                ax.set_ylabel("equilibrium error")
                fig.tight_layout()
                fig.savefig(d / "cointegration_ect.png", dpi=150)
                plt.close(fig)
                files.append("cointegration_ect.png")
            except Exception:
                pass
            (d / "vecm_summary.txt").write_text(str(vecm.summary()), encoding="utf-8")
            files.append("vecm_summary.txt")

        i1_note = ("" if (lvl_nonstat >= 1 and diff_stat >= 1) else
                   "；⚠ I(1) 前提存疑（levels 应非平稳、差分应平稳）——协整解读需谨慎")
        if full_rank:
            verdict = (f"协整秩 r={r} = 序列数 → levels 近似平稳(I(0))，不是 I(1) 协整系统；"
                       "协整/VECM 不适用，宜直接对 levels 建模(VAR)")
        elif r >= 1:
            verdict = (f"检出 {r} 个协整关系（Johansen trace 序贯检验，95%）；{longrun}"
                       f"调整速度 α₁={estimates.get('adjustment_speed_eq1')}（负=向均衡回拉）；"
                       f"ECT 回均值（ADF p={estimates.get('ect_adf_pvalue','—')}；注：基于估计的协整向量，p 偏乐观）")
        else:
            verdict = (f"未检出协整关系（Johansen trace r=0，trace={trace[0]:.2f} vs CV95={cv95[0]:.2f}；"
                       f"Engle-Granger 首对 p={eg_p:.3g}）——序列各自漂移、无长期均衡，宜对差分建模(VAR/ARIMA)")
        summary.append(
            f"{entry.method} 完成：{len(series)} 个序列 × {n} 期（diff 阶数 k={k}）。{verdict}。"
            f" ⚠ 协整要求各序列 I(1)（已查：{lvl_nonstat}/{len(series)} levels 非平稳、"
            f"{diff_stat}/{len(series)} 差分平稳{i1_note}）；Johansen 对滞后阶/确定性项设定敏感；"
            "秩检验与 VECM 均采用『无约束常数(co)』设定（statsmodels 的 Johansen 秩检验仅提供 "
            "nc/co/lo 临界值表，无约束常数外置，允许水平序列有非零均值）；"
            "johansen_trace.csv 的逐行 reject 列为逐点比较，非权威结果——权威协整秩以序贯 "
            "select_coint_rank（即上文 r）为准；长期关系是统计均衡、非结构因果。"
        )
        code += [
            "from statsmodels.tsa.vector_ar.vecm import select_coint_rank, VECM  # 协整 + VECM",
            f"# r=select_coint_rank(data, det_order=0, k_ar_diff={k}, signif=0.05).rank (序贯); "
            f"VECM(data, k_ar_diff={k}, coint_rank=r, deterministic='co').fit()",
        ]
    except Exception as err:
        summary.append(f"协整/VECM 失败：{err}")
