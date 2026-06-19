"""Branch handlers for the timeseries family (migrated from the run.py monolith).

Each handler unpacks ctx into the same local names run_analysis used and runs the
original branch body verbatim. See executor/_branch_api.py.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


@register("arima")
def _branch_arima(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    time_col = fp.time_col
    # value_col: forecast the first continuous column. Time columns are
    # datetime/id/count kind (never continuous), so they are never picked here.
    value_col = next((c.name for c in fp.columns if c.kind == "continuous"), None)

    if time_col is None or value_col is None:
        summary.append(
            "ARIMA 失败：未找到时间列或连续值列，请检查数据结构。"
        )
    else:
        try:
            from statsmodels.tsa.arima.model import ARIMA

            sorted_df = df.sort_values(time_col)
            dup = int(sorted_df[time_col].duplicated().sum())
            if dup:
                sorted_df = sorted_df.drop_duplicates(subset=time_col, keep="first")
                summary.append(f"注意：{dup} 个重复时间点已去重（保留首次）。")
            y = sorted_df[value_col].astype(float).reset_index(drop=True)
            if y.nunique() < 2 or len(y) < 10:
                raise ValueError(f"序列有效观测不足或近常数（n={len(y)}），无法拟合 ARIMA")

            model = ARIMA(y, order=(1, 1, 1)).fit()

            (d / "model_summary.txt").write_text(str(model.summary()), encoding="utf-8")
            files.append("model_summary.txt")

            steps = 10
            fc = model.forecast(steps=steps)
            import pandas as _pd
            fc_df = _pd.DataFrame({"step": list(range(1, steps + 1)), "forecast": fc.tolist()})
            fc_df.to_csv(d / "forecast.csv", index=False, encoding="utf-8")
            files.append("forecast.csv")

            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(range(len(y)), y, label="observed")
                fc_x = list(range(len(y), len(y) + steps))
                ax.plot(fc_x, fc.tolist(), color="red", linestyle="--", label="forecast")
                ax.set_xlabel("period index")
                ax.set_ylabel(value_col)
                ax.set_title(f"ARIMA(1,1,1) — {value_col}")
                ax.legend()
                fig.tight_layout()
                fig.savefig(d / "forecast.png", dpi=150)
                plt.close(fig)
                files.append("forecast.png")
            except Exception:
                pass

            estimates["aic"] = float(model.aic)
            summary.append(
                f"{entry.method} 完成：对 {value_col} 拟合 ARIMA(1,1,1)，"
                f"AIC={model.aic:.2f}，预测未来 {steps} 期"
            )
            code += [
                "from statsmodels.tsa.arima.model import ARIMA",
                f"y = df.sort_values('{time_col}')['{value_col}'].astype(float).reset_index(drop=True)",
                "model = ARIMA(y, order=(1, 1, 1)).fit()",
                "print(model.summary())",
                f"fc = model.forecast(steps={steps})",
            ]
        except Exception as err:
            summary.append(f"ARIMA 拟合失败：{err}")



@register("var_granger")
def _branch_var_granger(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    _excl = {fp.unit_col, fp.time_col}
    series = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl][:6]
    if len(series) < 2:
        summary.append("VAR/Granger 失败：需要 ≥2 个连续时间序列变量。")
    else:
        try:
            from statsmodels.tsa.api import VAR

            d2 = df.sort_values(fp.time_col) if (fp.time_col and fp.time_col in df.columns) else df
            data = d2[series].dropna().reset_index(drop=True)
            n = len(data)
            if n < 20:
                summary.append("VAR/Granger 失败：观测不足（<20），无法稳健拟合 VAR。")
            else:
                maxlags = max(1, min(8, n // (len(series) + 1) - 1))
                res = VAR(data).fit(maxlags=maxlags, ic="aic")
                if res.k_ar < 1:
                    res = VAR(data).fit(1)  # AIC picked 0 lags -> force lag 1 for Granger
                pmat = pd.DataFrame(np.nan, index=series, columns=series)  # rows=causing -> cols=caused
                for causing in series:
                    for caused in series:
                        if causing != caused:
                            try:
                                pmat.loc[causing, caused] = float(
                                    res.test_causality(caused, [causing]).pvalue
                                )
                            except Exception:
                                pass
                pmat.round(4).to_csv(d / "granger_pvalues.csv", encoding="utf-8")
                files.append("granger_pvalues.csv")
                links = [
                    f"{r}→{c}"
                    for r in series
                    for c in series
                    if r != c and pd.notna(pmat.loc[r, c]) and pmat.loc[r, c] < 0.05
                ]
                try:
                    import matplotlib

                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    mat = -np.log10(pmat.to_numpy(dtype=float).clip(1e-300, 1))
                    np.fill_diagonal(mat, np.nan)
                    fig, ax = plt.subplots(figsize=(5.5, 4.5))
                    im = ax.imshow(mat, cmap="Reds")
                    ax.set_xticks(range(len(series)))
                    ax.set_xticklabels(series, rotation=45, ha="right")
                    ax.set_yticks(range(len(series)))
                    ax.set_yticklabels(series)
                    ax.set_xlabel("caused →")
                    ax.set_ylabel("causing →")
                    ax.set_title("Granger causality  -log10(p)")
                    fig.colorbar(im, label="-log10(p)")
                    fig.tight_layout()
                    fig.savefig(d / "granger_heatmap.png", dpi=150)
                    plt.close(fig)
                    files.append("granger_heatmap.png")
                except Exception:
                    pass
                try:
                    fig = res.irf(10).plot()
                    fig.savefig(d / "irf.png", dpi=120)
                    import matplotlib.pyplot as plt

                    plt.close(fig)
                    files.append("irf.png")
                except Exception:
                    pass
                # active stationarity check (ADF) — non-stationary series give
                # spurious Granger causality; flag loudly, not just in prose (Opus catch).
                n_nonstat = 0
                try:
                    from statsmodels.tsa.stattools import adfuller

                    for s in series:
                        if adfuller(data[s].to_numpy(dtype=float), autolag="AIC")[1] > 0.05:
                            n_nonstat += 1
                except Exception:
                    n_nonstat = -1
                estimates["selected_lag"] = float(res.k_ar)
                estimates["n_series"] = float(len(series))
                estimates["n_causal_links"] = float(len(links))
                estimates["n_nonstationary"] = float(n_nonstat)
                stat_warn = (
                    f"；⚠ ADF 检验：{n_nonstat}/{len(series)} 个序列非平稳，Granger 结果恐为伪因果——请先差分/平稳化再解读"
                    if n_nonstat > 0
                    else ""
                )
                time_warn = "" if fp.time_col else "；⚠ 无时间列，按行序当作时间序列处理（请确认行序即时序）"
                summary.append(
                    f"{entry.method} 完成：{len(series)} 个序列 × {n} 期，VAR 阶数={res.k_ar}（AIC 选）；"
                    f"Granger 因果 p 值矩阵见 granger_pvalues.csv；显著(p<0.05)有向因果："
                    f"{('、'.join(links) if links else '无')}{stat_warn}{time_warn}。"
                    f"按{'时间列 ' + str(fp.time_col) if fp.time_col else '行序'}排序；"
                    "Granger 因果是「预测性」非结构因果。"
                )
                code += [
                    "from statsmodels.tsa.api import VAR  # VAR + Granger 因果",
                    "# VAR(data).fit(ic='aic'); res.test_causality(caused, [causing]).pvalue; res.irf().plot()",
                ]
        except Exception as err:
            summary.append(f"VAR/Granger 失败：{err}")


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
        pd.DataFrame({"r_le": list(range(len(trace))), "trace_stat": np.round(trace, 3),
                      "crit_95": np.round(cv95, 3), "reject_(coint>r)": trace > cv95}
                     ).to_csv(d / "johansen_trace.csv", index=False, encoding="utf-8")
        files.append("johansen_trace.csv")

        longrun = ""
        full_rank = r >= len(series)  # r == #vars -> levels stationary (I(0)), not a cointegrated I(1) system
        if 1 <= r < len(series):
            vecm = VECM(data, k_ar_diff=k, coint_rank=r, deterministic="ci").fit()
            beta = np.asarray(vecm.beta)[:, 0].astype(float)
            alpha = np.asarray(vecm.alpha)[:, 0].astype(float)
            beta_n = beta / beta[0] if abs(beta[0]) > 1e-12 else beta
            terms = " ".join(f"{'+' if b >= 0 else '-'}{abs(b):.3f}·{s}" for b, s in zip(beta_n, series))
            longrun = f"长期均衡关系（标准化 {series[0]}=1）：{terms} ≈ 0；"
            estimates["adjustment_speed_eq1"] = round(float(alpha[0]), 4)
            # deterministic="ci": a restricted constant lives INSIDE the cointegration -> include it in
            # the ECT so the equilibrium error is centered correctly (matters for the plot).
            const = np.ravel(getattr(vecm, "det_coef_coint", np.zeros(1)))
            ect = data.to_numpy() @ beta + (float(const[0]) if const.size else 0.0)
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
            "长期关系是统计均衡、非结构因果。"
        )
        code += [
            "from statsmodels.tsa.vector_ar.vecm import select_coint_rank, VECM  # 协整 + VECM",
            f"# r=select_coint_rank(data, det_order=0, k_ar_diff={k}, signif=0.05).rank (序贯); "
            f"VECM(data, k_ar_diff={k}, coint_rank=r, deterministic='ci').fit()",
        ]
    except Exception as err:
        summary.append(f"协整/VECM 失败：{err}")

