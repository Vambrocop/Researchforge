"""Branch handlers for the timeseries family (migrated from the run.py monolith).

Each handler unpacks ctx into the same local names run_analysis used and runs the
original branch body verbatim. See executor/_branch_api.py.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


def _periodogram_period(x, n):
    """Dominant seasonal period via the periodogram, or None if no SIGNIFICANT periodicity.
    Linearly detrends first (a trend's low-frequency power otherwise dominates), requires >=3
    cycles (period <= n/3), and applies Fisher's g-test (alpha=0.05) so pure noise/trend -> None."""
    import numpy as np

    x = np.asarray(x, dtype=float)
    idx = np.arange(n)
    c = np.polyfit(idx, x, 1)        # remove linear trend
    x = x - (c[0] * idx + c[1])
    if np.std(x) == 0:
        return None
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(n)
    mask = freqs >= 3.0 / n          # candidate seasonal freqs (period <= n/3)
    if not mask.any():
        return None
    pm = power[mask]
    m = len(pm)
    if m < 2 or pm.sum() <= 0:
        return None
    g = float(pm.max() / pm.sum())                    # Fisher's g statistic
    g_crit = 1.0 - (0.05 / m) ** (1.0 / (m - 1))      # alpha=0.05 critical value
    if g <= g_crit:                                   # no significant periodicity
        return None
    freq = freqs[mask][int(np.argmax(pm))]
    if freq <= 0:
        return None
    per = int(round(1.0 / freq))
    return per if 2 <= per <= n // 3 else None


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
                f"AIC={model.aic:.2f}，预测未来 {steps} 期。"
                "⚠ 阶数固定为 (1,1,1)（未做自动定阶/单位根检验，AIC 仅供参考）。"
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
                forced_lag1 = False
                if res.k_ar < 1:
                    res = VAR(data).fit(1)  # AIC picked 0 lags -> force lag 1 for Granger
                    forced_lag1 = True
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
                lag_note = "（AIC 选 0，已强制为 1 阶）" if forced_lag1 else "（AIC 选）"
                summary.append(
                    f"{entry.method} 完成：{len(series)} 个序列 × {n} 期，VAR 阶数={res.k_ar}{lag_note}；"
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


@register("garch")
def _branch_garch(ctx: Ctx) -> None:
    # GARCH(1,1) conditional-volatility model: captures volatility clustering in a series.
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import importlib.util

    import numpy as np

    _excl = {fp.unit_col, fp.time_col}
    value = cfg.get("value") if cfg.get("value") in df.columns else next(
        (c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl), None)
    if importlib.util.find_spec("arch") is None:
        summary.append("GARCH 需要 arch 包（未检测到）。安装：pip install arch。")
        return
    if value is None:
        summary.append("GARCH 失败：需要一个连续序列（收益/波动序列）。config['value'] 可指定。")
        return
    try:
        import pandas as pd
        from arch import arch_model
        from statsmodels.stats.diagnostic import het_arch

        d2 = df.sort_values(fp.time_col) if (fp.time_col and fp.time_col in df.columns) else df
        y = d2[value].astype(float).dropna().reset_index(drop=True)
        n = len(y)
        if n < 50 or y.nunique() < 5:
            summary.append("GARCH 失败：观测不足（<50）或近常数序列。")
            return
        # arch fits best when data are scaled ~[1, 1000]; rescale OUT-of-band series (tiny or huge)
        # to a single multiplicative scale (target std ~10) and restore volatility after — divide-back
        # stays exact for any scale. In-band series are left as-is.
        s = float(y.std())
        scale = 1.0 if 0.1 <= s <= 1000.0 else 10.0 / s
        ys = y * scale
        try:
            arch_lm_p = float(het_arch(ys - ys.mean(), nlags=min(10, n // 5))[1])
        except Exception:
            arch_lm_p = float("nan")
        res = arch_model(ys, mean="Constant", vol="GARCH", p=1, q=1).fit(disp="off")
        conv_note = "；⚠ GARCH 优化器未收敛，系数不可靠" if getattr(res, "convergence_flag", 0) else ""
        a, b = float(res.params.get("alpha[1]", 0.0)), float(res.params.get("beta[1]", 0.0))
        omega = float(res.params.get("omega", 0.0))
        persistence = a + b
        cond_vol = np.asarray(res.conditional_volatility, dtype=float) / scale  # back to original scale
        uncond = float(np.sqrt(omega / (1 - persistence)) / scale) if persistence < 1 else float("nan")
        estimates.update({
            "alpha1": round(a, 4), "beta1": round(b, 4), "persistence": round(persistence, 4),
            "omega": round(omega, 6), "arch_lm_pvalue": round(arch_lm_p, 4) if arch_lm_p == arch_lm_p else float("nan"),
            "uncond_volatility": round(uncond, 6) if uncond == uncond else float("nan"),
            "aic": round(float(res.aic), 2), "n_obs": float(n),
        })
        pd.DataFrame({"period": range(n), "cond_volatility": np.round(cond_vol, 6)}).to_csv(
            d / "garch_volatility.csv", index=False, encoding="utf-8")
        files.append("garch_volatility.csv")
        (d / "garch_summary.txt").write_text(str(res.summary()), encoding="utf-8")
        files.append("garch_summary.txt")
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 3.8))
            ax.plot(cond_vol, color="#C44E52", label="conditional volatility σ_t")
            ax.plot(np.abs(y - y.mean()).to_numpy(), color="#bbbbbb", lw=0.6, alpha=0.7, label="|series - mean|")
            ax.set_xlabel("period index")
            ax.set_ylabel(f"volatility of {value}")
            ax.set_title("GARCH(1,1) conditional volatility")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "garch_volatility.png", dpi=150)
            plt.close(fig)
            files.append("garch_volatility.png")
        except Exception:
            pass
        no_arch = arch_lm_p == arch_lm_p and arch_lm_p > 0.05
        arch_note = "；⚠ ARCH-LM 不显著(p>0.05)：无明显波动聚集，GARCH 或非必要" if no_arch else ""
        pers_note = "；⚠ α+β≥1：波动近单位根(IGARCH)，无条件方差不存在" if persistence >= 1 else ""
        omega_note = f"（在 ×{scale:g} 缩放序列的尺度上）" if scale != 1 else ""
        scale_note = f"（拟合时已×{scale:g}，条件/无条件波动率已还原原尺度）" if scale != 1 else ""
        summary.append(
            f"{entry.method} 完成：{value} GARCH(1,1){scale_note}；"
            f"ω={omega:.4g}{omega_note}、α₁={a:.3f}、β₁={b:.3f}；波动持续性 α+β={persistence:.3f}（越近 1 越持久）；"
            f"ARCH-LM p={arch_lm_p:.3g}（检波动聚集）；AIC={res.aic:.1f}。{conv_note}{arch_note}{pers_note}"
            " ⚠ GARCH 建模条件异方差（波动聚集），假定均值方程已设定、序列(弱)平稳；α+β<1 才有有限无条件方差；"
            "正态新息默认（厚尾可换 t 分布）；ω 是缩放序列上的方差截距（α/β 无量纲、波动率已还原，ω 未还原）。"
        )
        code += [
            "from arch import arch_model  # GARCH 条件波动率",
            "# arch_model(y, mean='Constant', vol='GARCH', p=1, q=1).fit(); 持续性=α₁+β₁",
        ]
    except Exception as err:
        summary.append(f"GARCH 拟合失败：{err}")


@register("structural_breaks")
def _branch_structural_breaks(ctx: Ctx) -> None:
    # Multiple structural-break (change-point) detection in a series' MEAN level via ruptures PELT
    # (Bai-Perron-style), with a ~2xBIC penalty (penalty_mult, default 2.0) auto-selecting the breaks.
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import importlib.util

    import numpy as np

    _excl = {fp.unit_col, fp.time_col}
    value = cfg.get("value") if cfg.get("value") in df.columns else next(
        (c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl), None)
    if importlib.util.find_spec("ruptures") is None:
        summary.append("结构突变检测需要 ruptures 包（未检测到）。安装：pip install ruptures。")
        return
    if value is None:
        summary.append("结构突变检测失败：需要一个连续序列。config['value'] 可指定。")
        return
    try:
        import pandas as pd
        import ruptures as rpt

        d2 = df.sort_values(fp.time_col) if (fp.time_col and fp.time_col in df.columns) else df
        y = d2[value].astype(float).dropna().reset_index(drop=True)
        n = len(y)
        if n < 30 or y.nunique() < 5:
            summary.append("结构突变检测失败：观测不足（<30）或近常数序列。")
            return
        sig = y.to_numpy()
        # noise variance from first differences (immune to mean shifts -> not inflated by the breaks)
        sigma2 = float(np.var(np.diff(sig)) / 2.0) if n > 2 else float(np.var(sig))
        try:
            mult = float(cfg.get("penalty_mult", 2.0))
        except (TypeError, ValueError):
            mult = 2.0
        pen = mult * np.log(n) * max(sigma2, 1e-12)
        min_size = max(5, n // 20)
        nb = cfg.get("n_breaks")
        if isinstance(nb, int) and nb >= 1:
            bkps = rpt.Dynp(model="l2", min_size=min_size).fit(sig).predict(n_bkps=nb)
            sel = f"固定 {nb} 个断点 (Dynp)"
        else:
            bkps = rpt.Pelt(model="l2", min_size=min_size).fit(sig).predict(pen=pen)
            sel = f"PELT 自动选 (~2×BIC 惩罚 pen={pen:.3g}，penalty_mult 默认 2.0，越大越少断点)"
        breaks = [int(b) for b in bkps if b < n]  # segment boundaries (drop the trailing n)
        bounds = [0] + breaks + [n]
        seg = [{"start": bounds[i], "end": bounds[i + 1], "n": bounds[i + 1] - bounds[i],
                "mean": round(float(sig[bounds[i]:bounds[i + 1]].mean()), 4),
                "sd": round(float(sig[bounds[i]:bounds[i + 1]].std()), 4)}
               for i in range(len(bounds) - 1)]
        time_vals = None
        if fp.time_col and fp.time_col in d2.columns:
            tv = d2[fp.time_col].reset_index(drop=True)
            time_vals = [tv.iloc[b] for b in breaks if b < len(tv)]
        # trend confound: l2 detects MEAN shifts; a strong linear trend gets approximated by steps
        idx = np.arange(n)
        trend_r = float(abs(np.corrcoef(idx, sig)[0, 1])) if np.std(sig) > 0 else 0.0
        estimates.update({"n_breaks": float(len(breaks)), "n_obs": float(n),
                          "penalty": round(float(pen), 4), "trend_abs_corr": round(trend_r, 3)})
        pd.DataFrame(seg).to_csv(d / "segments.csv", index=False, encoding="utf-8")
        files.append("segments.csv")
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 3.8))
            ax.plot(sig, color="#bbbbbb", lw=0.8)
            for s in seg:
                ax.hlines(s["mean"], s["start"], s["end"], color="#4C72B0", lw=2)
            for b in breaks:
                ax.axvline(b, color="#C44E52", ls="--", lw=1)
            ax.set_xlabel("period index")
            ax.set_ylabel(value)
            ax.set_title(f"Structural breaks — {len(breaks)} change point(s) in mean")
            fig.tight_layout()
            fig.savefig(d / "structural_breaks.png", dpi=150)
            plt.close(fig)
            files.append("structural_breaks.png")
        except Exception:
            pass
        shift_txt = ""
        if len(seg) >= 2:
            shifts = [abs(seg[i + 1]["mean"] - seg[i]["mean"]) for i in range(len(seg) - 1)]
            j = int(np.argmax(shifts))
            bt = (f"≈{fp.time_col}={time_vals[j]}" if time_vals and j < len(time_vals) else f"index {breaks[j]}")
            shift_txt = f"最大均值跳变在断点 #{j + 1}（{bt}）：{seg[j]['mean']}→{seg[j + 1]['mean']}；"
        trend_note = ("；⚠ 序列有强线性趋势（|r|=%.2f）——均值突变检测可能在用台阶逼近趋势，"
                      "建议先去趋势/差分再检测" % trend_r) if trend_r > 0.7 else ""
        loc_txt = "、".join(str(b) for b in breaks) if breaks else "无"
        summary.append(
            f"{entry.method} 完成：{value}（n={n}）检出 {len(breaks)} 个结构突变点（{sel}）；"
            f"断点位置(index)：{loc_txt}；{shift_txt}段均值见 segments.csv 与图。{trend_note}"
            " ⚠ 检测的是均值水平突变（非斜率/方差突变）；惩罚越大断点越少"
            "（config penalty_mult 调，或 n_breaks 固定个数）；突变点是数据驱动的探索性结果，"
            "需结合事件/政策时点佐证、非因果。"
        )
        code += [
            "import ruptures as rpt  # 结构突变(变点)检测",
            f"# rpt.Pelt(model='l2', min_size={min_size}).fit(y).predict(pen={pen:.3g})  # 段均值/断点",
        ]
    except Exception as err:
        summary.append(f"结构突变检测失败：{err}")


@register("stl_decomposition")
def _branch_stl_decomposition(ctx: Ctx) -> None:
    # STL (Seasonal-Trend decomposition via Loess): split a series into trend + seasonal + residual,
    # with Hyndman seasonal/trend strength measures. Descriptive (not a forecast/test).
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np

    _excl = {fp.unit_col, fp.time_col}
    value = cfg.get("value") if cfg.get("value") in df.columns else next(
        (c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl), None)
    if value is None:
        summary.append("STL 分解失败：需要一个连续序列。config['value'] 可指定。")
        return
    try:
        import pandas as pd
        from statsmodels.tsa.seasonal import STL

        d2 = df.sort_values(fp.time_col) if (fp.time_col and fp.time_col in df.columns) else df
        y = d2[value].astype(float).dropna().reset_index(drop=True)
        n = len(y)
        if n < 20 or y.nunique() < 5:
            summary.append("STL 分解失败：观测不足（<20）或近常数序列。")
            return
        period, auto = None, False
        try:
            cp = int(cfg["period"]) if cfg.get("period") is not None else None
            if cp and 2 <= cp <= n // 2:
                period = cp
        except (TypeError, ValueError):
            period = None
        if period is None:
            period, auto = _periodogram_period(y.to_numpy(), n), True
        if period is None:
            summary.append("STL 分解失败：未检出明显季节周期，请用 config['period'] 指定"
                           "（如月度=12、季度=4、周=7）。")
            return
        res = STL(y.to_numpy(), period=period, robust=True).fit()
        tr, se, rs = np.asarray(res.trend), np.asarray(res.seasonal), np.asarray(res.resid)
        Fs = max(0.0, 1 - np.var(rs) / np.var(se + rs)) if np.var(se + rs) > 0 else 0.0
        Ft = max(0.0, 1 - np.var(rs) / np.var(tr + rs)) if np.var(tr + rs) > 0 else 0.0
        estimates.update({"period": float(period), "seasonal_strength": round(float(Fs), 3),
                          "trend_strength": round(float(Ft), 3), "n_obs": float(n)})
        pd.DataFrame({"index": range(n), "observed": np.round(y.to_numpy(), 4),
                      "trend": np.round(tr, 4), "seasonal": np.round(se, 4), "resid": np.round(rs, 4)}
                     ).to_csv(d / "stl_components.csv", index=False, encoding="utf-8")
        files.append("stl_components.csv")
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(4, 1, figsize=(8, 7), sharex=True)
            for ax, dat, lab, col in zip(
                axes, [y.to_numpy(), tr, se, rs],
                ["observed", "trend", "seasonal", "resid"],
                ["#333333", "#4C72B0", "#55A868", "#bbbbbb"],
            ):
                ax.plot(dat, color=col, lw=1)
                ax.set_ylabel(lab, fontsize=9)
            axes[-1].set_xlabel("period index")
            axes[0].set_title(f"STL decomposition — {value} (period={period})")
            fig.tight_layout()
            fig.savefig(d / "stl_decomposition.png", dpi=150)
            plt.close(fig)
            files.append("stl_decomposition.png")
        except Exception:
            pass
        seas_word = "强" if Fs >= 0.6 else ("中等" if Fs >= 0.3 else "弱")
        trend_word = "强" if Ft >= 0.6 else ("中等" if Ft >= 0.3 else "弱")
        weak_note = "；⚠ 季节强度弱(Fs<0.3)：该周期下季节性不明显，确认 period 是否合适" if Fs < 0.3 else ""
        src = (f"周期图自动检出={period}（建议人工确认）" if auto else f"config 指定={period}")
        summary.append(
            f"{entry.method} 完成：{value}（n={n}）STL 分解（{src}）；"
            f"季节强度 Fs={Fs:.3f}（{seas_word}）、趋势强度 Ft={Ft:.3f}（{trend_word}）；"
            f"分量见 stl_components.csv 与四联图。{weak_note}"
            " ⚠ STL 是描述性分解（趋势+季节+余项），非预测/检验；周期需正确"
            "（自动检出基于周期图主峰，可 config['period'] 覆盖）；robust=True 降异常值影响。"
        )
        code += [
            "from statsmodels.tsa.seasonal import STL  # STL 季节-趋势分解",
            f"# STL(y, period={period}, robust=True).fit(); 季节强度=1-Var(resid)/Var(seasonal+resid)",
        ]
    except Exception as err:
        summary.append(f"STL 分解失败：{err}")


@register("ardl_bounds")
def _branch_ardl_bounds(ctx: Ctx) -> None:
    # ARDL bounds test (Pesaran-Shin-Smith) for a long-run relationship valid under a mix of
    # I(0)/I(1) regressors, plus the error-correction speed and long-run coefficients.
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    _excl = {fp.unit_col, fp.time_col}
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
    outcome = cfg["outcome"] if cfg.get("outcome") in cont else (cont[0] if cont else None)
    forced = [c for c in (cfg.get("predictors") or cfg.get("regressors") or [])
              if c in df.columns and c != outcome and c not in _excl]
    regs = (forced if forced else [c for c in cont if c != outcome])[:5]
    if outcome is None or not regs:
        summary.append("ARDL 边界检验失败：需要 1 个连续结果 + ≥1 个连续回归变量"
                       "（config outcome/predictors 可指定）。")
        return
    try:
        import pandas as pd
        from statsmodels.tsa.ardl import UECM, ardl_select_order

        d2 = df.sort_values(fp.time_col) if (fp.time_col and fp.time_col in df.columns) else df
        data = d2[[outcome, *regs]].dropna().reset_index(drop=True).astype(float)
        n = len(data)
        if n < 30:
            summary.append("ARDL 边界检验失败：观测不足（<30）。")
            return
        maxlag = max(1, min(4, n // 20))
        sel = ardl_select_order(data[outcome], maxlag=maxlag, exog=data[regs],
                                maxorder=maxlag, ic="aic", trend="c")
        fellback = False
        try:
            ur = UECM.from_ardl(sel.model).fit()
            used_order = sel.model.ardl_order
        except Exception:
            # AIC can drop the exog (0 lags) when there is no relationship -> from_ardl fails;
            # fall back to a forced order-1 ARDL so the bounds test still has the level (x.L1) term.
            fellback = True
            p = max(1, int(sel.model.ardl_order[0]) if (sel.model.ardl_order and sel.model.ardl_order[0]) else 1)
            ur = UECM(data[outcome], lags=p, exog=data[regs], order=1, trend="c").fit()
            used_order = (p,) + (1,) * len(regs)
        # I(2) screen (mirrors cointegration_vecm): ADF on first differences; if a differenced series
        # is still non-stationary it may be I(2), which invalidates the bounds test.
        i2_flag = 0
        try:
            from statsmodels.tsa.stattools import adfuller

            for c_ in [outcome, *regs]:
                if adfuller(data[c_].diff().dropna().to_numpy(), autolag="AIC")[1] > 0.05:
                    i2_flag += 1
        except Exception:
            i2_flag = -1
        bt = ur.bounds_test(case=3)
        F = float(bt.stat)
        lo95 = float(bt.crit_vals.loc[95.0, "lower"])
        up95 = float(bt.crit_vals.loc[95.0, "upper"])
        if F > up95:
            concl = "存在长期(协整)关系（F>I(1)上界）"
        elif F < lo95:
            concl = "无长期关系（F<I(0)下界）"
        else:
            concl = "不确定（F 落在 I(0)/I(1) 界之间）"
        ec = float(ur.params.get(f"{outcome}.L1", float("nan")))  # error-correction speed
        lr = {}
        for r_ in regs:
            key = f"{r_}.L1"
            if key in ur.params.index and ec == ec and abs(ec) > 1e-9:
                lr[r_] = round(-float(ur.params[key]) / ec, 4)
        estimates.update({
            "bounds_F": round(F, 3), "crit_lower_95": round(lo95, 3), "crit_upper_95": round(up95, 3),
            "speed_of_adjustment": round(ec, 4) if ec == ec else float("nan"),
            "ardl_p": float(used_order[0]) if used_order else 1.0,
            "maybe_i2": float(i2_flag), "n_obs": float(n),
        })
        for r_, v in lr.items():
            estimates[f"longrun_{r_}"] = v
        (d / "ardl_uecm_summary.txt").write_text(str(ur.summary()), encoding="utf-8")
        files.append("ardl_uecm_summary.txt")
        pd.DataFrame([{"regressor": r_, "longrun_coef": v} for r_, v in lr.items()]).to_csv(
            d / "ardl_longrun.csv", index=False, encoding="utf-8")
        files.append("ardl_longrun.csv")
        lr_txt = "；".join(f"{r_}={v}" for r_, v in lr.items()) or "—"
        ec_note = ("（负且回拉=支持长期关系）" if (ec == ec and ec < 0)
                   else "（⚠ EC 项非负，长期关系存疑）")
        fb_note = "；注：AIC 删除外生项，已强制 order-1 ARDL 以做边界检验" if fellback else ""
        i2_note = f"；⚠ {i2_flag} 个序列差分后仍非平稳(疑似 I(2))，边界检验或失效" if i2_flag > 0 else ""
        summary.append(
            f"{entry.method} 完成：{outcome} ~ {len(regs)} 个回归变量（ARDL{used_order}，"
            f"trend=c，n={n}）；边界检验 F={F:.3f}（95% 界 [{lo95:.2f}, {up95:.2f}]）→ {concl}；"
            f"误差修正速度 EC={ec:.3f}{ec_note}；长期系数 {lr_txt}。{fb_note}{i2_note}"
            " ⚠ ARDL 边界检验适用 I(0)/I(1) 混合（任一变量 I(2) 则失效）；case=3（不受限常数）；"
            "对滞后阶/确定性项设定敏感；长期关系是统计均衡、非结构因果。"
        )
        code += [
            "from statsmodels.tsa.ardl import ardl_select_order, UECM  # ARDL 边界检验 + ECM",
            "# UECM.from_ardl(ardl_select_order(y, exog=X, ic='aic', trend='c').model).fit().bounds_test(case=3)",
        ]
    except Exception as err:
        summary.append(f"ARDL 边界检验失败：{err}")

