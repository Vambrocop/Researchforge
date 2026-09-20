"""Vector autoregression + Granger causality.

One analysis, one module (see branches/__init__.py — walk_packages auto-registers it).
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


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
