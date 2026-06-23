"""Branch handlers for the time_series_diagnostics family.

Three single-series diagnostic methods (pure Python: statsmodels / numpy / scipy /
pandas — no R) that look at the autocorrelation structure / memory of ONE numeric
series, respecting time order via fp.time_col:

  * acf_pacf        — ACF + PACF up to nlags, with CIs and an AR/MA order hint
  * ljung_box       — Ljung-Box white-noise / residual autocorrelation test
  * hurst_exponent  — Hurst exponent via rescaled-range (R/S) analysis (long memory)

Each handler resolves the series (config column else first continuous, sorted by
fp.time_col when present), degrades honestly (series too short / non-numeric /
constant / import missing -> Chinese "<方法>跳过：<原因>" and RETURN; never crash
or fabricate), writes CSV + PNG (matplotlib Agg, ENGLISH plot labels) in try/except,
fills float `estimates`, appends a Chinese `summary` with ⚠ disclosures, and mutates
ctx (never rebinds). See executor/_branch_api.py, branches/timeseries.py and CLAUDE.md.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


# ─────────────────────────────────────────────────────────────────────────────
# Shared single-series resolution (mirrors timeseries.py STL/GARCH idiom):
# config column (value/column) else first continuous; sorted by fp.time_col.
# Returns (series_name, y_numpy, n, time_sorted, problem_msg). When problem_msg is
# not None the caller should append it to summary and RETURN (honest degrade).
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_series(ctx: Ctx, min_n: int, label: str):
    import numpy as np
    import pandas as pd

    df, fp, cfg = ctx.df, ctx.fp, ctx.cfg
    _excl = {fp.unit_col, fp.time_col}
    requested = cfg.get("column") or cfg.get("value")
    if requested and requested in df.columns:
        col = requested
    else:
        col = next(
            (c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl),
            None,
        )
    if col is None:
        return None, None, 0, False, f"{label}跳过：未找到数值序列列（config['column'] 可指定）。"

    time_sorted = bool(fp.time_col and fp.time_col in df.columns)
    d2 = df.sort_values(fp.time_col) if time_sorted else df
    s = d2[col]
    # honest non-numeric guard
    y = pd.to_numeric(s, errors="coerce").dropna()
    if y.size == 0:
        return None, None, 0, time_sorted, f"{label}跳过：列「{col}」非数值或全缺失。"
    yv = y.to_numpy(dtype=float)
    n = int(yv.size)
    if n < min_n:
        return None, None, n, time_sorted, f"{label}跳过：有效观测不足（n={n}<{min_n}）。"
    if float(np.nanstd(yv)) == 0.0 or np.unique(yv).size < 2:
        return None, None, n, time_sorted, f"{label}跳过：序列近常数（无方差），无法做诊断。"
    return col, yv, n, time_sorted, None


def _time_note(time_sorted: bool) -> str:
    return "" if time_sorted else "；⚠ 无时间列，按行序当作时间序列（请确认行序即时序）"


# ─────────────────────────────────────────────────────────────────────────────
# (A) acf_pacf — autocorrelation + partial autocorrelation up to nlags
# ─────────────────────────────────────────────────────────────────────────────
@register("acf_pacf")
def _branch_acf_pacf(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import importlib.util

    if importlib.util.find_spec("statsmodels") is None:
        summary.append("ACF/PACF 跳过：需要 statsmodels 包（未检测到）。安装：pip install statsmodels。")
        return

    col, y, n, time_sorted, problem = _resolve_series(ctx, min_n=10, label="ACF/PACF")
    if problem is not None:
        summary.append(problem)
        return
    try:
        import numpy as np
        import pandas as pd
        from statsmodels.tsa.stattools import acf, pacf

        # nlags: config else min(40, n//2 - 1); clamp to [1, n//2 - 1]
        default_nlags = min(40, n // 2 - 1)
        try:
            nlags = int(cfg["nlags"]) if cfg.get("nlags") is not None else default_nlags
        except (TypeError, ValueError):
            nlags = default_nlags
        nlags = max(1, min(nlags, n // 2 - 1))

        acf_vals = acf(y, nlags=nlags, fft=True)            # index 0 = lag 0 (=1.0)
        # pacf: ywm (Yule-Walker, modified) is robust and requires nlags < n//2
        pacf_vals = pacf(y, nlags=nlags, method="ywm")
        band = 1.96 / np.sqrt(n)                            # white-noise ±band

        lags = list(range(1, nlags + 1))
        acf_l = acf_vals[1:nlags + 1]
        pacf_l = pacf_vals[1:nlags + 1]

        sig_acf = [k for k in range(nlags) if abs(acf_l[k]) > band]
        sig_pacf = [k for k in range(nlags) if abs(pacf_l[k]) > band]
        n_sig_acf = len(sig_acf)
        # order hints from the CUTOFF = the initial consecutive-significant run (Box-Jenkins
        # reading), robust to spurious high-lag spikes that the "last significant lag" rule
        # would wrongly inflate. acf_l[k] is lag k+1, so a run k=0,1,.. = lags 1,2,..
        def _cutoff(sig_list):
            sset = set(sig_list)
            m = 0
            while m in sset:
                m += 1
            return float(m)

        suggested_ar = _cutoff(sig_pacf)   # PACF cutoff -> AR order
        suggested_ma = _cutoff(sig_acf)    # ACF cutoff -> MA order

        acf_lag1 = float(acf_l[0])
        pacf_lag1 = float(pacf_l[0])

        estimates.update({
            "acf_lag1": round(acf_lag1, 4),
            "pacf_lag1": round(pacf_lag1, 4),
            "n_sig_acf_lags": float(n_sig_acf),
            "suggested_ar_order": suggested_ar,
            "suggested_ma_order": suggested_ma,
            "n": float(n),
        })

        # CSV: lag / acf / acf_ci(±band) / pacf / pacf_ci(±band)
        try:
            pd.DataFrame({
                "lag": lags,
                "acf": np.round(acf_l, 5),
                "acf_ci": round(float(band), 5),
                "pacf": np.round(pacf_l, 5),
                "pacf_ci": round(float(band), 5),
            }).to_csv(d / "acf_pacf.csv", index=False, encoding="utf-8")
            files.append("acf_pacf.csv")
        except Exception:
            pass

        # PNG: ACF + PACF stem plots with the ±band
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            for ax, vals, ttl in (
                (ax1, acf_l, "ACF (autocorrelation)"),
                (ax2, pacf_l, "PACF (partial autocorrelation)"),
            ):
                markerline, stemlines, baseline = ax.stem(lags, vals)
                plt.setp(stemlines, color="#4C72B0")
                plt.setp(markerline, color="#4C72B0", markersize=4)
                plt.setp(baseline, color="black", lw=0.8)
                ax.axhline(band, color="#C44E52", ls="--", lw=1)
                ax.axhline(-band, color="#C44E52", ls="--", lw=1)
                ax.axhspan(-band, band, color="#C44E52", alpha=0.08)
                ax.set_xlabel("lag")
                ax.set_ylabel("correlation")
                ax.set_title(ttl)
            fig.suptitle(f"ACF / PACF — {col} (n={n}, nlags={nlags})")
            fig.tight_layout()
            fig.savefig(d / "acf_pacf.png", dpi=150)
            plt.close(fig)
            files.append("acf_pacf.png")
        except Exception:
            pass

        nonstat = "；⚠ 滞后1 ACF≈1（近单位根/趋势）——请先差分/去趋势再解读" if acf_lag1 > 0.95 else ""
        hint = (
            f"PACF 在滞后{int(suggested_ar)}后截断 → AR({int(suggested_ar)}) 倾向"
            if suggested_ar and suggested_ma < suggested_ar
            else ""
        )
        order_txt = (
            f"阶数提示：AR≈{int(suggested_ar)}（PACF 初始连续显著段的截断处）、"
            f"MA≈{int(suggested_ma)}（ACF 初始连续显著段的截断处）"
        )
        summary.append(
            f"{entry.method} 完成：{col}（n={n}，nlags={nlags}）；"
            f"滞后1 ACF={acf_lag1:.3f}、滞后1 PACF={pacf_lag1:.3f}；"
            f"显著 ACF 滞后数={n_sig_acf}（超出 ±1.96/√n={band:.3f} 带）；{order_txt}"
            f"{('（' + hint + '）') if hint else ''}。{nonstat}{_time_note(time_sorted)}"
            " ⚠ ACF/PACF 假定（弱）平稳——若有趋势请先差分/去趋势；"
            "显著性带为白噪声近似 ±1.96/√n（逐滞后、未做多重比较校正）；阶数提示仅作建模起点。"
        )
        code += [
            "from statsmodels.tsa.stattools import acf, pacf  # ACF / PACF + 显著性带",
            f"acf_vals = acf(y, nlags={nlags}, fft=True); pacf_vals = pacf(y, nlags={nlags}, method='ywm')",
            "band = 1.96/np.sqrt(len(y))  # white-noise ±band; |r|>band = significant lag",
        ]
    except Exception as err:
        summary.append(f"ACF/PACF 失败：{err}")


# ─────────────────────────────────────────────────────────────────────────────
# (B) ljung_box — Ljung-Box test for autocorrelation (white-noise / residual check)
# ─────────────────────────────────────────────────────────────────────────────
@register("ljung_box")
def _branch_ljung_box(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import importlib.util

    if importlib.util.find_spec("statsmodels") is None:
        summary.append("Ljung-Box 跳过：需要 statsmodels 包（未检测到）。安装：pip install statsmodels。")
        return

    col, y, n, time_sorted, problem = _resolve_series(ctx, min_n=12, label="Ljung-Box")
    if problem is not None:
        summary.append(problem)
        return
    try:
        import numpy as np  # noqa: F401  (kept for parity / potential reuse)
        import pandas as pd
        from statsmodels.stats.diagnostic import acorr_ljungbox

        # lags: config (int or list) else [10] (or up to n//5 when n is small)
        cfg_lags = cfg.get("lags")
        if isinstance(cfg_lags, int) and cfg_lags >= 1:
            lags = [cfg_lags]
        elif isinstance(cfg_lags, (list, tuple)) and cfg_lags:
            lags = sorted({int(x) for x in cfg_lags if int(x) >= 1})
        else:
            default_lag = 10 if n >= 50 else max(1, n // 5)
            lags = [default_lag]
        # a lag must be < n; clamp/drop invalid ones
        lags = [int(L) for L in lags if 1 <= int(L) <= n - 1]
        if not lags:
            lags = [max(1, min(10, n - 1))]
        lags = sorted(set(lags))

        lb = acorr_ljungbox(y, lags=lags, return_df=True)
        # acorr_ljungbox is indexed by lag; columns lb_stat / lb_pvalue
        rows = []
        for L in lags:
            Q = float(lb.loc[L, "lb_stat"])
            p = float(lb.loc[L, "lb_pvalue"])
            rows.append({"lag": L, "Q": round(Q, 4), "p": round(p, 5)})
        res_df = pd.DataFrame(rows)

        pvals = [r["p"] for r in rows]
        min_p = float(min(pvals))
        is_white_noise = 1.0 if all(p >= 0.05 for p in pvals) else 0.0

        # estimate at lag 10 specifically (NaN if 10 not testable)
        lb10 = acorr_ljungbox(y, lags=[10], return_df=True) if 10 <= n - 1 else None
        lb_stat10 = float(lb10.loc[10, "lb_stat"]) if lb10 is not None else float("nan")
        lb_p10 = float(lb10.loc[10, "lb_pvalue"]) if lb10 is not None else float("nan")

        estimates.update({
            "lb_stat_lag10": round(lb_stat10, 4) if lb_stat10 == lb_stat10 else float("nan"),
            "lb_p_lag10": round(lb_p10, 5) if lb_p10 == lb_p10 else float("nan"),
            "min_p": round(min_p, 5),
            "n_lags_tested": float(len(lags)),
            "is_white_noise": is_white_noise,
            "n": float(n),
        })

        try:
            res_df.to_csv(d / "ljung_box.csv", index=False, encoding="utf-8")
            files.append("ljung_box.csv")
        except Exception:
            pass

        # PNG: p-value by lag with the 0.05 line
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot([r["lag"] for r in rows], pvals, "o-", color="#4C72B0", label="Ljung-Box p-value")
            ax.axhline(0.05, color="#C44E52", ls="--", lw=1, label="0.05 significance")
            ax.set_ylim(-0.02, max(1.0, max(pvals) + 0.05))
            ax.set_xlabel("lag")
            ax.set_ylabel("p-value")
            ax.set_title(f"Ljung-Box test — {col} (white-noise check)")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "ljung_box.png", dpi=150)
            plt.close(fig)
            files.append("ljung_box.png")
        except Exception:
            pass

        verdict = (
            "未拒绝白噪声（所有滞后 p≥0.05，无显著自相关）"
            if is_white_noise
            else f"拒绝白噪声（存在显著自相关，最小 p={min_p:.3g}<0.05）—— 序列仍有结构"
        )
        lag_txt = "、".join(f"lag{r['lag']}: Q={r['Q']}, p={r['p']}" for r in rows)
        summary.append(
            f"{entry.method} 完成：{col}（n={n}，检验滞后 {lags}）；{lag_txt}；{verdict}。"
            f"{_time_note(time_sorted)}"
            " ⚠ Ljung-Box 原假设是「到滞后 k 无自相关」，拒绝 ⇒ 仍有结构/非白噪声；"
            "用于模型残差时应按估计参数个数调整自由度（df）；滞后数会影响结论（config['lags'] 可指定）。"
        )
        code += [
            "from statsmodels.stats.diagnostic import acorr_ljungbox  # Ljung-Box 白噪声检验",
            f"lb = acorr_ljungbox(y, lags={lags}, return_df=True)  # any p<0.05 ⇒ autocorrelation remains",
        ]
    except Exception as err:
        summary.append(f"Ljung-Box 失败：{err}")


# ─────────────────────────────────────────────────────────────────────────────
# (C) hurst_exponent — Hurst exponent via rescaled-range (R/S) analysis
# ─────────────────────────────────────────────────────────────────────────────
def _rescaled_range(y, window_sizes):
    """Average rescaled range (R/S) for each window size via classic R/S analysis.

    For a window size w: split the series into non-overlapping chunks of length w;
    in each chunk, compute the range R of the cumulative deviations from the chunk
    mean, divide by the chunk standard deviation S, and average R/S over chunks.
    Returns (ws_used, rs_used) keeping only windows with a finite positive R/S.
    """
    import numpy as np

    n = len(y)
    ws_used, rs_used = [], []
    for w in window_sizes:
        if w < 8 or w > n // 2:
            continue
        n_chunks = n // w
        if n_chunks < 1:
            continue
        rs_vals = []
        for c in range(n_chunks):
            chunk = y[c * w:(c + 1) * w]
            mean = chunk.mean()
            dev = chunk - mean
            Z = np.cumsum(dev)             # cumulative deviation
            R = Z.max() - Z.min()          # range
            S = chunk.std(ddof=0)          # standard deviation
            if S > 0 and R > 0:
                rs_vals.append(R / S)
        if rs_vals:
            ws_used.append(w)
            rs_used.append(float(np.mean(rs_vals)))
    return ws_used, rs_used


@register("hurst_exponent")
def _branch_hurst_exponent(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    # R/S Hurst needs a reasonable span; require at least ~32 obs to fit a slope
    col, y, n, time_sorted, problem = _resolve_series(ctx, min_n=32, label="Hurst 指数")
    if problem is not None:
        summary.append(problem)
        return
    try:
        import numpy as np
        import pandas as pd

        # geometric grid of window sizes from 8 up to n//2
        max_w = max(8, n // 2)
        ws = sorted({int(x) for x in np.unique(
            np.floor(np.logspace(np.log10(8), np.log10(max_w), num=20)).astype(int)
        ) if 8 <= int(x) <= n // 2})
        ws_used, rs_used = _rescaled_range(y, ws)

        if len(ws_used) < 3:
            summary.append(
                f"Hurst 指数跳过：可用窗口尺寸不足（n={n} 太短，R/S 至少需约 32 点且多个尺度）。"
            )
            return

        log_w = np.log(np.asarray(ws_used, dtype=float))
        log_rs = np.log(np.asarray(rs_used, dtype=float))
        # Hurst H = slope of log(R/S) ~ log(window); fit + R^2
        slope, intercept = np.polyfit(log_w, log_rs, 1)
        H = float(slope)
        pred = slope * log_w + intercept
        ss_res = float(np.sum((log_rs - pred) ** 2))
        ss_tot = float(np.sum((log_rs - log_rs.mean()) ** 2))
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

        # regime encoding: 0 = mean-revert/anti-persistent (<0.45),
        # 1 = random walk (0.45..0.55), 2 = trending/persistent (>0.55)
        if H < 0.45:
            regime = 0.0
            regime_word = "均值回复/反持久（H<0.5）"
        elif H > 0.55:
            regime = 2.0
            regime_word = "趋势/持久（H>0.5，长记忆）"
        else:
            regime = 1.0
            regime_word = "近随机游走（H≈0.5，无长记忆）"

        estimates.update({
            "hurst": round(H, 4),
            "rs_fit_r2": round(r2, 4) if r2 == r2 else float("nan"),
            "regime": regime,
            "n": float(n),
        })

        try:
            pd.DataFrame({
                "window_size": ws_used,
                "log_window": np.round(log_w, 5),
                "log_rs": np.round(log_rs, 5),
            }).to_csv(d / "hurst_rs.csv", index=False, encoding="utf-8")
            files.append("hurst_rs.csv")
        except Exception:
            pass

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(log_w, log_rs, color="#4C72B0", label="log(R/S) observed")
            ax.plot(log_w, pred, color="#C44E52", ls="--",
                    label=f"fit slope H={H:.3f} (R^2={r2:.3f})")
            ax.set_xlabel("log(window size)")
            ax.set_ylabel("log(R/S)")
            ax.set_title(f"Rescaled-range (R/S) Hurst — {col} (n={n})")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "hurst_rs.png", dpi=150)
            plt.close(fig)
            files.append("hurst_rs.png")
        except Exception:
            pass

        short_note = "；⚠ 序列较短（n<100），R/S Hurst 估计偏差较大，请谨慎解读" if n < 100 else ""
        fit_note = "；⚠ log-log 拟合 R² 偏低，标度关系不稳，H 估计不可靠" if (r2 == r2 and r2 < 0.8) else ""
        summary.append(
            f"{entry.method} 完成：{col}（n={n}）；R/S 标度法估得 Hurst H={H:.3f}"
            f"（log-log 拟合 R²={r2:.3f}，用 {len(ws_used)} 个窗口尺度）；"
            f"判读：{regime_word}。{short_note}{fit_note}{_time_note(time_sorted)}"
            " ⚠ H=0.5 随机游走、<0.5 均值回复/反持久、>0.5 趋势/持久（长记忆）；"
            "R/S 法对样本量敏感、短序列有偏（建议 n≥~100）；不同估计法（R/S、DFA 等）结果有差异（此处用 R/S，已披露）。"
        )
        code += [
            "import numpy as np  # Hurst via rescaled-range (R/S) analysis",
            "# for each window w: R = range(cumsum(dev)), S = std(chunk); avg R/S over chunks",
            "# H = slope of log(mean R/S) ~ log(window); H=0.5 random, <0.5 mean-revert, >0.5 persistent",
        ]
    except Exception as err:
        summary.append(f"Hurst 指数失败：{err}")
