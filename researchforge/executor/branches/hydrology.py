"""Branch handlers for the HYDROLOGY / ENVIRONMENTAL family.

Three workhorse methods for hydrologists, climatologists and water-resource /
environmental analysts working with time series of flow, rainfall and water
quality:

  - mann_kendall_trend  — non-parametric monotonic-trend test (Mann-Kendall S/Z/p,
                          with the ties correction) + Theil-Sen (Sen's) slope and
                          its rank-based (Gilbert 1987) confidence interval.
                          INFERENCE-BEARING (a hypothesis test →派审).
  - flow_duration_curve — exceedance-probability (flow-duration) curve via Weibull
                          plotting positions; key percentile flows + variability
                          indices. DETERMINISTIC (sort / percentile arithmetic).
  - idf_curve           — rainfall Intensity-Duration-Frequency fit of the Talbot
                          form i = a/(d+b)^c by non-linear least squares
                          (scipy.optimize.curve_fit); honest degrade when no
                          duration+intensity structure is present.

Conventions (CLAUDE.md「引擎约定」):
  * config overrides → cfg.get("<key>"); else auto-detect; defaults still run.
  * estimates is dict[str, float] — ONLY plain floats; float("nan") for N/A.
  * Honest degrade → Chinese "<方法> 跳过：<原因>" appended to summary + return
    (never crash / fabricate).
  * Products: CSV + PNG (matplotlib Agg, ENGLISH plot labels, best-effort
    try/except), Chinese ``summary`` with ⚠ assumption / bias disclosures.
  * The profiler may classify a flow/value/time column as continuous / count /
    datetime / id — we are tolerant and coerce with pd.to_numeric(errors="coerce").

Pure Python (numpy / pandas / matplotlib; scipy.stats.norm for Φ and
scipy.optimize.curve_fit for the IDF fit — both degrade honestly if absent).
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _numeric_cols(ctx: Ctx, exclude=()):
    """Names of columns usable as a numeric value / flow / rainfall series.

    Accepts continuous, count AND id kinds: an integer flow/time column with all-
    distinct values is misclassified as ``id`` by the profiler (CLAUDE.md「id 陷阱」),
    and a small integer stream profiles as ``count`` — both are legitimate numbers.
    """
    excl = set(exclude)
    out = []
    for c in ctx.fp.columns:
        if c.name in excl:
            continue
        if c.kind in ("continuous", "count", "id"):
            out.append(c.name)
    return out


def _num(df, col):
    """Coerce a column to a float numpy array (non-numeric → NaN, never raises)."""
    import pandas as pd

    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _save_fig(d, fname, files, build):
    """best-effort matplotlib figure (Agg). build(plt) draws on the current figure."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        build(plt)
        plt.tight_layout()
        plt.savefig(d / fname, dpi=150)
        plt.close("all")
        files.append(fname)
    except Exception:
        pass


def _norm_cdf(z: float) -> float:
    """Standard-normal CDF Φ(z). Prefer scipy; fall back to math.erf."""
    try:
        from scipy.stats import norm

        return float(norm.cdf(z))
    except Exception:
        import math

        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Standard-normal inverse CDF Φ⁻¹(p). Prefer scipy; rational-approx fallback."""
    try:
        from scipy.stats import norm

        return float(norm.ppf(p))
    except Exception:
        # Acklam's rational approximation (abs error < 1.15e-9).
        import math

        if p <= 0.0:
            return float("-inf")
        if p >= 1.0:
            return float("inf")
        a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
        b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
        dd = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
              3.754408661907416e+00]
        plow, phigh = 0.02425, 1 - 0.02425
        if p < plow:
            q = math.sqrt(-2 * math.log(p))
            return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                   ((((dd[0] * q + dd[1]) * q + dd[2]) * q + dd[3]) * q + 1)
        if p > phigh:
            q = math.sqrt(-2 * math.log(1 - p))
            return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                   ((((dd[0] * q + dd[1]) * q + dd[2]) * q + dd[3]) * q + 1)
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def _resolve_value(ctx: Ctx, key_aliases=("value", "flow", "magnitude")):
    """Resolve the numeric value column.

    config[<alias>] (first alias present in cfg AND in df) → that column; else the
    first continuous column; else the first numeric (continuous/count/id) column.
    Returns (col_name, auto_note) or (None, None) if none found.
    """
    df, cfg, fp = ctx.df, ctx.cfg, ctx.fp
    for alias in key_aliases:
        name = cfg.get(alias)
        if name in df.columns:
            return name, ""
    # prefer a continuous column; else any numeric (count/id tolerated)
    cont = [c.name for c in fp.columns if c.kind == "continuous"]
    if cont:
        chosen = cont[0]
    else:
        nums = _numeric_cols(ctx)
        if not nums:
            return None, None
        chosen = nums[0]
    note = (f"（⚠ 未指定 config['value']，自动取数值列 {chosen} 作为序列；"
            "config['value'] 可显式指定）")
    return chosen, note


def _order_series(ctx: Ctx, vcol: str):
    """Return (x_values, t_index, time_note) ordered by config['time'] / fp.time_col
    / natural row order. x is the value array (NaN-dropped); t_index is the 1..n rank
    position used by Sen's slope (equal spacing — we DISCLOSE that the slope's x is
    the observation rank, not the raw time coordinate)."""
    import numpy as np

    df, cfg, fp = ctx.df, ctx.cfg, ctx.fp
    tcol = cfg.get("time")
    if tcol not in df.columns:
        tcol = fp.time_col if (fp.time_col and fp.time_col in df.columns) else None

    x = _num(df, vcol)
    note = ""
    if tcol is not None:
        # try a numeric/datetime ordering key
        import pandas as pd

        tser = df[tcol]
        tnum = pd.to_numeric(tser, errors="coerce")
        if tnum.notna().all():
            order_key = tnum.to_numpy(dtype=float)
            note = f"（按数值时间列 {tcol} 排序）"
        else:
            tdt = pd.to_datetime(tser, errors="coerce")
            if tdt.notna().all():
                order_key = tdt.astype("int64").to_numpy(dtype=float)
                note = f"（按日期时间列 {tcol} 排序）"
            else:
                order_key = np.arange(len(df), dtype=float)
                note = f"（时间列 {tcol} 无法解析为数值/日期，退回原始行序）"
        order = np.argsort(order_key, kind="mergesort")
        x = x[order]
    # drop NaN values (keep order)
    mask = np.isfinite(x)
    x = x[mask]
    t = np.arange(1, x.size + 1, dtype=float)  # equally-spaced rank positions
    return x, t, note


# ===========================================================================
# 1) mann_kendall_trend — non-parametric monotonic trend + Sen's slope
#    Refs: Mann (1945); Kendall (1975); Sen (1968) Theil-Sen estimator;
#          Gilbert (1987) "Statistical Methods for Environmental Pollution
#          Monitoring" (rank-based Sen-slope CI); Hamed & Rao (1998) modified MK.
#    INFERENCE-BEARING → dispatch an inference-reviewer.
# ===========================================================================
@register("mann_kendall_trend")
def _branch_mann_kendall_trend(ctx: Ctx) -> None:
    d, files, summary, estimates, code = ctx.d, ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    vcol, vnote = _resolve_value(ctx)
    if vcol is None:
        summary.append("Mann-Kendall 趋势检验 跳过：未找到数值序列列（用 config['value'] 指定）。")
        return
    try:
        x, t, tnote = _order_series(ctx, vcol)
        n = int(x.size)
        if n < 4:
            summary.append(
                f"Mann-Kendall 趋势检验 跳过：有效观测过少（n={n}<4），无法稳定估计 Var(S)。")
            return

        try:
            alpha = float(ctx.cfg.get("alpha", 0.05))
        except (TypeError, ValueError):
            alpha = 0.05
        if not (0.0 < alpha < 1.0):
            alpha = 0.05

        # ---- Mann-Kendall S = Σ_{i<j} sign(x_j − x_i) -----------------------
        # for each j, sum sign(x_j − x_i) over i<j (equivalently #greater − #less).
        s = 0
        for j in range(1, n):
            diff = x[j] - x[:j]
            s += int(np.sum(np.sign(diff)))
        s = float(s)

        # ---- Var(S) with the tie correction --------------------------------
        # Var(S) = [n(n−1)(2n+5) − Σ_t t(t−1)(2t+5)] / 18, the inner sum over
        # tied groups of size t. (Distinct values contribute 0.)
        _, counts = np.unique(x, return_counts=True)
        tie_term = float(np.sum([c * (c - 1) * (2 * c + 5) for c in counts if c > 1]))
        var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0

        # ---- continuity-corrected Z ----------------------------------------
        if var_s <= 0:
            z = 0.0
        elif s > 0:
            z = (s - 1.0) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1.0) / np.sqrt(var_s)
        else:
            z = 0.0
        z = float(z)
        p = float(2.0 * (1.0 - _norm_cdf(abs(z))))
        p = min(max(p, 0.0), 1.0)

        if p < alpha and s > 0:
            trend = "increasing"
            trend_cn = "上升趋势（显著）"
        elif p < alpha and s < 0:
            trend = "decreasing"
            trend_cn = "下降趋势（显著）"
        else:
            trend = "no-trend"
            trend_cn = "无显著趋势"

        # ---- Theil-Sen (Sen's) slope: median of pairwise slopes ------------
        # slope_ij = (x_j − x_i) / (t_j − t_i) for i<j (t = equally-spaced ranks).
        slopes = []
        for j in range(1, n):
            dt = t[j] - t[:j]
            with np.errstate(divide="ignore", invalid="ignore"):
                sl = (x[j] - x[:j]) / dt
            slopes.append(sl[np.isfinite(sl)])
        all_slopes = np.concatenate(slopes) if slopes else np.array([])
        all_slopes = np.sort(all_slopes)
        n_pairs = int(all_slopes.size)
        sen_slope = float(np.median(all_slopes)) if n_pairs else float("nan")
        # intercept = median(x) − slope·median(t)  (Sen's intercept)
        intercept = float(np.median(x) - sen_slope * np.median(t)) if n_pairs else float("nan")

        # ---- rank-based Sen-slope CI (Gilbert 1987) ------------------------
        # C_alpha = Z_{1−alpha/2} · sqrt(Var(S)); lower/upper rank limits into the
        # sorted N pairwise slopes: M1 = (N − C)/2, M2 = (N + C)/2 (1-based ranks),
        # CI = [ slope at rank M1 , slope at rank M2+1 ].  Robust for small N.
        sen_lo = float("nan")
        sen_hi = float("nan")
        if n_pairs >= 1 and var_s > 0:
            z_crit = _norm_ppf(1.0 - alpha / 2.0)
            c_alpha = z_crit * np.sqrt(var_s)
            m1 = (n_pairs - c_alpha) / 2.0
            m2 = (n_pairs + c_alpha) / 2.0
            # 1-based ranks M1 and (M2+1) → 0-based indices (M1-1) and M2, clamped.
            lo_idx = int(round(m1)) - 1
            hi_idx = int(round(m2))
            lo_idx = min(max(lo_idx, 0), n_pairs - 1)
            hi_idx = min(max(hi_idx, 0), n_pairs - 1)
            sen_lo = float(all_slopes[lo_idx])
            sen_hi = float(all_slopes[hi_idx])

        # ---- products ------------------------------------------------------
        pd.DataFrame({
            "rank_t": t,
            "value": np.round(x, 8),
            "sen_fit": np.round(intercept + sen_slope * t, 8) if n_pairs else np.full(n, np.nan),
        }).to_csv(d / "mann_kendall_series.csv", index=False, encoding="utf-8")
        files.append("mann_kendall_series.csv")

        estimates.update({
            "mk_s": round(s, 6),
            "mk_z": round(z, 6),
            "mk_p": round(p, 8),
            "sen_slope": round(sen_slope, 8) if sen_slope == sen_slope else float("nan"),
            "sen_slope_low": round(sen_lo, 8) if sen_lo == sen_lo else float("nan"),
            "sen_slope_high": round(sen_hi, 8) if sen_hi == sen_hi else float("nan"),
            "sen_intercept": round(intercept, 8) if intercept == intercept else float("nan"),
            "var_s": round(float(var_s), 6),
            "n": float(n),
            "alpha": round(alpha, 6),
        })

        def _plot(plt):
            fig, ax = plt.subplots(figsize=(8, 4.2))
            ax.plot(t, x, color="#4C72B0", lw=1.0, marker="o", ms=3, label="series")
            if n_pairs:
                ax.plot(t, intercept + sen_slope * t, color="#C44E52", lw=1.8,
                        label=f"Sen slope = {sen_slope:.4g}/step")
            ax.set_xlabel("time order (rank)")
            ax.set_ylabel(str(vcol))
            ax.set_title(f"Mann-Kendall trend: {trend} (Z={z:.2f}, p={p:.3g})")
            ax.legend(fontsize=8)

        _save_fig(d, "mann_kendall.png", files, _plot)

        ci_txt = (f"[{sen_lo:.4g}, {sen_hi:.4g}]" if sen_lo == sen_lo else "（不可估）")
        summary.append(
            f"{ctx.entry.method} 完成：序列={vcol}（n={n}）"
            f"{(' ' + tnote) if tnote else ''}{vnote}；"
            f"Mann-Kendall S={s:.0f}、连续性校正 Z={z:.4f}、双侧 p={p:.4g}（α={alpha:g}）"
            f"→ 判定：{trend_cn}（{trend}）。"
            f"Sen(Theil-Sen)斜率={sen_slope:.6g}/步、{int(round(100 * (1 - alpha)))}% 秩基置信区间 {ci_txt}"
            f"（基于 {n_pairs} 个两两斜率，Gilbert 1987 秩限法）；截距={intercept:.6g}。"
            "明细见 mann_kendall_series.csv 与图。"
            " ⚠ Mann-Kendall 假定观测相互独立——序列自相关（持续性）会虚增显著性、"
            "使 p 偏小。若数据存在自相关，应先做预白化（pre-whitening）或改用"
            "修正 MK（Hamed-Rao 方差修正）来校正有效样本量。"
            " ⚠ 斜率的 x 取等间距秩位 1,2,…,n（每「步」为一个观测间隔）；若实际时间间隔不等，"
            "斜率单位应按真实步长解释。Sen 斜率对异常值稳健，但仍假定趋势为单调（非周期/非突变）。"
        )
        code += [
            "import numpy as np",
            "# Mann-Kendall S = Σ_{i<j} sign(x_j − x_i)",
            "s = sum(np.sign(x[j]-x[:j]).sum() for j in range(1, n))",
            "# Var(S) 含并列校正; 连续性校正 Z; 双侧 p = 2(1−Φ(|Z|))",
            "var_s = (n*(n-1)*(2*n+5) - tie_term) / 18",
            "z = (s-1)/np.sqrt(var_s) if s>0 else ((s+1)/np.sqrt(var_s) if s<0 else 0)",
            "# Theil-Sen 斜率 = 所有两两斜率的中位数",
            "sen = np.median([(x[j]-x[i])/(j-i) for i in range(n) for j in range(i+1, n)])",
        ]
    except Exception as exc:
        summary.append(f"Mann-Kendall 趋势检验 计算失败：{exc}")


# ===========================================================================
# 2) flow_duration_curve — exceedance-probability (flow-duration) curve
#    DETERMINISTIC. Refs: Vogel & Fennessey (1994); Searcy (1959) USGS WSP-1542.
# ===========================================================================
@register("flow_duration_curve")
def _branch_flow_duration_curve(ctx: Ctx) -> None:
    d, files, summary, estimates, code = ctx.d, ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    vcol, vnote = _resolve_value(ctx)
    if vcol is None:
        summary.append("流量历时曲线(FDC) 跳过：未找到数值流量序列（用 config['value'] 指定）。")
        return
    try:
        x = _num(ctx.df, vcol)
        x = x[np.isfinite(x)]
        n = int(x.size)
        if n < 5:
            summary.append(f"流量历时曲线(FDC) 跳过：有效观测过少（n={n}<5）。")
            return

        # sort DESCENDING; Weibull plotting position P_i = i/(n+1) for rank i=1..n
        # (rank 1 = largest flow → smallest exceedance). P_i is the fraction of time
        # the flow is EXCEEDED.
        xs = np.sort(x)[::-1]
        ranks = np.arange(1, n + 1, dtype=float)
        exceed = ranks / (n + 1.0)  # exceedance probability in (0,1)

        def _q_at(p_target: float) -> float:
            """Flow exceeded p_target fraction of the time: interpolate the
            (exceedance → flow) relation. exceed is increasing in rank; flow xs is
            decreasing, so flow is interpolated as a function of exceedance."""
            return float(np.interp(p_target, exceed, xs))

        q5 = _q_at(0.05)
        q10 = _q_at(0.10)
        q33 = _q_at(0.33)
        q50 = _q_at(0.50)
        q66 = _q_at(0.66)
        q90 = _q_at(0.90)
        q95 = _q_at(0.95)

        low_flow_index = (q90 / q50) if q50 != 0 else float("nan")

        # FDC slope between Q33 and Q66 on a natural-log scale (flashiness):
        # slope = (ln(Q33) − ln(Q66)) / (0.66 − 0.33).  Larger ⇒ flashier (steeper
        # curve, more variable flow).  Undefined if a flow is non-positive.
        if q33 > 0 and q66 > 0:
            fdc_slope = float((np.log(q33) - np.log(q66)) / (0.66 - 0.33))
        else:
            fdc_slope = float("nan")

        pd.DataFrame({
            "exceedance_pct": np.round(exceed * 100.0, 6),
            "flow": np.round(xs, 8),
        }).to_csv(d / "flow_duration_curve.csv", index=False, encoding="utf-8")
        files.append("flow_duration_curve.csv")

        estimates.update({
            "q5": round(q5, 8), "q10": round(q10, 8), "q50": round(q50, 8),
            "q90": round(q90, 8), "q95": round(q95, 8),
            "low_flow_index": round(low_flow_index, 8) if low_flow_index == low_flow_index else float("nan"),
            "fdc_slope": round(fdc_slope, 8) if fdc_slope == fdc_slope else float("nan"),
            "n": float(n),
        })

        def _plot(plt):
            fig, ax = plt.subplots(figsize=(8, 4.4))
            ax.plot(exceed * 100.0, xs, color="#4C72B0", lw=1.6)
            if np.all(xs > 0):
                ax.set_yscale("log")
            for pp, qq in [(5, q5), (50, q50), (90, q90), (95, q95)]:
                ax.axvline(pp, color="#cccccc", ls=":", lw=0.8)
                ax.plot(pp, qq, "o", color="#C44E52", ms=4)
            ax.set_xlabel("exceedance probability (%)")
            ax.set_ylabel(f"{vcol} (log scale)")
            ax.set_title("Flow-Duration Curve (Weibull plotting positions)")

        _save_fig(d, "flow_duration_curve.png", files, _plot)

        lfi_txt = (f"{low_flow_index:.4g}" if low_flow_index == low_flow_index else "不可估")
        sl_txt = (f"{fdc_slope:.4g}" if fdc_slope == fdc_slope else "不可估(含非正流量)")
        summary.append(
            f"{ctx.entry.method} 完成：流量序列={vcol}（n={n}，Weibull 位置 P=i/(n+1)）{vnote}；"
            f"关键超越百分位流量 Q5={q5:.4g}、Q10={q10:.4g}、Q50(中位)={q50:.4g}、"
            f"Q90={q90:.4g}、Q95={q95:.4g}（Qx=被超越 x% 时间的流量）；"
            f"低流指数 Q90/Q50={lfi_txt}（越小越易枯水）、FDC 斜率(Q33–Q66, 对数)={sl_txt}"
            "（越大越「闪洪」/流量越多变）。曲线见 flow_duration_curve.png、明细见 CSV。"
            " ⚠ FDC 只刻画「量值-频率」关系，忽略时间次序与持续性（不是水文过程线）——"
            "两条时序完全不同的序列可有相同的 FDC，不能据此判断季节/事件时序。"
            " ⚠ 百分位由经验绘点位置插值得到，受样本期长度与代表性影响；对数纵轴要求流量为正"
            "（含 0 或负值时按线性轴绘制，且对数斜率不可估）。"
        )
        code += [
            "import numpy as np",
            "xs = np.sort(x)[::-1]                       # 降序",
            "exceed = np.arange(1, n+1) / (n+1)          # Weibull 超越概率 P=i/(n+1)",
            "Qx = np.interp(p, exceed, xs)               # 被超越 p 比例时间的流量",
            "low_flow_index = Q90 / Q50                  # 低流指数",
            "fdc_slope = (np.log(Q33)-np.log(Q66))/(0.66-0.33)  # 对数 FDC 斜率",
        ]
    except Exception as exc:
        summary.append(f"流量历时曲线(FDC) 计算失败：{exc}")


# ===========================================================================
# 3) idf_curve — rainfall Intensity-Duration-Frequency fit (Talbot family)
#    Refs: Talbot / Sherman intensity-duration formulas; Chow, Maidment & Mays
#          "Applied Hydrology" (IDF analysis). Honest degrade when no
#          duration+intensity structure is present.
# ===========================================================================
@register("idf_curve")
def _branch_idf_curve(ctx: Ctx) -> None:
    d, files, summary, estimates, code = ctx.d, ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    df, cfg, fp = ctx.df, ctx.cfg, ctx.fp

    # ---- resolve duration + intensity columns (honest degrade if absent) ---
    dur_col = cfg.get("duration")
    int_col = cfg.get("intensity")
    rp_col = cfg.get("return_period")

    DEGRADE = ("IDF 跳过：需要 历时(duration)+ 强度(intensity) 列（可选 return_period）；"
               "未检测到，用 config['duration']/config['intensity'] 指定。")

    if dur_col not in df.columns or int_col not in df.columns:
        # try a light auto-detect by column-name hint, else degrade honestly.
        def _find(*hints):
            for c in fp.columns:
                nm = str(c.name).lower()
                if any(h in nm for h in hints) and c.kind in ("continuous", "count", "id"):
                    return c.name
            return None

        if dur_col not in df.columns:
            dur_col = _find("durat", "dur_", "minutes", "time_min")
        if int_col not in df.columns:
            int_col = _find("intens", "intensity", "rain_int", "mm_hr", "mmhr")
        if dur_col not in df.columns or int_col not in df.columns:
            summary.append(DEGRADE)
            return
    if rp_col not in df.columns:
        rp_col = None

    try:
        dur = _num(df, dur_col)
        inten = _num(df, int_col)
        mask = np.isfinite(dur) & np.isfinite(inten) & (dur > 0) & (inten > 0)
        if rp_col is not None:
            rp_all = _num(df, rp_col)
            mask = mask & np.isfinite(rp_all)
        dur, inten = dur[mask], inten[mask]
        rp = rp_all[mask] if rp_col is not None else None
        n = int(dur.size)
        if n < 4:
            summary.append(
                f"IDF 跳过：有效 (历时,强度) 点过少（n={n}<4），无法拟合 3 参数曲线。")
            return

        # ---- Talbot/Sherman form: i = a / (d + b)^c -------------------------
        # 3-param non-linear least squares.  STOP-AND-REPORT decision: we fit the
        # 3-param Talbot form i=a/(d+b)^c as the default (documented in the entry).
        def _talbot(dd, a, b, c):
            return a / np.power(dd + b, c)

        def _fit_talbot(dvals, ivals):
            """Return (a, b, c, r2) or (nan,nan,nan,nan) on failure. Tries scipy
            curve_fit; falls back to a log-linear fit with b=0 (i≈a·d^(−c))."""
            try:
                from scipy.optimize import curve_fit

                # initial guesses: a~max intensity·duration^c, b small, c~0.7
                p0 = [float(np.max(ivals) * np.max(dvals) ** 0.7), 1.0, 0.7]
                popt, _ = curve_fit(
                    _talbot, dvals, ivals, p0=p0, maxfev=20000,
                    bounds=([1e-9, 0.0, 1e-3], [np.inf, np.inf, 10.0]),
                )
                a, b, c = (float(v) for v in popt)
                pred = _talbot(dvals, a, b, c)
            except Exception:
                # fallback: log-log linear fit ln(i) = ln(a) − c·ln(d)  (b forced 0)
                try:
                    lx = np.log(dvals)
                    ly = np.log(ivals)
                    slope, icpt = np.polyfit(lx, ly, 1)
                    c = float(-slope)
                    a = float(np.exp(icpt))
                    b = 0.0
                    pred = _talbot(dvals, a, b, c)
                except Exception:
                    return float("nan"), float("nan"), float("nan"), float("nan")
            ss_res = float(np.sum((ivals - pred) ** 2))
            ss_tot = float(np.sum((ivals - np.mean(ivals)) ** 2))
            r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
            return a, b, c, r2

        per_t_rows = []
        if rp_col is not None and np.unique(rp).size >= 2:
            # per-return-period fit; report each curve's params. STOP-AND-REPORT:
            # if a per-T fit is unstable (too few points / NaN), we skip that T and
            # disclose; the overall fit below is always reported as the headline.
            for T in np.unique(rp):
                sel = rp == T
                if int(np.sum(sel)) >= 4:
                    a, b, c, r2 = _fit_talbot(dur[sel], inten[sel])
                    if a == a:  # not NaN
                        per_t_rows.append({"return_period": float(T), "a": round(a, 6),
                                           "b": round(b, 6), "c": round(c, 6),
                                           "r2": round(r2, 6) if r2 == r2 else float("nan"),
                                           "n_points": float(np.sum(sel))})

        # headline overall fit (all points pooled)
        a_o, b_o, c_o, r2_o = _fit_talbot(dur, inten)
        if a_o != a_o:
            summary.append(
                "IDF 跳过：3 参数 Talbot 曲线 i=a/(d+b)^c 拟合失败（数据可能不符该函数形式，"
                "或历时点过于集中）。")
            return

        # predicted intensities at standard durations within observed range
        std_durs = [d_ for d_ in (5.0, 10.0, 30.0, 60.0, 120.0)
                    if dur.min() <= d_ <= dur.max()]
        if not std_durs:
            std_durs = [float(v) for v in np.round(np.linspace(dur.min(), dur.max(), 5), 4)]
        pred_std = [float(_talbot(np.array([dd]), a_o, b_o, c_o)[0]) for dd in std_durs]

        # ---- products ------------------------------------------------------
        out_df = pd.DataFrame({
            "duration": np.round(dur, 6),
            "intensity_obs": np.round(inten, 6),
            "intensity_fit": np.round(_talbot(dur, a_o, b_o, c_o), 6),
        })
        if rp_col is not None:
            out_df.insert(0, "return_period", np.round(rp, 6))
        out_df.to_csv(d / "idf_fit.csv", index=False, encoding="utf-8")
        files.append("idf_fit.csv")
        if per_t_rows:
            pd.DataFrame(per_t_rows).to_csv(d / "idf_params_by_T.csv", index=False,
                                            encoding="utf-8")
            files.append("idf_params_by_T.csv")

        estimates.update({
            "idf_a": round(a_o, 6), "idf_b": round(b_o, 6), "idf_c": round(c_o, 6),
            "idf_r2": round(r2_o, 6) if r2_o == r2_o else float("nan"),
            "n": float(n),
        })

        def _plot(plt):
            fig, ax = plt.subplots(figsize=(8, 4.6))
            grid = np.linspace(max(dur.min(), 1e-6), dur.max(), 100)
            if rp_col is not None and per_t_rows:
                colors = ["#4C72B0", "#C44E52", "#55A868", "#8172B3", "#CCB974",
                          "#64B5CD", "#937860"]
                for k, row in enumerate(per_t_rows):
                    sel = rp == row["return_period"]
                    col = colors[k % len(colors)]
                    ax.scatter(dur[sel], inten[sel], s=18, color=col, alpha=0.7)
                    ax.plot(grid, _talbot(grid, row["a"], row["b"], row["c"]),
                            color=col, lw=1.4, label=f"T={row['return_period']:g}")
            else:
                ax.scatter(dur, inten, s=20, color="#4C72B0", alpha=0.7, label="observed")
                ax.plot(grid, _talbot(grid, a_o, b_o, c_o), color="#C44E52", lw=1.8,
                        label=f"fit i=a/(d+b)^c (R2={r2_o:.3f})")
            if np.all(dur > 0) and np.all(inten > 0):
                ax.set_xscale("log")
                ax.set_yscale("log")
            ax.set_xlabel("duration (log scale)")
            ax.set_ylabel("intensity (log scale)")
            ax.set_title("Intensity-Duration-Frequency (IDF) curve")
            ax.legend(fontsize=8)

        _save_fig(d, "idf_curve.png", files, _plot)

        std_txt = "、".join(f"d={dd:g}→i={pp:.4g}" for dd, pp in zip(std_durs, pred_std))
        rp_note = ""
        if rp_col is not None:
            if per_t_rows:
                rp_note = (f" 检测到重现期列 {rp_col}，已按重现期分别拟合 {len(per_t_rows)} 条曲线"
                           "（参数见 idf_params_by_T.csv）；下列总体参数为汇集全部点的拟合，供概览。")
            else:
                rp_note = (f" 检测到重现期列 {rp_col}，但各重现期点数不足/分 T 拟合不稳定，"
                           "已退回汇集全部点的总体拟合并披露（STOP-AND-REPORT）。")
        summary.append(
            f"{ctx.entry.method} 完成：历时={dur_col}、强度={int_col}"
            f"{('、重现期=' + str(rp_col)) if rp_col is not None else ''}（n={n} 个点）。"
            f"拟合 Talbot 形 i=a/(d+b)^c：a={a_o:.6g}、b={b_o:.6g}、c={c_o:.6g}、"
            f"R²={r2_o:.4f}。标准历时预测强度：{std_txt}。"
            f"{rp_note} 拟合点见 idf_fit.csv、曲线见 idf_curve.png。"
            " ⚠ 外推到观测历时/重现期范围之外有风险（曲线在端点行为对 b、c 极敏感）；"
            "需要足够多、覆盖范围广的历时点，且结果依赖所选函数形式（此处为 3 参数 Talbot；"
            "Sherman/其它形式可能更合适）。"
            " ⚠ IDF 假定降雨强度-历时关系在该重现期下稳定；若历时点稀疏或集中，参数不可靠、"
            "R² 可能虚高。重现期本身的估计（频率分析）未在此完成——本方法只拟合给定点的曲线形状。"
        )
        code += [
            "import numpy as np; from scipy.optimize import curve_fit",
            "talbot = lambda d, a, b, c: a / (d + b)**c   # 强度-历时关系",
            "(a, b, c), _ = curve_fit(talbot, duration, intensity, p0=[..,1,.7])",
            "pred = talbot(duration, a, b, c)",
            "r2 = 1 - np.sum((intensity-pred)**2)/np.sum((intensity-intensity.mean())**2)",
        ]
    except Exception as exc:
        summary.append(f"IDF 计算失败：{exc}")
