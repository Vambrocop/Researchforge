"""STL seasonal-trend decomposition.

One analysis, one module (see branches/__init__.py — walk_packages auto-registers it).
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.branches.timeseries._seasonal import (
    _periodogram_period,
)


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
