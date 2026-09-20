"""Structural-break detection (Bai-Perron / Chow style).

One analysis, one module (see branches/__init__.py — walk_packages auto-registers it).
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


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
