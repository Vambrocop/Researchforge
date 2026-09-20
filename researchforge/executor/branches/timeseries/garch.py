"""GARCH volatility modelling.

One analysis, one module (see branches/__init__.py — walk_packages auto-registers it).
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


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
