"""Branch handler: efa (statistics family).

Each handler unpacks ctx into the same local names run_analysis used and runs the
original branch body verbatim. See executor/_branch_api.py.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.run import _varimax


@register("efa")
def _branch_efa(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    _excl = {fp.unit_col, fp.time_col}
    items = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
    if len(items) < 3:
        summary.append("EFA 失败：需要 ≥3 个连续变量（题项）。")
    else:
        try:
            from sklearn.decomposition import FactorAnalysis
            from sklearn.preprocessing import StandardScaler

            sub = df[items].dropna()
            z = StandardScaler().fit_transform(sub.to_numpy(dtype=float))
            # sampling adequacy (factor_analyzer's KMO/Bartlett work despite its
            # FactorAnalyzer.fit sklearn-compat bug; extraction via sklearn instead).
            kmo = bartlett_p = float("nan")
            try:
                from factor_analyzer.factor_analyzer import (
                    calculate_bartlett_sphericity,
                    calculate_kmo,
                )

                _, bartlett_p = calculate_bartlett_sphericity(sub)
                _, kmo = calculate_kmo(sub)
            except Exception:
                pass
            ev = np.sort(np.linalg.eigvalsh(np.corrcoef(z, rowvar=False)))[::-1]
            n_factors = max(1, int((ev > 1).sum()))  # Kaiser criterion
            fa = FactorAnalysis(n_components=n_factors, random_state=0).fit(z)
            load = _varimax(fa.components_.T)  # (items, factors), varimax-rotated
            ssl = (load**2).sum(axis=0)  # SS loadings per factor
            prop_var = ssl / len(items)
            load_df = pd.DataFrame(
                np.round(load, 4),
                index=items,
                columns=[f"F{i + 1}" for i in range(n_factors)],
            )
            load_df.to_csv(d / "loadings.csv", encoding="utf-8")
            files.append("loadings.csv")
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(range(1, len(ev) + 1), ev, "-o", color="#4C72B0")
                ax.axhline(1.0, color="#C44E52", ls="--", lw=0.8, label="Kaiser (eigenvalue=1)")
                ax.set_xlabel("factor")
                ax.set_ylabel("eigenvalue")
                ax.set_title(f"Scree plot (retained {n_factors} factors)")
                ax.legend(fontsize=8)
                fig.tight_layout()
                fig.savefig(d / "scree.png", dpi=150)
                plt.close(fig)
                files.append("scree.png")
            except Exception:
                pass
            estimates["n_factors"] = float(n_factors)
            estimates["kmo"] = round(float(kmo), 4) if kmo == kmo else -1.0
            estimates["cumulative_variance"] = round(float(prop_var.sum()), 4)
            kmo_txt = (
                f"KMO={kmo:.3f}（{'适合因子分析' if kmo >= 0.6 else '⚠ <0.6 数据不太适合'}）"
                if kmo == kmo
                else "KMO 不可用"
            )
            bart_txt = f"，Bartlett p={bartlett_p:.2g}" if bartlett_p == bartlett_p else ""
            summary.append(
                f"{entry.method} 完成：{len(items)} 个变量 → Kaiser 准则保留 {n_factors} 个因子"
                f"（累计解释方差 {prop_var.sum():.1%}）；{kmo_txt}{bart_txt}；varimax 旋转载荷见 loadings.csv。"
                "⚠ 因子数(特征值>1)是启发式,碎石/平行分析可能不同；EFA 为探索性,确证用 SEM/CFA；"
                "提取用 sklearn FactorAnalysis(factor_analyzer 提取器与本机 sklearn 不兼容)。"
            )
            code += [
                "from sklearn.decomposition import FactorAnalysis  # EFA",
                "# 标准化 -> 相关阵特征值定 Kaiser n -> FactorAnalysis -> 手写 varimax 旋转",
            ]
        except Exception as err:
            summary.append(f"EFA 失败：{err}")
