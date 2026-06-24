"""Experimental-design family branch handler: gge_biplot (split from experimental_design.py)."""
from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register

from ._shared import _ge_means_matrix, _pick_geno_env


@register("gge_biplot")
def _branch_gge_biplot(ctx: Ctx) -> None:
    """GGE biplot = SVD of the environment-centered Genotype + Genotype×Environment means.
    Unlike AMMI, the genotype main effect is KEPT (only the environment main effect is
    removed by column-centering). Reports PC1/PC2 variance explained and a which-won-where
    readout (winning genotype per environment from the biplot approximation)."""
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    _excl = {fp.unit_col, fp.time_col}
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
    y = cfg["outcome"] if cfg.get("outcome") in cont else (cont[0] if cont else None)
    genotype, environment, guessed = _pick_geno_env(fp, df, cfg, y)

    if y is None or genotype is None or environment is None or genotype == environment:
        summary.append(
            "GGE biplot 失败：需要 1 个连续结果 + 基因型 genotype + 环境 environment 两个因子。"
            '用 config={"outcome":..,"genotype":..,"environment":..} 指定角色。'
        )
        return

    sub = df[[y, genotype, environment]].dropna()
    mat = _ge_means_matrix(sub, genotype, environment, y)
    g, e = mat.shape
    if g < 3 or e < 3:
        summary.append(
            f"GGE biplot 失败：完整 G×E 均值表为 {g} 基因型 × {e} 环境（需各 ≥3）。"
            "缺格已删行/列；请提供更完整的多环境试验。"
        )
        return

    try:
        M = mat.to_numpy(dtype=float)
        # environment-centering: remove each environment (column) mean → retains G + G×E
        col_mean = M.mean(axis=0)
        centered = M - col_mean[None, :]

        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        # column-centering drops one df → true rank is min(g-1, e) (mirrors AMMI's min(g-1,e-1));
        # avoids a trailing 0%-variance PC row in gge_variance.csv when g <= e.
        rank = min(g - 1, e)
        S = S[:rank]
        U = U[:, :rank]
        Vt = Vt[:rank, :]
        ss = S ** 2
        total = float(ss.sum())
        pct = (ss / total * 100.0) if total > 1e-12 else np.zeros_like(ss)

        # symmetric scaling for biplot coordinates
        sqrtS = np.sqrt(S)
        g_scores = U * sqrtS[None, :]      # genotype coords
        e_scores = (Vt.T) * sqrtS[None, :]  # environment coords
        n_pc = int(rank)

        var_df = pd.DataFrame({"PC": [f"PC{i+1}" for i in range(n_pc)],
                               "pct_variance": pct})
        var_df.to_csv(d / "gge_variance.csv", index=False, encoding="utf-8")
        files.append("gge_variance.csv")

        # which-won-where: winning genotype per environment = genotype with the highest
        # biplot projection onto that environment vector (rank-2 GGE approximation).
        g2 = g_scores[:, :2] if n_pc >= 2 else np.column_stack([g_scores[:, 0], np.zeros(g)])
        e2 = e_scores[:, :2] if n_pc >= 2 else np.column_stack([e_scores[:, 0], np.zeros(e)])
        proj = g2 @ e2.T                  # g × e projection
        winners_idx = proj.argmax(axis=0)
        www = pd.DataFrame({
            "environment": [str(c) for c in mat.columns],
            "winning_genotype": [str(mat.index[i]) for i in winners_idx],
        })
        www.to_csv(d / "gge_which_won_where.csv", index=False, encoding="utf-8")
        files.append("gge_which_won_where.csv")

        estimates["n_genotypes"] = float(g)
        estimates["n_environments"] = float(e)
        estimates["PC1_pct"] = float(pct[0])
        if n_pc >= 2:
            estimates["PC2_pct"] = float(pct[1])
            estimates["PC1_PC2_pct"] = float(pct[0] + pct[1])
        estimates["n_winning_genotypes"] = float(len(set(winners_idx.tolist())))

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            gy = g2[:, 1]
            ey = e2[:, 1]
            fig, ax = plt.subplots(figsize=(6.5, 5.5))
            ax.axhline(0, color="#999", lw=0.8)
            ax.axvline(0, color="#999", lw=0.8)
            ax.scatter(g2[:, 0], gy, color="#4C72B0", marker="o", label="genotype")
            for i, lab in enumerate(mat.index):
                ax.annotate(str(lab), (g2[i, 0], gy[i]), fontsize=8, color="#1f3a66")
            # environments drawn as vectors from origin
            for j, lab in enumerate(mat.columns):
                ax.annotate("", xy=(e2[j, 0], ey[j]), xytext=(0, 0),
                            arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.0))
                ax.annotate(str(lab), (e2[j, 0], ey[j]), fontsize=8, color="#7a2a2c")
            ax.set_xlabel(f"PC1 ({pct[0]:.1f}%)")
            ax.set_ylabel(f"PC2 ({pct[1]:.1f}%)" if n_pc >= 2 else "PC2")
            ax.set_title(f"GGE biplot — {y}")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "gge_biplot.png", dpi=150)
            plt.close(fig)
            files.append("gge_biplot.png")
        except Exception:
            pass

        role_note = "（角色自动猜测，建议 config 明确 genotype/environment）" if guessed else ""
        two = f"，PC1+PC2 共 {pct[0]+pct[1]:.1f}%" if n_pc >= 2 else ""
        n_win = len(set(winners_idx.tolist()))
        win_note = (f"{n_win} 个基因型在不同环境胜出（存在 which-won-where 分区）"
                    if n_win > 1 else "单一基因型在所有环境胜出（无明显分区）")
        summary.append(
            f"{entry.method} 完成{role_note}：{g} 基因型 × {e} 环境（结果 {y}）。"
            f"环境中心化(去环境主效应、保留 G+G×E)后做 SVD：PC1 解释 {pct[0]:.1f}% 变异{two}。"
            f"Which-won-where：{win_note}（详见 gge_which_won_where.csv）。"
            " ⚠ GGE 是对 基因型+G×E 的**描述性双标图**（非推断检验）；中心化/标度方式（环境中心化、"
            "对称标度）会改变图形与解读；biplot 仅是低秩近似，which-won-where 是近似投影、非显著性判定；"
            "需重复的多环境试验、每 基因型×环境 格有均值；缺格已删整行/列。"
        )
        code += [
            "import numpy as np",
            "M = df.groupby([genotype, environment])[y].mean().unstack().dropna().to_numpy()",
            "centered = M - M.mean(0)[None,:]  # environment-centered → keeps G + GxE",
            "U,S,Vt = np.linalg.svd(centered, full_matrices=False)  # PC%% = S**2/sum(S**2)",
        ]
    except Exception as err:
        summary.append(f"GGE biplot 失败：{err}")
