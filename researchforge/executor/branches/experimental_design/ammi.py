"""Experimental-design family branch handler: ammi (split from experimental_design.py)."""
from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register

from ._shared import _ge_means_matrix, _pick_geno_env


@register("ammi")
def _branch_ammi(ctx: Ctx) -> None:
    """AMMI = Additive Main effects + Multiplicative Interaction. Two-way additive model
    (grand + genotype + environment main effects), then SVD of the G×E interaction
    residual matrix → IPCA axes ranked by % interaction explained, an AMMI-2 biplot, and
    a per-genotype stability readout (IPCA1 magnitude / interaction-residual norm)."""
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
            "AMMI 失败：需要 1 个连续结果 + 基因型 genotype + 环境 environment 两个因子。"
            '用 config={"outcome":..,"genotype":..,"environment":..} 指定角色。'
        )
        return

    sub = df[[y, genotype, environment]].dropna()
    mat = _ge_means_matrix(sub, genotype, environment, y)
    g, e = mat.shape
    if g < 3 or e < 3:
        summary.append(
            f"AMMI 失败：完整 G×E 均值表为 {g} 基因型 × {e} 环境（需各 ≥3 才有可分解的交互结构）。"
            "缺格已删行/列；请提供更完整的多环境试验。"
        )
        return

    try:
        M = mat.to_numpy(dtype=float)
        grand = float(M.mean())
        g_main = M.mean(axis=1) - grand          # genotype main effects
        e_main = M.mean(axis=0) - grand          # environment main effects
        # additive expectation; interaction residual = observed - additive (double-centered)
        additive = grand + g_main[:, None] + e_main[None, :]
        inter = M - additive                     # G×E interaction matrix (row & col centered)

        # SVD of the interaction matrix → IPCA axes
        U, S, Vt = np.linalg.svd(inter, full_matrices=False)
        # number of non-trivial interaction axes = min(g-1, e-1)
        rank = min(g - 1, e - 1)
        S = S[:rank]
        U = U[:, :rank]
        Vt = Vt[:rank, :]
        ss = S ** 2
        total_inter_ss = float(ss.sum())
        pct = (ss / total_inter_ss * 100.0) if total_inter_ss > 1e-12 else np.zeros_like(ss)

        # genotype / environment IPCA scores (symmetric scaling: sqrt(singular value))
        sqrtS = np.sqrt(S)
        g_scores = U * sqrtS[None, :]            # g × rank
        e_scores = (Vt.T) * sqrtS[None, :]       # e × rank

        n_axes = int(rank)
        ipca_df = pd.DataFrame(
            {f"IPCA{i+1}_pct": [pct[i]] for i in range(n_axes)}
        )
        ipca_df.insert(0, "axis_ss", [total_inter_ss])
        ipca_df.to_csv(d / "ammi_ipca_variance.csv", index=False, encoding="utf-8")
        files.append("ammi_ipca_variance.csv")

        # genotype scores + stability: AMMI stability value uses IPCA1/IPCA2 weighted by their %.
        ip1 = g_scores[:, 0]
        ip2 = g_scores[:, 1] if n_axes >= 2 else np.zeros(g)
        w1 = pct[0] / pct[1] if n_axes >= 2 and pct[1] > 1e-12 else 1.0
        asv = np.sqrt((w1 * ip1) ** 2 + ip2 ** 2)   # AMMI stability value (smaller = more stable)
        inter_norm = np.sqrt((inter ** 2).sum(axis=1))  # genotype interaction residual norm
        geno_tbl = pd.DataFrame({
            "genotype": [str(x) for x in mat.index],
            "mean": M.mean(axis=1),
            "main_effect": g_main,
            "IPCA1": ip1,
            "IPCA2": ip2,
            "ASV_stability": asv,            # smaller = more stable across environments
            "interaction_norm": inter_norm,
        }).sort_values("ASV_stability")
        geno_tbl.to_csv(d / "ammi_genotype_stability.csv", index=False, encoding="utf-8")
        files.append("ammi_genotype_stability.csv")

        estimates["n_genotypes"] = float(g)
        estimates["n_environments"] = float(e)
        estimates["n_ipca_axes"] = float(n_axes)
        estimates["interaction_ss"] = total_inter_ss
        estimates["IPCA1_pct"] = float(pct[0])
        if n_axes >= 2:
            estimates["IPCA2_pct"] = float(pct[1])
            estimates["IPCA1_IPCA2_pct"] = float(pct[0] + pct[1])
        # most stable genotype = smallest ASV
        most_stable = str(geno_tbl.iloc[0]["genotype"])

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6.5, 5.5))
            ax.axhline(0, color="#999", lw=0.8)
            ax.axvline(0, color="#999", lw=0.8)
            ax.scatter(g_scores[:, 0], ip2, color="#4C72B0", marker="o", label="genotype")
            for i, lab in enumerate(mat.index):
                ax.annotate(str(lab), (g_scores[i, 0], ip2[i]), fontsize=8, color="#1f3a66")
            e_ip2 = e_scores[:, 1] if n_axes >= 2 else np.zeros(e)
            ax.scatter(e_scores[:, 0], e_ip2, color="#C44E52", marker="^", label="environment")
            for j, lab in enumerate(mat.columns):
                ax.annotate(str(lab), (e_scores[j, 0], e_ip2[j]), fontsize=8, color="#7a2a2c")
            ax.set_xlabel(f"IPCA1 ({pct[0]:.1f}% of interaction)")
            ax.set_ylabel(f"IPCA2 ({pct[1]:.1f}% of interaction)" if n_axes >= 2 else "IPCA2")
            ax.set_title(f"AMMI-2 biplot — {y}")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "ammi_biplot.png", dpi=150)
            plt.close(fig)
            files.append("ammi_biplot.png")
        except Exception:
            pass

        role_note = "（角色自动猜测，建议 config 明确 genotype/environment）" if guessed else ""
        two = f"，IPCA1+IPCA2 共 {pct[0]+pct[1]:.1f}%" if n_axes >= 2 else ""
        summary.append(
            f"{entry.method} 完成{role_note}：{g} 基因型 × {e} 环境（结果 {y}）。"
            f"加性主效应(基因型+环境)分离后，对 G×E 交互残差做 SVD：共 {n_axes} 条 IPCA 轴，"
            f"IPCA1 解释 {pct[0]:.1f}% 的交互平方和{two}。"
            f"最稳定基因型（AMMI 稳定值 ASV 最小）：{most_stable}。"
            " ⚠ AMMI 是对 G×E 交互的**描述性乘法分解**（非推断检验）；ASV/IPCA 是稳定性度量、口径多样"
            "（也可用 Gauch F 检验定保留轴数，此处仅报 % 解释）；biplot 远离原点=交互大/不稳定，需谨慎解读；"
            "需重复的多环境试验、每 基因型×环境 格有均值；缺格已删整行/列。"
        )
        code += [
            "import numpy as np",
            "M = df.groupby([genotype, environment])[y].mean().unstack().dropna().to_numpy()",
            "inter = M - (M.mean() + (M.mean(1)-M.mean())[:,None] + (M.mean(0)-M.mean())[None,:])",
            "U,S,Vt = np.linalg.svd(inter, full_matrices=False)  # IPCA: % = S**2/sum(S**2)",
        ]
    except Exception as err:
        summary.append(f"AMMI 失败：{err}")
