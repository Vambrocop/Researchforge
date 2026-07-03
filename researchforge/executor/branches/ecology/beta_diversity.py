"""beta_diversity — Bray-Curtis dissimilarity across sites (pairwise turnover)."""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


@register("beta_diversity")
def _branch_beta_diversity(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    species = [
        c.name
        for c in fp.columns
        if c.kind == "count" and c.name not in {fp.unit_col, fp.time_col}
    ]
    if len(species) < 2 or len(df) < 2:
        summary.append("Beta 多样性跳过：需要 ≥2 个计数列（物种丰度）与 ≥2 个样点。")
    else:
        try:
            from scipy.spatial.distance import pdist, squareform

            mat = df[species].fillna(0).clip(lower=0).astype(float).values
            mat = mat[mat.sum(axis=1) > 0]  # Bray-Curtis undefined for empty sites
            if len(mat) < 2:
                summary.append("Beta 多样性跳过：有效（非空）样点不足 2 个。")
            else:
                dist = squareform(pdist(mat, metric="braycurtis"))
                labels = [f"site{i + 1}" for i in range(len(mat))]
                pd.DataFrame(np.round(dist, 4), index=labels, columns=labels).to_csv(
                    d / "bray_curtis.csv", encoding="utf-8"
                )
                files.append("bray_curtis.csv")

                iu = np.triu_indices(len(mat), k=1)
                mean_bc = float(np.nanmean(dist[iu])) if iu[0].size else 0.0
                estimates["mean_bray_curtis"] = round(mean_bc, 4)
                estimates["n_sites"] = float(len(mat))

                try:
                    import matplotlib

                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(5, 4))
                    im = ax.imshow(dist, cmap="viridis", vmin=0, vmax=1)
                    fig.colorbar(im)
                    ax.set_title("Bray-Curtis dissimilarity")
                    fig.tight_layout()
                    fig.savefig(d / "bray_curtis_heatmap.png", dpi=150)
                    plt.close(fig)
                    files.append("bray_curtis_heatmap.png")
                except Exception:
                    pass

                summary.append(
                    f"{entry.method} 完成：{len(species)} 个物种 × {len(mat)} 个样点，"
                    f"平均 Bray-Curtis 相异度={mean_bc:.3f}"
                )
                code += [
                    "from scipy.spatial.distance import pdist, squareform",
                    f"mat = df[{species!r}].fillna(0).values",
                    "mat = mat[mat.sum(axis=1) > 0]  # Bray-Curtis undefined for empty sites",
                    "dist = squareform(pdist(mat, metric='braycurtis'))",
                ]
        except Exception as err:
            summary.append(f"Beta 多样性失败：{err}")
