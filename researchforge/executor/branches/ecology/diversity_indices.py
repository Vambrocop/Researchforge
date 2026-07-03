"""diversity_indices — Shannon, Simpson, richness, total abundance per site."""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


@register("diversity_indices")
def _branch_diversity_indices(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    species = [
        c.name
        for c in fp.columns
        if c.kind == "count" and c.name not in {fp.unit_col, fp.time_col}
    ]
    if len(species) < 2:
        summary.append("多样性指数跳过：未找到 ≥2 个计数列（物种丰度矩阵）。")
    else:
        mat = df[species].fillna(0).clip(lower=0).astype(float)

        def _shannon(counts):
            total = counts.sum()
            if total <= 0:
                return 0.0
            p = counts[counts > 0] / total
            return float(-(p * np.log(p)).sum())

        def _simpson(counts):
            total = counts.sum()
            if total <= 0:
                return 0.0
            p = counts / total
            return float(1.0 - (p ** 2).sum())

        div = pd.DataFrame(
            {
                "shannon": mat.apply(_shannon, axis=1).round(4),
                "simpson": mat.apply(_simpson, axis=1).round(4),
                "richness": (mat > 0).sum(axis=1).astype(int),
                "total_abundance": mat.sum(axis=1),
            }
        )
        div.to_csv(d / "diversity.csv", encoding="utf-8")
        files.append("diversity.csv")
        estimates["mean_shannon"] = float(div["shannon"].mean())
        estimates["mean_richness"] = float(div["richness"].mean())
        estimates["n_species"] = float(len(species))
        summary.append(
            f"{entry.method} 完成：{len(species)} 个物种 × {len(df)} 个样点，"
            f"平均 Shannon={div['shannon'].mean():.3f}，平均丰富度={div['richness'].mean():.2f}"
        )
        code += [
            "import numpy as np",
            f"mat = df[{species!r}].fillna(0)",
            "# 每行(样点): Shannon=-sum(p*ln p), Simpson=1-sum(p^2), richness=present species",
        ]
