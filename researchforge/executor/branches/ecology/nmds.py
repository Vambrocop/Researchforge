"""nmds — non-metric multidimensional scaling ordination (Bray-Curtis, sklearn MDS)."""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


@register("nmds")
def _branch_nmds(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    species = [
        c.name
        for c in fp.columns
        if c.kind == "count" and c.name not in {fp.unit_col, fp.time_col}
    ]
    if len(species) < 2 or len(df) < 4:
        summary.append("NMDS 跳过：需要 ≥2 个计数列（物种丰度）与 ≥4 个样点。")
    else:
        try:
            import numpy as np
            import pandas as pd
            from scipy.spatial.distance import pdist, squareform
            from sklearn.manifold import MDS

            mat = df[species].fillna(0).clip(lower=0).astype(float)
            mat = mat[mat.sum(axis=1) > 0]  # Bray-Curtis undefined for empty sites
            if len(mat) < 4:
                summary.append("NMDS 跳过：有效（非空）样点不足 4 个。")
            else:
                dist = squareform(pdist(mat.values, metric="braycurtis"))
                _mds_kwargs = dict(
                    n_components=2,
                    metric=False,
                    dissimilarity="precomputed",
                    random_state=0,
                    n_init=4,
                    max_iter=300,
                )
                try:
                    # sklearn>=1.4: normalized_stress -> Kruskal Stress-1 (comparable to
                    # the ecological <0.1/<0.2 rule of thumb); older sklearn lacks the
                    # kwarg and returns raw (un-normalized) stress instead.
                    mds = MDS(normalized_stress=True, **_mds_kwargs)
                    stress_kind = "normalized (Kruskal Stress-1)"
                except TypeError:
                    mds = MDS(**_mds_kwargs)
                    stress_kind = "raw / un-normalized（旧版 sklearn 无 normalized_stress，"\
                        "数值可能 >1，不可直接套用 <0.1/<0.2 经验阈值）"
                coords = mds.fit_transform(dist)
                labels = [f"site{i + 1}" for i in range(len(mat))]
                pd.DataFrame(
                    np.round(coords, 4), index=labels, columns=["NMDS1", "NMDS2"]
                ).to_csv(d / "nmds_coords.csv", encoding="utf-8")
                files.append("nmds_coords.csv")
                estimates["stress"] = round(float(mds.stress_), 4)
                estimates["n_sites"] = float(len(mat))

                try:
                    import matplotlib

                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.scatter(coords[:, 0], coords[:, 1], s=25)
                    ax.set_xlabel("NMDS1")
                    ax.set_ylabel("NMDS2")
                    ax.set_title(f"NMDS ordination (stress={mds.stress_:.3f})")
                    fig.tight_layout()
                    fig.savefig(d / "nmds_ordination.png", dpi=150)
                    plt.close(fig)
                    files.append("nmds_ordination.png")
                except Exception:
                    pass

                summary.append(
                    f"{entry.method} 完成：{len(species)} 物种 × {len(mat)} 样点 → 2D 排序，"
                    f"stress={mds.stress_:.4f}（{stress_kind}）。"
                    "⚠ stress 的可比性依赖于是否归一化：Kruskal Stress-1（新版 sklearn 默认）"
                    "才适用常见的 <0.1 优/<0.2 可用经验阈值，raw stress 与之量纲不同。"
                )
                code += [
                    "from sklearn.manifold import MDS",
                    "from scipy.spatial.distance import pdist, squareform",
                    "dist = squareform(pdist(mat.values, metric='braycurtis'))",
                    "coords = MDS(n_components=2, metric=False, dissimilarity='precomputed',",
                    "             normalized_stress=True).fit_transform(dist)  # Kruskal Stress-1",
                ]
        except Exception as err:
            summary.append(f"NMDS 失败：{err}")
