"""Branch handlers for the soil family (migrated from the run.py monolith).

Each handler unpacks ctx into the same local names run_analysis used and runs the
original branch body verbatim. See executor/_branch_api.py.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.run import (
    _usda_texture,
)


@register("soil_texture")
def _branch_soil_texture(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    def _find(kw):
        # name-locked to the texture fraction; accept any numeric kind
        # (whole-number distinct % columns can profile as "id").
        return next(
            (
                c.name
                for c in fp.columns
                if kw in c.name.lower() and c.kind in {"continuous", "count", "id"}
            ),
            None,
        )

    sand_c, silt_c, clay_c = _find("sand"), _find("silt"), _find("clay")
    if not (sand_c and silt_c and clay_c):
        summary.append("土壤质地分类失败：需要 sand/silt/clay（砂/粉/黏粒）百分比列。")
    else:
        raw = df[[sand_c, silt_c, clay_c]].dropna().astype(float)
        raw = raw[raw.sum(axis=1) > 0]
        norm = raw.div(raw.sum(axis=1), axis=0) * 100.0  # renormalise rows to sum 100
        classes = [
            _usda_texture(float(r[sand_c]), float(r[silt_c]), float(r[clay_c]))
            for _, r in norm.iterrows()
        ]
        res = norm.round(2)
        res["usda_texture"] = classes
        res.to_csv(d / "soil_texture.csv", index=False, encoding="utf-8")
        files.append("soil_texture.csv")
        dist = pd.Series(classes).value_counts()
        dist.rename_axis("texture_class").reset_index(name="count").to_csv(
            d / "texture_distribution.csv", index=False, encoding="utf-8"
        )
        files.append("texture_distribution.csv")
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            cats = list(dist.index)
            cmap = plt.get_cmap("tab20")
            cidx = {c: cmap(i % 20) for i, c in enumerate(cats)}
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            # soil texture triangle (clay apex at top)
            cl = norm[clay_c].to_numpy()
            si = norm[silt_c].to_numpy()
            x = si + 0.5 * cl
            y = cl * (np.sqrt(3) / 2)
            tri = np.array([[0, 0], [100, 0], [50, 100 * np.sqrt(3) / 2], [0, 0]])
            axes[0].plot(tri[:, 0], tri[:, 1], color="#444", lw=1)
            for c in cats:
                m = np.array([k == c for k in classes])
                axes[0].scatter(x[m], y[m], s=22, color=cidx[c], label=c, edgecolor="#333", linewidth=0.3)
            axes[0].text(0, -4, "sand", ha="center")
            axes[0].text(100, -4, "silt", ha="center")
            axes[0].text(50, 100 * np.sqrt(3) / 2 + 3, "clay", ha="center")
            axes[0].set_title("USDA soil texture triangle")
            axes[0].axis("off")
            axes[0].legend(fontsize=6, loc="upper right", ncol=2)
            axes[1].barh([str(c) for c in cats][::-1], dist.values[::-1], color="#55A868")
            axes[1].set_xlabel("count")
            axes[1].set_title("Texture class distribution")
            fig.tight_layout()
            fig.savefig(d / "soil_texture.png", dpi=150)
            plt.close(fig)
            files.append("soil_texture.png")
        except Exception:
            pass
        dominant = str(dist.index[0])
        estimates["n_samples"] = float(len(norm))
        estimates["n_classes"] = float(len(dist))
        estimates["dominant_class_pct"] = round(100.0 * float(dist.iloc[0]) / len(norm), 2)
        summary.append(
            f"{entry.method} 完成：{len(norm)} 个土样按 USDA 质地三角分入 {len(dist)} 类；"
            f"最多为「{dominant}」（{100.0 * dist.iloc[0] / len(norm):.0f}%）；"
            f"质地三角图见 soil_texture.png。（各行已归一化至砂+粉+黏=100%）"
        )
        code += [
            "# USDA 质地三角分类: 按 sand/silt/clay% 的标准边界规则判类",
            "# silt+1.5*clay<15 -> sand; ... clay>=40&silt<40&sand<=45 -> clay 等 12 类",
        ]

