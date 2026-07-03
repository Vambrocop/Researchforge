"""indicator_species — IndVal (Dufrene & Legendre 1997) indicator-value analysis."""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


@register("indicator_species")
def _branch_indicator_species(ctx: Ctx) -> None:
    """IndVal (Dufrene & Legendre 1997) indicator-value analysis: for each taxon and
    each group, IndVal = specificity (A, mean-abundance concentration in the group) x
    fidelity (B, frequency of occurrence within the group), with a permutation
    p-value. Reports the best-group indicator value per taxon. Pure numpy."""
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    _excl = {fp.unit_col, fp.time_col}
    auto_species = [
        c.name for c in fp.columns if c.kind == "count" and c.name not in _excl
    ]
    # group default: first binary/categorical column (fewest levels first for cats)
    bin_cols = [c.name for c in fp.columns if c.kind == "binary" and c.name not in _excl]
    cat_cols = [
        c.name for c in fp.columns if c.kind == "categorical" and c.name not in _excl
    ]
    cat_cols.sort(key=lambda name: int(df[name].nunique()))
    auto_group = (bin_cols + cat_cols)[0] if (bin_cols + cat_cols) else None

    group_col = cfg.get("group") or auto_group
    species = list(cfg.get("species") or cfg.get("abundance") or auto_species)
    species = [c for c in species if c in df.columns and c != group_col]
    try:
        n_perm = int(cfg.get("n_perm") or 999)
    except (TypeError, ValueError):
        n_perm = 999
    n_perm = max(99, min(n_perm, 9999))

    if not group_col or group_col not in df.columns or len(species) < 1:
        summary.append(
            "指示种(IndVal)跳过：需要 ≥1 个物种/丰度列（计数）+ 一个分组列。"
            f"（自动检出 species={species[:3]}，group={group_col}；可用 config "
            "group/species 指定。）"
        )
        return

    sub = df[species + [group_col]].dropna()
    g = sub[group_col].astype(str).to_numpy()
    groups = sorted(set(g))
    if len(groups) < 2:
        summary.append("指示种(IndVal)跳过：分组变量需 ≥2 组。")
        return
    mat = sub[species].clip(lower=0).astype(float).to_numpy()
    N = len(sub)
    if N < 6:
        summary.append("指示种(IndVal)跳过：去缺失后有效样本 <6。")
        return

    try:
        masks = {grp: (g == grp) for grp in groups}
        group_sizes = {grp: int(m.sum()) for grp, m in masks.items()}

        def _indval_all(labels_masks: dict) -> np.ndarray:
            """IndVal per (species, group): A * B * 100. Returns array [n_sp, n_grp]."""
            # A_kj = mean abundance of sp k in group j / sum over groups of those means
            mean_abund = np.empty((mat.shape[1], len(groups)))
            present = (mat > 0).astype(float)
            freq = np.empty((mat.shape[1], len(groups)))
            for j, grp in enumerate(groups):
                m = labels_masks[grp]
                ng = max(int(m.sum()), 1)
                mean_abund[:, j] = mat[m].mean(axis=0) if m.any() else 0.0
                freq[:, j] = present[m].sum(axis=0) / ng  # B_kj fidelity
            denom = mean_abund.sum(axis=1, keepdims=True)
            A = np.divide(
                mean_abund, denom, out=np.zeros_like(mean_abund), where=denom > 0
            )
            return A * freq * 100.0  # IndVal_kj in [0, 100]

        iv = _indval_all(masks)  # [n_sp, n_grp]
        best_j = np.argmax(iv, axis=1)
        best_iv = iv[np.arange(iv.shape[0]), best_j]
        best_grp = [groups[j] for j in best_j]

        # permutation p-value: permute group labels, recompute max IndVal per species,
        # count permuted-max >= observed best IndVal (Dufrene-Legendre randomization).
        rng = np.random.default_rng(0)
        ge = np.ones(mat.shape[1])  # +1 (observed) convention
        for _ in range(n_perm):
            perm = rng.permutation(g)
            pmasks = {grp: (perm == grp) for grp in groups}
            iv_p = _indval_all(pmasks)
            ge += (iv_p.max(axis=1) >= best_iv).astype(float)
        pvals = ge / (n_perm + 1)

        res = pd.DataFrame(
            {
                "taxon": species,
                "indicator_group": best_grp,
                "indval": np.round(best_iv, 2),
                "p_value": np.round(pvals, 4),
            }
        )
        res["significant"] = res["p_value"] < 0.05
        res = res.sort_values(
            ["significant", "indval"], ascending=[False, False]
        ).reset_index(drop=True)
        res.to_csv(d / "indicator_species.csv", index=False, encoding="utf-8")
        files.append("indicator_species.csv")

        n_sig = int(res["significant"].sum())
        estimates["n_significant_indicators"] = float(n_sig)
        estimates["n_taxa"] = float(len(species))
        estimates["max_indval"] = round(float(best_iv.max()), 2)

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            top = res.head(min(15, len(res)))[::-1]
            colors = ["#C44E52" if s else "#999999" for s in top["significant"]]
            fig, ax = plt.subplots(figsize=(6, max(3, 0.35 * len(top) + 1)))
            ax.barh(
                [f"{t} ({grp})" for t, grp in zip(top["taxon"], top["indicator_group"])],
                top["indval"],
                color=colors,
            )
            ax.set_xlabel("IndVal (specificity x fidelity x 100)")
            ax.set_title("Indicator species (red: p<0.05)")
            fig.tight_layout()
            fig.savefig(d / "indicator_species.png", dpi=150)
            plt.close(fig)
            files.append("indicator_species.png")
        except Exception:
            pass

        per_grp = res[res["significant"]].groupby("indicator_group").size().to_dict()
        grp_desc = (
            "，".join(f"{k}:{v}" for k, v in per_grp.items()) if per_grp else "无"
        )
        summary.append(
            f"{entry.method} 完成：{len(species)} 个物种 × {len(groups)} 组"
            f"（{group_col}，组样本 {group_sizes}），{n_sig} 个显著指示种（p<0.05）；"
            f"各组显著指示种数：{grp_desc}。"
            "⚠ IndVal 基于置换检验，p 值依采样与组定义；"
            "⚠ 分组定义直接决定结果（不同分组得到不同指示种）；"
            "⚠ 稀有类群与组内样本量小时指示值不稳（置换 p 的分辨率受组规模限制）；"
            "⚠ 描述性关联、非因果，A×B 高表示该物种既偏好该组又在组内常见。"
        )
        code += [
            "import numpy as np  # Dufrene-Legendre (1997) IndVal",
            "# A = mean-abundance concentration in group; B = within-group occurrence freq",
            "# IndVal = A * B * 100; permute group labels for the p-value",
        ]
    except Exception as err:
        summary.append(f"指示种(IndVal)失败：{err}")
