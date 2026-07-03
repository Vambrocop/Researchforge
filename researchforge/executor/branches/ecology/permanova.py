"""permanova — pseudo-F test for community composition differences by group
(Anderson 2001), permutation p-value on Bray-Curtis distances."""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


@register("permanova")
def _branch_permanova(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd
    from scipy.spatial.distance import pdist, squareform

    species = [
        c.name
        for c in fp.columns
        if c.kind == "count" and c.name not in {fp.unit_col, fp.time_col}
    ]
    _excl = {fp.unit_col, fp.time_col}
    bin_cols = [c.name for c in fp.columns if c.kind == "binary" and c.name not in _excl]
    cat_cols = [c.name for c in fp.columns if c.kind == "categorical" and c.name not in _excl]
    cat_cols.sort(key=lambda name: int(df[name].nunique()))
    group_col = (bin_cols + cat_cols)[0] if (bin_cols + cat_cols) else None

    if len(species) < 2 or group_col is None or len(df) < 6:
        summary.append(
            "PERMANOVA 跳过：需要 ≥2 个计数列（物种丰度）、一个分组变量，以及 ≥6 个样点。"
        )
    else:
        try:
            sub = df[species + [group_col]].dropna()
            mat = sub[species].clip(lower=0).astype(float).values
            g = sub[group_col].astype(str).values
            keep = mat.sum(axis=1) > 0  # Bray-Curtis undefined for empty rows
            mat, g = mat[keep], g[keep]
            groups = sorted(set(g))
            a = len(groups)
            N = len(g)
            if a < 2 or N < 6:
                summary.append("PERMANOVA 跳过：有效样本分组后不足两组或总样本 <6。")
            else:
                D = squareform(pdist(mat, metric="braycurtis"))
                D2 = D ** 2

                # ---- pseudo-F (Anderson 2001) ----
                def _ss_within(D2, labels):
                    ss = 0.0
                    for lev in set(labels):
                        idx = np.where(labels == lev)[0]
                        ng = len(idx)
                        if ng > 1:
                            sub2 = D2[np.ix_(idx, idx)]
                            ss += sub2[np.triu_indices(ng, k=1)].sum() / ng
                    return ss

                SS_total = D2[np.triu_indices(N, k=1)].sum() / N

                def _pseudo_F(labels):
                    ssw = _ss_within(D2, labels)
                    ssa = SS_total - ssw
                    # F = (SSA/(a-1)) / (SSW/(N-a)); guard ssw==0
                    denom = ssw / (N - a)
                    return float((ssa / (a - 1)) / denom) if denom > 0 else float("nan")

                F_obs = _pseudo_F(np.asarray(g))
                rng = np.random.default_rng(0)
                n_perm = 999
                count = 0
                labs = np.asarray(g)
                for _ in range(n_perm):
                    perm = rng.permutation(labs)
                    if _pseudo_F(perm) >= F_obs:
                        count += 1
                p_value = (count + 1) / (n_perm + 1)

                pd.DataFrame(
                    [{
                        "pseudo_F": round(F_obs, 4),
                        "p_value": round(p_value, 4),
                        "n_groups": a,
                        "N": N,
                        "n_perm": n_perm,
                    }]
                ).to_csv(d / "permanova_result.csv", index=False, encoding="utf-8")
                files.append("permanova_result.csv")

                estimates["pseudo_F"] = round(F_obs, 4)
                estimates["p_value"] = round(p_value, 4)
                summary.append(
                    f"{entry.method} 完成：按 {group_col} 分 {a} 组，"
                    f"pseudo-F={F_obs:.3f}，p={p_value:.3f}（{n_perm} 次置换）。"
                    "⚠ PERMANOVA 检验的是组间质心（centroid/位置）差异，但对组内**离散度**"
                    "（dispersion）不均一同样敏感——离散度不同也可能给出显著的 pseudo-F，"
                    "常需配合 PERMDISP/betadisper 检验组内离散度是否同质，"
                    "以判断显著结果反映的是位置差异、离散度差异，还是二者皆有；"
                    "⚠ 距离矩阵为原始计数上的 Bray-Curtis（未做总丰度标准化/转换）。"
                )
                code += [
                    "import numpy as np",
                    "from scipy.spatial.distance import pdist, squareform",
                    f"species = {species!r}",
                    f"sub = df[species + ['{group_col}']].dropna()",
                    "mat = sub[species].clip(lower=0).astype(float).values",
                    f"g = sub['{group_col}'].astype(str).values",
                    "keep = mat.sum(axis=1) > 0",
                    "mat, g = mat[keep], g[keep]",
                    "D = squareform(pdist(mat, metric='braycurtis'))",
                    "D2 = D ** 2",
                    "N = len(g)",
                    "a = len(set(g))",
                    "SS_total = D2[np.triu_indices(N, k=1)].sum() / N",
                    "def ss_within(labels):",
                    "    tot = 0.0",
                    "    for lev in set(labels):",
                    "        idx = np.where(labels == lev)[0]; ng = len(idx)",
                    "        if ng > 1:",
                    "            tot += D2[np.ix_(idx, idx)][np.triu_indices(ng, k=1)].sum() / ng",
                    "    return tot",
                    "def pseudo_F(labels):",
                    "    w = ss_within(labels); return ((SS_total - w)/(a-1)) / (w/(N-a))",
                    "rng = np.random.default_rng(0); g = np.asarray(g)",
                    "F_obs = pseudo_F(g)",
                    "p = (sum(pseudo_F(rng.permutation(g)) >= F_obs for _ in range(999)) + 1) / 1000",
                    "print('pseudo-F =', round(F_obs, 4), 'p =', round(p, 4))",
                ]
        except Exception as err:
            summary.append(f"PERMANOVA 失败：{err}")
