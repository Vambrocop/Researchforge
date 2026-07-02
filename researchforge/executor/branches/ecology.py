"""Branch handlers for the ecology family (migrated from the run.py monolith).

Each handler unpacks ctx into the same local names run_analysis used and runs the
original branch body verbatim. See executor/_branch_api.py.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.run import (
    _diff_abundance_aldex2_via_r,
)


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



@register("differential_abundance")
def _branch_differential_abundance(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import multipletests

    _excl = {fp.unit_col, fp.time_col}
    taxa = [c.name for c in fp.columns if c.kind == "count" and c.name not in _excl]
    group_col = next(
        (
            c.name
            for c in fp.columns
            if c.kind in {"binary", "categorical"}
            and c.name not in _excl
            and df[c.name].dropna().nunique() == 2
        ),
        None,
    )
    if len(taxa) < 2 or group_col is None:
        summary.append("差异丰度失败：需要 ≥2 个计数列（物种/OTU）+ 一个 2 水平分组变量。")
    else:
        sub = df[[*taxa, group_col]].dropna()
        grps = list(pd.Series(sub[group_col].astype(str)).unique())
        if len(grps) != 2:
            summary.append("差异丰度失败：分组变量需恰好 2 组。")
        else:
            # config={"da_method": ...}: "aldex2" (R 金标准, MC-CLR + Welch) /
            # "clr_mw" (默认, CLR+Mann-Whitney) / "clr_welch" (CLR+Welch t)。
            # "ancombc" 桥待接 → 诚实降级。请求 R 法而包/桥不可用也诚实降级。
            from researchforge.executor import rbridge

            da_method = str(cfg.get("da_method") or "clr_mw").lower()
            _degrade_note = ""
            use_aldex2 = False
            ar = None
            if da_method == "aldex2" and not rbridge.r_names_safe([*taxa, group_col]):
                _degrade_note = "；⚠ ALDEx2 需要标识符式列名（字母/数字/. _），已降级 CLR+Mann-Whitney"
            elif da_method == "aldex2":
                if rbridge.r_available() and rbridge.r_package_available("ALDEx2"):
                    _csv = d / "_da_input.csv"
                    sub[[*taxa, group_col]].to_csv(_csv, index=False)
                    try:
                        ar = _diff_abundance_aldex2_via_r(_csv, taxa, group_col)
                        use_aldex2 = True
                    except Exception as err:
                        _degrade_note = f"；⚠ ALDEx2 运行失败（{err}），已降级 CLR+Mann-Whitney"
                    finally:
                        try:
                            _csv.unlink()
                        except OSError:
                            pass
                else:
                    _degrade_note = (
                        "；⚠ 请求 ALDEx2 但未检测到（装：BiocManager::install('ALDEx2')），"
                        "已用 CLR+Mann-Whitney 保底"
                    )
            elif da_method in {"ancombc", "ancom-bc"}:
                _degrade_note = (
                    "；⚠ ANCOM-BC 专用桥尚未接（API 需 TreeSummarizedExperiment，待接，"
                    "见 loop-decisions），已用 CLR+Mann-Whitney 保底；如需 ALDEx2 请 da_method=aldex2"
                )
            if not use_aldex2 and da_method not in {"clr_mw", "clr_welch"}:
                da_method = "clr_mw"  # unknown / degraded → default

            if use_aldex2:
                method_label = "ALDEx2 (R, MC-CLR + Welch)"
                effect_col = f"median_CLR_diff_{grps[1]}_vs_{grps[0]}"
                x_label = f"median CLR difference ({grps[1]} vs {grps[0]})"
                taxa_out = ar["taxon"].tolist()
                effect_vals = ar["diff_btw"].to_numpy(dtype=float)
                pvals = ar["p_value"].to_numpy(dtype=float)
                qvals = ar["q_value"].to_numpy(dtype=float)
            else:
                use_welch = da_method == "clr_welch"
                method_label = "CLR+Welch t" if use_welch else "CLR+Mann-Whitney"
                effect_col = f"log2FC_{grps[1]}_vs_{grps[0]}"
                x_label = f"log2 fold-change ({grps[1]} vs {grps[0]})"
                taxa_out = taxa
                mat = sub[taxa].clip(lower=0).to_numpy(dtype=float)
                rel = mat / mat.sum(axis=1, keepdims=True).clip(min=1e-12)
                logm = np.log(mat + 0.5)  # CLR (compositional-aware), pseudocount 0.5
                clr = logm - logm.mean(axis=1, keepdims=True)
                g = sub[group_col].astype(str).to_numpy()
                ma, mb = g == grps[0], g == grps[1]
                pvals, l2fc = [], []
                if use_welch:
                    from scipy.stats import ttest_ind
                for j in range(len(taxa)):
                    try:
                        if use_welch:
                            _, p = ttest_ind(clr[ma, j], clr[mb, j], equal_var=False)
                        else:
                            _, p = mannwhitneyu(clr[ma, j], clr[mb, j], alternative="two-sided")
                        if not np.isfinite(p):
                            p = 1.0
                    except ValueError:
                        p = 1.0
                    pvals.append(p)
                    l2fc.append(np.log2((rel[mb, j].mean() + 1e-9) / (rel[ma, j].mean() + 1e-9)))
                pvals = np.array(pvals)
                qvals = multipletests(pvals, method="fdr_bh")[1]
                effect_vals = np.array(l2fc)

            res = pd.DataFrame(
                {
                    "taxon": taxa_out,
                    effect_col: np.round(effect_vals, 4),
                    "p_value": np.round(pvals, 4),
                    "q_value": np.round(qvals, 4),
                }
            )
            res["significant"] = res["q_value"] < 0.05
            res = res.sort_values("q_value").reset_index(drop=True)
            res.to_csv(d / "differential_abundance.csv", index=False, encoding="utf-8")
            files.append("differential_abundance.csv")
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fc = np.asarray(effect_vals, dtype=float)
                nlq = -np.log10(np.clip(qvals, 1e-300, 1.0))
                sig = np.asarray(qvals) < 0.05
                fig, ax = plt.subplots(figsize=(6, 4.5))
                ax.scatter(fc[~sig], nlq[~sig], s=18, c="#999999", label="ns")
                ax.scatter(fc[sig], nlq[sig], s=24, c="#C44E52", label="q<0.05")
                ax.axhline(-np.log10(0.05), color="grey", ls="--", lw=0.8)
                ax.axvline(0, color="grey", ls="--", lw=0.6)
                ax.set_xlabel(x_label)
                ax.set_ylabel("-log10(q)")
                ax.set_title(f"Differential abundance ({method_label})")
                ax.legend(fontsize=8)
                fig.tight_layout()
                fig.savefig(d / "volcano.png", dpi=150)
                plt.close(fig)
                files.append("volcano.png")
            except Exception:
                pass
            n_sig = int(res["significant"].sum())
            estimates["n_significant"] = float(n_sig)
            estimates["n_taxa"] = float(len(taxa_out))
            _caveat = (
                "ALDEx2 用 Monte-Carlo Dirichlet 采样 + CLR，组成性严谨（金标准）。"
                if use_aldex2
                else "⚠ 组成性数据：相对丰度受总和约束，本法用 CLR 缓解但非金标准；"
                "CLR 各物种共享每样本分母、非独立，BH-FDR 的独立性假定略被违反；"
                "严格分析可 da_method=aldex2（R ALDEx2）。"
            )
            summary.append(
                f"{entry.method} 完成：{len(taxa_out)} 个物种 × {len(sub)} 样本，比较 "
                f"{grps[0]} vs {grps[1]}；{n_sig} 个物种丰度差异显著（q<0.05，{method_label}+BH-FDR）。"
                + _caveat + _degrade_note
            )
            code += [
                f"# 差异丰度 ({method_label}); config da_method: aldex2 / clr_mw / clr_welch",
                "# ALDEx2: aldex(counts, conds, test='t', effect=TRUE); 纯Py: CLR + 检验 + BH-FDR",
            ]



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



@register("rarefaction")
def _branch_rarefaction(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    from scipy.special import gammaln

    species = [
        c.name
        for c in fp.columns
        if c.kind == "count" and c.name not in {fp.unit_col, fp.time_col}
    ]
    if len(species) < 2:
        summary.append("稀释曲线跳过：需要 ≥2 个计数列（物种/OTU 丰度）。")
    else:
        mat = df[species].fillna(0).clip(lower=0).to_numpy(dtype=float)
        site_N = mat.sum(axis=1)
        keep = site_N > 0
        mat, site_N = mat[keep], site_N[keep]
        n_sites = len(mat)
        if n_sites == 0:
            summary.append("稀释曲线失败：所有样点总丰度为 0。")
        else:

            def hurlbert(counts: np.ndarray, m: int) -> float:
                # E[S_m] = Σ_i [1 - C(N-N_i, m)/C(N, m)]  (Hurlbert 1971, analytic)
                counts = counts[counts > 0]
                total = counts.sum()
                if m >= total:
                    return float(len(counts))  # full depth -> observed richness
                log_cnm = gammaln(total + 1) - gammaln(m + 1) - gammaln(total - m + 1)
                valid = (total - counts) >= m  # else C(N-N_i,m)=0 -> term contributes 1
                out = float((~valid).sum())
                cv = counts[valid]
                if len(cv):
                    lt = gammaln(total - cv + 1) - gammaln(m + 1) - gammaln(total - cv - m + 1) - log_cnm
                    out += float((1.0 - np.exp(lt)).sum())
                return out

            max_n = int(site_N.max())
            grid = sorted(set(int(round(g)) for g in np.linspace(1, max_n, min(30, max_n))))
            rows_out = []
            richness = []
            for s in range(n_sites):
                counts = mat[s]
                n_s = int(site_N[s])
                richness.append(float((counts > 0).sum()))
                for m in grid:
                    if m <= n_s:
                        rows_out.append((s, m, round(hurlbert(counts, m), 4)))
            import pandas as pd

            tab = pd.DataFrame(rows_out, columns=["site", "depth", "expected_richness"])
            tab.to_csv(d / "rarefaction.csv", index=False, encoding="utf-8")
            files.append("rarefaction.csv")

            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(6, 4))
                shown = min(n_sites, 25)  # keep the legend/figure readable
                for s in range(shown):
                    sub = tab[tab["site"] == s]
                    ax.plot(sub["depth"], sub["expected_richness"], lw=1.2, alpha=0.8)
                ax.set_xlabel("sampling depth (individuals)")
                ax.set_ylabel("expected richness E[S]")
                ax.set_title(f"Rarefaction curves ({shown} of {n_sites} sites)")
                fig.tight_layout()
                fig.savefig(d / "rarefaction_curves.png", dpi=150)
                plt.close(fig)
                files.append("rarefaction_curves.png")
            except Exception:
                pass

            estimates["min_depth"] = float(int(site_N.min()))
            estimates["mean_observed_richness"] = round(float(np.mean(richness)), 2)
            estimates["n_sites"] = float(n_sites)
            summary.append(
                f"{entry.method} 完成：{n_sites} 个样点 × {len(species)} 个物种，"
                f"平均观测丰度 {np.mean(richness):.1f}，最浅样点深度 {int(site_N.min())}"
                "（曲线趋平=采样充分；仍上升=需加深采样）"
            )
            code += [
                "import numpy as np  # Hurlbert (1971) analytic rarefaction",
                "# E[S_m] = sum_i (1 - comb(N-N_i, m)/comb(N, m)), per site over depth grid",
            ]


@register("mantel_test")
def _branch_mantel_test(ctx: Ctx) -> None:
    """Mantel test: correlation between two distance matrices with a permutation
    p-value. Community/species distances (Bray-Curtis or Euclidean) vs
    environmental/spatial distances (Euclidean), correlated on the lower triangles
    with a label-permutation null. Pure numpy/scipy."""
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd
    from scipy.spatial.distance import pdist, squareform

    _excl = {fp.unit_col, fp.time_col}
    # Auto-defaults: community = count columns (species abundances); env/spatial =
    # continuous columns (measured gradients / coordinates). Overridable via config.
    auto_comm = [
        c.name for c in fp.columns if c.kind == "count" and c.name not in _excl
    ]
    auto_env = [
        c.name
        for c in fp.columns
        if c.kind in {"continuous", "geo"} and c.name not in _excl
    ]
    comm_cols = list(cfg.get("community") or cfg.get("species") or auto_comm)
    env_cols = list(cfg.get("env") or cfg.get("spatial") or auto_env)
    # config metric for the community matrix: "braycurtis" (default for counts) or
    # "euclidean". The second (env/spatial) matrix is always Euclidean.
    comm_metric = str(cfg.get("metric") or "braycurtis").lower()
    if comm_metric not in {"braycurtis", "euclidean"}:
        comm_metric = "braycurtis"
    method = str(cfg.get("corr") or "pearson").lower()
    if method not in {"pearson", "spearman"}:
        method = "pearson"
    try:
        n_perm = int(cfg.get("n_perm") or 999)
    except (TypeError, ValueError):
        n_perm = 999
    n_perm = max(99, min(n_perm, 9999))

    # keep only columns that actually exist
    comm_cols = [c for c in comm_cols if c in df.columns]
    env_cols = [c for c in env_cols if c in df.columns]

    if len(comm_cols) < 2 or len(env_cols) < 1:
        summary.append(
            "Mantel 检验跳过：需要 ≥2 个群落/物种列（计数）与 ≥1 个环境/空间列（连续）。"
            f"（自动检出 community={comm_cols}，env={env_cols}；可用 config "
            "community/env 指定。）"
        )
        return

    sub = df[comm_cols + env_cols].dropna()
    if len(sub) < 4:
        summary.append("Mantel 检验跳过：去缺失后有效样点 <4，距离矩阵自由度不足。")
        return

    comm_mat = sub[comm_cols].astype(float).to_numpy()
    if comm_metric == "braycurtis":
        comm_mat = np.clip(comm_mat, 0, None)
        keep = comm_mat.sum(axis=1) > 0  # Bray-Curtis undefined for empty rows
        comm_mat = comm_mat[keep]
        env_mat = sub[env_cols].astype(float).to_numpy()[keep]
    else:
        env_mat = sub[env_cols].astype(float).to_numpy()
    n = len(comm_mat)
    if n < 4:
        summary.append("Mantel 检验跳过：有效（非空）样点 <4。")
        return

    try:
        D_comm = squareform(pdist(comm_mat, metric=comm_metric))
        # z-score each env column before euclidean pdist: unstandardized columns on
        # different scales let the large-magnitude column dominate the distance,
        # making r/p unit-dependent (guard std==0 -> leave that column centered at 0).
        env_std = env_mat.std(axis=0)
        env_std_safe = np.where(env_std > 0, env_std, 1.0)
        env_z = (env_mat - env_mat.mean(axis=0)) / env_std_safe
        D_env = squareform(pdist(env_z, metric="euclidean"))

        iu = np.triu_indices(n, k=1)
        x = D_comm[iu]
        y = D_env[iu]

        if method == "spearman":
            from scipy.stats import rankdata

            x = rankdata(x)
            y = rankdata(y)

        # standardized Mantel statistic r = corr(lower-triangle vectors)
        def _corr(a: np.ndarray, b: np.ndarray) -> float:
            a = a - a.mean()
            b = b - b.mean()
            denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
            return float((a * b).sum() / denom) if denom > 0 else float("nan")

        r_obs = _corr(x, y)
        if not np.isfinite(r_obs):
            summary.append("Mantel 检验失败：距离矩阵无变异（常数距离），相关无定义。")
            return

        # permutation null: jointly permute rows/cols of one matrix (Mantel 1967),
        # two-sided test for association (|r| >= |r_obs|).
        rng = np.random.default_rng(0)
        ge = 0
        r_abs = abs(r_obs)
        for _ in range(n_perm):
            perm = rng.permutation(n)
            xp = D_comm[np.ix_(perm, perm)][iu]
            if method == "spearman":
                from scipy.stats import rankdata as _rd

                xp = _rd(xp)
            if abs(_corr(xp, y)) >= r_abs:
                ge += 1
        p_value = (ge + 1) / (n_perm + 1)

        pd.DataFrame(
            [{
                "mantel_r": round(r_obs, 4),
                "p_value": round(p_value, 4),
                "correlation": method,
                "community_metric": comm_metric,
                "env_metric": "euclidean",
                "n_sites": n,
                "n_perm": n_perm,
            }]
        ).to_csv(d / "mantel_result.csv", index=False, encoding="utf-8")
        files.append("mantel_result.csv")

        estimates["mantel_r"] = round(r_obs, 4)
        estimates["p_value"] = round(p_value, 4)
        estimates["n_sites"] = float(n)

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(D_comm[iu], D_env[iu], s=14, alpha=0.6, c="#4C72B0")
            ax.set_xlabel(f"community distance ({comm_metric})")
            ax.set_ylabel("environmental/spatial distance (euclidean)")
            ax.set_title(f"Mantel test (r={r_obs:.3f}, p={p_value:.3f})")
            fig.tight_layout()
            fig.savefig(d / "mantel_scatter.png", dpi=150)
            plt.close(fig)
            files.append("mantel_scatter.png")
        except Exception:
            pass

        sig = "显著" if p_value < 0.05 else "不显著"
        summary.append(
            f"{entry.method} 完成：{n} 个样点，群落距离({comm_metric}) vs "
            f"环境/空间距离(euclidean) 的 Mantel {method} r={r_obs:.3f}，"
            f"p={p_value:.3f}（{n_perm} 次置换，{sig}）。"
            "⚠ Mantel 检验的是两个距离矩阵的相关、非因果；"
            "⚠ 距离度量选择会改变结论（计数群落默认 Bray-Curtis，可 config metric/corr 切换）；"
            "⚠ 空间自相关会使 Mantel r 偏高/p 偏小（样点非独立）——若 env 即地理距离，"
            "考虑偏 Mantel（控制空间）或谨慎解读；"
            "⚠ 环境/空间变量在计算欧氏距离前已逐列标准化（z-score），避免量纲/尺度不同时"
            "大数值列主导距离矩阵；"
            f"⚠ 列分组按 community={comm_cols[:3]}{'...' if len(comm_cols) > 3 else ''} / "
            f"env={env_cols} 自动划分，可用 config 覆盖。"
        )
        code += [
            "import numpy as np",
            "from scipy.spatial.distance import pdist, squareform",
            f"D_comm = squareform(pdist(df[{comm_cols!r}].values, metric={comm_metric!r}))",
            f"env_mat = df[{env_cols!r}].values",
            "env_z = (env_mat - env_mat.mean(axis=0)) / env_mat.std(axis=0)  # z-score env cols",
            "D_env = squareform(pdist(env_z, metric='euclidean'))",
            "iu = np.triu_indices(len(D_comm), k=1); x, y = D_comm[iu], D_env[iu]",
            "r_obs = np.corrcoef(x, y)[0, 1]  # Mantel statistic (Pearson on triangles)",
            "# permutation: jointly permute rows/cols of one matrix, recompute |r|, count >= |r_obs|",
        ]
    except Exception as err:
        summary.append(f"Mantel 检验失败：{err}")


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


def _rda_via_r(csv_path, comm_cols: list[str], pred_cols: list[str]):
    """Redundancy analysis via R vegan::rda (constrained ordination) + anova.cca for
    global and per-axis/term significance. Community matrix constrained by
    environmental predictors. Returns a dict with variance partition, scores, and
    significance. Column names are identifier-guarded upstream. Raises so the caller
    can degrade honestly."""
    from researchforge.executor import rbridge

    csv_r = str(csv_path).replace("\\", "/")
    comm_r = ", ".join(f'"{c}"' for c in comm_cols)
    pred_r = " + ".join(pred_cols)
    rcode = (
        "suppressMessages(library(vegan))\n"
        f'd <- read.csv("{csv_r}", check.names=FALSE)\n'
        f"comm <- d[, c({comm_r})]\n"
        f"m <- rda(comm ~ {pred_r}, data = d)\n"
        "tot <- m$tot.chi\n"
        "constr <- if (is.null(m$CCA)) 0 else m$CCA$tot.chi\n"
        "unconstr <- if (is.null(m$CA)) 0 else m$CA$tot.chi\n"
        # adjusted R^2 (Ezekiel) — honest fit measure; raw constrained% inflates with #predictors.
        # NA when n is too small relative to constraints; emit -999 sentinel for that case.
        "r2 <- tryCatch(RsquareAdj(m)$adj.r.squared, error = function(e) NA)\n"
        "set.seed(0)\n"
        "ag <- anova(m, permutations = 999)\n"
        "set.seed(0)\n"
        "aax <- tryCatch(anova(m, by = 'axis', permutations = 999), error = function(e) NULL)\n"
        "set.seed(0)\n"
        "atm <- tryCatch(anova(m, by = 'terms', permutations = 999), error = function(e) NULL)\n"
        'cat("##VAR\\n")\n'
        'cat(sprintf("total|%.6f\\n", tot))\n'
        'cat(sprintf("constrained|%.6f\\n", constr))\n'
        'cat(sprintf("unconstrained|%.6f\\n", unconstr))\n'
        'cat(sprintf("r2adj|%.6f\\n", ifelse(is.na(r2), -999, r2)))\n'
        'cat("##GLOBAL\\n")\n'
        'cat(sprintf("F|%.6f\\np|%.6g\\n", ag$F[1], ag$"Pr(>F)"[1]))\n'
        'cat("##EIG\\n")\n'
        "eig <- if (is.null(m$CCA)) NULL else m$CCA$eig\n"
        "if (!is.null(eig)) for (i in seq_along(eig)) "
        'cat(sprintf("%s|%.6f\\n", names(eig)[i], eig[i]))\n'
        'cat("##AXIS\\n")\n'
        "if (!is.null(aax)) { nm <- rownames(aax); "
        "for (i in seq_len(nrow(aax)-1)) "
        'cat(sprintf("%s|%.6f|%.6g\\n", nm[i], aax$F[i], aax$"Pr(>F)"[i])) }\n'
        'cat("##TERMS\\n")\n'
        "if (!is.null(atm)) { nm <- rownames(atm); "
        "for (i in seq_len(nrow(atm)-1)) "
        'cat(sprintf("%s|%.6f|%.6g\\n", nm[i], atm$F[i], atm$"Pr(>F)"[i])) }\n'
        "nax <- if (is.null(eig)) 1 else min(2, max(1, length(eig)))\n"
        'cat("##SITES\\n")\n'
        "sc <- as.matrix(scores(m, display = 'sites', choices = 1:nax))\n"
        "for (i in seq_len(nrow(sc))) "
        'cat(sprintf("%d|%s\\n", i, paste(sprintf("%.6f", sc[i, ]), collapse="|")))\n'
        'cat("##BP\\n")\n'
        "bp <- tryCatch(as.matrix(scores(m, display = 'bp', choices = 1:nax)), "
        "error = function(e) NULL)\n"
        "if (!is.null(bp)) for (i in seq_len(nrow(bp))) "
        'cat(sprintf("%s|%s\\n", rownames(bp)[i], paste(sprintf("%.6f", bp[i, ]), collapse="|")))\n'
    )
    out = rbridge.run_r(rcode, timeout=300)
    parsed: dict = {
        "var": {},
        "global": {},
        "eig": [],
        "axis": [],
        "terms": [],
        "sites": [],
        "bp": [],
    }
    section = None
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("##"):
            section = s[2:]
            continue
        if "|" not in s:
            continue
        parts = s.split("|")
        if section == "VAR":
            parsed["var"][parts[0]] = float(parts[1])
        elif section == "GLOBAL":
            parsed["global"][parts[0]] = float(parts[1])
        elif section == "EIG":
            parsed["eig"].append((parts[0], float(parts[1])))
        elif section == "AXIS":
            parsed["axis"].append((parts[0], float(parts[1]), float(parts[2])))
        elif section == "TERMS":
            parsed["terms"].append((parts[0], float(parts[1]), float(parts[2])))
        elif section == "SITES":
            parsed["sites"].append([float(x) for x in parts[1:]])
        elif section == "BP":
            parsed["bp"].append((parts[0], [float(x) for x in parts[1:]]))
    if not parsed["var"]:
        raise RuntimeError("vegan::rda 未返回方差分解")
    return parsed


@register("rda")
def _branch_rda(ctx: Ctx) -> None:
    """Redundancy analysis (RDA) — constrained ordination via R vegan::rda. Community
    matrix constrained by environmental predictors; reports constrained vs
    unconstrained variance, global + axis/term significance (anova.cca), and a
    triplot. Optional R bridge with honest graceful degrade to nmds/permanova."""
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import re

    import numpy as np
    import pandas as pd

    from researchforge.executor import rbridge

    _excl = {fp.unit_col, fp.time_col}
    auto_comm = [
        c.name for c in fp.columns if c.kind == "count" and c.name not in _excl
    ]
    auto_pred = [
        c.name
        for c in fp.columns
        if c.kind in {"continuous", "geo"} and c.name not in _excl
    ]
    comm_cols = list(cfg.get("community") or cfg.get("species") or auto_comm)
    pred_cols = list(cfg.get("predictors") or cfg.get("env") or auto_pred)
    comm_cols = [c for c in comm_cols if c in df.columns]
    pred_cols = [c for c in pred_cols if c in df.columns and c not in comm_cols]

    # identifier guard before column names enter an R formula (injection / parse safety)
    _ident = re.compile(r"[A-Za-z.][A-Za-z0-9._]*")
    bad = [c for c in comm_cols + pred_cols if not _ident.fullmatch(str(c))]

    if len(comm_cols) < 2 or len(pred_cols) < 1:
        summary.append(
            "RDA 跳过：需要 ≥2 个群落/物种列（计数）与 ≥1 个环境预测变量（连续）。"
            f"（自动检出 community={comm_cols[:3]}，predictors={pred_cols}；可用 config "
            "community/predictors 指定。）"
        )
        return

    sub = df[comm_cols + pred_cols].dropna()
    if len(sub) < max(6, len(pred_cols) + 2):
        summary.append(
            f"RDA 跳过：去缺失后有效样点 {len(sub)} 个，对 {len(pred_cols)} 个约束变量过少"
            "（需 n > 预测数 + 2）。"
        )
        return

    degrade_to = "可改用 nmds（无约束排序）或 permanova（分组成分检验）作为纯 Python 替代。"
    if bad:
        summary.append(
            f"RDA 跳过：列名 {bad} 含非法字符（R 公式守卫拦截）。请重命名为 "
            "字母/点开头、仅含字母数字._ 的标识符。" + degrade_to
        )
        return

    if not (rbridge.r_available() and rbridge.r_package_available("vegan")):
        summary.append(
            "⚠ RDA 需要 R 包 vegan（约束排序金标准），当前环境未检测到 R 或 vegan，"
            "已诚实跳过（装：install.packages('vegan')）。" + degrade_to
        )
        return

    _csv = d / "_rda_input.csv"
    try:
        sub.to_csv(_csv, index=False, encoding="utf-8")
        parsed = _rda_via_r(_csv, comm_cols, pred_cols)
    except Exception as err:
        summary.append(
            f"⚠ vegan::rda 运行失败（{err}），已跳过。" + degrade_to
        )
        return
    finally:
        try:
            _csv.unlink()
        except OSError:
            pass

    try:
        tot = parsed["var"].get("total", 0.0)
        constr = parsed["var"].get("constrained", 0.0)
        unconstr = parsed["var"].get("unconstrained", 0.0)
        pct_constr = 100.0 * constr / tot if tot > 0 else 0.0
        pct_unconstr = 100.0 * unconstr / tot if tot > 0 else 0.0
        r2adj_raw = parsed["var"].get("r2adj", -999.0)
        r2adj = r2adj_raw if r2adj_raw > -900.0 else float("nan")  # -999 sentinel = NA in R
        g_F = parsed["global"].get("F", float("nan"))
        g_p = parsed["global"].get("p", float("nan"))

        # variance / significance tables
        rows = [
            {"component": "constrained (RDA)", "variance": round(constr, 4),
             "pct_of_total": round(pct_constr, 2)},
            {"component": "unconstrained (PCA residual)", "variance": round(unconstr, 4),
             "pct_of_total": round(pct_unconstr, 2)},
        ]
        pd.DataFrame(rows).to_csv(
            d / "rda_variance.csv", index=False, encoding="utf-8"
        )
        files.append("rda_variance.csv")

        sig_rows = [{"test": "global", "F": round(g_F, 4), "p_value": round(g_p, 4)}]
        for nm, F, p in parsed["axis"]:
            sig_rows.append({"test": f"axis:{nm}", "F": round(F, 4), "p_value": round(p, 4)})
        for nm, F, p in parsed["terms"]:
            sig_rows.append({"test": f"term:{nm}", "F": round(F, 4), "p_value": round(p, 4)})
        pd.DataFrame(sig_rows).to_csv(
            d / "rda_significance.csv", index=False, encoding="utf-8"
        )
        files.append("rda_significance.csv")

        estimates["constrained_variance_pct"] = round(pct_constr, 2)
        estimates["unconstrained_variance_pct"] = round(pct_unconstr, 2)
        import math as _math

        if _math.isfinite(r2adj):
            estimates["adjusted_r_squared"] = round(float(r2adj), 4)
        estimates["global_F"] = round(float(g_F), 4)
        estimates["global_p"] = round(float(g_p), 4)
        n_sig_terms = sum(1 for _, _, p in parsed["terms"] if p < 0.05)
        estimates["n_significant_predictors"] = float(n_sig_terms)

        # triplot: site scores + predictor biplot arrows (first 2 constrained axes)
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            sites = np.asarray(parsed["sites"], dtype=float)
            if sites.ndim == 2 and sites.shape[1] >= 2 and len(sites) > 0:
                fig, ax = plt.subplots(figsize=(6, 5.5))
                ax.scatter(sites[:, 0], sites[:, 1], s=22, c="#4C72B0",
                           alpha=0.7, label="sites")
                bp2 = [(nm, vec) for nm, vec in parsed["bp"] if len(vec) >= 2]
                if bp2:
                    smax = float(np.abs(sites[:, :2]).max()) or 1.0
                    arr = np.array([vec[:2] for _, vec in bp2], dtype=float)
                    bmax = float(np.abs(arr).max()) or 1.0
                    scale = 0.85 * smax / bmax
                    for name, vec in bp2:
                        ax.arrow(0, 0, vec[0] * scale, vec[1] * scale,
                                 color="#C44E52", width=0.0, head_width=0.04 * smax,
                                 length_includes_head=True)
                        ax.text(vec[0] * scale * 1.08, vec[1] * scale * 1.08,
                                name, color="#C44E52", fontsize=8)
                ax.axhline(0, color="grey", lw=0.6, ls="--")
                ax.axvline(0, color="grey", lw=0.6, ls="--")
                ax.set_xlabel("RDA1")
                ax.set_ylabel("RDA2")
                ax.set_title(f"RDA triplot (constrained={pct_constr:.1f}%)")
                fig.tight_layout()
                fig.savefig(d / "rda_triplot.png", dpi=150)
                plt.close(fig)
                files.append("rda_triplot.png")
        except Exception:
            pass

        sig = "显著" if (np.isfinite(g_p) and g_p < 0.05) else "不显著"
        r2adj_txt = (
            f"，调整 R²={r2adj:.3f}（Ezekiel 校正预测变量数后的诚实拟合）"
            if _math.isfinite(r2adj) else "（调整 R² 不可估：样点对约束变量过少）"
        )
        summary.append(
            f"{entry.method} 完成（R vegan::rda）：{len(comm_cols)} 物种 × {len(sub)} 样点，"
            f"受 {len(pred_cols)} 个环境变量约束。约束方差占比 {pct_constr:.1f}%、"
            f"非约束（残差）{pct_unconstr:.1f}%{r2adj_txt}；全局检验 F={g_F:.3f}，p={g_p:.3f}"
            f"（999 次置换，{sig}）；{n_sig_terms}/{len(pred_cols)} 个预测变量显著。"
            "⚠ RDA 是线性约束排序，假定物种对梯度线性响应——长环境梯度（单峰响应）应优先 CCA；"
            "⚠ 原始约束方差占比随预测变量数膨胀（过拟合风险），应以上面的**调整 R²**为准；"
            "⚠ 统计关联、非因果；预测变量按连续列自动选取，可用 config predictors 覆盖。"
        )
        code += [
            "library(vegan)  # R; redundancy analysis (constrained ordination)",
            f"# m <- rda(comm ~ {' + '.join(pred_cols)}, data = d)",
            "# anova(m, permutations=999) 全局; by='axis' 轴显著; by='terms' 各预测变量",
        ]
    except Exception as err:
        summary.append(f"RDA 结果解析失败：{err}。" + degrade_to)

