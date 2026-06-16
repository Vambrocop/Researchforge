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
            if da_method == "aldex2":
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
                mds = MDS(
                    n_components=2,
                    metric=False,
                    dissimilarity="precomputed",
                    random_state=0,
                    n_init=4,
                    max_iter=300,
                )
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
                    f"stress={mds.stress_:.4f}"
                )
                code += [
                    "from sklearn.manifold import MDS",
                    "from scipy.spatial.distance import pdist, squareform",
                    "dist = squareform(pdist(mat.values, metric='braycurtis'))",
                    "coords = MDS(n_components=2, metric=False, dissimilarity='precomputed').fit_transform(dist)",
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
                    f"pseudo-F={F_obs:.3f}，p={p_value:.3f}（{n_perm} 次置换）"
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

