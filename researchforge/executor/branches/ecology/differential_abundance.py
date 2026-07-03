"""differential_abundance — CLR+Mann-Whitney/Welch (default, pure Python) or R ALDEx2
(config da_method=aldex2, MC-CLR + Welch, gold standard) with honest degrade."""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.run import (
    _diff_abundance_aldex2_via_r,
)


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
