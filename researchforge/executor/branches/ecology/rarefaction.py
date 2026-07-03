"""rarefaction — Hurlbert (1971) analytic expected-richness curves per site."""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


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
