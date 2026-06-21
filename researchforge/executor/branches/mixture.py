"""Branch handlers for the mixture / density / robust-outlier family.

Three unsupervised methods (sklearn, pure Python — no R):

  * ``gaussian_mixture``       — GMM clustering, k by BIC + covariance_type compare.
  * ``dbscan_clustering``      — density clustering, eps from the k-distance elbow.
  * ``mahalanobis_outliers``   — robust (MCD) multivariate outlier flagging.

Idioms copied from ``branches/ml.py`` ``_branch_kmeans_clustering``: continuous
columns minus unit/time, ``df[features].dropna()`` (keep index), honest skip on too
few features/rows, CSV + matplotlib(Agg, English labels) best-effort, float-only
``estimates``, Chinese ``summary`` with ⚠ disclosure. New family file → auto-registered
by ``branches/__init__.py`` (walk_packages).
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


def _features(ctx: Ctx) -> list[str]:
    """Continuous columns excluding the unit/time roles (kmeans convention)."""
    fp = ctx.fp
    return [
        c.name
        for c in fp.columns
        if c.kind == "continuous" and c.name not in {fp.unit_col, fp.time_col}
    ]


# --------------------------------------------------------------------------- #
# 1) Gaussian mixture model — soft probabilistic clustering, k chosen by BIC.
# --------------------------------------------------------------------------- #
@register("gaussian_mixture")
def _branch_gaussian_mixture(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    features = cfg.get("features") if isinstance(cfg.get("features"), list) else None
    if features:
        features = [c for c in features if c in df.columns]
    else:
        features = _features(ctx)

    X = df[features].dropna() if features else df[[]]
    # GMM with "full" covariance needs n > components*p-ish; ask for a sane minimum.
    if len(features) < 2 or len(X) < 10:
        summary.append("高斯混合模型 跳过：连续特征不足（需 ≥2）或有效样本太少（需 ≥10）。")
        return

    try:
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        from sklearn.mixture import GaussianMixture
        from sklearn.metrics import silhouette_score
        from sklearn.decomposition import PCA

        # STANDARDIZE: GMM (esp. spherical/diag/tied) is scale-sensitive — disclosed.
        Xs = StandardScaler().fit_transform(X)
        n = len(Xs)

        # k search range (config k_range; default 1..6, capped so each component can
        # be estimated). covariance_type(s) to compare (config; default just "full").
        k_lo, k_hi = 1, 6
        kr = cfg.get("k_range")
        if isinstance(kr, (list, tuple)) and len(kr) == 2:
            k_lo, k_hi = int(kr[0]), int(kr[1])
        k_lo = max(1, k_lo)
        k_hi = max(k_lo, min(k_hi, n - 1))

        cov_types = cfg.get("covariance_type", "full")
        if isinstance(cov_types, str):
            cov_types = [cov_types]
        cov_types = [c for c in cov_types if c in {"full", "tied", "diag", "spherical"}] or ["full"]

        n_init = int(cfg.get("n_init", 5))  # guard against local optima

        rows = []  # BIC curve (k, cov_type, bic)
        best = None  # (bic, k, cov_type, model)
        for cov in cov_types:
            for k in range(k_lo, k_hi + 1):
                try:
                    gm = GaussianMixture(
                        n_components=k, covariance_type=cov,
                        n_init=n_init, random_state=0, max_iter=300,
                    ).fit(Xs)
                except Exception:
                    continue
                bic = float(gm.bic(Xs))
                rows.append({"k": k, "covariance_type": cov, "bic": bic})
                if best is None or bic < best[0]:
                    best = (bic, k, cov, gm)

        if best is None:
            summary.append("高斯混合模型 未能拟合任何候选（数据可能近常数）。")
            return

        bic_best, k_best, cov_best, gm = best
        labels = gm.predict(Xs)               # hard assignment (argmax posterior)
        proba = gm.predict_proba(Xs)          # soft responsibilities
        max_proba = proba.max(axis=1)         # membership certainty per point

        bic_df = pd.DataFrame(rows).sort_values(["covariance_type", "k"])
        bic_df.to_csv(d / "gmm_bic_curve.csv", index=False, encoding="utf-8")
        files.append("gmm_bic_curve.csv")

        # per-component weights/means (means reported on the STANDARDIZED scale).
        comp = pd.DataFrame(
            gm.means_, columns=[f"mean_{c}" for c in features]
        )
        comp.insert(0, "weight", gm.weights_)
        comp.insert(0, "component", range(k_best))
        comp.to_csv(d / "gmm_components.csv", index=False, encoding="utf-8")
        files.append("gmm_components.csv")

        # cluster labels + membership certainty (CSV).
        assign = pd.DataFrame(
            {"row": X.index, "component": labels, "max_proba": max_proba}
        )
        assign.to_csv(d / "gmm_assignments.csv", index=False, encoding="utf-8")
        files.append("gmm_assignments.csv")

        # silhouette of the HARD assignment (only defined for >=2 distinct clusters).
        n_eff = len(set(labels))
        sil = None
        if n_eff >= 2:
            sil = float(silhouette_score(Xs, labels))
            estimates["silhouette"] = sil

        estimates["k_selected"] = float(k_best)
        estimates["bic"] = bic_best
        estimates["mean_max_proba"] = float(max_proba.mean())
        estimates["min_weight"] = float(gm.weights_.min())

        # 2D scatter (first 2 features or PCA) colored by component, best-effort.
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            ncomp = min(2, len(features))
            coords = PCA(n_components=ncomp).fit_transform(Xs)
            fig, ax = plt.subplots(figsize=(6, 5))
            if ncomp == 2:
                sc = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=20)
                ax.set_ylabel("PC2")
            else:
                sc = ax.scatter(coords[:, 0], [0] * len(coords), c=labels, cmap="tab10", s=20)
            ax.set_xlabel("PC1")
            ax.set_title(f"GMM (k={k_best}, {cov_best}) — PCA projection")
            fig.colorbar(sc, ax=ax, label="component")
            fig.tight_layout()
            fig.savefig(d / "gmm_scatter.png", dpi=150)
            plt.close(fig)
            files.append("gmm_scatter.png")

            # BIC curve plot
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            for cov in cov_types:
                sub = bic_df[bic_df["covariance_type"] == cov]
                if not sub.empty:
                    ax2.plot(sub["k"], sub["bic"], marker="o", label=cov)
            ax2.axvline(k_best, color="grey", ls="--", lw=1)
            ax2.set_xlabel("n_components (k)")
            ax2.set_ylabel("BIC (lower is better)")
            ax2.set_title("GMM model selection by BIC")
            ax2.legend(title="covariance")
            fig2.tight_layout()
            fig2.savefig(d / "gmm_bic_curve.png", dpi=150)
            plt.close(fig2)
            files.append("gmm_bic_curve.png")
        except Exception:
            pass

        sil_txt = f"，silhouette={sil:.4f}" if sil is not None else "（单一成分，silhouette 不适用）"
        summary.append(
            f"{entry.method} 完成：在 {len(features)} 个标准化连续特征上，BIC 选出 "
            f"k={k_best}（covariance_type={cov_best}，BIC={bic_best:.1f}）；"
            f"平均隶属确定度 max_proba={max_proba.mean():.3f}{sil_txt}。"
            " ⚠ GMM 是软概率聚类，假定各成分服从高斯分布（与 k-means 的球形硬划分不同）；"
            "k 由 BIC 选定（一种建模选择，已附 BIC 曲线，请自行核验）；EM 可能落入局部最优"
            f"（n_init={n_init} 缓解，仍非全局保证）；已做标准化。"
        )
        code += [
            "from sklearn.preprocessing import StandardScaler",
            "from sklearn.mixture import GaussianMixture",
            f"features = {features!r}",
            "X = df[features].dropna()",
            "Xs = StandardScaler().fit_transform(X)",
            f"gm = GaussianMixture(n_components={k_best}, covariance_type={cov_best!r},",
            f"                     n_init={n_init}, random_state=0).fit(Xs)",
            "labels = gm.predict(Xs); proba = gm.predict_proba(Xs)",
            "print('BIC:', gm.bic(Xs), 'weights:', gm.weights_)",
        ]
    except Exception as err:  # honest failure
        summary.append(f"高斯混合模型 执行失败：{err}")


# --------------------------------------------------------------------------- #
# 2) DBSCAN — density clustering, eps auto from the k-distance elbow.
# --------------------------------------------------------------------------- #
@register("dbscan_clustering")
def _branch_dbscan_clustering(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    features = cfg.get("features") if isinstance(cfg.get("features"), list) else None
    if features:
        features = [c for c in features if c in df.columns]
    else:
        features = _features(ctx)

    X = df[features].dropna() if features else df[[]]
    if len(features) < 2 or len(X) < 10:
        summary.append("DBSCAN 跳过：连续特征不足（需 ≥2）或有效样本太少（需 ≥10）。")
        return

    try:
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import DBSCAN
        from sklearn.neighbors import NearestNeighbors
        from sklearn.decomposition import PCA

        # STANDARDIZE: DBSCAN is a Euclidean-distance method — scale matters. Disclosed.
        Xs = StandardScaler().fit_transform(X)
        n, p = Xs.shape

        # min_samples default = 2*n_features (Ester et al. rule of thumb), config override.
        min_samples = cfg.get("min_samples")
        if min_samples is None:
            min_samples = max(2, 2 * p)
        min_samples = int(min_samples)
        min_samples = min(min_samples, n - 1)  # can't ask for more neighbours than exist

        # eps: config, else the knee of the sorted k-distance curve (k = min_samples).
        # Heuristic: sort each point's distance to its k-th nearest neighbour ascending;
        # the "elbow" (max distance from the line joining first & last point) marks where
        # density drops off. DISCLOSED as a heuristic.
        eps_cfg = cfg.get("eps")
        kdist = None
        knee_idx = None
        if eps_cfg is not None:
            eps = float(eps_cfg)
        else:
            k = min_samples  # k-th neighbour distance (k=min_samples per common practice)
            nn = NearestNeighbors(n_neighbors=k + 1).fit(Xs)  # +1: first neighbour is self (dist 0)
            dists, _ = nn.kneighbors(Xs)
            kdist = np.sort(dists[:, k])  # ascending k-distance curve
            # knee via max distance from the chord (first->last); robust, dependency-free.
            x0, y0 = 0.0, float(kdist[0])
            x1, y1 = float(len(kdist) - 1), float(kdist[-1])
            denom = np.hypot(x1 - x0, y1 - y0)
            if denom == 0:
                knee_idx = len(kdist) - 1
            else:
                xs = np.arange(len(kdist), dtype=float)
                # perpendicular distance from each (xs, kdist) to the chord line
                num = np.abs((y1 - y0) * xs - (x1 - x0) * kdist + x1 * y0 - y1 * x0)
                knee_idx = int(np.argmax(num / denom))
            eps = float(kdist[knee_idx])
            if eps <= 0:  # degenerate (duplicate points) — fall back to a small positive eps
                pos = kdist[kdist > 0]
                eps = float(pos[0]) if len(pos) else 0.5

        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Xs)
        # DBSCAN's -1 label = NOISE (points in no dense region) — a key feature.
        noise_mask = labels == -1
        n_noise = int(noise_mask.sum())
        cluster_ids = sorted(set(labels) - {-1})
        n_clusters = len(cluster_ids)

        assign = pd.DataFrame({"row": X.index, "cluster": labels})
        assign.to_csv(d / "dbscan_assignments.csv", index=False, encoding="utf-8")
        files.append("dbscan_assignments.csv")

        # cluster sizes (including the noise group, labelled -1).
        sizes = pd.Series(labels).value_counts().sort_index()
        size_df = sizes.rename_axis("cluster").reset_index(name="size")
        size_df.to_csv(d / "dbscan_cluster_sizes.csv", index=False, encoding="utf-8")
        files.append("dbscan_cluster_sizes.csv")

        estimates["n_clusters"] = float(n_clusters)
        estimates["n_noise"] = float(n_noise)
        estimates["noise_frac"] = float(n_noise / n)
        estimates["eps"] = float(eps)
        estimates["min_samples"] = float(min_samples)

        # scatter colored by cluster, NOISE in grey, best-effort.
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            ncomp = min(2, len(features))
            coords = PCA(n_components=ncomp).fit_transform(Xs)
            fig, ax = plt.subplots(figsize=(6, 5))
            if ncomp == 2:
                cx, cy = coords[:, 0], coords[:, 1]
                ax.set_ylabel("PC2")
            else:
                cx, cy = coords[:, 0], np.zeros(len(coords))
            # noise grey first, clusters colored on top
            if n_noise:
                ax.scatter(cx[noise_mask], cy[noise_mask], c="lightgrey", s=18,
                           label="noise (-1)")
            if n_clusters:
                ax.scatter(cx[~noise_mask], cy[~noise_mask], c=labels[~noise_mask],
                           cmap="tab10", s=20)
            ax.set_xlabel("PC1")
            ax.set_title(f"DBSCAN (eps={eps:.3f}, min_samples={min_samples})")
            if n_noise:
                ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(d / "dbscan_scatter.png", dpi=150)
            plt.close(fig)
            files.append("dbscan_scatter.png")

            # k-distance elbow plot (only when eps was auto-derived).
            if kdist is not None:
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                ax2.plot(np.arange(len(kdist)), kdist, lw=1.2)
                if knee_idx is not None:
                    ax2.axhline(eps, color="red", ls="--", lw=1,
                                label=f"eps={eps:.3f} (knee)")
                    ax2.axvline(knee_idx, color="grey", ls=":", lw=1)
                ax2.set_xlabel("points sorted by k-distance")
                ax2.set_ylabel(f"distance to {min_samples}-th NN")
                ax2.set_title("k-distance elbow (eps heuristic)")
                ax2.legend(loc="best")
                fig2.tight_layout()
                fig2.savefig(d / "dbscan_kdistance.png", dpi=150)
                plt.close(fig2)
                files.append("dbscan_kdistance.png")
        except Exception:
            pass

        eps_src = "config 指定" if eps_cfg is not None else "k-距离肘部启发式自动选取"
        summary.append(
            f"{entry.method} 完成：在 {len(features)} 个标准化连续特征上，找到 "
            f"{n_clusters} 个密度簇，{n_noise} 个噪声点（占 {n_noise / n:.1%}，DBSCAN 的 -1 标签）；"
            f"eps={eps:.4f}（{eps_src}），min_samples={min_samples}。"
            " ⚠ DBSCAN 发现任意形状的密度簇并把离群点标记为噪声（无需预设簇数 k）；"
            "eps 与 min_samples 决定一切结果（已报告取值与所用启发式）；"
            "对密度差异大的数据表现欠佳；已做标准化。"
        )
        code += [
            "from sklearn.preprocessing import StandardScaler",
            "from sklearn.cluster import DBSCAN",
            f"features = {features!r}",
            "X = df[features].dropna()",
            "Xs = StandardScaler().fit_transform(X)",
            f"labels = DBSCAN(eps={eps:.4f}, min_samples={min_samples}).fit_predict(Xs)",
            "print('clusters:', set(labels) - {-1}, 'noise:', (labels == -1).sum())",
        ]
    except Exception as err:
        summary.append(f"DBSCAN 执行失败：{err}")


# --------------------------------------------------------------------------- #
# 3) Mahalanobis outliers — robust (MCD) covariance + chi-square cutoff.
# --------------------------------------------------------------------------- #
@register("mahalanobis_outliers")
def _branch_mahalanobis_outliers(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    features = cfg.get("features") if isinstance(cfg.get("features"), list) else None
    if features:
        features = [c for c in features if c in df.columns]
    else:
        features = _features(ctx)

    X = df[features].dropna() if features else df[[]]
    p = len(features)
    n = len(X)
    # MCD needs n > ~2*p to estimate a robust covariance; require a clear margin.
    if p < 2 or n < max(10, 2 * p + 1):
        summary.append(
            f"Mahalanobis 稳健离群检测 跳过：需要 ≥2 个连续特征且样本量 n > ~2·p"
            f"（当前 p={p}, n={n}）。"
        )
        return

    try:
        import numpy as np
        import pandas as pd
        from scipy import stats
        from sklearn.covariance import MinCovDet, EmpiricalCovariance

        # NOTE: NO standardization — Mahalanobis distance is scale-invariant
        # (it whitens by the covariance), unlike the distance methods above.
        Xv = X.to_numpy(dtype=float)
        alpha = float(cfg.get("alpha", 0.975))  # chi-square quantile for the cutoff

        # robust mean/cov (MCD) — high-breakdown, resistant to ~h outliers.
        mcd = MinCovDet(random_state=0).fit(Xv)
        d2_robust = mcd.mahalanobis(Xv)            # robust squared Mahalanobis dist
        # classical (empirical) cov for comparison — outliers inflate it.
        emp = EmpiricalCovariance().fit(Xv)
        d2_classical = emp.mahalanobis(Xv)

        cutoff = float(stats.chi2.ppf(alpha, df=p))  # chi-square(p) threshold on d^2
        outlier_mask = d2_robust > cutoff
        out_idx = X.index[outlier_mask].tolist()
        n_out = int(outlier_mask.sum())

        # distances CSV (robust + classical, plus the flag).
        dist_df = pd.DataFrame(
            {
                "row": X.index,
                "d2_robust": d2_robust,
                "d2_classical": d2_classical,
                "outlier": outlier_mask,
            }
        )
        dist_df.to_csv(d / "mahalanobis_distances.csv", index=False, encoding="utf-8")
        files.append("mahalanobis_distances.csv")

        # robust vs classical covariance comparison (CSV) — show outlier inflation.
        rob_cov = pd.DataFrame(mcd.covariance_, index=features, columns=features)
        cla_cov = pd.DataFrame(emp.covariance_, index=features, columns=features)
        rob_cov.to_csv(d / "robust_covariance.csv", encoding="utf-8")
        cla_cov.to_csv(d / "classical_covariance.csv", encoding="utf-8")
        files.append("robust_covariance.csv")
        files.append("classical_covariance.csv")

        # generalized-variance (det of cov) ratio — a scalar inflation summary.
        det_rob = float(np.linalg.det(mcd.covariance_))
        det_cla = float(np.linalg.det(emp.covariance_))
        gv_ratio = (det_cla / det_rob) if det_rob > 0 else float("nan")

        estimates["n_outliers"] = float(n_out)
        estimates["outlier_frac"] = float(n_out / n)
        estimates["chi2_cutoff"] = cutoff
        estimates["max_d2_robust"] = float(d2_robust.max())
        estimates["det_cov_robust"] = det_rob
        estimates["det_cov_classical"] = det_cla
        if np.isfinite(gv_ratio):
            estimates["gen_var_ratio_classical_over_robust"] = float(gv_ratio)

        # distance plot with the chi-square threshold line, best-effort.
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            order = np.argsort(d2_robust)
            fig, ax = plt.subplots(figsize=(7, 4.5))
            xs = np.arange(n)
            d2_sorted = d2_robust[order]
            flagged = outlier_mask[order]
            ax.scatter(xs[~flagged], d2_sorted[~flagged], s=16, c="steelblue",
                       label="inlier")
            if n_out:
                ax.scatter(xs[flagged], d2_sorted[flagged], s=24, c="crimson",
                           label="outlier")
            ax.axhline(cutoff, color="black", ls="--", lw=1,
                       label=f"chi2({p}, {alpha}) = {cutoff:.2f}")
            ax.set_xlabel("points sorted by robust Mahalanobis d^2")
            ax.set_ylabel("robust Mahalanobis d^2")
            ax.set_title("Robust (MCD) Mahalanobis outlier detection")
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(d / "mahalanobis_distances.png", dpi=150)
            plt.close(fig)
            files.append("mahalanobis_distances.png")
        except Exception:
            pass

        gv_txt = (
            f"经典协方差广义方差(det)是稳健估计的 {gv_ratio:.2f} 倍"
            if np.isfinite(gv_ratio) else "经典与稳健协方差对比见 CSV"
        )
        summary.append(
            f"{entry.method} 完成：在 {p} 个连续特征上，用 MCD 稳健协方差计算 Mahalanobis 距离，"
            f"以 chi2(p={p}, {alpha})={cutoff:.2f} 为阈值，标记出 {n_out} 个离群点"
            f"（占 {n_out / n:.1%}）；{gv_txt}（离群点会膨胀经典估计）。"
            " ⚠ MCD 给出高崩溃点的稳健协方差（可抵抗约 h 个离群点）；"
            "chi-square 阈值假定干净数据服从多元正态（一种启发式阈值，已报告标记比例）；"
            f"需 n > ~2·p（当前 n={n}, p={p}）；Mahalanobis 距离尺度不变，"
            "故不需要标准化（与上述距离类方法不同）。"
        )
        code += [
            "from scipy import stats",
            "from sklearn.covariance import MinCovDet, EmpiricalCovariance",
            f"features = {features!r}",
            "X = df[features].dropna().to_numpy(float)",
            "mcd = MinCovDet(random_state=0).fit(X)",
            "d2 = mcd.mahalanobis(X)  # robust squared Mahalanobis distances",
            f"cutoff = stats.chi2.ppf({alpha}, df={p})",
            "outliers = d2 > cutoff",
            "print('n outliers:', outliers.sum())",
        ]
    except Exception as err:
        summary.append(f"Mahalanobis 稳健离群检测 执行失败：{err}")
