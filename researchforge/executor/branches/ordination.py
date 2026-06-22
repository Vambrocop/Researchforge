"""Branch handlers for the ordination / dimension-reduction family.

Three methods:

- ``mds``                    — metric Multidimensional Scaling (sklearn) on a
                               precomputed Euclidean distance matrix of standardized
                               features; reports the 2D embedding, Kruskal stress-1,
                               a Shepard-diagram correlation, and a labelled scatter.
- ``correspondence_analysis`` — CA (2 categorical cols) / MCA (>=3) via ``prince``;
                               reports inertia per dimension, row/column coordinates,
                               the chi-square test of independence (CA), and a biplot.
- ``pls_regression``          — Partial Least Squares regression (sklearn) for high-
                               dimensional / collinear predictors; CV-selected
                               components, per-component X/Y variance, VIP scores,
                               a CV-R2 curve, fitted-vs-actual, and coefficients.

Each handler unpacks ctx into the same local names run_analysis uses and MUTATES
summary/estimates/files/code (never rebinds). See executor/_branch_api.py. This
family file is auto-registered by branches/__init__.py (pkgutil.walk_packages).
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


# ---------------------------------------------------------------------------
# Shared column-role helpers (local to this family — generic split rules).
# ---------------------------------------------------------------------------

def _continuous_cols(fp) -> list[str]:
    """Continuous columns, excluding the profiled unit/time columns."""
    return [
        c.name
        for c in fp.columns
        if c.kind == "continuous" and c.name not in {fp.unit_col, fp.time_col}
    ]


def _categorical_cols(fp, df) -> list[str]:
    """Categorical/binary string-like columns for CA/MCA, lowest-cardinality first.

    Accepts ``categorical`` and ``binary`` (a 2-level string column profiles as
    ``binary``). Excludes the profiled unit/time columns and high-cardinality
    ``id`` columns. Sorted by nunique so the most informative low-cardinality
    factors come first.
    """
    excl = {fp.unit_col, fp.time_col}
    cols = [
        c.name
        for c in fp.columns
        if c.kind in {"categorical", "binary"} and c.name not in excl
    ]
    cols.sort(key=lambda name: int(df[name].nunique()))
    return cols


def _id_label_col(fp, df, used: set[str]) -> str | None:
    """An id/label column to annotate scatter points, if a short one exists."""
    for c in fp.columns:
        if c.name in used:
            continue
        if c.kind in {"id", "categorical", "binary"} and 2 <= int(df[c.name].nunique()) <= 40:
            return c.name
    return None


# ===========================================================================
# 1. MDS — Multidimensional Scaling (metric)
# ===========================================================================

@register("mds")
def _branch_mds(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    cont = _continuous_cols(fp)
    forced = [c for c in (cfg.get("features") or []) if c in df.columns]
    features = forced if forced else cont
    try:
        n_components = int(cfg.get("n_components", 2))
    except (TypeError, ValueError):
        n_components = 2
    n_components = max(2, min(n_components, 5))
    metric = bool(cfg.get("metric", True))

    if len(features) < 2:
        summary.append("MDS 跳过：需要 ≥2 个连续特征。设 config features。")
        return

    # keep an aligned subset so we can label points by an id/label column
    label_col = cfg.get("label") if cfg.get("label") in df.columns else _id_label_col(fp, df, set(features))
    keep = features + ([label_col] if label_col else [])
    sub = df[keep].dropna()
    X = sub[features]
    n = len(X)
    if n < 4:
        summary.append(f"MDS 跳过：有效样本太少（n={n} < 4）。")
        return
    if n_components >= n:
        n_components = max(2, n - 1)

    try:
        import numpy as np
        import pandas as pd
        from scipy.spatial.distance import pdist, squareform
        from sklearn.manifold import MDS
        from sklearn.preprocessing import StandardScaler

        Xs = StandardScaler().fit_transform(X.values.astype(float))
        # Pairwise Euclidean distances on the STANDARDIZED features.
        D = squareform(pdist(Xs, metric="euclidean"))

        mds = MDS(
            n_components=n_components,
            dissimilarity="precomputed",
            metric=metric,
            random_state=0,
            n_init=4,
            max_iter=300,
        )
        emb = mds.fit_transform(D)

        # --- Goodness of fit ------------------------------------------------
        # Kruskal stress-1 computed from the embedding vs the ORIGINAL distances:
        #   stress1 = sqrt( sum_{i<j} (d_ij - dhat_ij)^2 / sum_{i<j} d_ij^2 )
        # where d_ij = input distance, dhat_ij = distance in the low-D embedding.
        iu = np.triu_indices(n, k=1)
        d_orig = D[iu]
        d_emb = squareform(pdist(emb, metric="euclidean"))[iu]
        denom = float(np.sum(d_orig ** 2))
        stress1 = float(np.sqrt(np.sum((d_orig - d_emb) ** 2) / denom)) if denom > 0 else float("nan")
        # Shepard-diagram correlation: original vs fitted distances (monotone fit quality).
        if d_orig.std() > 0 and d_emb.std() > 0:
            shepard_r = float(np.corrcoef(d_orig, d_emb)[0, 1])
        else:
            shepard_r = float("nan")

        # Kruskal (1964) stress-1 interpretation bands.
        if stress1 != stress1:
            band = "无法计算"
        elif stress1 < 0.025:
            band = "极佳(excellent)"
        elif stress1 < 0.05:
            band = "很好(good)"
        elif stress1 < 0.10:
            band = "尚可(fair)"
        elif stress1 < 0.20:
            band = "勉强(poor)"
        else:
            band = "差(unacceptable)"

        # --- Embedding CSV --------------------------------------------------
        dim_cols = [f"dim{i+1}" for i in range(n_components)]
        emb_df = pd.DataFrame(emb, columns=dim_cols, index=X.index)
        if label_col:
            emb_df.insert(0, "label", sub[label_col].values)
        emb_df.to_csv(d / "mds_embedding.csv", index=True, encoding="utf-8")
        files.append("mds_embedding.csv")

        # --- 2D scatter -----------------------------------------------------
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 5))
            if label_col and sub[label_col].nunique() <= 12:
                for lv in sub[label_col].dropna().unique().tolist():
                    m = (sub[label_col] == lv).values
                    ax.scatter(emb[m, 0], emb[m, 1], s=20, alpha=0.75, label=str(lv))
                ax.legend(title=str(label_col), fontsize=8)
            else:
                ax.scatter(emb[:, 0], emb[:, 1], s=20, alpha=0.75)
                if label_col and n <= 30:
                    for i, lab in enumerate(sub[label_col].astype(str).values):
                        ax.annotate(lab, (emb[i, 0], emb[i, 1]), fontsize=7, alpha=0.7)
            ax.set_xlabel("MDS dim 1")
            ax.set_ylabel("MDS dim 2")
            ax.set_title(f"MDS 2D embedding (stress-1={stress1:.3f})")
            fig.tight_layout()
            fig.savefig(d / "mds_scatter.png", dpi=150)
            plt.close(fig)
            files.append("mds_scatter.png")
        except Exception:
            pass

        # --- Shepard diagram ------------------------------------------------
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5.5, 5))
            ax.scatter(d_orig, d_emb, s=8, alpha=0.4)
            lim = [0, max(float(d_orig.max()), float(d_emb.max())) * 1.05]
            ax.plot(lim, lim, "r--", lw=1, label="y=x")
            ax.set_xlabel("original distance")
            ax.set_ylabel("embedding distance")
            ax.set_title(f"Shepard diagram (corr={shepard_r:.3f})")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "mds_shepard.png", dpi=150)
            plt.close(fig)
            files.append("mds_shepard.png")
        except Exception:
            pass

        estimates["stress1"] = round(stress1, 6) if stress1 == stress1 else float("nan")
        estimates["shepard_corr"] = round(shepard_r, 6) if shepard_r == shepard_r else float("nan")
        estimates["n_components"] = float(n_components)
        estimates["n"] = float(n)

        kind = "度量(metric)" if metric else "非度量(non-metric)"
        summary.append(
            f"{entry.method} 完成（{kind} MDS，{len(features)} 个特征 × {n} 个样本 → {n_components}D）："
            f"Kruskal stress-1={stress1:.3f}（{band}，越低越好），"
            f"Shepard 相关={shepard_r:.3f}（拟合距离 vs 原始距离）。"
            f"⚠ MDS 把点摆在低维使其距离逼近高维距离；stress 度量失真——stress 偏高(>0.20)说明 2D 是劣总结，"
            f"试 3D 或换方法；度量 MDS 假定距离为区间尺度（有序数据应用非度量 MDS，设 config metric=false）；"
            f"特征已标准化（避免量纲主导距离）。"
        )
        code += [
            "from scipy.spatial.distance import pdist, squareform",
            "from sklearn.manifold import MDS",
            "from sklearn.preprocessing import StandardScaler",
            f"features = {features!r}",
            "Xs = StandardScaler().fit_transform(df[features].dropna())",
            "D = squareform(pdist(Xs, metric='euclidean'))",
            f"emb = MDS(n_components={n_components}, dissimilarity='precomputed', "
            f"metric={metric}, random_state=0).fit_transform(D)",
            "# Kruskal stress-1 = sqrt(sum((d-dhat)^2)/sum(d^2))",
        ]
    except Exception as err:
        summary.append(f"MDS 失败：{err}")


# ===========================================================================
# 2. Correspondence analysis (CA / MCA)
# ===========================================================================

@register("correspondence_analysis")
def _branch_correspondence_analysis(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import importlib.util

    forced = [c for c in (cfg.get("columns") or []) if c in df.columns]
    columns = forced if forced else _categorical_cols(fp, df)

    if importlib.util.find_spec("prince") is None:
        summary.append("对应分析需要 prince 包（未检测到）。安装：pip install prince。")
        return
    if len(columns) < 2:
        summary.append("对应分析跳过：需要 ≥2 个分类变量。设 config columns。")
        return

    sub = df[columns].dropna()
    if len(sub) < 4:
        summary.append(f"对应分析跳过：有效样本太少（n={len(sub)} < 4）。")
        return
    # every selected column needs >= 2 levels to carry association.
    if any(sub[c].nunique() < 2 for c in columns):
        summary.append("对应分析跳过：所选分类变量需各 ≥2 个水平。")
        return

    try:
        import numpy as np
        import pandas as pd
        import prince

        is_ca = len(columns) == 2
        method_tag = "CA" if is_ca else "MCA"

        if is_ca:
            col_r, col_c = columns[0], columns[1]
            ct = pd.crosstab(sub[col_r], sub[col_c])
            n_dim = max(1, min(2, min(ct.shape) - 1))
            model = prince.CA(n_components=n_dim, n_iter=10, random_state=0)
            model = model.fit(ct)
            fit_in = ct
        else:
            n_levels_total = sum(int(sub[c].nunique()) for c in columns)
            n_dim = max(1, min(2, n_levels_total - len(columns)))
            model = prince.MCA(n_components=n_dim, n_iter=10, random_state=0)
            model = model.fit(sub)
            fit_in = sub

        # --- Inertia / eigenvalues (version-robust accessors) ---------------
        eig = _prince_eigenvalues(model)
        pov = _prince_pct_variance(model, eig)
        total_inertia = float(np.sum(eig)) if eig is not None and len(eig) else float("nan")

        n_eig = len(eig) if eig is not None else 0
        inertia_df = pd.DataFrame({
            "dimension": [f"Dim{i+1}" for i in range(n_eig)],
            "eigenvalue_inertia": np.asarray(eig, dtype=float) if n_eig else [],
            "pct_of_inertia": np.asarray(pov, dtype=float) if n_eig else [],
            "cumulative_pct": np.cumsum(np.asarray(pov, dtype=float)) if n_eig else [],
        })
        inertia_df.to_csv(d / "ca_inertia.csv", index=False, encoding="utf-8")
        files.append("ca_inertia.csv")

        # --- Row / column coordinates --------------------------------------
        row_coords = model.row_coordinates(fit_in)
        try:
            col_coords = model.column_coordinates(fit_in)
        except Exception:
            col_coords = None
        row_coords.to_csv(d / "ca_row_coordinates.csv", encoding="utf-8")
        files.append("ca_row_coordinates.csv")
        if col_coords is not None:
            col_coords.to_csv(d / "ca_column_coordinates.csv", encoding="utf-8")
            files.append("ca_column_coordinates.csv")

        # --- Chi-square test of independence (CA only) ---------------------
        chi2 = chi2_p = chi2_df = phi2 = None
        if is_ca:
            from scipy.stats import chi2_contingency

            chi2, chi2_p, chi2_df, _ = chi2_contingency(ct.values)
            chi2 = float(chi2)
            chi2_p = float(chi2_p)
            chi2_df = int(chi2_df)
            n_total = int(ct.values.sum())
            # total inertia in CA = chi-square / n  (Pearson mean-square contingency phi^2)
            phi2 = chi2 / n_total if n_total else float("nan")
            chi_df = pd.DataFrame([{
                "chi_square": chi2, "df": chi2_df, "p_value": chi2_p,
                "n": n_total, "total_inertia_phi2": phi2,
            }])
            chi_df.to_csv(d / "ca_chi_square.csv", index=False, encoding="utf-8")
            files.append("ca_chi_square.csv")

        # --- Biplot (rows + columns, symmetric map) ------------------------
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            rc = np.asarray(row_coords.values, dtype=float)
            fig, ax = plt.subplots(figsize=(6.5, 5.5))
            if rc.shape[1] >= 2:
                ax.scatter(rc[:, 0], rc[:, 1], c="tab:blue", s=22, alpha=0.6,
                           label="rows" + (" (obs)" if not is_ca else ""))
                if is_ca and rc.shape[0] <= 30:
                    for lab, (x, y) in zip(row_coords.index.astype(str), rc[:, :2]):
                        ax.annotate(lab, (x, y), fontsize=7, color="tab:blue")
                if col_coords is not None:
                    cc = np.asarray(col_coords.values, dtype=float)
                    if cc.shape[1] >= 2:
                        ax.scatter(cc[:, 0], cc[:, 1], c="tab:red", marker="^", s=40,
                                   alpha=0.85, label="columns")
                        if cc.shape[0] <= 60:
                            for lab, (x, y) in zip(col_coords.index.astype(str), cc[:, :2]):
                                ax.annotate(lab, (x, y), fontsize=7, color="tab:red")
                ax.axhline(0, color="grey", lw=0.5)
                ax.axvline(0, color="grey", lw=0.5)
                ax.set_xlabel("Dim 1")
                ax.set_ylabel("Dim 2")
                ax.set_title(f"{method_tag} biplot (symmetric map)")
                ax.legend(fontsize=8)
                fig.tight_layout()
                fig.savefig(d / "ca_biplot.png", dpi=150)
                files.append("ca_biplot.png")
            plt.close(fig)
        except Exception:
            pass

        if n_eig:
            estimates["dim1_inertia_pct"] = float(pov[0])
            if n_eig > 1:
                estimates["dim2_inertia_pct"] = float(pov[1])
            estimates["total_inertia"] = round(total_inertia, 6)
            estimates["n_dimensions"] = float(n_eig)
        if is_ca and chi2 is not None:
            estimates["chi_square"] = round(chi2, 4)
            estimates["chi_square_p"] = round(chi2_p, 6)
            estimates["chi_square_df"] = float(chi2_df)

        dim1_txt = f"{pov[0]:.1f}%" if n_eig else "—"
        cum2 = (float(np.cumsum(pov)[min(1, n_eig - 1)]) if n_eig else float("nan"))
        if is_ca:
            sig = "显著" if (chi2_p == chi2_p and chi2_p < 0.05) else "不显著"
            assoc = (
                f"卡方独立性检验 χ²={chi2:.3f}（df={chi2_df}，p={chi2_p:.3g}，{sig}）；"
                f"总惯量=χ²/n={phi2:.4f}。"
            )
            method_desc = f"CA（2 个分类变量 {col_r} × {col_c} 的列联表）"
        else:
            assoc = "MCA 推广到 ≥3 个分类变量（指示矩阵的对应分析）。"
            method_desc = f"MCA（{len(columns)} 个分类变量）"
        summary.append(
            f"{entry.method} 完成（{method_desc}，n={len(sub)}）："
            f"Dim1 解释惯量 {dim1_txt}、前两维累计 {cum2:.1%}（详见 ca_inertia.csv）；{assoc}"
            f"⚠ CA/MCA 把分类变量间的关联画成低维地图（点越近=关联越强）；CA 中惯量=χ²/n；"
            f"通常只有前几维可解释；行列坐标采用 prince 默认的对称映射(symmetric map)——"
            f"对称图里行点与列点的距离不能直接互比（同集合内比较才严谨）。"
        )
        code += [
            "import prince",
            "from scipy.stats import chi2_contingency  # CA 独立性检验",
            f"# columns = {columns!r}",
            ("ct = pd.crosstab(df[c1], df[c2]); model = prince.CA(n_components=2).fit(ct)"
             if is_ca else
             "model = prince.MCA(n_components=2).fit(df[columns])"),
            "model.eigenvalues_; model.row_coordinates(X); model.column_coordinates(X)",
        ]
    except Exception as err:
        summary.append(f"对应分析失败：{err}")


def _prince_eigenvalues(model):
    """Eigenvalues (inertia per dim) across prince versions.

    v0.7-style exposes ``eigenvalues_`` (list); v0.19 exposes an
    ``eigenvalues_summary`` DataFrame and an ``eigenvalues_`` array. Fall back to
    ``percentage_of_variance_`` x total if needed.
    """
    import numpy as np

    ev = getattr(model, "eigenvalues_", None)
    if ev is not None:
        try:
            arr = np.asarray(list(ev), dtype=float)
            if arr.ndim == 1 and arr.size:
                return arr
        except Exception:
            pass
    summ = getattr(model, "eigenvalues_summary", None)
    if summ is not None:
        try:
            for cand in ("eigenvalue", "eigenvalues", "Eigenvalue"):
                if cand in getattr(summ, "columns", []):
                    return np.asarray(summ[cand].values, dtype=float)
            return np.asarray(summ.iloc[:, 0].values, dtype=float)
        except Exception:
            pass
    return None


def _prince_pct_variance(model, eig):
    """Percentage of variance/inertia per dim across prince versions."""
    import numpy as np

    pov = getattr(model, "percentage_of_variance_", None)
    if pov is not None:
        try:
            arr = np.asarray(list(pov), dtype=float)
            if arr.ndim == 1 and arr.size:
                return arr
        except Exception:
            pass
    summ = getattr(model, "eigenvalues_summary", None)
    if summ is not None:
        try:
            for cand in ("% of variance", "percentage of variance", "% of inertia"):
                if cand in getattr(summ, "columns", []):
                    raw = summ[cand]
                    # prince may store these as "12.34%" strings.
                    vals = [float(str(v).replace("%", "")) for v in raw.values]
                    return np.asarray(vals, dtype=float)
        except Exception:
            pass
    # derive from eigenvalues as a last resort.
    if eig is not None and len(eig):
        tot = float(np.sum(eig))
        if tot > 0:
            return np.asarray(eig, dtype=float) / tot * 100.0
    return np.asarray([], dtype=float)


# ===========================================================================
# 3. PLS regression — Partial Least Squares
# ===========================================================================

@register("pls_regression")
def _branch_pls_regression(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import importlib.util

    cont = _continuous_cols(fp)
    outcome = cfg["outcome"] if cfg.get("outcome") in cont else (cont[0] if cont else None)
    forced = [c for c in (cfg.get("predictors") or []) if c in df.columns and c != outcome]
    if forced:
        predictors = forced
    else:
        predictors = [
            c.name for c in fp.columns
            if c.kind in {"continuous", "count", "binary"}
            and c.name not in {outcome, fp.unit_col, fp.time_col}
        ]

    if importlib.util.find_spec("sklearn") is None:
        summary.append("PLS 回归需要 scikit-learn（未检测到）。安装：pip install scikit-learn。")
        return
    if outcome is None:
        summary.append("PLS 回归跳过：需要 1 个连续结果变量。设 config outcome。")
        return
    if len(predictors) < 2:
        summary.append("PLS 回归跳过：需要 ≥2 个预测变量。设 config predictors。")
        return

    sub = df[[outcome] + predictors].dropna()
    n = len(sub)
    if n < 10:
        summary.append(f"PLS 回归跳过：有效样本太少（n={n} < 10）。")
        return

    try:
        import numpy as np
        import pandas as pd
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import KFold, cross_val_predict
        from sklearn.metrics import r2_score
        from sklearn.preprocessing import StandardScaler

        X = sub[predictors].values.astype(float)
        y = sub[outcome].values.astype(float).reshape(-1, 1)
        Xs = StandardScaler().fit_transform(X)
        ys = StandardScaler().fit_transform(y)

        p = len(predictors)
        max_comp = min(p, n - 1, 10)
        # --- select n_components by cross-validated R^2 ---------------------
        k_folds = max(2, min(5, n // 5))
        cv = KFold(n_splits=k_folds, shuffle=True, random_state=0)
        cv_rows = []
        forced_k = cfg.get("n_components")
        try:
            forced_k = int(forced_k) if forced_k is not None else None
        except (TypeError, ValueError):
            forced_k = None

        best_k, best_cv = 1, -np.inf
        for k in range(1, max_comp + 1):
            try:
                yhat_cv = cross_val_predict(PLSRegression(n_components=k), Xs, ys.ravel(), cv=cv)
                cv_r2 = float(r2_score(ys.ravel(), yhat_cv))
            except Exception:
                cv_r2 = float("nan")
            cv_rows.append({"n_components": k, "cv_r2": cv_r2})
            if cv_r2 == cv_r2 and cv_r2 > best_cv:
                best_cv, best_k = cv_r2, k

        n_comp = forced_k if (forced_k is not None and 1 <= forced_k <= max_comp) else best_k
        cv_df = pd.DataFrame(cv_rows)
        cv_df.to_csv(d / "pls_cv_r2.csv", index=False, encoding="utf-8")
        files.append("pls_cv_r2.csv")
        selected_cv_r2 = next((r["cv_r2"] for r in cv_rows if r["n_components"] == n_comp), float("nan"))

        # --- fit final model on standardized data --------------------------
        pls = PLSRegression(n_components=n_comp)
        pls.fit(Xs, ys.ravel())
        yhat = pls.predict(Xs).ravel()
        insample_r2 = float(r2_score(ys.ravel(), yhat))

        # --- per-component X / Y variance explained ------------------------
        # X-variance per comp: var of each X-score column / total X variance.
        x_scores = pls.x_scores_              # (n, n_comp)
        x_loadings = pls.x_loadings_          # (p, n_comp)
        total_x_var = float(np.sum(np.var(Xs, axis=0)))
        # reconstructed X contribution per component = score_k outer loading_k
        x_var_expl = []
        for kk in range(n_comp):
            recon_k = np.outer(x_scores[:, kk], x_loadings[:, kk])
            x_var_expl.append(float(np.sum(np.var(recon_k, axis=0)) / total_x_var) if total_x_var else float("nan"))
        # Y-variance per comp: incremental R^2 in y from cumulative components.
        y_var_expl = []
        prev_r2 = 0.0
        for kk in range(1, n_comp + 1):
            pls_k = PLSRegression(n_components=kk).fit(Xs, ys.ravel())
            r2_k = float(r2_score(ys.ravel(), pls_k.predict(Xs).ravel()))
            y_var_expl.append(max(0.0, r2_k - prev_r2))
            prev_r2 = r2_k

        comp_df = pd.DataFrame({
            "component": [f"Comp{i+1}" for i in range(n_comp)],
            "x_variance_explained": x_var_expl,
            "y_variance_explained": y_var_expl,
            "y_cumulative": np.cumsum(y_var_expl),
        })
        comp_df.to_csv(d / "pls_component_variance.csv", index=False, encoding="utf-8")
        files.append("pls_component_variance.csv")

        # --- VIP scores -----------------------------------------------------
        # VIP_j = sqrt( p * sum_k ( (w_jk/||w_k||)^2 * SSY_k ) / sum_k SSY_k )
        # where SSY_k = q_k^2 * (t_k^T t_k), w = x_weights_, q = y_loadings_, t = x_scores.
        w = pls.x_weights_                    # (p, n_comp)
        q = pls.y_loadings_.ravel()           # (n_comp,)
        t = pls.x_scores_                      # (n, n_comp)
        ssy = (q ** 2) * np.sum(t ** 2, axis=0)   # (n_comp,)
        ssy_total = float(np.sum(ssy))
        vip = np.zeros(p)
        if ssy_total > 0:
            wnorm2 = np.sum(w ** 2, axis=0)
            wnorm2[wnorm2 == 0] = 1e-12
            for j in range(p):
                contrib = np.sum((w[j, :] ** 2 / wnorm2) * ssy)
                vip[j] = float(np.sqrt(p * contrib / ssy_total))
        vip_df = pd.DataFrame({
            "predictor": predictors,
            "VIP": vip,
            "coefficient_std": pls.coef_.ravel(),
        }).sort_values("VIP", ascending=False)
        vip_df.to_csv(d / "pls_vip.csv", index=False, encoding="utf-8")
        files.append("pls_vip.csv")

        # --- regression coefficients on the ORIGINAL scale -----------------
        x_std = X.std(axis=0, ddof=0)
        x_std[x_std == 0] = 1e-12
        y_std = float(y.std(ddof=0)) or 1e-12
        coef_std = pls.coef_.ravel()
        coef_orig = coef_std * (y_std / x_std)
        intercept_orig = float(y.mean() - np.sum(coef_orig * X.mean(axis=0)))
        coef_df = pd.DataFrame({
            "predictor": predictors,
            "coef_standardized": coef_std,
            "coef_original_scale": coef_orig,
        })
        coef_df.to_csv(d / "pls_coefficients.csv", index=False, encoding="utf-8")
        files.append("pls_coefficients.csv")

        # --- CV-R2 curve ----------------------------------------------------
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(cv_df["n_components"], cv_df["cv_r2"], "o-")
            ax.axvline(n_comp, color="r", ls="--", lw=1, label=f"selected k={n_comp}")
            ax.set_xlabel("n_components")
            ax.set_ylabel("cross-validated R2")
            ax.set_title("PLS: CV R2 vs n_components")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "pls_cv_curve.png", dpi=150)
            plt.close(fig)
            files.append("pls_cv_curve.png")
        except Exception:
            pass

        # --- fitted vs actual ----------------------------------------------
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            y_actual = y.ravel()
            y_fitted = yhat * y_std + float(y.mean())  # back to original scale
            fig, ax = plt.subplots(figsize=(5.5, 5))
            ax.scatter(y_actual, y_fitted, s=14, alpha=0.6)
            lim = [min(y_actual.min(), y_fitted.min()), max(y_actual.max(), y_fitted.max())]
            ax.plot(lim, lim, "r--", lw=1, label="y=x")
            ax.set_xlabel(f"actual {outcome}")
            ax.set_ylabel(f"fitted {outcome}")
            ax.set_title(f"PLS fitted vs actual (in-sample R2={insample_r2:.3f})")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "pls_fitted_vs_actual.png", dpi=150)
            plt.close(fig)
            files.append("pls_fitted_vs_actual.png")
        except Exception:
            pass

        estimates["n_components"] = float(n_comp)
        estimates["cv_r2"] = round(float(selected_cv_r2), 6) if selected_cv_r2 == selected_cv_r2 else float("nan")
        estimates["insample_r2"] = round(insample_r2, 6)
        estimates["best_cv_r2"] = round(float(best_cv), 6) if best_cv == best_cv and best_cv != -np.inf else float("nan")
        n_vip1 = int(np.sum(vip >= 1.0))
        estimates["n_vip_above_1"] = float(n_vip1)

        top = vip_df.head(3)["predictor"].tolist()
        summary.append(
            f"{entry.method} 完成（PLS，{p} 个预测变量 → {n_comp} 个潜成分，由 {k_folds}-折 CV 选定，n={n}）："
            f"CV R²={selected_cv_r2:.3f}（样本外）、样本内 R²={insample_r2:.3f}；"
            f"VIP≥1 的重要预测变量 {n_vip1} 个，最重要前 3：{top}（详见 pls_vip.csv）。"
            f"⚠ PLS 找与 Y 协方差最大的潜成分（p≫n / 多重共线下优于 OLS）；"
            f"成分数是一个选择（已用 CV 选，见 pls_cv_curve.png 曲线，注意 CV R² 与样本内 R² 的差距）；"
            f"VIP>1 为「重要」的经验法则（非显著性检验）；预测变量已标准化（VIP/标准化系数才可比）；预测性非因果。"
        )
        code += [
            "from sklearn.cross_decomposition import PLSRegression",
            "from sklearn.model_selection import KFold, cross_val_predict",
            "from sklearn.metrics import r2_score",
            f"# outcome={outcome!r}; predictors={predictors!r}",
            "# select n_components by CV R^2; VIP_j=sqrt(p*sum_k((w_jk/||w_k||)^2*SSY_k)/sum_k SSY_k)",
            f"pls = PLSRegression(n_components={n_comp}).fit(Xs, ys)",
        ]
    except Exception as err:
        summary.append(f"PLS 回归失败：{err}")
