"""Branch handlers for the dimensionality-reduction / latent-structure family.

Four methods (scikit-learn / numpy / pandas / scipy — NO R; umap is an OPTIONAL
backend with honest degrade):

- ``tsne``                — t-SNE 2-D embedding (sklearn TSNE) for VISUALIZATION
                            only. Reports the embedding coordinates + the final KL
                            divergence; if a low-cardinality label column exists it
                            colors by it and reports a (descriptive) silhouette of that
                            label in the STANDARDIZED INPUT feature space (NOT the
                            embedding — embedding distances are non-metric). Embedding
                            distances/positions are NOT metric.
- ``umap``                — UMAP 2-D embedding (umap-learn, OPTIONAL backend) for
                            VISUALIZATION only. Same disclosure shape as t-SNE: a
                            descriptive silhouette of a label is computed on the
                            STANDARDIZED INPUT space, never the embedding. Honest
                            degrade ("跳过：需要 umap-learn …") when umap is absent;
                            suggests tsne/pca/mds as substitutes.
- ``factor_analysis``     — exploratory factor analysis (sklearn FactorAnalysis) with
                            an n_factors rule (config / parallel-analysis / Kaiser>1 on
                            the correlation eigenvalues), varimax rotation if feasible,
                            per-factor loadings, communalities, and variance explained.
- ``linear_discriminant`` — LDA as supervised dimension reduction + classification.
                            Needs a categorical/binary target + continuous features.
                            Reports the discriminant axes, % between-class variance per
                            axis, k-fold CV accuracy, and the class means on LD1.

Each handler unpacks ctx into the same local names run_analysis uses and MUTATES
summary/estimates/files/code (never rebinds). See executor/_branch_api.py. This
family file is auto-registered by branches/__init__.py (pkgutil.walk_packages).

RNG note: every stochastic step uses a fixed random_state (0 / 42 disclosed in the
summary) so results are reproducible; t-SNE remains seed-sensitive by nature.
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


def _label_col(fp, df, used: set[str], max_levels: int = 20) -> str | None:
    """A low-cardinality categorical/binary/id column usable as a label/target.

    Accepts ``categorical``/``binary``/``id`` (a 2-level string profiles as binary;
    a small integer label can profile as id). Excludes the profiled unit/time cols.
    Lowest-cardinality first so the most informative grouping comes first.
    """
    excl = {fp.unit_col, fp.time_col} | used
    cands = [
        c.name
        for c in fp.columns
        if c.kind in {"categorical", "binary", "id"}
        and c.name not in excl
        and 2 <= int(df[c.name].nunique()) <= max_levels
    ]
    cands.sort(key=lambda name: int(df[name].nunique()))
    return cands[0] if cands else None


# ===========================================================================
# 1. t-SNE — 2-D embedding for VISUALIZATION only
# ===========================================================================

@register("tsne")
def _branch_tsne(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import importlib.util

    cont = _continuous_cols(fp)
    forced = [c for c in (cfg.get("features") or []) if c in df.columns]
    features = forced if forced else cont

    if importlib.util.find_spec("sklearn") is None:
        summary.append("t-SNE 跳过：需要 scikit-learn（未检测到）。安装：pip install scikit-learn。")
        return
    if len(features) < 2:
        summary.append("t-SNE 跳过：需要 ≥2 个连续特征。设 config features。")
        return

    # An optional label column to color the scatter / compute a descriptive silhouette.
    label_col = cfg.get("label") if cfg.get("label") in df.columns else _label_col(fp, df, set(features))
    keep = features + ([label_col] if label_col else [])
    sub = df[keep].dropna()
    X = sub[features]
    n = len(X)
    if n < 5:
        summary.append(f"t-SNE 跳过：有效样本太少（n={n} < 5）。")
        return

    # Constant features carry no information and break standardization scaling.
    nonconst = [c for c in features if float(X[c].std(ddof=0)) > 0]
    if len(nonconst) < 2:
        summary.append("t-SNE 跳过：非常量连续特征不足 2 个。")
        return
    features = nonconst
    X = sub[features]

    # perplexity must be < n; default min(30, (n-1)/3) per spec.
    try:
        forced_perp = cfg.get("perplexity")
        forced_perp = float(forced_perp) if forced_perp is not None else None
    except (TypeError, ValueError):
        forced_perp = None
    auto_perp = min(30.0, (n - 1) / 3.0)
    perplexity = forced_perp if (forced_perp is not None and forced_perp > 0) else auto_perp
    # sklearn requires perplexity < n; clamp to a safe positive value.
    perplexity = max(1.0, min(perplexity, float(n - 1)))

    try:
        import pandas as pd
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler

        Xs = StandardScaler().fit_transform(X.values.astype(float))

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=42,
        )
        emb = tsne.fit_transform(Xs)
        kl = float(getattr(tsne, "kl_divergence_", float("nan")))

        # --- descriptive silhouette of the LABEL in INPUT space (not a fit metric) ---
        # Computed on the standardized INPUT features `Xs` (the matrix fed to t-SNE),
        # NOT on the 2-D embedding: the embedding's distances are non-metric, so a
        # silhouette there would contradict our own disclosure. This is an input-space
        # separability descriptor (how distinct the labels are in feature space).
        sil = float("nan")
        if label_col is not None:
            try:
                from sklearn.metrics import silhouette_score

                lab = sub[label_col].values
                n_lab = int(pd.Series(lab).nunique())
                if 2 <= n_lab < n:
                    sil = float(silhouette_score(Xs, lab))
            except Exception:
                sil = float("nan")

        # --- embedding CSV (coords + label) --------------------------------
        emb_df = pd.DataFrame(emb, columns=["tsne1", "tsne2"], index=X.index)
        if label_col is not None:
            emb_df.insert(0, "label", sub[label_col].values)
        emb_df.to_csv(d / "tsne_embedding.csv", index=True, encoding="utf-8")
        files.append("tsne_embedding.csv")

        # --- 2-D scatter (colored by label if present) ---------------------
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 5))
            if label_col is not None and sub[label_col].nunique() <= 12:
                for lv in sub[label_col].dropna().unique().tolist():
                    m = (sub[label_col] == lv).values
                    ax.scatter(emb[m, 0], emb[m, 1], s=18, alpha=0.75, label=str(lv))
                ax.legend(title=str(label_col), fontsize=8)
            else:
                ax.scatter(emb[:, 0], emb[:, 1], s=18, alpha=0.75)
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            ax.set_title(f"t-SNE 2D embedding (perplexity={perplexity:.0f})")
            fig.tight_layout()
            fig.savefig(d / "tsne_scatter.png", dpi=150)
            plt.close(fig)
            files.append("tsne_scatter.png")
        except Exception:
            pass

        estimates["kl_divergence"] = round(kl, 6) if kl == kl else float("nan")
        estimates["perplexity"] = float(perplexity)
        estimates["n_features"] = float(len(features))
        estimates["n"] = float(n)
        estimates["silhouette_by_label"] = round(sil, 6) if sil == sil else float("nan")

        lab_txt = (
            f"，按 {label_col} 着色，该标签在标准化输入特征空间的轮廓系数={sil:.3f}"
            f"（输入空间可分性描述，非来自嵌入；仅描述性）"
            if (label_col is not None and sil == sil)
            else "，无低基数标签列可着色"
        )
        kl_txt = f"{kl:.4f}" if kl == kl else "N/A"
        summary.append(
            f"{entry.method} 完成（t-SNE，{len(features)} 个特征 × {n} 个样本 → 2D，"
            f"perplexity={perplexity:.0f}，KL 散度={kl_txt}{lab_txt}）。"
            f"⚠ t-SNE 仅用于可视化——低维图中的距离、簇大小、簇间相对位置都不具度量意义，"
            f"不要据此算距离或做下游度量分析；结果对 perplexity 敏感且为随机优化"
            f"（已固定 random_state=42，换种子图会变），别把它当稳定坐标；"
            f"全部连续特征为默认特征、已标准化——可用 config features/perplexity 覆盖。"
        )
        code += [
            "from sklearn.manifold import TSNE",
            "from sklearn.preprocessing import StandardScaler",
            f"features = {features!r}",
            "Xs = StandardScaler().fit_transform(df[features].dropna())",
            f"tsne = TSNE(n_components=2, perplexity={perplexity:.1f}, init='pca', "
            f"learning_rate='auto', random_state=42)",
            "emb = tsne.fit_transform(Xs); print('KL:', tsne.kl_divergence_)",
        ]
    except Exception as err:
        summary.append(f"t-SNE 失败：{err}")


# ===========================================================================
# 2. Factor analysis — exploratory factor analysis (EFA)
# ===========================================================================

@register("factor_analysis")
def _branch_factor_analysis(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import importlib.util

    cont = _continuous_cols(fp)
    # Likert items profile as count — accept count too so EFA can run on item batteries.
    if len(cont) < 3:
        cont = [
            c.name for c in fp.columns
            if c.kind in {"continuous", "count"} and c.name not in {fp.unit_col, fp.time_col}
        ]
    forced = [c for c in (cfg.get("features") or []) if c in df.columns]
    features = forced if forced else cont

    if importlib.util.find_spec("sklearn") is None:
        summary.append("因子分析跳过：需要 scikit-learn（未检测到）。安装：pip install scikit-learn。")
        return
    if len(features) < 3:
        summary.append("因子分析跳过：需要 ≥3 个连续/计数特征。设 config features。")
        return

    sub = df[features].dropna()
    n = len(sub)
    p = len(features)
    if n < 10:
        summary.append(f"因子分析跳过：有效样本太少（n={n} < 10）。")
        return

    # drop constant columns (zero variance breaks correlation / scaling).
    nonconst = [c for c in features if float(sub[c].std(ddof=0)) > 0]
    if len(nonconst) < 3:
        summary.append("因子分析跳过：非常量特征不足 3 个。")
        return
    features = nonconst
    sub = sub[features]
    p = len(features)

    try:
        import numpy as np
        import pandas as pd
        from sklearn.decomposition import FactorAnalysis
        from sklearn.preprocessing import StandardScaler

        Xs = StandardScaler().fit_transform(sub.values.astype(float))

        # --- choose n_factors -----------------------------------------------
        # eigenvalues of the correlation matrix drive both Kaiser (>1) and parallel analysis.
        R = np.corrcoef(Xs, rowvar=False)
        eigvals = np.sort(np.real(np.linalg.eigvalsh(R)))[::-1]

        forced_k = cfg.get("n_factors")
        try:
            forced_k = int(forced_k) if forced_k is not None else None
        except (TypeError, ValueError):
            forced_k = None

        rng = np.random.default_rng(0)
        # Horn's parallel analysis: factors whose observed eigenvalue exceeds the 95th
        # percentile of eigenvalues from random data of the same shape.
        n_perm = 50
        rand_eigs = np.zeros((n_perm, p))
        for i in range(n_perm):
            Z = rng.standard_normal((n, p))
            Rr = np.corrcoef(Z, rowvar=False)
            rand_eigs[i] = np.sort(np.real(np.linalg.eigvalsh(Rr)))[::-1]
        pa_thresh = np.percentile(rand_eigs, 95, axis=0)
        pa_k = int(np.sum(eigvals > pa_thresh))
        kaiser_k = int(np.sum(eigvals > 1.0))

        if forced_k is not None and 1 <= forced_k <= p - 1:
            n_factors = forced_k
            rule = f"config 指定 n_factors={n_factors}"
        elif pa_k >= 1:
            n_factors = pa_k
            rule = f"平行分析(Horn parallel analysis, 95th pct) 选 {pa_k}"
        elif kaiser_k >= 1:
            n_factors = kaiser_k
            rule = f"Kaiser 准则(特征值>1) 选 {kaiser_k}"
        else:
            n_factors = 1
            rule = "回退至 1 因子（无规则建议 ≥1）"
        n_factors = max(1, min(n_factors, p - 1))

        # --- fit EFA ---------------------------------------------------------
        fa = FactorAnalysis(n_components=n_factors, rotation=None, random_state=0)
        fa.fit(Xs)
        loadings = fa.components_.T  # (p, n_factors)

        # --- varimax rotation (orthogonal) if >1 factor ---------------------
        rotated = False
        if n_factors > 1:
            try:
                loadings = _varimax(loadings)
                rotated = True
            except Exception:
                rotated = False

        fac_cols = [f"Factor{i+1}" for i in range(n_factors)]
        # communality_j = sum of squared loadings across factors (shared variance).
        communalities = np.sum(loadings ** 2, axis=1)
        # variance explained per factor = sum of squared loadings down each column / p.
        var_per_factor = np.sum(loadings ** 2, axis=0)
        var_explained = var_per_factor / p

        load_df = pd.DataFrame(loadings, index=features, columns=fac_cols)
        load_df["communality"] = communalities
        load_df.to_csv(d / "factor_loadings.csv", encoding="utf-8")
        files.append("factor_loadings.csv")

        var_df = pd.DataFrame({
            "factor": fac_cols,
            "ss_loadings": var_per_factor,
            "proportion_var": var_explained,
            "cumulative_var": np.cumsum(var_explained),
        })
        var_df.to_csv(d / "factor_variance.csv", index=False, encoding="utf-8")
        files.append("factor_variance.csv")

        # --- loadings heatmap -----------------------------------------------
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(1.6 + 1.1 * n_factors, 0.45 * p + 1.5))
            im = ax.imshow(loadings, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(n_factors))
            ax.set_xticklabels(fac_cols, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(p))
            ax.set_yticklabels([str(f) for f in features], fontsize=8)
            ax.set_title("EFA loadings" + (" (varimax)" if rotated else ""))
            for i in range(p):
                for j in range(n_factors):
                    ax.text(j, i, f"{loadings[i, j]:.2f}", ha="center", va="center",
                            fontsize=7, color="black")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(d / "factor_loadings_heatmap.png", dpi=150)
            plt.close(fig)
            files.append("factor_loadings_heatmap.png")
        except Exception:
            pass

        total_var = float(np.sum(var_explained))
        estimates["n_factors"] = float(n_factors)
        estimates["total_var_explained"] = round(total_var, 6)
        estimates["max_communality"] = round(float(np.max(communalities)), 6)
        estimates["min_communality"] = round(float(np.min(communalities)), 6)
        estimates["n"] = float(n)

        rot_txt = "varimax 旋转后" if rotated else "未旋转"
        min_comm = float(np.min(communalities))
        max_comm = float(np.max(communalities))
        tv_txt = f"{total_var:.1%}" if total_var == total_var else "N/A"
        comm_txt = (
            f"{min_comm:.2f}–{max_comm:.2f}"
            if (min_comm == min_comm and max_comm == max_comm)
            else "N/A"
        )
        summary.append(
            f"{entry.method} 完成（探索性因子分析 EFA，{p} 个指标 × {n} 个样本 → {n_factors} 个因子，"
            f"{rule}）：{rot_txt}共解释 {tv_txt} 的方差，"
            f"共同度(communality) 范围 {comm_txt}"
            f"（详见 factor_loadings.csv）。"
            f"⚠ EFA 假定存在线性的潜在结构并需足够样本（n/p={n/p:.1f}；KMO 抽样适切性未计算——请自行评估）；"
            f"旋转只帮助解释、不改变拟合优度；因子数是一个选择（此处规则：{rule}，已固定 random_state=0）；"
            f"EFA 建模的是共同方差，与 PCA（建模总方差）不同——可用 config features/n_factors 覆盖。"
            f" ⚠ 载荷的符号是任意的（整列可同时翻转，不改变解）——解释时只看相对模式，别纠结正负。"
        )
        # Conditional Heywood / low-communality warning (only when triggered).
        comm_flags = []
        if min_comm == min_comm and min_comm < 0.2:
            comm_flags.append(f"最低共同度仅 {min_comm:.2f}（<0.2），该指标几乎未被因子解释（可能不属于本结构）")
        if max_comm == max_comm and max_comm > 0.98:
            comm_flags.append(
                f"最高共同度达 {max_comm:.2f}（接近/超过 1，疑似 Heywood case），解可能不稳定，请谨慎对待"
            )
        if comm_flags:
            summary.append("⚠ 共同度告警：" + "；".join(comm_flags) + "。")
        code += [
            "from sklearn.decomposition import FactorAnalysis",
            "from sklearn.preprocessing import StandardScaler",
            f"features = {features!r}",
            "Xs = StandardScaler().fit_transform(df[features].dropna())",
            "# n_factors via Horn parallel analysis / Kaiser>1 on correlation eigenvalues",
            f"fa = FactorAnalysis(n_components={n_factors}, random_state=0).fit(Xs)",
            "loadings = fa.components_.T  # then varimax-rotate; communality = row sum of squares",
        ]
    except Exception as err:
        summary.append(f"因子分析失败：{err}")


def _varimax(loadings, gamma: float = 1.0, q: int = 100, tol: float = 1e-6):
    """Kaiser varimax rotation of a (p x k) loadings matrix (orthogonal).

    Iteratively rotates to maximize the variance of the squared loadings within each
    factor (simple structure). Returns the rotated loadings (same shape).
    """
    import numpy as np

    L = np.asarray(loadings, dtype=float)
    p, k = L.shape
    if k < 2:
        return L
    R = np.eye(k)
    var_prev = 0.0
    for _ in range(q):
        Lr = L @ R
        # Kaiser update
        u, s, vt = np.linalg.svd(
            L.T @ (Lr ** 3 - (gamma / p) * Lr @ np.diag(np.sum(Lr ** 2, axis=0)))
        )
        R = u @ vt
        var_now = float(np.sum(s))
        if var_prev != 0.0 and abs(var_now - var_prev) < tol:
            break
        var_prev = var_now
    return L @ R


# ===========================================================================
# 3. Linear discriminant analysis (LDA) — supervised dim reduction + classify
# ===========================================================================

@register("linear_discriminant")
def _branch_linear_discriminant(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import importlib.util

    cont = _continuous_cols(fp)
    forced_feat = [c for c in (cfg.get("features") or []) if c in df.columns]

    if importlib.util.find_spec("sklearn") is None:
        summary.append("LDA 跳过：需要 scikit-learn（未检测到）。安装：pip install scikit-learn。")
        return

    # target: config outcome else a low-cardinality categorical/binary/id column.
    target = cfg.get("outcome") if cfg.get("outcome") in df.columns else None
    if target is None:
        target = _label_col(fp, df, set(forced_feat))
    if target is None:
        summary.append("LDA 跳过：需要 1 个分类/二值目标变量（无低基数类别列）。设 config outcome。")
        return

    features = forced_feat if forced_feat else [c for c in cont if c != target]
    features = [c for c in features if c != target]
    if len(features) < 2:
        summary.append("LDA 跳过：需要 ≥2 个连续特征。设 config features。")
        return

    sub = df[features + [target]].dropna()
    n = len(sub)
    if n < 10:
        summary.append(f"LDA 跳过：有效样本太少（n={n} < 10）。")
        return

    y = sub[target].astype(str)
    n_classes = int(y.nunique())
    if n_classes < 2:
        summary.append(f"LDA 跳过：目标 {target} 只有 1 个类别。")
        return
    if n_classes > 20:
        summary.append(f"LDA 跳过：目标 {target} 类别过多（{n_classes}>20），不像分类目标。设 config outcome。")
        return

    # need at least 2 samples per class for a meaningful stratified CV split.
    vc = y.value_counts()
    if int(vc.min()) < 2:
        summary.append(f"LDA 跳过：存在样本数<2 的类别（{target}），无法做交叉验证。")
        return

    # constant features carry no discriminant information.
    nonconst = [c for c in features if float(sub[c].std(ddof=0)) > 0]
    if len(nonconst) < 2:
        summary.append("LDA 跳过：非常量连续特征不足 2 个。")
        return
    features = nonconst
    sub = sub[features + [target]]
    y = sub[target].astype(str)

    try:
        import numpy as np
        import pandas as pd
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline

        X = sub[features].values.astype(float)
        Xs = StandardScaler().fit_transform(X)

        # number of discriminant axes = min(n_classes-1, n_features)
        n_components = max(1, min(n_classes - 1, len(features)))

        lda = LinearDiscriminantAnalysis(n_components=n_components)
        lda.fit(Xs, y.values)
        scores = lda.transform(Xs)  # (n, n_components)

        # % between-class variance per discriminant axis (sklearn exposes this).
        evr = getattr(lda, "explained_variance_ratio_", None)
        if evr is not None and len(evr):
            evr = np.asarray(evr, dtype=float)
        else:
            evr = np.array([float("nan")] * n_components)

        # --- k-fold CV classification accuracy (the honest estimate) -------
        k_folds = max(2, min(5, int(vc.min())))
        cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=0)
        # standardize inside the CV pipeline to avoid leakage.
        pipe = make_pipeline(
            StandardScaler(),
            LinearDiscriminantAnalysis(n_components=n_components),
        )
        try:
            cv_scores = cross_val_score(pipe, X, y.values, cv=cv, scoring="accuracy")
            cv_acc = float(np.mean(cv_scores))
        except Exception:
            cv_acc = float("nan")

        # --- per-sample LD scores CSV --------------------------------------
        score_cols = [f"LD{i+1}" for i in range(n_components)]
        score_df = pd.DataFrame(scores, columns=score_cols, index=sub.index)
        score_df.insert(0, "class", y.values)
        score_df.to_csv(d / "lda_scores.csv", index=True, encoding="utf-8")
        files.append("lda_scores.csv")

        # --- class means on LD1 --------------------------------------------
        ld1 = scores[:, 0]
        means_df = (
            pd.DataFrame({"class": y.values, "LD1": ld1})
            .groupby("class")["LD1"].mean().reset_index()
            .rename(columns={"LD1": "ld1_class_mean"})
        )
        means_df.to_csv(d / "lda_class_means_ld1.csv", index=False, encoding="utf-8")
        files.append("lda_class_means_ld1.csv")

        # --- LD scatter -----------------------------------------------------
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 5))
            classes = pd.Series(y.values).unique().tolist()
            if n_components >= 2:
                for cl in classes:
                    m = (y.values == cl)
                    ax.scatter(scores[m, 0], scores[m, 1], s=18, alpha=0.75, label=str(cl))
                ax.set_xlabel("LD1")
                ax.set_ylabel("LD2")
            else:
                # single axis: jittered strip plot by class on LD1.
                rng = np.random.default_rng(0)
                for idx, cl in enumerate(classes):
                    m = (y.values == cl)
                    jitter = rng.uniform(-0.15, 0.15, int(m.sum())) + idx
                    ax.scatter(scores[m, 0], jitter, s=18, alpha=0.75, label=str(cl))
                ax.set_xlabel("LD1")
                ax.set_ylabel("class (jittered)")
            ax.legend(title=str(target), fontsize=8)
            ax.set_title(f"LDA discriminant space (CV acc={cv_acc:.2f})")
            fig.tight_layout()
            fig.savefig(d / "lda_scatter.png", dpi=150)
            plt.close(fig)
            files.append("lda_scatter.png")
        except Exception:
            pass

        var_ld1 = float(evr[0]) if len(evr) and evr[0] == evr[0] else float("nan")
        estimates["cv_accuracy"] = round(cv_acc, 6) if cv_acc == cv_acc else float("nan")
        estimates["n_components"] = float(n_components)
        estimates["var_explained_ld1"] = round(var_ld1, 6) if var_ld1 == var_ld1 else float("nan")
        estimates["n_classes"] = float(n_classes)
        estimates["n"] = float(n)

        # baseline = majority-class rate, for honest CV-accuracy context.
        baseline = float(vc.max()) / float(n)
        cv_txt = f"{cv_acc:.3f}" if cv_acc == cv_acc else "N/A"
        var_ld1_txt = f"{var_ld1:.1%}" if var_ld1 == var_ld1 else "N/A"
        summary.append(
            f"{entry.method} 完成（LDA 监督降维 + 分类，目标={target}，{n_classes} 类，"
            f"{len(features)} 个特征 × {n} 个样本 → {n_components} 个判别轴）："
            f"{k_folds}-折交叉验证准确率={cv_txt}（多数类基线={baseline:.3f}），"
            f"LD1 解释 {var_ld1_txt} 的类间方差（详见 lda_scores.csv / lda_class_means_ld1.csv）。"
            f"⚠ LDA 假定各类协方差相等且特征近似正态——若违背请改用 QDA；"
            f"交叉验证准确率是诚实的样本外估计（与基线对比才有意义），样本内拟合会偏乐观；"
            f"需要分类目标——可用 config outcome/features 覆盖；特征已在 CV 管线内标准化（防泄漏，已固定 random_state=0）。"
            f" ⚠ 判别轴（LD1/LD2…）的符号是任意的（整轴可同时翻转，不改变可分性）——解释方向时只看相对位置。"
        )
        code += [
            "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis",
            "from sklearn.model_selection import StratifiedKFold, cross_val_score",
            "from sklearn.preprocessing import StandardScaler",
            "from sklearn.pipeline import make_pipeline",
            f"# target={target!r}; features={features!r}",
            f"lda = LinearDiscriminantAnalysis(n_components={n_components}).fit(Xs, y)",
            "pipe = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())",
            "cv_acc = cross_val_score(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()",
        ]
    except Exception as err:
        summary.append(f"LDA 失败：{err}")


# ===========================================================================
# 4. UMAP — 2-D embedding for VISUALIZATION only (OPTIONAL backend: umap-learn)
# ===========================================================================

@register("umap")
def _branch_umap(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import importlib.util

    cont = _continuous_cols(fp)
    forced = [c for c in (cfg.get("features") or []) if c in df.columns]
    features = forced if forced else cont

    # OPTIONAL backend — umap-learn imports as the module `umap`. Honest degrade if
    # absent (do NOT hard-fail) and point at the always-available substitutes.
    if importlib.util.find_spec("umap") is None:
        summary.append(
            "UMAP 跳过：需要 umap-learn（未检测到）。安装：pip install umap-learn；"
            "可用替代 tsne / pca / mds（均为内置纯 Python 嵌入/降维）。"
        )
        return
    if importlib.util.find_spec("sklearn") is None:
        summary.append("UMAP 跳过：需要 scikit-learn（未检测到）。安装：pip install scikit-learn。")
        return
    if len(features) < 2:
        summary.append("UMAP 跳过：需要 ≥2 个连续特征。设 config features。")
        return

    # An optional label column to color the scatter / compute a descriptive silhouette.
    label_col = cfg.get("label") if cfg.get("label") in df.columns else _label_col(fp, df, set(features))
    keep = features + ([label_col] if label_col else [])
    sub = df[keep].dropna()
    X = sub[features]
    n = len(X)
    if n < 5:
        summary.append(f"UMAP 跳过：有效样本太少（n={n} < 5）。")
        return

    # Constant features carry no information and break standardization scaling.
    nonconst = [c for c in features if float(X[c].std(ddof=0)) > 0]
    if len(nonconst) < 2:
        summary.append("UMAP 跳过：非常量连续特征不足 2 个。")
        return
    features = nonconst
    X = sub[features]

    # n_neighbors must be < n; default min(15, n-1). min_dist defaults to 0.1.
    try:
        forced_nn = cfg.get("n_neighbors")
        forced_nn = int(forced_nn) if forced_nn is not None else None
    except (TypeError, ValueError):
        forced_nn = None
    auto_nn = min(15, n - 1)
    n_neighbors = forced_nn if (forced_nn is not None and forced_nn >= 2) else auto_nn
    n_neighbors = max(2, min(int(n_neighbors), n - 1))

    try:
        forced_md = cfg.get("min_dist")
        min_dist = float(forced_md) if forced_md is not None else 0.1
    except (TypeError, ValueError):
        min_dist = 0.1
    min_dist = max(0.0, min(min_dist, 0.99))

    try:
        import pandas as pd
        import umap
        from sklearn.preprocessing import StandardScaler

        Xs = StandardScaler().fit_transform(X.values.astype(float))

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42,
        )
        emb = reducer.fit_transform(Xs)

        # --- descriptive silhouette of the LABEL in INPUT space (not a fit metric) ---
        # Computed on the standardized INPUT features `Xs` (the matrix fed to UMAP),
        # NOT on the 2-D embedding — UMAP distances are non-metric (it preserves local
        # > global structure), so a silhouette there would contradict our disclosure.
        # This is an input-space separability descriptor only (mirrors the t-SNE fix).
        sil = float("nan")
        if label_col is not None:
            try:
                from sklearn.metrics import silhouette_score

                lab = sub[label_col].values
                n_lab = int(pd.Series(lab).nunique())
                if 2 <= n_lab < n:
                    sil = float(silhouette_score(Xs, lab))
            except Exception:
                sil = float("nan")

        # --- embedding CSV (coords + label) --------------------------------
        emb_df = pd.DataFrame(emb, columns=["umap1", "umap2"], index=X.index)
        if label_col is not None:
            emb_df.insert(0, "label", sub[label_col].values)
        emb_df.to_csv(d / "umap_embedding.csv", index=True, encoding="utf-8")
        files.append("umap_embedding.csv")

        # --- 2-D scatter (colored by label if present) ---------------------
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 5))
            if label_col is not None and sub[label_col].nunique() <= 12:
                for lv in sub[label_col].dropna().unique().tolist():
                    m = (sub[label_col] == lv).values
                    ax.scatter(emb[m, 0], emb[m, 1], s=18, alpha=0.75, label=str(lv))
                ax.legend(title=str(label_col), fontsize=8)
            else:
                ax.scatter(emb[:, 0], emb[:, 1], s=18, alpha=0.75)
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.set_title(f"UMAP 2D embedding (n_neighbors={n_neighbors}, min_dist={min_dist:.2f})")
            fig.tight_layout()
            fig.savefig(d / "umap_scatter.png", dpi=150)
            plt.close(fig)
            files.append("umap_scatter.png")
        except Exception:
            pass

        estimates["n_neighbors"] = float(n_neighbors)
        estimates["min_dist"] = float(min_dist)
        estimates["n_features"] = float(len(features))
        estimates["n"] = float(n)
        estimates["silhouette_by_label"] = round(sil, 6) if sil == sil else float("nan")

        lab_txt = (
            f"，按 {label_col} 着色，该标签在标准化输入特征空间的轮廓系数={sil:.3f}"
            f"（输入空间可分性描述，非来自嵌入；仅描述性）"
            if (label_col is not None and sil == sil)
            else "，无低基数标签列可着色"
        )
        summary.append(
            f"{entry.method} 完成（UMAP，{len(features)} 个特征 × {n} 个样本 → 2D，"
            f"n_neighbors={n_neighbors}，min_dist={min_dist:.2f}{lab_txt}）。"
            f"⚠ UMAP 仅用于可视化——其距离为**非度量**（保留局部结构优于全局结构），"
            f"低维图中的距离、簇大小、簇间相对位置都不具度量意义，不要据此算距离或做下游度量分析；"
            f"结果对 n_neighbors / min_dist 敏感且为随机优化"
            f"（已固定 random_state=42，换种子/参数图会变），别把它当稳定坐标；"
            f"全部连续特征为默认特征、已标准化——可用 config features/label/n_neighbors/min_dist 覆盖。"
        )
        code += [
            "import umap  # pip install umap-learn",
            "from sklearn.preprocessing import StandardScaler",
            f"features = {features!r}",
            "Xs = StandardScaler().fit_transform(df[features].dropna())",
            f"reducer = umap.UMAP(n_components=2, n_neighbors={n_neighbors}, "
            f"min_dist={min_dist:.2f}, random_state=42)",
            "emb = reducer.fit_transform(Xs)",
        ]
    except Exception as err:
        summary.append(f"UMAP 失败：{err}")
