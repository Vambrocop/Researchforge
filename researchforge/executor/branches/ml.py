"""Branch handlers for the ml family (migrated from the run.py monolith).

Each handler unpacks ctx into the same local names run_analysis used and runs the
original branch body verbatim. See executor/_branch_api.py.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.run import (
    _bart_via_r,
    _conformal_prediction,
    _network_via_nx,
    _plotly_scatter,
    _silhouette_plot,
)


@register("bart")
def _branch_bart(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import re

    from researchforge.executor import rbridge

    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in {fp.unit_col, fp.time_col}]
    y = cfg["outcome"] if cfg.get("outcome") in cont else (cont[0] if cont else None)
    forced = [c for c in (cfg.get("predictors") or []) if c in df.columns and c != y]
    if forced:
        preds = forced[:20]
    else:
        preds = [
            c.name for c in fp.columns
            if c.kind in {"continuous", "count", "binary"} and c.name not in {y, fp.unit_col, fp.time_col}
        ][:15]
    try:
        ntree = max(10, int(cfg.get("ntree", 100)))
    except (TypeError, ValueError):
        ntree = 100
    try:
        seed = int(cfg.get("seed", 0))
    except (TypeError, ValueError):
        seed = 0
    names_safe = y is not None and all(re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", str(c)) for c in [y, *preds])
    if y is None or not preds:
        summary.append("BART 失败：需要 1 个连续结果变量 + ≥1 个预测变量。")
    elif not (rbridge.r_available() and rbridge.r_package_available("dbarts")):
        summary.append("BART 需要 R 的 dbarts 包（未检测到）。安装：install.packages('dbarts')；或用 random_forest / gam。")
    elif not names_safe:
        summary.append("BART 失败：列名需为标识符式（字母/数字/. _），R 列选择要求。")
    else:
        import pandas as pd

        sub = df[[y, *preds]].dropna()
        csv = d / "_bart_input.csv"
        sub.to_csv(csv, index=False)
        try:
            meta, varimp = _bart_via_r(csv, y, preds, ntree, seed, d / "bart_varimp.png")
            varimp.to_csv(d / "bart_variable_importance.csv", index=False, encoding="utf-8")
            files.append("bart_variable_importance.csv")
            if (d / "bart_varimp.png").exists():
                files.append("bart_varimp.png")
            r2, rmse, sigma, n = meta["r2"], meta["rmse"], meta["sigma"], int(meta["n"])
            estimates["r_squared_insample"] = round(r2, 4)
            estimates["rmse"] = round(rmse, 4)
            estimates["sigma"] = round(sigma, 4)
            estimates["n"] = float(n)
            top = varimp.head(3)["predictor"].tolist()
            (d / "bart_summary.txt").write_text(
                f"BART 贝叶斯加性回归树（dbarts，{ntree} 树，seed={seed}）：{y} ~ {len(preds)} 个预测变量\n"
                f"样本内 R² = {r2:.4f}，RMSE = {rmse:.4f}，残差 σ = {sigma:.4f}，n={n}\n"
                f"变量重要性（分裂占比）前 3：{top}\n"
                "BART = 正则化先验下的树之和，自动建非线性 + 交互，给后验不确定性。\n"
                "注：R² 为样本内（偏乐观，无交叉验证）；分裂占比是粗略的重要性度量"
                "（相关变量间会摊分）；BART 是预测性的，非因果。\n\n"
                + varimp.to_string(index=False),
                encoding="utf-8",
            )
            files.append("bart_summary.txt")
            summary.append(
                f"{entry.method} 完成（R/dbarts，{ntree} 树）：{y} ~ {len(preds)} 个预测变量；"
                f"样本内 R²={r2:.3f}，RMSE={rmse:.3f}，残差 σ={sigma:.3f}（n={n}）；"
                f"最重要预测变量 {top}。⚠ 样本内 R² 偏乐观（无 CV）；分裂占比为粗略重要性；预测性非因果。"
            )
            code += [
                "library(dbarts)  # 贝叶斯加性回归树 BART",
                f"# bart(x.train=X, y.train=y, ntree={ntree}); yhat.train.mean + varcount 重要性",
            ]
        except Exception as err:
            summary.append(f"BART 拟合失败：{err}")
        finally:
            try:
                csv.unlink()
            except OSError:
                pass



@register("conformal_prediction")
def _branch_conformal_prediction(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import importlib.util

    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in {fp.unit_col, fp.time_col}]
    y = cfg["outcome"] if cfg.get("outcome") in cont else (cont[0] if cont else None)
    forced = [c for c in (cfg.get("predictors") or []) if c in df.columns and c != y]
    if forced:
        preds = forced[:15]
    else:
        preds = [
            c.name for c in fp.columns
            if c.kind in {"continuous", "count", "binary"} and c.name not in {y, fp.unit_col, fp.time_col}
        ][:12]
    try:
        alpha = min(0.5, max(0.01, float(cfg.get("alpha", 0.1))))
    except (TypeError, ValueError):
        alpha = 0.1
    try:
        seed = int(cfg.get("seed", 0))
    except (TypeError, ValueError):
        seed = 0
    if importlib.util.find_spec("sklearn") is None:
        summary.append("保形预测需要 scikit-learn（未检测到）。安装：pip install scikit-learn。")
    elif y is None or not preds:
        summary.append("保形预测失败：需要 1 个连续结果变量 + ≥1 个预测变量。")
    elif len(df[[y, *preds]].dropna()) < 40:
        summary.append("保形预测失败：有效样本 <40，无法做 训练/校准/测试 三分。")
    else:
        try:
            import pandas as pd

            res = _conformal_prediction(df, y, preds, alpha, seed, d / "conformal.png")
            if (d / "conformal.png").exists():
                files.append("conformal.png")
            pd.DataFrame([res]).to_csv(d / "conformal_metrics.csv", index=False, encoding="utf-8")
            files.append("conformal_metrics.csv")
            tc, ec = res["target_coverage"], res["empirical_coverage"]
            w, qv = res["mean_interval_width"], res["conformal_q"]
            for k_, v_ in res.items():
                estimates[k_] = float(v_)
            # honest disclosure: when the calibration set is too small the 1-alpha
            # finite-sample guarantee is UNATTAINABLE (cap fired) — never claim it then,
            # and don't let cov_ok print "达标" for a void guarantee (inference-reviewer must-fix).
            too_small = bool(res["cal_too_small"])
            if too_small:
                n_c = int(res["n_calibration"])
                ceil_cov = n_c / (n_c + 1)
                cov_ok = "保证不可达：校准集过小"
                guarantee = (
                    f"⚠ 校准集仅 {n_c} 个样本，对 α={alpha} 而言 ceil((n_cal+1)(1−α))>n_cal，"
                    f"1−α 的有限样本覆盖保证**无法达到**；已退化为「最大校准残差」作区间上界"
                    f"（可达覆盖上限≈n_cal/(n_cal+1)={ceil_cov:.1%}）。请增大样本或调高 α。"
                )
            else:
                cov_ok = "达标" if ec >= tc - 0.05 else "偏低（样本/可交换性存疑）"
                guarantee = "保形预测给**分布无关、有限样本**的边际覆盖保证（任意基模型）。"
            (d / "conformal_summary.txt").write_text(
                f"分裂保形预测（split conformal，RandomForest 基模型，α={alpha}）：{y} ~ {len(preds)} 个预测变量\n"
                f"目标覆盖 {tc:.0%}，测试集经验覆盖 {ec:.1%}（{cov_ok}）\n"
                f"保形阈值 q={qv:.4f}，平均区间宽 {w:.4f}（=2q）；测试 R²={res['test_r2']:.3f}；"
                f"校准 {res['n_calibration']} / 测试 {res['n_test']}\n"
                f"{guarantee}"
                "区间等宽（非自适应）；保证是边际而非条件覆盖；假定数据可交换(iid)。\n",
                encoding="utf-8",
            )
            files.append("conformal_summary.txt")
            summary.append(
                f"{entry.method} 完成（split conformal，RF 基模型，α={alpha}）：{y} ~ {len(preds)} 预测变量；"
                f"目标覆盖 {tc:.0%}、测试经验覆盖 {ec:.1%}（{cov_ok}）；平均区间宽 {w:.3f}，测试 R²={res['test_r2']:.3f}。"
                + (guarantee if too_small
                   else "⚠ 分布无关有限样本覆盖保证，但为**边际**(非条件)覆盖、区间等宽；假定可交换(iid)。")
            )
            code += [
                "from sklearn.ensemble import RandomForestRegressor  # 分裂保形预测",
                "# 训练拟合 -> 校准 |残差| 的 ceil((n+1)(1-α)) 分位 q -> 区间 ŷ±q（覆盖≥1-α）",
            ]
        except Exception as err:
            summary.append(f"保形预测失败：{err}")



@register("hierarchical_clustering")
def _branch_hierarchical_clustering(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    features = [
        c.name
        for c in fp.columns
        if c.kind == "continuous" and c.name not in {fp.unit_col, fp.time_col}
    ]
    X = df[features].dropna()
    if len(features) < 2 or len(X) < 5:
        summary.append("层次聚类跳过：连续特征不足或样本太少。")
    else:
        try:
            import numpy as np
            import pandas as pd
            from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
            from scipy.spatial.distance import pdist
            from sklearn.preprocessing import StandardScaler

            Xs = StandardScaler().fit_transform(X)
            n = len(Xs)
            k = max(2, min(4, n // 5))
            Z = linkage(Xs, method="ward")
            labels = fcluster(Z, t=k, criterion="maxclust")
            pd.DataFrame({"row": X.index, "cluster": labels}).to_csv(
                d / "cluster_assignments.csv", index=False, encoding="utf-8"
            )
            files.append("cluster_assignments.csv")
            coph, _ = cophenet(Z, pdist(Xs))

            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(7, 4))
                dendrogram(Z, ax=ax, no_labels=(n > 30))
                ax.set_title(f"Hierarchical clustering (Ward, k={k})")
                fig.tight_layout()
                fig.savefig(d / "dendrogram.png", dpi=150)
                plt.close(fig)
                files.append("dendrogram.png")
            except Exception:
                pass

            _silhouette_plot(Xs, labels, d / "silhouette.png")
            if (d / "silhouette.png").exists():
                files.append("silhouette.png")

            estimates["n_clusters"] = float(len(set(labels)))
            estimates["cophenetic_corr"] = round(float(coph), 4)
            summary.append(
                f"{entry.method} 完成：{len(features)} 个特征 × {n} 个样本聚成 "
                f"{len(set(labels))} 类（cophenetic 相关={coph:.3f}）"
            )
            code += [
                "from scipy.cluster.hierarchy import linkage, fcluster",
                "Z = linkage(Xs, method='ward')",
                f"labels = fcluster(Z, t={k}, criterion='maxclust')",
            ]
        except Exception as err:
            summary.append(f"层次聚类失败：{err}")



@register("kmeans_clustering")
def _branch_kmeans_clustering(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    features = [
        c.name
        for c in fp.columns
        if c.kind == "continuous" and c.name not in {fp.unit_col, fp.time_col}
    ]
    X = df[features].dropna()  # keep X.index for alignment
    if len(features) < 2 or len(X) < 4:
        summary.append("K-means 跳过：连续特征不足或有效样本太少。")
    else:
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            from sklearn.decomposition import PCA
            import numpy as np
            import pandas as pd

            Xs = StandardScaler().fit_transform(X)
            n = len(Xs)
            k_max = max(2, min(6, n // 10))
            best = None  # (score, k, labels)
            for k in range(2, min(k_max, n - 1) + 1):
                labels = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(Xs)
                if len(set(labels)) < 2:
                    continue
                score = silhouette_score(Xs, labels)
                if best is None or score > best[0]:
                    best = (score, k, labels)

            if best is None:
                summary.append("K-means 未能形成有效聚类（数据可能近常数）。")
            else:
                score, k, labels = best
                k = len(set(labels))  # actual cluster count (KMeans may collapse on duplicate points)
                assign = pd.DataFrame({"row": X.index, "cluster": labels})
                assign.to_csv(d / "cluster_assignments.csv", index=False, encoding="utf-8")
                files.append("cluster_assignments.csv")

                profile_out = X.groupby(labels).mean()
                size = X.groupby(labels).size()
                profile_out["size"] = size.values
                profile_out.to_csv(d / "cluster_profile.csv", encoding="utf-8")
                files.append("cluster_profile.csv")

                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    n_components = min(2, len(features))
                    pca_coords = PCA(n_components=n_components).fit_transform(Xs)
                    fig, ax = plt.subplots(figsize=(6, 5))
                    if n_components == 2:
                        ax.scatter(pca_coords[:, 0], pca_coords[:, 1], c=labels, cmap="tab10", s=20)
                    else:
                        ax.scatter(pca_coords[:, 0], [0] * len(pca_coords), c=labels, cmap="tab10", s=20)
                    ax.set_xlabel("PC1")
                    ax.set_ylabel("PC2" if n_components == 2 else "")
                    ax.set_title(f"K-means (k={k}) — PCA projection")
                    fig.tight_layout()
                    fig.savefig(d / "pca_scatter.png", dpi=150)
                    plt.close(fig)
                    files.append("pca_scatter.png")
                    _plotly_scatter(
                        pca_coords, labels, d / "cluster_scatter.html",
                        f"K-means (k={k}) — interactive", "PC1",
                        "PC2" if n_components == 2 else "",
                    )
                    if (d / "cluster_scatter.html").exists():
                        files.append("cluster_scatter.html")
                except Exception:
                    pass

                _silhouette_plot(Xs, labels, d / "silhouette.png")
                if (d / "silhouette.png").exists():
                    files.append("silhouette.png")

                estimates["silhouette"] = float(score)
                estimates["k"] = float(k)
                summary.append(
                    f"{entry.method} 完成：在 {len(features)} 个连续特征上聚成 {k} 类，silhouette={score:.4f}"
                )
                code += [
                    "from sklearn.preprocessing import StandardScaler",
                    "from sklearn.cluster import KMeans",
                    "from sklearn.metrics import silhouette_score",
                    f"features = {features!r}",
                    "X = df[features].dropna()",
                    "Xs = StandardScaler().fit_transform(X)",
                    f"labels = KMeans(n_clusters={k}, random_state=0, n_init=10).fit_predict(Xs)",
                    "print('silhouette:', silhouette_score(Xs, labels))",
                ]
        except Exception as err:
            summary.append(f"K-means 执行失败：{err}")



@register("network_analysis")
def _branch_network_analysis(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import importlib.util

    # node-identifier columns for an edge list: config source/target, else the
    # first two id/categorical columns.
    id_cols = [c.name for c in fp.columns if c.kind in {"id", "categorical"} and c.name != fp.time_col]
    source = cfg.get("source") or (id_cols[0] if id_cols else None)
    target = cfg.get("target") or (id_cols[1] if len(id_cols) > 1 else None)
    weight = cfg.get("weight")
    if weight and (weight not in df.columns or weight in {source, target}):
        weight = None
    directed = bool(cfg.get("directed", False))
    if importlib.util.find_spec("networkx") is None:
        summary.append("网络分析需要 networkx 包（未检测到）。安装：pip install networkx。")
    elif source is None or target is None or source == target or source not in df.columns or target not in df.columns:
        summary.append(
            "网络分析失败：需要两列节点标识（边的 source / target）。"
            "用 config={\"source\":\"<列>\",\"target\":\"<列>\"} 指定（可选 weight）。"
        )
    else:
        try:
            import pandas as pd

            metrics, cent = _network_via_nx(df, source, target, weight, directed, d / "network.png")
            if (d / "network.png").exists():
                files.append("network.png")
            pd.DataFrame(list(metrics.items()), columns=["metric", "value"]).to_csv(
                d / "network_metrics.csv", index=False, encoding="utf-8"
            )
            files.append("network_metrics.csv")
            cent.to_csv(d / "node_centrality.csv", index=False, encoding="utf-8")
            files.append("node_centrality.csv")
            for k in ("n_nodes", "n_edges", "density", "avg_clustering", "n_communities", "modularity"):
                if k in metrics:
                    estimates[k] = float(metrics[k])
            top = cent.head(3)["node"].astype(str).tolist()
            dia = metrics.get("diameter_largest", "—")
            apl = metrics.get("avg_path_len_largest", "—")
            (d / "network_summary.txt").write_text(
                f"网络分析（networkx，{'有向' if directed else '无向'}图）：边 {source}→{target}"
                + (f"，权重 {weight}" if weight else "") + "\n"
                f"节点 {metrics['n_nodes']}，边 {metrics['n_edges']}，密度 {metrics['density']}，"
                f"平均度 {metrics['avg_degree']}，平均聚类系数 {metrics['avg_clustering']}\n"
                f"连通分量 {metrics['n_components']}（最大占比 {metrics['largest_component_frac']}），"
                f"最大分量 直径={dia}、平均路径长={apl}，度同配性 {metrics.get('degree_assortativity')}\n"
                f"社团（Louvain）{metrics['n_communities']} 个，模块度 {metrics['modularity']}\n"
                f"度中心性最高节点：{top}\n"
                "注：网络指标是结构性/描述性的（非因果）；社团划分依赖算法与分辨率，"
                "Louvain 有随机性（已固定 seed）。\n\n"
                "节点中心性（前 20）：\n" + cent.head(20).to_string(index=False),
                encoding="utf-8",
            )
            files.append("network_summary.txt")
            summary.append(
                f"{entry.method} 完成（networkx，{'有向' if directed else '无向'}）：边 {source}→{target}；"
                f"{metrics['n_nodes']} 节点、{metrics['n_edges']} 边，密度 {metrics['density']}，"
                f"聚类系数 {metrics['avg_clustering']}；{metrics['n_components']} 个连通分量；"
                f"Louvain 社团 {metrics['n_communities']} 个（模块度 {metrics['modularity']}）；"
                f"度中心性最高 {top}。⚠ 结构性描述（非因果）；社团划分依算法/分辨率。"
            )
            code += [
                "import networkx as nx  # 网络/图分析",
                f"# G=nx.from_pandas_edgelist(df,{source!r},{target!r}); 中心性 + louvain_communities + modularity",
            ]
        except Exception as err:
            summary.append(f"网络分析失败：{err}")



@register("pca")
def _branch_pca(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    features = [
        c.name
        for c in fp.columns
        if c.kind == "continuous" and c.name not in {fp.unit_col, fp.time_col}
    ]
    X = df[features].dropna()
    if len(features) < 2 or len(X) < 3:
        summary.append("PCA 跳过：连续特征不足或样本太少。")
    else:
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
            import numpy as np
            import pandas as pd

            Xs = StandardScaler().fit_transform(X)
            n_comp = min(len(features), 10, len(X) - 1)
            pca = PCA(n_components=n_comp).fit(Xs)
            evr = pca.explained_variance_ratio_

            # explained_variance.csv: component (PC1..), explained_variance_ratio, cumulative
            ev_df = pd.DataFrame({
                "component": [f"PC{i+1}" for i in range(n_comp)],
                "explained_variance_ratio": evr,
                "cumulative": np.cumsum(evr),
            })
            ev_df.to_csv(d / "explained_variance.csv", index=False, encoding="utf-8")
            files.append("explained_variance.csv")

            # loadings.csv: rows=features, cols=PC1..n
            load_df = pd.DataFrame(
                pca.components_.T,
                index=features,
                columns=[f"PC{i+1}" for i in range(n_comp)],
            )
            load_df.to_csv(d / "loadings.csv", encoding="utf-8")
            files.append("loadings.csv")

            # scree plot (bar of evr) -> pca_scree.png
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar([f"PC{i+1}" for i in range(n_comp)], evr)
                ax.set_xlabel("component")
                ax.set_ylabel("explained variance ratio")
                ax.set_title("PCA scree plot")
                fig.tight_layout()
                fig.savefig(d / "pca_scree.png", dpi=150)
                plt.close(fig)
                files.append("pca_scree.png")
            except Exception:
                pass

            estimates["pc1_explained_ratio"] = float(evr[0])
            estimates["n_components"] = float(n_comp)
            estimates["cum_explained_top2"] = float(np.cumsum(evr)[min(1, n_comp - 1)])
            summary.append(
                f"{entry.method} 完成：{len(features)} 个连续特征 -> {n_comp} 个主成分，"
                f"PC1 解释方差={evr[0]:.1%}"
            )
            code += [
                "from sklearn.preprocessing import StandardScaler",
                "from sklearn.decomposition import PCA",
                "import numpy as np",
                f"features = {features!r}",
                "X = df[features].dropna()",
                "Xs = StandardScaler().fit_transform(X)",
                f"pca = PCA(n_components={n_comp}).fit(Xs)",
                "print('explained variance ratio:', pca.explained_variance_ratio_)",
            ]
        except Exception as err:
            summary.append(f"PCA 执行失败：{err}")



@register("random_forest")
def _branch_random_forest(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    cont_cols = [c.name for c in fp.columns if c.kind == "continuous"]
    binary_cols = [c.name for c in fp.columns if c.kind == "binary"]

    # Prefer a continuous outcome (regression). Classify a binary outcome only
    # when there is no continuous column — a lone binary is usually a
    # treatment / flag *feature*, not the prediction target. This prevents
    # silently running the wrong analysis on the common "outcome + indicator" shape.
    if cont_cols:
        outcome, is_clf = cont_cols[0], False
    elif binary_cols:
        outcome, is_clf = binary_cols[0], True
    else:
        outcome, is_clf = None, False

    exclude = {outcome, fp.unit_col, fp.time_col}
    features = [
        c.name
        for c in fp.columns
        if c.kind in {"continuous", "count", "binary"} and c.name not in exclude
    ]

    if outcome is None:
        summary.append("随机森林失败：未找到合适的结果变量（需要连续型或二值列）。")
    elif not features:
        summary.append("随机森林失败：未找到可用的特征列。")
    else:
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.model_selection import train_test_split

            mask = df[features].notna().all(axis=1) & df[outcome].notna()
            X = df.loc[mask, features]
            y = df.loc[mask, outcome]

            if y.nunique() < 2:
                raise ValueError(f"结果变量 {outcome} 取值不足两类，无法建模")

            split_kwargs = {"test_size": 0.25, "random_state": 0}
            if is_clf and int(y.value_counts().min()) >= 2:
                split_kwargs["stratify"] = y
            X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)

            model = (
                RandomForestClassifier(n_estimators=200, random_state=0)
                if is_clf
                else RandomForestRegressor(n_estimators=200, random_state=0)
            )

            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)

            import pandas as pd
            imp_df = pd.DataFrame(
                {"feature": features, "importance": model.feature_importances_}
            ).sort_values("importance", ascending=False)
            imp_df.to_csv(d / "feature_importances.csv", index=False, encoding="utf-8")
            files.append("feature_importances.csv")

            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(6, max(3, len(features) * 0.4)))
                ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1])
                ax.set_xlabel("importance")
                ax.set_title(f"Feature importances — {outcome}")
                fig.tight_layout()
                fig.savefig(d / "feature_importances.png", dpi=150)
                plt.close(fig)
                files.append("feature_importances.png")
            except Exception:
                pass

            estimates["test_score"] = float(score)
            score_label = "accuracy" if is_clf else "R²"
            task_label = "分类" if is_clf else "回归"
            summary.append(
                f"{entry.method} 完成：{task_label}预测 {outcome}，"
                f"测试集得分={score:.4f}（{score_label}）"
            )
            code += [
                "from sklearn.ensemble import "
                + ("RandomForestClassifier" if is_clf else "RandomForestRegressor"),
                "from sklearn.model_selection import train_test_split",
                f"features = {features!r}",
                f"X = df[features].dropna()",
                f"y = df.loc[X.index, '{outcome}'].dropna()",
                f"X = X.loc[y.index]",
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)",
                "model = "
                + ("RandomForestClassifier" if is_clf else "RandomForestRegressor")
                + "(n_estimators=200, random_state=0)",
                "model.fit(X_train, y_train)",
                f"print('score:', model.score(X_test, y_test))",
            ]
        except Exception as err:
            summary.append(f"随机森林执行失败：{err}")



@register("xgboost")
def _branch_xgboost(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    cont_cols = [c.name for c in fp.columns if c.kind == "continuous"]
    binary_cols = [c.name for c in fp.columns if c.kind == "binary"]

    # Prefer a continuous outcome (regression). Classify a binary outcome only
    # when there is no continuous column — a lone binary is usually a
    # treatment / flag *feature*, not the prediction target. This prevents
    # silently running the wrong analysis on the common "outcome + indicator" shape.
    if cont_cols:
        outcome, is_clf = cont_cols[0], False
    elif binary_cols:
        outcome, is_clf = binary_cols[0], True
    else:
        outcome, is_clf = None, False

    exclude = {outcome, fp.unit_col, fp.time_col}
    features = [
        c.name
        for c in fp.columns
        if c.kind in {"continuous", "count", "binary"} and c.name not in exclude
    ]

    if outcome is None:
        summary.append("XGBoost 失败：未找到合适的结果变量（需要连续型或二值列）。")
    elif not features:
        summary.append("XGBoost 失败：未找到可用的特征列。")
    else:
        try:
            from xgboost import XGBClassifier, XGBRegressor
            from sklearn.model_selection import train_test_split

            mask = df[features].notna().all(axis=1) & df[outcome].notna()
            X = df.loc[mask, features]
            y = df.loc[mask, outcome]

            if y.nunique() < 2:
                raise ValueError(f"结果变量 {outcome} 取值不足两类，无法建模")

            split_kwargs = {"test_size": 0.25, "random_state": 0}
            if is_clf and int(y.value_counts().min()) >= 2:
                split_kwargs["stratify"] = y
            X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)

            model = (
                XGBClassifier(n_estimators=200, random_state=0, verbosity=0)
                if is_clf
                else XGBRegressor(n_estimators=200, random_state=0, verbosity=0)
            )

            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)

            import pandas as pd
            imp_df = pd.DataFrame(
                {"feature": features, "importance": model.feature_importances_}
            ).sort_values("importance", ascending=False)
            imp_df.to_csv(d / "feature_importances.csv", index=False, encoding="utf-8")
            files.append("feature_importances.csv")

            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(6, max(3, len(features) * 0.4)))
                ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1])
                ax.set_xlabel("importance")
                ax.set_title(f"Feature importances — {outcome}")
                fig.tight_layout()
                fig.savefig(d / "feature_importances.png", dpi=150)
                plt.close(fig)
                files.append("feature_importances.png")
            except Exception:
                pass

            estimates["test_score"] = float(score)
            score_label = "accuracy" if is_clf else "R²"
            task_label = "分类" if is_clf else "回归"
            summary.append(
                f"{entry.method} 完成：{task_label}预测 {outcome}，"
                f"测试集得分={score:.4f}（{score_label}）"
            )
            code += [
                "from xgboost import "
                + ("XGBClassifier" if is_clf else "XGBRegressor"),
                "from sklearn.model_selection import train_test_split",
                f"features = {features!r}",
                f"X = df[features].dropna()",
                f"y = df.loc[X.index, '{outcome}'].dropna()",
                f"X = X.loc[y.index]",
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)",
                "model = "
                + ("XGBClassifier" if is_clf else "XGBRegressor")
                + "(n_estimators=200, random_state=0, verbosity=0)",
                "model.fit(X_train, y_train)",
                f"print('score:', model.score(X_test, y_test))",
            ]
        except Exception as err:
            summary.append(f"XGBoost 执行失败：{err}")

