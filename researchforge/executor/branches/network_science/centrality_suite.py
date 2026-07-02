"""centrality_suite — 5 centralities per node + their Spearman agreement."""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.branches.network_science._common import _build_graph, _resolve_edges


@register("centrality_suite")
def _branch_centrality_suite(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    source, target, weight, directed, problem = _resolve_edges(ctx)
    if problem is not None:
        summary.append(problem)
        return
    try:
        import networkx as nx
        import numpy as np
        import pandas as pd

        G, _ = _build_graph(df, source, target, weight, directed)
        w = weight if weight else None
        n = G.number_of_nodes()
        nodes = list(G.nodes())

        measures = ["degree", "betweenness", "closeness", "eigenvector", "pagerank"]
        size_by = cfg.get("size_by")
        if size_by not in measures:
            size_by = "betweenness"
        top_n = int(cfg.get("top_n", 5))

        deg = nx.degree_centrality(G)  # NOTE: unweighted by construction (normalized degree)
        bet = nx.betweenness_centrality(G, weight=w, seed=0)
        clo = nx.closeness_centrality(G, distance=w)  # use weights as distance (consistent w/ betweenness)
        # eigenvector may fail to converge on some graphs -> flag & fall back to NaN
        eig_ok = True
        try:
            eig = nx.eigenvector_centrality_numpy(G, weight=w)
        except Exception:
            eig_ok = False
            eig = {x: float("nan") for x in nodes}
        try:
            pr = nx.pagerank(G, weight=w)
        except Exception:
            pr = {x: float("nan") for x in nodes}

        cent = pd.DataFrame({
            "node": nodes,
            "degree": [round(deg[x], 5) for x in nodes],
            "betweenness": [round(bet[x], 5) for x in nodes],
            "closeness": [round(clo[x], 5) for x in nodes],
            "eigenvector": [round(eig[x], 5) if eig[x] == eig[x] else float("nan") for x in nodes],
            "pagerank": [round(pr[x], 5) if pr[x] == pr[x] else float("nan") for x in nodes],
        })
        cent_sorted = cent.sort_values("degree", ascending=False).reset_index(drop=True)
        cent_sorted.to_csv(d / "node_centrality.csv", index=False, encoding="utf-8")
        files.append("node_centrality.csv")

        # top-N nodes by each measure
        top_rows = []
        for mlabel in measures:
            ordered = cent.sort_values(mlabel, ascending=False).head(top_n)
            for rank, (_, r) in enumerate(ordered.iterrows(), start=1):
                top_rows.append({"measure": mlabel, "rank": rank, "node": r["node"], "value": r[mlabel]})
        top_df = pd.DataFrame(top_rows)
        top_df.to_csv(d / "top_nodes.csv", index=False, encoding="utf-8")
        files.append("top_nodes.csv")

        # Spearman correlation among the 5 centralities (do they agree on importance?)
        spear = cent[measures].corr(method="spearman")
        spear.to_csv(d / "centrality_spearman.csv", encoding="utf-8")
        files.append("centrality_spearman.csv")
        # mean off-diagonal agreement (ignore NaN, e.g. when eigenvector failed)
        sv = spear.values.astype(float)
        offdiag = sv[~np.eye(len(measures), dtype=bool)]
        offdiag = offdiag[~np.isnan(offdiag)]
        mean_agreement = float(np.mean(offdiag)) if offdiag.size else float("nan")

        # graph plot sized by chosen centrality
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            UG = G.to_undirected() if directed else G
            Gp = UG if n <= 400 else UG.subgraph(max(nx.connected_components(UG), key=len))
            pos = nx.spring_layout(Gp, seed=0)
            cmap = {"degree": deg, "betweenness": bet, "closeness": clo, "eigenvector": eig, "pagerank": pr}[size_by]
            raw = [cmap.get(x, 0.0) for x in Gp.nodes()]
            raw = [0.0 if (v != v) else v for v in raw]  # NaN -> 0
            mx = max(raw) if raw and max(raw) > 0 else 1.0
            sizes = [40 + 700 * (v / mx) for v in raw]
            fig, ax = plt.subplots(figsize=(7, 6))
            nx.draw_networkx_edges(Gp, pos, alpha=0.25, ax=ax)
            nx.draw_networkx_nodes(Gp, pos, node_size=sizes, node_color="#4C72B0", ax=ax)
            ax.set_title(f"Network sized by {size_by} centrality ({Gp.number_of_nodes()} nodes)")
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(d / "centrality_graph.png", dpi=150)
            plt.close(fig)
            files.append("centrality_graph.png")
        except Exception:
            pass

        estimates["n_nodes"] = float(n)
        estimates["n_edges"] = float(G.number_of_edges())
        estimates["mean_spearman_agreement"] = round(mean_agreement, 4) if mean_agreement == mean_agreement else float("nan")
        estimates["eigenvector_converged"] = 1.0 if eig_ok else 0.0
        estimates["max_degree_centrality"] = round(float(cent["degree"].max()), 5)
        estimates["max_betweenness"] = round(float(cent["betweenness"].max()), 5)

        def _topk(mlabel):
            return cent.sort_values(mlabel, ascending=False).head(3)["node"].astype(str).tolist()

        eig_note = "" if eig_ok else "（特征向量中心性未收敛，已置 NaN 并标记）"
        (d / "centrality_summary.txt").write_text(
            f"中心性套件（networkx）：边 {source}→{target}"
            + (f"，权重 {weight}" if weight else "") + (f"，{'有向' if directed else '无向'}图") + "\n"
            f"节点 {n}，边 {G.number_of_edges()}\n"
            f"度中心性最高：{_topk('degree')}\n"
            f"介数（中介/桥接）最高：{_topk('betweenness')}\n"
            f"接近中心性最高：{_topk('closeness')}\n"
            f"特征向量中心性最高：{_topk('eigenvector')} {eig_note}\n"
            f"PageRank 最高：{_topk('pagerank')}\n"
            f"5 种中心性的平均 Spearman 一致度：{round(mean_agreement, 4) if mean_agreement == mean_agreement else 'NaN'}\n"
            "注：每种中心性刻画不同的「重要」——度=连接数；介数=桥接/中介；接近=可达性；"
            "特征向量/PageRank=经由重要邻居的影响力；它们常不一致（见 Spearman 矩阵）。"
            "特征向量中心性在某些图上可能不收敛（已回退/标记）。\n\n"
            "Spearman 相关矩阵：\n" + spear.round(3).to_string() + "\n\n"
            "节点中心性（前 20）：\n" + cent_sorted.head(20).to_string(index=False),
            encoding="utf-8",
        )
        files.append("centrality_summary.txt")

        summary.append(
            f"{entry.method} 完成（networkx）：边 {source}→{target}；{n} 节点、{G.number_of_edges()} 边；"
            f"度最高 {_topk('degree')}，介数最高 {_topk('betweenness')}；"
            f"5 种中心性平均 Spearman 一致度 "
            f"{round(mean_agreement, 4) if mean_agreement == mean_agreement else 'NaN'}"
            + ("" if eig_ok else "；⚠ 特征向量中心性未收敛")
            + "。⚠ 各中心性刻画不同的重要性，常不一致（见相关矩阵）；特征向量可能不收敛（已标记）。"
            + ("" if not w else
               "⚠ 权重语义：度中心性按构造**不计权重**；介数/接近度把权重当**距离/成本**"
               "（值越大=越远，若你的权重是关系强度请先取倒数）；特征向量/PageRank 把权重当**连接强度**"
               "（值越大=越强）——同一表里两种相反语义，解读注意。")
        )
        code += [
            "import networkx as nx",
            f"G = nx.from_pandas_edgelist(df, {source!r}, {target!r})",
            "deg = nx.degree_centrality(G); bet = nx.betweenness_centrality(G, seed=0)",
            "clo = nx.closeness_centrality(G); eig = nx.eigenvector_centrality_numpy(G)",
            "pr = nx.pagerank(G)  # 5 centralities; compare via Spearman correlation",
        ]
    except Exception as err:
        summary.append(f"中心性套件失败：{err}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. epidemic_model — network-based stochastic SIR/SIS diffusion
# ─────────────────────────────────────────────────────────────────────────────
