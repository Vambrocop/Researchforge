"""community_detection — Louvain modules + modularity Q (python-louvain)."""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.branches.network_science._common import _build_graph, _resolve_edges


@register("community_detection")
def _branch_community_detection(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    source, target, weight, directed, problem = _resolve_edges(ctx)
    if problem is not None:
        summary.append(problem)
        return
    try:
        import community as community_louvain  # python-louvain
        import networkx as nx
        import pandas as pd
        from collections import Counter

        _, UG = _build_graph(df, source, target, weight, directed)
        w = weight if weight else None

        # Louvain (greedily maximises modularity; seeded -> reproducible).
        # python-louvain BREAKS on weight=None (induced_graph does **{None:..} →
        # "keywords must be strings"); pass the actual weight column name when weighted,
        # else the literal "weight" (no such edge attr → .get defaults to 1 = unweighted).
        wkey = w if w else "weight"
        partition = community_louvain.best_partition(UG, weight=wkey, random_state=0)
        q = community_louvain.modularity(partition, UG, weight=wkey)

        node2comm = dict(partition)
        # re-index communities by descending size for stable, readable labels
        size_by_raw = Counter(node2comm.values())
        order = [cid for cid, _ in size_by_raw.most_common()]
        relabel = {cid: i for i, cid in enumerate(order)}
        node2comm = {n: relabel[c] for n, c in node2comm.items()}
        sizes = Counter(node2comm.values())
        n_comm = len(sizes)

        n_nodes = UG.number_of_nodes()
        # node -> community CSV
        nc = pd.DataFrame(
            {"node": list(UG.nodes()), "community": [node2comm[n] for n in UG.nodes()]}
        ).sort_values(["community", "node"]).reset_index(drop=True)
        nc.to_csv(d / "node_communities.csv", index=False, encoding="utf-8")
        files.append("node_communities.csv")

        # community sizes CSV
        cs = pd.DataFrame(
            {"community": list(range(n_comm)), "size": [int(sizes[i]) for i in range(n_comm)]}
        )
        cs.to_csv(d / "community_sizes.csv", index=False, encoding="utf-8")
        files.append("community_sizes.csv")

        # Optional cross-check vs networkx greedy modularity (different heuristic).
        greedy_n = greedy_q = None
        try:
            greedy = list(nx.community.greedy_modularity_communities(UG, weight=w))
            greedy_n = len(greedy)
            greedy_q = float(nx.community.modularity(UG, greedy, weight=w))
        except Exception:
            pass

        # graph plot coloured by community
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            Gp = UG if n_nodes <= 400 else UG.subgraph(max(nx.connected_components(UG), key=len))
            pos = nx.spring_layout(Gp, seed=0)
            colors = [node2comm.get(n, 0) for n in Gp.nodes()]
            fig, ax = plt.subplots(figsize=(7, 6))
            nx.draw_networkx_edges(Gp, pos, alpha=0.25, ax=ax)
            nx.draw_networkx_nodes(Gp, pos, node_color=colors, node_size=120, cmap="tab20", ax=ax)
            ax.set_title(f"Communities (Louvain): {n_comm} modules, Q={q:.3f}")
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(d / "communities.png", dpi=150)
            plt.close(fig)
            files.append("communities.png")
        except Exception:
            pass

        estimates["n_nodes"] = float(n_nodes)
        estimates["n_edges"] = float(UG.number_of_edges())
        estimates["n_communities"] = float(n_comm)
        estimates["modularity"] = round(float(q), 4)
        estimates["largest_community_size"] = float(max(sizes.values()) if sizes else 0)
        if greedy_n is not None:
            estimates["greedy_n_communities"] = float(greedy_n)
            estimates["greedy_modularity"] = round(greedy_q, 4)

        meaningful = "有意义的社团结构（Q>0.3）" if q > 0.3 else "弱/无明显社团结构（Q≤0.3）"
        topsizes = ", ".join(str(int(sizes[i])) for i in range(min(n_comm, 6)))
        (d / "community_summary.txt").write_text(
            f"社团发现（Louvain, python-louvain）：边 {source}→{target}"
            + (f"，权重 {weight}" if weight else "") + "\n"
            f"节点 {n_nodes}，边 {UG.number_of_edges()}\n"
            f"社团数 {n_comm}，模块度 Q={round(float(q), 4)} —— {meaningful}\n"
            f"前几大社团规模：{topsizes}\n"
            + (f"对照 networkx 贪心模块度：{greedy_n} 个社团，Q={round(greedy_q, 4)}\n" if greedy_n is not None else "")
            + "注：Louvain 贪心最大化模块度（启发式，已固定 random_state=0 求可复现）；"
            "模块度有分辨率极限（很小的社团可能被并入大社团）；社团是结构性划分，"
            "并非外部验证过的真实分组。\n\n"
            "节点→社团（前 30）：\n" + nc.head(30).to_string(index=False),
            encoding="utf-8",
        )
        files.append("community_summary.txt")

        summary.append(
            f"{entry.method} 完成（Louvain）：边 {source}→{target}；{n_nodes} 节点、"
            f"{UG.number_of_edges()} 边；发现 {n_comm} 个社团，模块度 Q={round(float(q), 4)}"
            f"（{meaningful}）"
            + (f"；贪心法对照 {greedy_n} 社团 Q={round(greedy_q, 4)}" if greedy_n is not None else "")
            + "。⚠ Louvain 是启发式（已固定 seed）；模块度有分辨率极限；社团为结构性划分（非验证分组）。"
        )
        code += [
            "import community as community_louvain  # python-louvain",
            "import networkx as nx",
            f"G = nx.from_pandas_edgelist(df, {source!r}, {target!r})",
            "part = community_louvain.best_partition(G, random_state=0)",
            "Q = community_louvain.modularity(part, G)  # >0.3 = meaningful structure",
        ]
    except Exception as err:
        summary.append(f"社团发现失败：{err}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. centrality_suite — 5 centralities + Spearman agreement matrix
# ─────────────────────────────────────────────────────────────────────────────
