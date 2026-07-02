"""Shared edge-list helpers for the network_science family: resolve two node-identifier
columns (config source/target, optional weight, directed) into a graph, degrading
honestly. Used by every network_science method module."""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx


def _resolve_edges(ctx: Ctx):
    import importlib.util

    fp, cfg, df = ctx.fp, ctx.cfg, ctx.df
    id_cols = [c.name for c in fp.columns if c.kind in {"id", "categorical"} and c.name != fp.time_col]
    source = cfg.get("source") or (id_cols[0] if id_cols else None)
    target = cfg.get("target") or (id_cols[1] if len(id_cols) > 1 else None)
    weight = cfg.get("weight")
    if weight and (weight not in df.columns or weight in {source, target}):
        weight = None
    directed = bool(cfg.get("directed", False))

    if importlib.util.find_spec("networkx") is None:
        return None, None, None, directed, "网络分析需要 networkx 包（未检测到）。安装：pip install networkx。"
    if source is None or target is None or source == target or source not in df.columns or target not in df.columns:
        return None, None, None, directed, (
            "网络分析失败：需要两列节点标识（边的 source / target）。"
            "用 config={\"source\":\"<列>\",\"target\":\"<列>\"} 指定（可选 weight）。"
        )
    return source, target, weight, directed, None


def _build_graph(df, source, target, weight, directed):
    """Build a networkx graph from the edge list. Raises if <3 nodes so the caller
    can degrade honestly. Returns (G, UG) — G respects `directed`, UG is undirected
    (Louvain / clustering / epidemic spread run on UG)."""
    import networkx as nx

    cols = [source, target] + ([weight] if weight else [])
    sub = df[cols].dropna()
    create = nx.DiGraph if directed else nx.Graph
    G = nx.from_pandas_edgelist(
        sub, source, target, edge_attr=(weight if weight else None), create_using=create()
    )
    if G.number_of_nodes() < 3:
        raise RuntimeError("有效节点 <3，无法做网络分析")
    UG = G.to_undirected() if directed else G
    return G, UG


# ─────────────────────────────────────────────────────────────────────────────
# 1. community_detection — Louvain modules + modularity Q
# ─────────────────────────────────────────────────────────────────────────────
