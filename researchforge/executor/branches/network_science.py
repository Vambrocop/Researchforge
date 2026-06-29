"""Branch handlers for the network_science family.

Network/graph-science methods that build a graph from an EDGE LIST (two
node-identifier columns + optional weight, like the existing ml._branch_network_analysis):

  * community_detection     — Louvain modules + modularity Q (python-louvain `community`)
  * centrality_suite        — 5 centralities per node + their Spearman agreement
  * epidemic_model          — network-based stochastic SIR/SIS diffusion simulation
  * link_prediction         — CN/Jaccard/Adamic-Adar/RA/PA scores, held-out-edge AUC + top-K
  * stochastic_block_model  — spectral (ASE→KMeans) SBM, MLE block matrix, ICL block selection
  * ergm                    — exponential random graph model (R statnet/ergm) + CUG-test degrade

Each handler resolves the edge list, degrades honestly (no networkx / too few nodes /
no edge list -> skip with a Chinese ⚠ message), writes CSV + PNG (matplotlib Agg,
ENGLISH plot labels), fills float `estimates`, appends a Chinese `summary` with ⚠
disclosures, and mutates ctx (never rebinds). See executor/_branch_api.py and CLAUDE.md.

networkx + python-louvain (imported as `community`) are installed. ergm delegates to R's
gold-standard statnet/ergm package and degrades to a pure-Python CUG test when R/ergm is
absent (R is optional + graceful degrade; R code is audited fixed strings, never fetched).
"""

from __future__ import annotations

import re

from researchforge.executor._branch_api import Ctx, register


# ─────────────────────────────────────────────────────────────────────────────
# Shared edge-list resolution (same idiom as ml._branch_network_analysis).
# Returns (source, target, weight, directed, problem_msg).  When problem_msg is
# not None the caller should append it to summary and return (honest degrade).
# ─────────────────────────────────────────────────────────────────────────────
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
def _simulate_epidemic(UG, model, beta, gamma, initial_infected, steps, n_runs, seed):
    """Discrete-time stochastic SIR/SIS on the network UG, averaged over n_runs.

    Each step (synchronous update): every S node becomes I with prob
    1-(1-beta)^(#infected neighbours); every I node recovers with prob gamma
    (SIR -> R, immune; SIS -> S, susceptible again).
    Returns (mean_curve dict t->(S,I,R), peak_I, time_to_peak, attack_rate)."""
    import random

    nodes = list(UG.nodes())
    N = len(nodes)
    neighbors = {u: list(UG.neighbors(u)) for u in nodes}

    runs_curves = []
    attack_rates = []
    for run in range(n_runs):
        rng = random.Random(seed + run)
        seeds = rng.sample(nodes, min(initial_infected, N))
        state = {u: "S" for u in nodes}
        for s in seeds:
            state[s] = "I"
        ever_infected = set(seeds)

        curve = []  # list of (S, I, R)
        for _t in range(steps + 1):
            S = sum(1 for u in nodes if state[u] == "S")
            I = sum(1 for u in nodes if state[u] == "I")
            R = N - S - I
            curve.append((S, I, R))
            if I == 0:
                # epidemic died out: pad remaining steps with the steady state
                last = (S, I, R)
                while len(curve) < steps + 1:
                    curve.append(last)
                break
            # compute new states (synchronous update)
            new_state = dict(state)
            for u in nodes:
                if state[u] == "I":
                    if rng.random() < gamma:
                        new_state[u] = "R" if model == "sir" else "S"
                elif state[u] == "S":
                    inf_nb = sum(1 for v in neighbors[u] if state[v] == "I")
                    if inf_nb:
                        p = 1.0 - (1.0 - beta) ** inf_nb
                        if rng.random() < p:
                            new_state[u] = "I"
                            ever_infected.add(u)
            state = new_state
        runs_curves.append(curve)
        attack_rates.append(len(ever_infected) / N)

    # average the curves across runs (all padded to steps+1)
    T = steps + 1
    mean_curve = {}
    for t in range(T):
        s = sum(c[t][0] for c in runs_curves) / n_runs
        i = sum(c[t][1] for c in runs_curves) / n_runs
        r = sum(c[t][2] for c in runs_curves) / n_runs
        mean_curve[t] = (s, i, r)

    peak_I = max(mean_curve[t][1] for t in range(T))
    time_to_peak = max(range(T), key=lambda t: mean_curve[t][1])
    attack_rate = sum(attack_rates) / n_runs
    return mean_curve, peak_I, time_to_peak, attack_rate


@register("epidemic_model")
def _branch_epidemic_model(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    source, target, weight, directed, problem = _resolve_edges(ctx)
    if problem is not None:
        summary.append(problem)
        return
    try:
        import numpy as np
        import pandas as pd

        # spread runs on the undirected contact graph
        _, UG = _build_graph(df, source, target, weight, directed)
        N = UG.number_of_nodes()

        model = str(cfg.get("model", "sir")).lower()
        if model not in {"sir", "sis"}:
            model = "sir"
        beta = float(cfg.get("beta", 0.05))
        gamma = float(cfg.get("gamma", 0.1))
        initial_infected = max(1, int(cfg.get("initial_infected", 1)))
        steps = int(cfg.get("steps", 60))
        n_runs = max(1, int(cfg.get("n_runs", 10)))
        seed = int(cfg.get("seed", 0))

        mean_curve, peak_I, time_to_peak, attack_rate = _simulate_epidemic(
            UG, model, beta, gamma, initial_infected, steps, n_runs, seed
        )

        T = steps + 1
        curve_df = pd.DataFrame({
            "t": list(range(T)),
            "S": [round(mean_curve[t][0], 3) for t in range(T)],
            "I": [round(mean_curve[t][1], 3) for t in range(T)],
            "R": [round(mean_curve[t][2], 3) for t in range(T)],
        })
        curve_df.to_csv(d / "epidemic_curve.csv", index=False, encoding="utf-8")
        files.append("epidemic_curve.csv")

        # R0 proxy = beta * <k> / gamma  (rough; true threshold ~ <k^2>/<k>)
        degs = [dg for _, dg in UG.degree()]
        mean_k = float(np.mean(degs)) if degs else 0.0
        r0_proxy = (beta * mean_k / gamma) if gamma > 0 else float("inf")

        # epidemic curve plot
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot(curve_df["t"], curve_df["S"], label="S (susceptible)", color="#4C72B0")
            ax.plot(curve_df["t"], curve_df["I"], label="I (infected)", color="#C44E52")
            if model == "sir":
                ax.plot(curve_df["t"], curve_df["R"], label="R (recovered)", color="#55A868")
            ax.axvline(time_to_peak, ls="--", lw=1, color="grey", alpha=0.7)
            ax.set_xlabel("time step")
            ax.set_ylabel("number of nodes")
            ax.set_title(f"Network {model.upper()} epidemic curve (mean of {n_runs} runs)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(d / "epidemic_curve.png", dpi=150)
            plt.close(fig)
            files.append("epidemic_curve.png")
        except Exception:
            pass

        estimates["n_nodes"] = float(N)
        estimates["mean_degree"] = round(mean_k, 3)
        estimates["beta"] = round(beta, 4)
        estimates["gamma"] = round(gamma, 4)
        estimates["peak_infected"] = round(float(peak_I), 3)
        estimates["peak_infected_frac"] = round(float(peak_I) / N, 4) if N else 0.0
        estimates["time_to_peak"] = float(time_to_peak)
        estimates["attack_rate"] = round(float(attack_rate), 4)
        estimates["r0_proxy"] = round(float(r0_proxy), 4) if r0_proxy != float("inf") else float("inf")

        spread_note = "可能扩散（R0 代理 >1）" if r0_proxy > 1 else "趋于熄灭（R0 代理 ≤1）"
        (d / "epidemic_summary.txt").write_text(
            f"网络传播模拟（{model.upper()}，基于接触网络的离散时间随机过程）：边 {source}→{target}\n"
            f"节点 {N}，平均度 <k>={round(mean_k, 3)}\n"
            f"参数：beta（每接触每步传播概率）={beta}，gamma（每步康复概率）={gamma}，"
            f"初始感染 {initial_infected} 个随机种子（已固定 seed={seed}），"
            f"模拟 {steps} 步、{n_runs} 次取平均\n"
            f"峰值感染 {round(float(peak_I), 2)}（占 {round(float(peak_I) / N * 100, 1) if N else 0}%），"
            f"达峰时间 t={time_to_peak}\n"
            f"最终攻击率（曾被感染比例）={round(float(attack_rate), 4)}\n"
            f"R0 代理 = beta·<k>/gamma = {round(float(r0_proxy), 3) if r0_proxy != float('inf') else 'inf'} —— {spread_note}\n"
            "注：网络 SIR/SIS 取决于接触结构（度的异质性驱动传播——hub 加速扩散）；"
            "beta/gamma 是假定参数（已报告）；过程是随机的（多次取平均、seed 已固定）；"
            "R0 代理是粗略值，真实流行阈值取决于度分布方差 <k^2>/<k>。\n\n"
            "流行曲线（前 20 步）：\n" + curve_df.head(20).to_string(index=False),
            encoding="utf-8",
        )
        files.append("epidemic_summary.txt")

        summary.append(
            f"{entry.method} 完成（网络 {model.upper()}）：边 {source}→{target}；{N} 节点、平均度 {round(mean_k, 3)}；"
            f"beta={beta}, gamma={gamma}；峰值感染 {round(float(peak_I), 1)}"
            f"（占 {round(float(peak_I) / N * 100, 1) if N else 0}%）于 t={time_to_peak}；"
            f"最终攻击率 {round(float(attack_rate), 4)}；R0 代理 "
            f"{round(float(r0_proxy), 3) if r0_proxy != float('inf') else 'inf'}（{spread_note}）。"
            "⚠ 依赖接触结构（hub 加速扩散）；beta/gamma 为假定（已报告）；随机（多次取平均、seed 固定）；R0 代理粗略。"
        )
        code += [
            "import networkx as nx, random",
            f"G = nx.from_pandas_edgelist(df, {source!r}, {target!r})",
            f"# discrete-time stochastic {model.upper()}: S->I w.p. 1-(1-beta)^(#inf nb), I->{'R' if model == 'sir' else 'S'} w.p. gamma",
            f"# beta={beta}, gamma={gamma}, {n_runs} runs averaged (seed fixed); R0 proxy = beta*<k>/gamma",
        ]
    except Exception as err:
        summary.append(f"传播模拟失败：{err}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. link_prediction — neighbourhood similarity scores + held-out-edge AUC + top-K
# ─────────────────────────────────────────────────────────────────────────────
def _lp_index_scores(fn, g, pairs):
    """Score node pairs with a networkx link-prediction index function (yields
    (u, v, score)); return scores aligned to `pairs`."""
    dd = {(u, v): s for u, v, s in fn(g, pairs)}
    return [dd[(u, v)] for u, v in pairs]


@register("link_prediction")
def _branch_link_prediction(ctx: Ctx) -> None:
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
        from sklearn.metrics import roc_auc_score

        # Link prediction runs on the undirected SIMPLE graph (topological scores
        # ignore edge weights / direction). _build_graph raises if <3 nodes.
        _, UG = _build_graph(df, source, target, weight, directed)
        UG = nx.Graph(UG)  # collapse to a simple undirected graph
        UG.remove_edges_from(nx.selfloop_edges(UG))
        n = UG.number_of_nodes()
        m = UG.number_of_edges()
        if m < 10:
            summary.append("链路预测失败：有效边 <10，留出评估不稳定（需要更大的网络）。")
            return

        test_frac = float(cfg.get("test_frac", 0.15))
        test_frac = min(max(test_frac, 0.05), 0.4)
        seed = int(cfg.get("seed", 0))
        top_k = int(cfg.get("top_k", 15))
        rng = np.random.default_rng(seed)

        # Hold out a random fraction of edges as positive test pairs; the rest train.
        edges = list(UG.edges())
        k = max(1, int(round(test_frac * len(edges))))
        order = rng.permutation(len(edges))
        test_pos = [edges[int(i)] for i in order[:k]]
        Gtr = UG.copy()
        Gtr.remove_edges_from(test_pos)

        # Sample an equal number of negatives = non-edges of the FULL graph (1:1 balance).
        nodes = list(UG.nodes())
        neg: set = set()
        attempts, max_attempts = 0, k * 400 + 1000
        while len(neg) < k and attempts < max_attempts:
            attempts += 1
            ai, bi = rng.choice(len(nodes), 2, replace=False)
            a, b = nodes[int(ai)], nodes[int(bi)]
            if not UG.has_edge(a, b) and (a, b) not in neg and (b, a) not in neg:
                neg.add((a, b))
        neg_list = list(neg)
        if not neg_list:
            summary.append("链路预测失败：无法采样负样本（图近乎完全图）。")
            return

        pairs = list(test_pos) + neg_list
        labels = [1] * len(test_pos) + [0] * len(neg_list)

        # Five neighbourhood-based predictors scored on the TRAINING graph.
        cn = [len(list(nx.common_neighbors(Gtr, u, v))) for u, v in pairs]
        preds = {
            "common_neighbors": cn,
            "jaccard": _lp_index_scores(nx.jaccard_coefficient, Gtr, pairs),
            "adamic_adar": _lp_index_scores(nx.adamic_adar_index, Gtr, pairs),
            "resource_allocation": _lp_index_scores(nx.resource_allocation_index, Gtr, pairs),
            "preferential_attachment": _lp_index_scores(nx.preferential_attachment, Gtr, pairs),
        }
        aucs = {name: float(roc_auc_score(labels, sc)) for name, sc in preds.items()}
        best = max(aucs, key=lambda kk: aucs[kk])

        auc_df = (
            pd.DataFrame({"predictor": list(aucs), "auc": [round(aucs[p], 4) for p in aucs]})
            .sort_values("auc", ascending=False)
            .reset_index(drop=True)
        )
        auc_df.to_csv(d / "link_prediction_auc.csv", index=False, encoding="utf-8")
        files.append("link_prediction_auc.csv")

        # Predicted NEW links: re-score candidate non-edges on the FULL graph with the
        # best predictor. Candidates = non-adjacent pairs sharing >=1 common neighbour
        # (the natural support of neighbourhood predictors; cheap and meaningful).
        index_fns = {
            "jaccard": nx.jaccard_coefficient,
            "adamic_adar": nx.adamic_adar_index,
            "resource_allocation": nx.resource_allocation_index,
            "preferential_attachment": nx.preferential_attachment,
        }
        cand: set = set()
        cap = 200_000
        for wnode in UG.nodes():
            nb = list(UG.neighbors(wnode))
            for i in range(len(nb)):
                for j in range(i + 1, len(nb)):
                    a, b = nb[i], nb[j]
                    if not UG.has_edge(a, b):
                        cand.add((a, b) if str(a) <= str(b) else (b, a))
            if len(cand) > cap:
                break
        cand_list = list(cand)
        pred_rows = []
        if cand_list:
            if best == "common_neighbors":
                cscores = [len(list(nx.common_neighbors(UG, u, v))) for u, v in cand_list]
            else:
                cscores = _lp_index_scores(index_fns[best], UG, cand_list)
            top_idx = np.argsort(cscores)[::-1][:top_k]
            pred_rows = [
                {"source": cand_list[int(i)][0], "target": cand_list[int(i)][1],
                 "score": round(float(cscores[int(i)]), 5)}
                for i in top_idx
            ]
            pd.DataFrame(pred_rows).to_csv(d / "predicted_links.csv", index=False, encoding="utf-8")
            files.append("predicted_links.csv")

        # AUC bar chart (ENGLISH labels).
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7, 4.2))
            ax.barh(auc_df["predictor"][::-1], auc_df["auc"][::-1], color="#4C72B0")
            ax.axvline(0.5, ls="--", lw=1, color="grey", label="random (0.5)")
            ax.set_xlim(0, 1)
            ax.set_xlabel("held-out AUC")
            ax.set_title(f"Link-prediction AUC by predictor (n={n}, {k} held-out edges)")
            ax.legend(loc="lower right")
            fig.tight_layout()
            fig.savefig(d / "link_prediction_auc.png", dpi=150)
            plt.close(fig)
            files.append("link_prediction_auc.png")
        except Exception:
            pass

        estimates["n_nodes"] = float(n)
        estimates["n_edges"] = float(m)
        estimates["n_test_edges"] = float(k)
        for name, a in aucs.items():
            estimates[f"auc_{name}"] = round(a, 4)
        estimates["best_predictor_auc"] = round(aucs[best], 4)
        estimates["n_candidate_links"] = float(len(cand_list))

        verdict = ("明显优于随机（AUC>0.7）" if aucs[best] > 0.7
                   else "中等（0.6<AUC≤0.7）" if aucs[best] > 0.6
                   else "弱/接近随机（AUC≤0.6）")
        toplinks = "、".join(f"{r['source']}–{r['target']}" for r in pred_rows[:5]) if pred_rows else "（无候选）"
        (d / "link_prediction_summary.txt").write_text(
            f"链路预测（邻域相似度 + 留出边 AUC）：边 {source}→{target}\n"
            f"节点 {n}，边 {m}；留出 {k} 条真实边作正样本 + 等量非边作负样本（1:1，seed={seed}）\n"
            f"各预测器留出 AUC：\n" + auc_df.to_string(index=False) + "\n"
            f"最佳预测器：{best}（AUC={round(aucs[best], 4)}，{verdict}）\n"
            f"预测的潜在新连边（最佳预测器在全图上 top-{top_k}）：{toplinks}\n"
            "注：AUC=随机取一对(真实边, 非边)、真实边得分更高的概率（0.5=随机）；"
            "评估是单次随机划分（seed 已固定），AUC 会随划分波动；"
            "把当前缺失的边当作「真实/未来应存在的连边」是链路预测的标准假设；"
            "负样本按 1:1 采样（真实网络极稀疏，正例罕见）；"
            "top-K 候选限定在「与已有节点有共同邻居」的非边上（邻域预测器的自然支撑集）。\n\n"
            + ("候选新连边（前 15）：\n" + pd.DataFrame(pred_rows).to_string(index=False) if pred_rows else ""),
            encoding="utf-8",
        )
        files.append("link_prediction_summary.txt")

        summary.append(
            f"{entry.method} 完成：边 {source}→{target}；{n} 节点、{m} 边；留出 {k} 条边评估。"
            f"最佳预测器 {best}（留出 AUC={round(aucs[best], 4)}，{verdict}）；"
            f"common_neighbors={round(aucs['common_neighbors'], 3)}、adamic_adar={round(aucs['adamic_adar'], 3)}、"
            f"jaccard={round(aucs['jaccard'], 3)}、preferential_attachment={round(aucs['preferential_attachment'], 3)}。"
            + (f"预测潜在新连边 top：{toplinks}。" if pred_rows else "")
            + " ⚠ 单次随机划分（seed 固定）AUC 会波动；把缺失边当未来真连边是标准假设；"
            "负样本 1:1 采样；top-K 候选限于有共同邻居的非边。"
        )
        code += [
            "import networkx as nx; from sklearn.metrics import roc_auc_score",
            f"G = nx.Graph(nx.from_pandas_edgelist(df, {source!r}, {target!r}))",
            "# hold out 15% of edges as positives, sample equal non-edges as negatives",
            "# score with common_neighbors / jaccard / adamic_adar / resource_allocation / preferential_attachment",
            "# AUC = roc_auc_score(labels, scores) on the held-out set (0.5 = random)",
        ]
    except Exception as err:
        summary.append(f"链路预测失败：{err}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. stochastic_block_model — spectral (ASE→KMeans) SBM + MLE block matrix + ICL
# ─────────────────────────────────────────────────────────────────────────────
def _fit_sbm(A, K, seed):
    """Fit a K-block SBM by adjacency spectral embedding (top-K eigenvectors of A by
    |eigenvalue|, scaled by sqrt|eigenvalue|) → KMeans hard assignment, then the MLE
    Bernoulli block-connection matrix B and the resulting SBM log-likelihood.
    Returns (labels, B[K,K], loglik). K=1 is the Erdos-Renyi baseline."""
    import numpy as np

    n = A.shape[0]
    if K <= 1:
        labels = np.zeros(n, dtype=int)
    else:
        from sklearn.cluster import KMeans

        w, V = np.linalg.eigh(A)
        sel = np.argsort(-np.abs(w))[:K]
        X = V[:, sel] * np.sqrt(np.abs(w[sel]))
        labels = KMeans(n_clusters=K, n_init=10, random_state=seed).fit_predict(X).astype(int)

    B = np.full((K, K), np.nan)
    ll = 0.0
    for r in range(K):
        ir = np.where(labels == r)[0]
        for s in range(r, K):
            isd = np.where(labels == s)[0]
            if r == s:
                if len(ir) < 2:
                    continue
                poss = len(ir) * (len(ir) - 1) / 2.0
                m = A[np.ix_(ir, ir)].sum() / 2.0
            else:
                poss = float(len(ir) * len(isd))
                m = float(A[np.ix_(ir, isd)].sum())
            if poss <= 0:
                continue
            b = m / poss
            B[r, s] = B[s, r] = b
            if 0.0 < b < 1.0:
                ll += m * np.log(b) + (poss - m) * np.log(1.0 - b)
    return labels, B, float(ll)


def _estimate_n_blocks(A, max_blocks):
    """Spectral estimate of the block count: the number of adjacency eigenvalues whose
    magnitude exceeds the random-matrix bulk edge 2*sqrt(mean degree) (Lei-Rinaldo /
    Le-Levina). Informative eigenvalues stick out beyond the semicircle bulk; the
    others are noise. This is ROBUST to within-block degree heterogeneity — unlike a
    likelihood/ICL search over K, which over-splits a single dense block into pieces
    because the plain (non-degree-corrected) SBM fits residual degree structure.
    Returns (K, eigen-evidence rows, bulk_threshold)."""
    import numpy as np

    n = A.shape[0]
    w = np.linalg.eigvalsh(A)
    wabs = np.sort(np.abs(w))[::-1]
    dbar = float(A.sum()) / n
    thr = 2.0 * np.sqrt(max(dbar, 1e-9))
    k = int(np.sum(wabs > thr))
    k = max(1, min(k, max_blocks))
    rows = [{"rank": i + 1, "abs_eigenvalue": round(float(wabs[i]), 4),
             "bulk_threshold": round(float(thr), 4), "beyond_bulk": bool(wabs[i] > thr)}
            for i in range(min(len(wabs), max_blocks + 2))]
    return k, rows, float(thr)


@register("stochastic_block_model")
def _branch_stochastic_block_model(ctx: Ctx) -> None:
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

        _, UG = _build_graph(df, source, target, weight, directed)
        UG = nx.Graph(UG)
        UG.remove_edges_from(nx.selfloop_edges(UG))
        nodes = list(UG.nodes())
        n = len(nodes)
        if n < 10:
            summary.append("随机块模型失败：有效节点 <10，块结构估计不可靠（需要更大的网络）。")
            return
        if n > 2000:
            summary.append("随机块模型跳过：节点 >2000，稠密谱嵌入开销过大（请抽样子图或用 community_detection）。")
            return

        A = nx.to_numpy_array(UG, nodelist=nodes, weight=None)  # binary, symmetric, 0 diagonal
        npairs = n * (n - 1) / 2.0

        forced_k = cfg.get("n_blocks")
        max_blocks = int(cfg.get("max_blocks", min(8, max(2, n // 5))))
        seed = int(cfg.get("seed", 0))

        # Select the block count K spectrally (robust), then fit the SBM at K.
        kspec, eig_rows, bulk_thr = _estimate_n_blocks(A, max_blocks)
        K = int(forced_k) if forced_k else kspec
        K = max(1, min(K, n))
        labels, B, ll = _fit_sbm(A, K, seed)
        # ICL of the chosen fit (reported as a diagnostic, NOT the selector — the plain
        # SBM likelihood over-splits on degree-heterogeneous graphs, hence spectral K).
        nparams = K * (K + 1) / 2.0
        icl = ll - 0.5 * nparams * np.log(npairs) - 0.5 * (K - 1) * np.log(n)

        sel_df = pd.DataFrame(eig_rows)
        sel_df.to_csv(d / "sbm_block_selection.csv", index=False, encoding="utf-8")
        files.append("sbm_block_selection.csv")

        # block assignment CSV (relabel blocks by descending size for readability)
        from collections import Counter

        sizes = Counter(labels.tolist())
        relabel = {old: new for new, (old, _) in enumerate(sizes.most_common())}
        lab2 = np.array([relabel[int(x)] for x in labels])
        # reorder B consistently with the relabelling (K is the fitted block count)
        Bre = np.full((K, K), np.nan)
        inv = {v: k for k, v in relabel.items()}
        for r in range(K):
            for s in range(K):
                Bre[r, s] = B[inv[r], inv[s]]
        nb = pd.DataFrame({"node": nodes, "block": lab2.tolist()}).sort_values(
            ["block", "node"]).reset_index(drop=True)
        nb.to_csv(d / "sbm_node_blocks.csv", index=False, encoding="utf-8")
        files.append("sbm_node_blocks.csv")

        bm = pd.DataFrame(np.round(Bre, 4),
                          index=[f"block{i}" for i in range(K)],
                          columns=[f"block{i}" for i in range(K)])
        bm.to_csv(d / "sbm_block_matrix.csv", encoding="utf-8")
        files.append("sbm_block_matrix.csv")

        # within- vs between-block connection probability → assortative vs not
        diag = np.array([Bre[i, i] for i in range(K)], dtype=float)
        offmask = ~np.eye(K, dtype=bool)
        off = Bre[offmask].astype(float)
        mean_within = float(np.nanmean(diag)) if K >= 1 else float("nan")
        mean_between = float(np.nanmean(off)) if off.size else float("nan")
        if K == 1:
            structure = "单块（与 Erdős–Rényi 随机图无异，未发现块结构）"
        elif np.isnan(mean_between) or mean_within >= mean_between:
            structure = "同配（块内连接强于块间 → 社区/模块结构）"
        else:
            structure = "异配（块间连接强于块内 → 二部/核心-边缘类结构）"

        # block-connection heatmap (ENGLISH labels)
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5.5, 4.8))
            im = ax.imshow(Bre, cmap="viridis", vmin=0, vmax=np.nanmax(Bre) if np.nanmax(Bre) > 0 else 1)
            ax.set_xticks(range(K)); ax.set_yticks(range(K))
            ax.set_xticklabels([f"b{i}" for i in range(K)])
            ax.set_yticklabels([f"b{i}" for i in range(K)])
            for r in range(K):
                for s in range(K):
                    if not np.isnan(Bre[r, s]):
                        ax.text(s, r, f"{Bre[r, s]:.2f}", ha="center", va="center",
                                color="white" if Bre[r, s] < np.nanmax(Bre) * 0.6 else "black", fontsize=8)
            ax.set_title(f"SBM block-connection matrix B (K={K}, spectral-selected)")
            fig.colorbar(im, ax=ax, fraction=0.046, label="edge probability")
            fig.tight_layout()
            fig.savefig(d / "sbm_block_matrix.png", dpi=150)
            plt.close(fig)
            files.append("sbm_block_matrix.png")
        except Exception:
            pass

        estimates["n_nodes"] = float(n)
        estimates["n_edges"] = float(UG.number_of_edges())
        estimates["n_blocks"] = float(K)
        estimates["loglik"] = round(ll, 3)
        estimates["icl"] = round(float(icl), 3)
        estimates["mean_within_block_prob"] = round(mean_within, 4) if mean_within == mean_within else float("nan")
        estimates["mean_between_block_prob"] = round(mean_between, 4) if mean_between == mean_between else float("nan")
        estimates["assortativity"] = (round(mean_within - mean_between, 4)
                                      if (mean_within == mean_within and mean_between == mean_between)
                                      else float("nan"))
        estimates["largest_block_size"] = float(max(Counter(lab2.tolist()).values()))

        bsizes = ", ".join(str(int(v)) for _, v in Counter(lab2.tolist()).most_common())
        ksrc = f"配置强制 K={K}" if forced_k else f"谱估计 K={K}（谱方法估计 {kspec}）"
        (d / "sbm_summary.txt").write_text(
            f"随机块模型（谱估计块数 + 谱嵌入 ASE→KMeans 分配 + 块连接概率 MLE）：边 {source}→{target}\n"
            f"节点 {n}，边 {UG.number_of_edges()}\n"
            f"块数 {ksrc}（块规模：{bsizes}）\n"
            f"块内平均连边概率={round(mean_within, 4)}，块间平均={round(mean_between, 4)} → {structure}\n"
            f"对数似然={round(ll, 3)}，ICL（诊断）={round(float(icl), 3)}\n"
            "块连接概率矩阵 B：\n" + bm.to_string() + "\n"
            f"块数选择证据（邻接谱：|特征值| 超出随机矩阵体边界 {round(bulk_thr, 3)}=2√平均度 即为信息维）：\n"
            + sel_df.to_string(index=False) + "\n"
            "注：块数由**邻接谱**选择——信息性特征值会超出随机图的半圆体边界 2√⟨k⟩（Lei-Rinaldo/"
            "Le-Levina），对块内度异质性稳健；**刻意不用似然/ICL 选块数**，因为非度修正 SBM 的"
            "似然会把单个稠密块过度切分（ICL 仅作诊断报告）。块分配用谱嵌入(ASE)+KMeans 硬分配"
            "（**非完整变分 EM**，后者给软分配、通常更准但更重）；KMeans 已固定 seed；"
            "二值无向 SBM（未拟合度修正/加权变体）；块是模型潜结构、非外部验证分组。\n\n"
            + "节点→块（前 30）：\n" + nb.head(30).to_string(index=False),
            encoding="utf-8",
        )
        files.append("sbm_summary.txt")

        summary.append(
            f"{entry.method} 完成（谱选块数 + 谱嵌入分配）：边 {source}→{target}；{n} 节点、"
            f"{UG.number_of_edges()} 边；{ksrc}，共 {K} 个块（规模 {bsizes}）；"
            f"块内平均连边概率 {round(mean_within, 4)} vs 块间 {round(mean_between, 4)} → {structure}；"
            f"对数似然 {round(ll, 3)}、ICL（诊断）{round(float(icl), 3)}。"
            "⚠ 块数由邻接谱选（信息特征值超出 2√⟨k⟩ 随机体边界，对度异质性稳健；"
            "刻意不用 ICL 选块数因非度修正 SBM 会过度切分）；分配用谱嵌入(ASE)+KMeans 硬分配"
            "（非完整变分 EM，seed 固定）；二值无向 SBM；块为模型潜结构（非验证分组）。"
        )
        code += [
            "import networkx as nx, numpy as np; from sklearn.cluster import KMeans",
            f"A = nx.to_numpy_array(nx.Graph(nx.from_pandas_edgelist(df, {source!r}, {target!r})))",
            "w = np.linalg.eigvalsh(A); K = int((abs(w) > 2*np.sqrt(A.sum()/len(A))).sum())  # spectral block count",
            "V = np.linalg.eigh(A)[1]; X = V[:, np.argsort(-abs(w))[:K]] * np.sqrt(abs(np.sort(-abs(w))[:K]))",
            "labels = KMeans(K, random_state=0).fit_predict(X)  # ASE -> hard block assignment; B[r,s]=edges/pairs",
        ]
    except Exception as err:
        summary.append(f"随机块模型失败：{err}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. ergm — exponential random graph model (R statnet/ergm) + CUG-test degrade
# ─────────────────────────────────────────────────────────────────────────────
# Curated allowlist of safe ergm term *names* (config terms are validated against this
# to keep the R formula injection-safe; node identifiers never enter the formula — they
# are written to a temp CSV with fixed column names src/tgt).
_ERGM_TERMS = {
    "edges", "mutual", "triangle", "triangles", "gwesp", "gwdegree", "gwdsp", "gwnsp",
    "kstar", "istar", "ostar", "twopath", "density", "isolates", "balance",
    "transitiveties", "cyclicalties", "degree", "idegree", "odegree",
}
_ERGM_TERM_RE = re.compile(r"^[a-zA-Z]+(\([0-9.,=\sA-Za-z]*\))?$")


def _ergm_terms_ok(terms: str) -> bool:
    """Validate a user-supplied ergm RHS like 'edges + gwesp(0.25, fixed=TRUE)' against
    the curated term allowlist + a strict token charset (no quotes/semicolons/backticks
    → no R injection through the formula)."""
    parts = [p.strip() for p in str(terms).split("+") if p.strip()]
    if not parts:
        return False
    for p in parts:
        if not _ERGM_TERM_RE.match(p):
            return False
        base = p.split("(", 1)[0]
        if base not in _ERGM_TERMS:
            return False
    return True


def _ergm_via_r(csv_path, directed, terms):
    """Fit an ERGM via R's statnet/ergm (MCMC-MLE). Returns (coef DataFrame[term,
    estimate,std_err,p_value], diag dict). Raises on failure so the caller degrades."""
    import pandas as pd

    from researchforge.executor import rbridge

    csv_r = str(csv_path).replace("\\", "/")
    dir_r = "TRUE" if directed else "FALSE"
    rcode = (
        "suppressMessages(library(ergm))\n"
        f'el <- read.csv("{csv_r}", colClasses="character")\n'
        f'net <- network(as.matrix(el[, c("src","tgt")]), matrix.type="edgelist", directed={dir_r})\n'
        f"m <- ergm(net ~ {terms}, control=control.ergm(seed=1, MCMC.samplesize=4096))\n"
        "s <- summary(m); ct <- s$coefficients\n"
        'pcol <- ncol(ct)\n'  # last column is Pr(>|z|) in ergm's coefficient table
        'cat("##COEF\\n")\n'
        'for (nm in rownames(ct)) cat(sprintf("%s|%.6f|%.6f|%.6g\\n", nm, ct[nm,1], ct[nm,2], ct[nm,pcol]))\n'
        'cat("##DIAG\\n")\n'
        'cat(sprintf("aic|%.6f\\nbic|%.6f\\n", AIC(m), BIC(m)))\n'
    )
    out = rbridge.run_r(rcode, timeout=300)
    section, crows, diag = None, [], {}
    for line in out.splitlines():
        s = line.strip()
        if s == "##COEF":
            section = "C"
        elif s == "##DIAG":
            section = "D"
        elif "|" in s and section == "C":
            crows.append(s.rsplit("|", 3))
        elif "|" in s and section == "D":
            k, v = s.split("|", 1)
            try:
                diag[k] = float(v)
            except ValueError:
                pass
    if not crows:
        raise RuntimeError("ergm 未返回系数")
    coef = pd.DataFrame(crows, columns=["term", "estimate", "std_err", "p_value"])
    for c in ("estimate", "std_err", "p_value"):
        coef[c] = pd.to_numeric(coef[c], errors="coerce")
    return coef, diag


def _cug_transitivity_test(UG, n_sim, seed):
    """Pure-Python conditional-uniform-graph (CUG) test: compare the observed global
    transitivity to its null distribution under random graphs with the SAME node count
    and edge count (G(n,m)). Returns (observed, null_mean, null_sd, z, p_one_sided)."""
    import networkx as nx
    import numpy as np

    n = UG.number_of_nodes()
    m = UG.number_of_edges()
    obs = nx.transitivity(UG)
    rng = np.random.RandomState(seed)
    null = np.empty(n_sim)
    for i in range(n_sim):
        Gr = nx.gnm_random_graph(n, m, seed=int(rng.randint(0, 2**31 - 1)))
        null[i] = nx.transitivity(Gr)
    mu, sd = float(null.mean()), float(null.std(ddof=1))
    z = (obs - mu) / sd if sd > 0 else float("nan")
    p = float((np.sum(null >= obs) + 1) / (n_sim + 1))  # one-sided (clustering above random)
    return float(obs), mu, sd, z, p


@register("ergm")
def _branch_ergm(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    from researchforge.executor import rbridge

    source, target, weight, directed, problem = _resolve_edges(ctx)
    if problem is not None:
        summary.append(problem)
        return
    try:
        import networkx as nx

        G, UG = _build_graph(df, source, target, weight, directed)
        UG = nx.Graph(UG)
        UG.remove_edges_from(nx.selfloop_edges(UG))
        n = UG.number_of_nodes()
        m = UG.number_of_edges()

        terms = str(cfg.get("terms", "edges + gwesp(0.25, fixed=TRUE)"))
        if not _ergm_terms_ok(terms):
            terms = "edges + gwesp(0.25, fixed=TRUE)"

        # ── Full ERGM via R statnet/ergm (MCMC-MLE) when available ──────────────
        if rbridge.r_available() and rbridge.r_package_available("ergm"):
            import pandas as pd

            sub = df[[source, target]].dropna()
            sub = sub[sub[source] != sub[target]]
            sub.columns = ["src", "tgt"]
            csv = d / "_ergm_input.csv"
            sub.to_csv(csv, index=False)
            try:
                coef, diag = _ergm_via_r(csv, directed, terms)
                coef.to_csv(d / "ergm_coefficients.csv", index=False, encoding="utf-8")
                files.append("ergm_coefficients.csv")

                for _, r in coef.iterrows():
                    estimates[f"coef_{r['term']}"] = round(float(r["estimate"]), 4)
                    estimates[f"p_{r['term']}"] = round(float(r["p_value"]), 4)
                estimates["n_nodes"] = float(n)
                estimates["n_edges"] = float(m)
                if "aic" in diag:
                    estimates["aic"] = round(diag["aic"], 3)
                if "bic" in diag:
                    estimates["bic"] = round(diag["bic"], 3)

                lines = [f"{r['term']}: 估计={round(float(r['estimate']), 4)}（SE={round(float(r['std_err']), 4)}, "
                         f"p={round(float(r['p_value']), 4)}）" for _, r in coef.iterrows()]
                (d / "ergm_summary.txt").write_text(
                    f"指数随机图模型 ERGM（R statnet/ergm, MCMC-MLE）：边 {source}→{target}"
                    f"（{'有向' if directed else '无向'}）\n"
                    f"节点 {n}，边 {m}\n公式：net ~ {terms}\n"
                    + "\n".join(lines) + "\n"
                    + (f"AIC={round(diag.get('aic', float('nan')), 3)}, BIC={round(diag.get('bic', float('nan')), 3)}\n"
                       if "aic" in diag else "")
                    + "注：系数为对数几率尺度——edges≈基线密度（类似截距）；gwesp>0=超出随机的"
                    "三元闭合/传递性（聚集）；正的结构项表示该构型比随机更常见。ERGM 为 MCMC-MLE，"
                    "需检查退化(degeneracy)与 MCMC 收敛；这里用固定 seed 与默认控制参数。\n",
                    encoding="utf-8",
                )
                files.append("ergm_summary.txt")

                edges_row = coef[coef["term"] == "edges"]
                gw = coef[coef["term"].str.startswith("gwesp")]
                msg = f"{entry.method} 完成（R ergm, MCMC-MLE）：边 {source}→{target}；{n} 节点、{m} 边；公式 net ~ {terms}。"
                if not edges_row.empty:
                    msg += f" edges（基线密度）log-odds={round(float(edges_row['estimate'].iloc[0]), 3)}；"
                if not gw.empty:
                    gv = float(gw["estimate"].iloc[0]); gp = float(gw["p_value"].iloc[0])
                    msg += (f"gwesp（传递性/三元闭合）={round(gv, 3)}（p={round(gp, 3)}，"
                            f"{'显著高于随机的聚集' if (gv > 0 and gp < 0.05) else '不显著'}）；")
                if "aic" in diag:
                    msg += f"AIC={round(diag['aic'], 1)}。"
                msg += " ⚠ ERGM 为 MCMC-MLE，需检查模型退化与 MCMC 收敛（固定 seed/默认控制）；系数为对数几率。"
                summary.append(msg)
                code += [
                    "library(ergm)  # R statnet",
                    'net <- network(el, matrix.type="edgelist", directed=FALSE)',
                    f"m <- ergm(net ~ {terms})  # MCMC-MLE; edges=baseline density, gwesp=transitivity",
                    "summary(m)  # coefficients on the log-odds scale",
                ]
                return
            finally:
                try:
                    csv.unlink()
                except OSError:
                    pass

        # ── Degrade: pure-Python CUG test on transitivity ──────────────────────
        import numpy as np
        import pandas as pd

        if n < 5 or m < 3:
            summary.append("ERGM 失败：网络过小，无法做 ERGM 或 CUG 检验。")
            return
        n_sim = int(cfg.get("n_sim", 300))
        seed = int(cfg.get("seed", 0))
        obs, mu, sd, z, p = _cug_transitivity_test(UG, n_sim, seed)
        density = nx.density(UG)

        cug = pd.DataFrame([{"statistic": "transitivity", "observed": round(obs, 5),
                             "null_mean": round(mu, 5), "null_sd": round(sd, 5),
                             "z": round(z, 4) if z == z else float("nan"),
                             "p_one_sided": round(p, 5)}])
        cug.to_csv(d / "ergm_cug_test.csv", index=False, encoding="utf-8")
        files.append("ergm_cug_test.csv")

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            rng = np.random.RandomState(seed)
            null = np.array([nx.transitivity(nx.gnm_random_graph(n, m, seed=int(rng.randint(0, 2**31 - 1))))
                             for _ in range(n_sim)])
            fig, ax = plt.subplots(figsize=(7, 4.2))
            ax.hist(null, bins=30, color="#B0B0B0", alpha=0.85, label="null G(n,m)")
            ax.axvline(obs, color="#C44E52", lw=2, label=f"observed ({obs:.3f})")
            ax.set_xlabel("global transitivity")
            ax.set_ylabel("count")
            ax.set_title(f"CUG test: transitivity vs random graphs (n={n}, m={m}, {n_sim} sims)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(d / "ergm_cug_test.png", dpi=150)
            plt.close(fig)
            files.append("ergm_cug_test.png")
        except Exception:
            pass

        estimates["n_nodes"] = float(n)
        estimates["n_edges"] = float(m)
        estimates["density"] = round(float(density), 5)
        estimates["transitivity_observed"] = round(obs, 5)
        estimates["transitivity_null_mean"] = round(mu, 5)
        estimates["transitivity_z"] = round(z, 4) if z == z else float("nan")
        estimates["transitivity_p"] = round(p, 5)

        verdict = ("聚集显著高于随机（存在三元闭合/传递性）" if (p < 0.05 and obs > mu)
                   else "聚集与随机图无显著差异")
        (d / "ergm_summary.txt").write_text(
            f"指数随机图模型 ERGM —— 降级为 CUG 条件均匀图检验（未检测到 R 的 ergm 包）：边 {source}→{target}\n"
            f"节点 {n}，边 {m}，密度={round(float(density), 5)}\n"
            f"观测全局传递性={round(obs, 5)}；同节点同边数随机图 G(n,m) 的零分布："
            f"均值={round(mu, 5)}、SD={round(sd, 5)}（{n_sim} 次模拟）\n"
            f"z={round(z, 4) if z == z else 'NaN'}，单侧 p={round(p, 5)} —— {verdict}\n"
            "注：完整 ERGM（系数/SE/退化诊断）需要 R 的 statnet/ergm（install.packages('ergm')）；"
            "这里给出纯 Python 的 CUG 检验作为替代——它检验「观测聚集是否超出仅由规模与密度决定的随机水平」，"
            "正是 ergm 的 gwesp 项所刻画的传递性效应的单变量版本（不控制其他结构项、无系数尺度）。\n",
            encoding="utf-8",
        )
        files.append("ergm_summary.txt")

        summary.append(
            f"{entry.method}（降级为 CUG 检验，未检测到 R 的 ergm 包）：边 {source}→{target}；"
            f"{n} 节点、{m} 边、密度 {round(float(density), 4)}；观测传递性 {round(obs, 4)} vs "
            f"随机零分布均值 {round(mu, 4)}（z={round(z, 3) if z == z else 'NaN'}, p={round(p, 4)}）—— {verdict}。"
            "⚠ 完整 ERGM（系数/退化诊断）需 R 的 statnet/ergm（install.packages('ergm')）；"
            "CUG 是单变量替代（仅检验聚集是否超随机，不控制其他结构项、无系数）。"
        )
        code += [
            "import networkx as nx, numpy as np",
            "# Full ERGM needs R statnet/ergm; pure-Python degrade = CUG test:",
            "obs = nx.transitivity(G)  # vs null distribution of G(n,m) random graphs",
            "# p = P(null transitivity >= observed); tests clustering beyond random density",
        ]
    except Exception as err:
        summary.append(f"ERGM/CUG 失败：{err}")
