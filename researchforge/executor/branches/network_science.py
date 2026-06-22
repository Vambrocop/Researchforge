"""Branch handlers for the network_science family.

Three network/graph-science methods that build a graph from an EDGE LIST (two
node-identifier columns + optional weight, like the existing ml._branch_network_analysis):

  * community_detection — Louvain modules + modularity Q (python-louvain `community`)
  * centrality_suite    — 5 centralities per node + their Spearman agreement
  * epidemic_model      — network-based stochastic SIR/SIS diffusion simulation

Each handler resolves the edge list, degrades honestly (no networkx / too few nodes /
no edge list -> skip with a Chinese ⚠ message), writes CSV + PNG (matplotlib Agg,
ENGLISH plot labels), fills float `estimates`, appends a Chinese `summary` with ⚠
disclosures, and mutates ctx (never rebinds). See executor/_branch_api.py and CLAUDE.md.

networkx + python-louvain (imported as `community`) are installed.
"""

from __future__ import annotations

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
