"""Causal family branch handler: dag_structure_learning — the PC algorithm.

Constraint-based causal-discovery (Spirtes-Glymour PC) that learns a CPDAG from
continuous/Gaussian observational data using pure numpy/scipy:

  1. Skeleton: start complete-undirected; for conditioning-set size m=0,1,2,...
     remove edge i-j if i ⟂ j | S for some S ⊆ adj(i)∖{j} of size m; record sepset.
  2. CI test (Gaussian): partial correlation ρ(i,j|S) from the inverse sub-covariance
     (precision) matrix; Fisher z = 0.5·ln((1+ρ)/(1-ρ)); stat = sqrt(n-|S|-3)·|z| ~ N(0,1);
     independent iff p > alpha.
  3. Orient v-structures: unshielded triple i-k-j (i,j non-adjacent) ⇒ i→k←j iff
     k NOT in sepset(i,j) (collider).
  4. Meek rules R1-R3 applied to a fixpoint to propagate orientations without making
     new colliders or cycles (R4 is omitted — it only fires with background-knowledge
     edges; R1-R3 already complete the CPDAG from the v-structures, and omitting R4 stays
     SOUND: it can only leave an edge undirected, never orient one the wrong way).
     Remaining undirected edges = unresolved Markov-equivalence.

Honest skip when <3 continuous variables or <50 rows. Output is a CPDAG under STRONG
assumptions (causal sufficiency, faithfulness, linearity+Gaussianity) — disclosed in the
Chinese summary. matplotlib Agg / ENGLISH labels; estimates are plain floats.

See executor/_branch_api.py and CLAUDE.md. networkx is used for the plot if importable,
else a matrix heatmap fallback.
"""
from __future__ import annotations

import math

from researchforge.executor._branch_api import Ctx, register


# ─────────────────────────────────────────────────────────────────────────────
# Gaussian conditional-independence test (partial correlation + Fisher z).
# ─────────────────────────────────────────────────────────────────────────────
def _partial_corr(corr, i: int, j: int, cond: tuple[int, ...]):
    """Partial correlation ρ(i,j | cond) from the correlation matrix.

    Uses the precision (inverse) of the sub-correlation matrix over {i, j, *cond}:
    ρ(i,j|cond) = -P[i,j] / sqrt(P[i,i]·P[j,j]). Falls back to the marginal Pearson
    correlation when cond is empty. Returns a float in (-1, 1) (clipped for stability).
    """
    import numpy as np

    if not cond:
        r = float(corr[i, j])
    else:
        idx = [i, j, *cond]
        sub = corr[np.ix_(idx, idx)]
        try:
            prec = np.linalg.inv(sub)
        except np.linalg.LinAlgError:
            prec = np.linalg.pinv(sub)
        denom = math.sqrt(abs(prec[0, 0] * prec[1, 1]))
        r = 0.0 if denom == 0 else float(-prec[0, 1] / denom)
    # numerical guard so the Fisher transform stays finite
    return max(min(r, 0.999999), -0.999999)


def _ci_pvalue(corr, n: int, i: int, j: int, cond: tuple[int, ...]) -> float:
    """Two-sided p-value of H0: ρ(i,j|cond)=0 via the Fisher z-transform.

    z = 0.5·ln((1+ρ)/(1-ρ)); statistic = sqrt(n - |cond| - 3)·|z| ~ N(0,1).
    df guard: if n - |cond| - 3 <= 0 the test is undecidable → treat as DEPENDENT
    (p=0) so the (under-powered) edge is conservatively kept.
    """
    from scipy import stats

    r = _partial_corr(corr, i, j, cond)
    dof = n - len(cond) - 3
    if dof <= 0:
        return 0.0
    z = 0.5 * math.log((1.0 + r) / (1.0 - r))
    stat = math.sqrt(dof) * abs(z)
    return float(2.0 * (1.0 - stats.norm.cdf(stat)))


# ─────────────────────────────────────────────────────────────────────────────
# PC skeleton (stable adjacency search with separating sets).
# ─────────────────────────────────────────────────────────────────────────────
def _pc_skeleton(corr, n: int, p: int, alpha: float):
    """Return (adj, sepset) where adj is a symmetric set-of-neighbours dict and
    sepset[(i,j)]=sepset[(j,i)] is the conditioning set that d-separated i,j.

    PC-stable: snapshot the adjacency at the start of each size-m pass so removals
    within a pass do not affect which conditioning sets are tested (order-independent).
    """
    from itertools import combinations

    adj = {i: set(range(p)) - {i} for i in range(p)}
    sepset: dict = {}
    m = 0
    while True:
        # PC-stable: freeze neighbourhoods for this whole conditioning-set size
        frozen = {i: set(adj[i]) for i in range(p)}
        any_eligible = False
        for i in range(p):
            for j in list(adj[i]):
                if j <= i:  # test each unordered pair once
                    continue
                # standard PC tests S ⊆ adj(i)\{j} AND adj(j)\{i}: condition on the
                # UNION of both frozen neighbourhoods (still order-independent / PC-stable),
                # otherwise a pair separable only via the higher-degree endpoint keeps a
                # spurious edge on denser graphs.
                others = list((frozen[i] | frozen[j]) - {i, j})
                if len(others) < m:
                    continue
                any_eligible = True
                for cond in combinations(others, m):
                    if i not in adj[j]:  # edge already removed this pass
                        break
                    p_val = _ci_pvalue(corr, n, i, j, cond)
                    if p_val > alpha:  # conditionally independent → remove edge
                        adj[i].discard(j)
                        adj[j].discard(i)
                        sepset[(i, j)] = set(cond)
                        sepset[(j, i)] = set(cond)
                        break
        if not any_eligible:
            break
        m += 1
    return adj, sepset


# ─────────────────────────────────────────────────────────────────────────────
# Orientation: v-structures + Meek rules → a CPDAG.
# directed[(a,b)] == True means a → b is oriented. An undirected edge i-j is
# represented by directed[(i,j)] == directed[(j,i)] == False while i,j adjacent.
# ─────────────────────────────────────────────────────────────────────────────
def _orient(adj, sepset, p: int):
    directed: dict = {}
    edges = set()
    for i in range(p):
        for j in adj[i]:
            edges.add((i, j))  # both directions present = undirected for now
            directed[(i, j)] = False

    def adjacent(a, b):
        return (a, b) in edges or (b, a) in edges

    def is_directed(a, b):
        """a → b is oriented (and the reverse is not)."""
        return directed.get((a, b), False) and not directed.get((b, a), False)

    def is_undirected(a, b):
        return adjacent(a, b) and not directed.get((a, b), False) and not directed.get((b, a), False)

    def orient(a, b):
        directed[(a, b)] = True
        directed[(b, a)] = False

    # 1) v-structures (colliders): unshielded triple i-k-j, i,j non-adjacent;
    #    collider iff k NOT in sepset(i,j) → i→k←j.
    for k in range(p):
        nbrs = list(adj[k])
        for a in range(len(nbrs)):
            for b in range(a + 1, len(nbrs)):
                i, j = nbrs[a], nbrs[b]
                if adjacent(i, j):
                    continue  # shielded → not a v-structure
                s = sepset.get((i, j))
                if s is not None and k not in s:
                    orient(i, k)
                    orient(j, k)

    # 2) Meek rules R1-R3 to a fixpoint (R4 omitted; see note below).
    changed = True
    while changed:
        changed = False
        for i in range(p):
            for j in range(p):
                if i == j or not is_undirected(i, j):
                    continue
                # R1: k→i, i-j, k,j non-adjacent ⇒ i→j (avoid a new collider at i)
                for k in range(p):
                    if k in (i, j):
                        continue
                    if is_directed(k, i) and not adjacent(k, j):
                        orient(i, j)
                        changed = True
                        break
                if not is_undirected(i, j):
                    continue
                # R2: i→k→j and i-j ⇒ i→j (acyclicity)
                for k in range(p):
                    if k in (i, j):
                        continue
                    if is_directed(i, k) and is_directed(k, j):
                        orient(i, j)
                        changed = True
                        break
                if not is_undirected(i, j):
                    continue
                # R3: i-k→j, i-l→j, k,l non-adjacent, i-k & i-l ⇒ i→j
                commons = [
                    k for k in range(p)
                    if k not in (i, j) and is_undirected(i, k) and is_directed(k, j)
                ]
                done_r3 = False
                for a in range(len(commons)):
                    for b in range(a + 1, len(commons)):
                        if not adjacent(commons[a], commons[b]):
                            orient(i, j)
                            changed = True
                            done_r3 = True
                            break
                    if done_r3:
                        break
                # R4 (Meek's fourth rule) is only needed when there are
                # background-knowledge orientations to propagate; in a pure
                # observational PC run R1-R3 already complete the CPDAG from the
                # v-structures, so it is intentionally omitted (omitting it keeps
                # the orientation SOUND — it can only leave an edge undirected,
                # never orient one wrongly). See module docstring.
    return directed, edges


@register("dag_structure_learning")
def _branch_dag_structure_learning(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    import numpy as np
    import pandas as pd

    alpha = float(cfg.get("alpha", 0.05))
    _excl = {fp.unit_col, fp.time_col}
    # accept continuous AND id-kind columns (the "id" trap: all-distinct numeric
    # columns profile as id but are perfectly valid Gaussian variables here).
    auto_cont = [
        c.name for c in fp.columns
        if c.kind in {"continuous", "id"} and c.name not in _excl
    ]
    requested = [c for c in (cfg.get("variables") or []) if c in df.columns]
    variables = requested or auto_cont
    # coerce to numeric, keep only columns that survive coercion, cap at 10
    numeric = []
    for c in variables:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= 1:
            numeric.append(c)
    variables = numeric[:10]

    if len(variables) < 3:
        summary.append(
            "因果发现（PC 算法）跳过：需要 ≥3 个连续/数值变量（用 "
            "config={\"variables\":[...]} 指定）。当前可用变量不足 3 个。"
        )
        return

    sub = df[variables].apply(pd.to_numeric, errors="coerce").dropna()
    n = int(len(sub))
    if n < 50:
        summary.append(
            f"因果发现（PC 算法）跳过：去缺失后仅 {n} 行，需 ≥50 行才能稳健做条件独立检验。"
        )
        return

    try:
        p = len(variables)
        corr = np.corrcoef(sub.values, rowvar=False)
        if not np.all(np.isfinite(corr)):  # a constant column → undefined correlation
            summary.append(
                "因果发现（PC 算法）失败：存在常数列或不可计算的相关（请移除零方差变量）。"
            )
            return

        adj, sepset = _pc_skeleton(corr, n, p, alpha)
        directed, edges = _orient(adj, sepset, p)

        # ── build the edge list (directed vs undirected), avoiding double-counting ──
        seen: set = set()
        rows = []
        adj_mat = np.zeros((p, p), dtype=int)
        n_directed = 0
        n_undirected = 0
        # strength = |marginal correlation| for reporting the strongest links
        for (i, j) in list(edges):
            if (i, j) in seen or (j, i) in seen:
                continue
            seen.add((i, j))
            di = directed.get((i, j), False)
            dj = directed.get((j, i), False)
            strength = abs(float(corr[i, j]))
            if di and not dj:
                rows.append((variables[i], variables[j], "directed", round(strength, 4)))
                adj_mat[i, j] = 1
                n_directed += 1
            elif dj and not di:
                rows.append((variables[j], variables[i], "directed", round(strength, 4)))
                adj_mat[j, i] = 1
                n_directed += 1
            else:  # undirected (unresolved equivalence)
                rows.append((variables[i], variables[j], "undirected", round(strength, 4)))
                adj_mat[i, j] = 1
                adj_mat[j, i] = 1
                n_undirected += 1

        n_edges = n_directed + n_undirected
        edf = pd.DataFrame(rows, columns=["source", "target", "type", "abs_corr"])
        edf = edf.sort_values("abs_corr", ascending=False).reset_index(drop=True)
        edf.to_csv(d / "dag_edges.csv", index=False, encoding="utf-8")
        files.append("dag_edges.csv")

        amat = pd.DataFrame(adj_mat, index=variables, columns=variables)
        amat.to_csv(d / "dag_adjacency.csv", encoding="utf-8")
        files.append("dag_adjacency.csv")

        estimates["n_variables"] = float(p)
        estimates["n"] = float(n)
        estimates["alpha"] = round(alpha, 4)
        estimates["n_edges"] = float(n_edges)
        estimates["n_directed_edges"] = float(n_directed)
        estimates["n_undirected_edges"] = float(n_undirected)

        # ── plot: networkx spring layout if available, else adjacency heatmap ──
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            try:
                import networkx as nx

                G = nx.DiGraph()
                G.add_nodes_from(variables)
                undirected_pairs = []
                for src, tgt, typ, _w in rows:
                    if typ == "directed":
                        G.add_edge(src, tgt)
                    else:
                        undirected_pairs.append((src, tgt))
                        G.add_edge(src, tgt)  # placeholder for layout only
                pos = nx.spring_layout(G, seed=0)
                fig, ax = plt.subplots(figsize=(6.5, 5.5))
                nx.draw_networkx_nodes(G, pos, node_color="#4C72B0", node_size=900, ax=ax)
                nx.draw_networkx_labels(G, pos, font_size=9, font_color="white", ax=ax)
                directed_edges = [(s, t) for (s, t, ty, _w) in rows if ty == "directed"]
                nx.draw_networkx_edges(
                    G, pos, edgelist=directed_edges, ax=ax,
                    arrows=True, arrowstyle="-|>", arrowsize=18, width=1.6,
                    edge_color="#C44E52", connectionstyle="arc3,rad=0.05",
                )
                nx.draw_networkx_edges(
                    G, pos, edgelist=undirected_pairs, ax=ax,
                    arrows=False, width=1.6, style="dashed", edge_color="grey",
                )
                ax.set_title(
                    f"PC CPDAG: {n_directed} directed (red), "
                    f"{n_undirected} undirected (grey dashed)"
                )
                ax.axis("off")
                fig.tight_layout()
                fig.savefig(d / "dag_graph.png", dpi=150)
                plt.close(fig)
                files.append("dag_graph.png")
            except Exception:
                # fallback: adjacency-matrix heatmap (no networkx)
                fig, ax = plt.subplots(figsize=(5.5, 5))
                im = ax.imshow(adj_mat, cmap="Blues", vmin=0, vmax=1)
                ax.set_xticks(range(p))
                ax.set_yticks(range(p))
                ax.set_xticklabels(variables, rotation=45, ha="right", fontsize=8)
                ax.set_yticklabels(variables, fontsize=8)
                ax.set_xlabel("target")
                ax.set_ylabel("source")
                ax.set_title("PC CPDAG adjacency (1 = edge source->target)")
                fig.colorbar(im, ax=ax, fraction=0.046)
                fig.tight_layout()
                fig.savefig(d / "dag_adjacency.png", dpi=150)
                plt.close(fig)
                files.append("dag_adjacency.png")
        except Exception:
            pass

        # ── headline: strongest directed edges ──
        strong_dir = [r for r in edf.itertuples() if r.type == "directed"][:3]
        if strong_dir:
            dir_txt = "；".join(f"{r.source}→{r.target}" for r in strong_dir)
            head = f"强方向边（按相关排序）：{dir_txt}"
        else:
            head = "未识别出有向边（全部为无向，方向无法仅从观测数据确定）"

        summary.append(
            f"{entry.method} 完成（PC 算法，纯 Python）：在 {p} 个变量、{n} 行上学得 CPDAG，"
            f"共 {n_edges} 条边（{n_directed} 有向、{n_undirected} 无向），α={alpha:g}。{head}。"
            "⚠ PC 假设极强：①因果充分性（无未观测混杂）②忠实性（独立=d-分离）③本检验为"
            "高斯偏相关，要求**线性+正态**。输出是 CPDAG = 一个马尔可夫等价类——无向边表示"
            "方向无法仅凭观测数据辨识，需要实验/时间/领域知识；所有边都是**统计关联，非因果保证**；"
            f"结果对 α（当前 {alpha:g}）和假设违背敏感，建议做敏感性分析。"
        )
        code += [
            "# PC algorithm (Spirtes-Glymour), pure numpy/scipy:",
            "import numpy as np; from scipy import stats",
            f"corr = np.corrcoef(df[{variables!r}].dropna().values, rowvar=False)",
            "# skeleton: remove i-j if partial-corr CI test (Fisher z) p>alpha for some S",
            "# z=0.5*ln((1+r)/(1-r)); stat=sqrt(n-|S|-3)*|z| ~ N(0,1)",
            "# orient v-structures (collider iff k NOT in sepset(i,j)) + Meek rules R1-R3",
            f"# alpha={alpha}",
        ]
    except Exception as err:
        summary.append(f"因果发现（PC 算法）失败：{err}")
