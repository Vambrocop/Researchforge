"""stochastic_block_model — spectral block-count + ASE->KMeans SBM + MLE block matrix."""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.branches.network_science._common import _build_graph, _resolve_edges


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
