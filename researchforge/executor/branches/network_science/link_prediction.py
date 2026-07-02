"""link_prediction — neighbourhood-similarity scores + held-out-edge AUC + top-K."""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.branches.network_science._common import _build_graph, _resolve_edges


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
