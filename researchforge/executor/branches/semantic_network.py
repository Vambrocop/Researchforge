"""Branch handler for the SEMANTIC NETWORK method (Wave P2) — text/nlp family.

`semantic_network` — a keyword co-occurrence network over a free-text column: tokenize
each document, keep the most frequent words as nodes, and connect two words with an edge
weighted by the number of documents they co-occur in. Reports degree / weighted-degree /
betweenness centrality and greedy-modularity communities, plus a spring-layout network
plot. This is a DESCRIPTIVE text-mining method (domain text, family nlp, goal explore).

Reuses the multilingual tokenizer in ``text_mining`` (jieba bridge + character-bigram
fallback), so it is Chinese-capable for free; the Chinese fallback caveat is disclosed
when jieba is absent.

⚠ Co-occurrence is a bag-of-words ASSOCIATION, not a semantic relation: it ignores word
order, negation and context, and the graph is sensitive to tokenization and the
``top_words`` / ``min_cooccur`` thresholds. Betweenness uses 1/weight as edge distance
(a stronger tie = a shorter distance); the layout seed is fixed for reproducibility.

networkx is OPTIONAL — absent → honest degrade. Auto-registered by branches/__init__.py.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.branches.text_mining import (
    _corpus_lang,
    _docs,
    _find_text_col,
    _jieba_available,
    _make_zh_analyzer,
)


def _doc_analyzer(docs, cfg):
    """(analyze(text)->list[str], lang, use_jieba) — the language-appropriate tokenizer,
    mirroring word_frequency: jieba/char-bigram for Chinese, english-stopword latin words
    otherwise."""
    lang = _corpus_lang(docs, cfg)
    if lang == "zh":
        use_jieba = _jieba_available()
        return _make_zh_analyzer(use_jieba), lang, use_jieba
    import re

    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as _EN_STOP
    except Exception:
        _EN_STOP = frozenset()

    def analyze(text):
        return [w for w in re.findall(r"[a-zA-Z]{2,}", str(text).lower()) if w not in _EN_STOP]

    return analyze, lang, True


@register("semantic_network")
def _branch_semantic_network(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    import importlib.util

    if importlib.util.find_spec("networkx") is None:
        summary.append("语义网络跳过：需要 networkx（未检测到）。安装：pip install networkx。")
        return

    text_col = _find_text_col(fp, df, cfg)
    if text_col is None:
        summary.append(
            "语义网络跳过：未找到自由文本列（需要一个多词/长字符串的高基数文本列）。"
            "可用 config text 指定文本列。"
        )
        return

    docs, _doc_idx = _docs(df, text_col)
    if len(docs) < 3:
        summary.append(f"语义网络跳过：有效文档太少（n={len(docs)} < 3）。")
        return

    def _cfg_int(key, default, lo=1):
        try:
            v = cfg.get(key)
            v = int(v) if v is not None else default
            return v if v >= lo else default
        except (TypeError, ValueError):
            return default

    top_words = _cfg_int("top_words", 30, lo=2)
    min_cooccur = _cfg_int("min_cooccur", 2, lo=1)
    min_count = _cfg_int("min_count", 2, lo=1)

    try:
        from collections import Counter
        from itertools import combinations

        import networkx as nx
        import pandas as pd

        analyze, lang, use_jieba = _doc_analyzer(docs, cfg)
        per_doc_tokens = [analyze(doc) for doc in docs]
        tf: Counter = Counter()
        for toks in per_doc_tokens:
            tf.update(toks)
        vocab = [w for w, c in tf.most_common() if c >= min_count][:top_words]
        if len(vocab) < 2:
            summary.append(
                f"语义网络跳过：满足 min_count={min_count} 的词少于 2 个（可调低 min_count）。"
            )
            return
        vocabset = set(vocab)

        # co-occurrence: per document, count each unordered vocab-word pair once.
        pair_docs: Counter = Counter()
        for toks in per_doc_tokens:
            present = sorted(set(toks) & vocabset)
            for a, b in combinations(present, 2):
                pair_docs[(a, b)] += 1
        edges = [(a, b, w) for (a, b), w in pair_docs.items() if w >= min_cooccur]
        if not edges:
            summary.append(
                f"语义网络跳过：没有一对词在 ≥{min_cooccur} 篇文档中共现（可调低 min_cooccur/min_count）。"
            )
            return

        G = nx.Graph()
        for w in vocab:
            G.add_node(w, count=int(tf[w]))
        for a, b, w in edges:
            G.add_edge(a, b, weight=int(w), distance=1.0 / float(w))

        deg_cent = nx.degree_centrality(G)
        wdeg = dict(G.degree(weight="weight"))
        try:
            btw = nx.betweenness_centrality(G, weight="distance")
        except Exception:
            btw = {n: 0.0 for n in G}
        try:
            comms = list(nx.community.greedy_modularity_communities(G, weight="weight"))
        except Exception:
            comms = [set(G.nodes())]
        node_comm = {n: i for i, cset in enumerate(comms) for n in cset}

        # --- nodes.csv ---------------------------------------------------------
        node_rows = [{
            "word": n,
            "count": int(tf[n]),
            "degree": round(float(deg_cent.get(n, 0.0)), 6),
            "weighted_degree": int(wdeg.get(n, 0)),
            "betweenness": round(float(btw.get(n, 0.0)), 6),
            "community": int(node_comm.get(n, 0)),
        } for n in G.nodes()]
        nodes_df = pd.DataFrame(node_rows).sort_values(
            ["community", "degree"], ascending=[True, False]
        )
        nodes_df.to_csv(d / "semantic_network_nodes.csv", index=False, encoding="utf-8")
        files.append("semantic_network_nodes.csv")

        # --- edges.csv ---------------------------------------------------------
        edges_df = pd.DataFrame(
            sorted(((a, b, w) for a, b, w in edges), key=lambda t: -t[2]),
            columns=["word1", "word2", "cooccur_count"],
        )
        edges_df.to_csv(d / "semantic_network_edges.csv", index=False, encoding="utf-8")
        files.append("semantic_network_edges.csv")

        # --- PNG: spring-layout network ---------------------------------------
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            pos = nx.spring_layout(G, seed=0, weight="weight")
            degrees = dict(G.degree())
            sizes = [120 + 380 * (degrees[n] / max(1, max(degrees.values()))) for n in G.nodes()]
            ncolors = [node_comm.get(n, 0) for n in G.nodes()]
            weights = [G[a][b]["weight"] for a, b in G.edges()]
            wmax = max(weights) if weights else 1
            widths = [0.5 + 3.0 * (w / wmax) for w in weights]

            fig, ax = plt.subplots(figsize=(8, 6.5))
            nx.draw_networkx_edges(G, pos, width=widths, edge_color="0.7", ax=ax)
            nx.draw_networkx_nodes(
                G, pos, node_size=sizes, node_color=ncolors, cmap="tab10", ax=ax
            )
            # label only the top ~15 nodes by degree to avoid clutter
            top_label = sorted(G.nodes(), key=lambda n: degrees[n], reverse=True)[:15]
            nx.draw_networkx_labels(
                G, pos, labels={n: n for n in top_label}, font_size=8, ax=ax
            )
            ax.set_title(f"Semantic co-occurrence network ({text_col})", fontsize=11)
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(d / "semantic_network.png", dpi=150)
            plt.close(fig)
            files.append("semantic_network.png")
        except Exception:
            pass

        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        estimates["n_docs"] = float(len(docs))
        estimates["vocab_size"] = float(n_nodes)
        estimates["n_edges"] = float(n_edges)
        estimates["n_communities"] = float(len(comms))
        estimates["density"] = round(float(nx.density(G)), 6)
        estimates["max_betweenness"] = round(float(max(btw.values()) if btw else 0.0), 6)

        top_deg = sorted(G.nodes(), key=lambda n: deg_cent.get(n, 0.0), reverse=True)[:8]
        fallback_warn = (
            "⚠ 未检测到 jieba，中文按字符二元组降级分词（词≈相邻两字，较粗）——"
            "装 jieba 更准：pip install jieba。"
            if (lang == "zh" and not use_jieba) else ""
        )
        summary.append(
            f"{entry.method} 完成（语义网络，文本列={text_col}，"
            f"{'中文' if lang == 'zh' else '英文'}分词，{len(docs)} 篇文档）："
            f"{n_nodes} 个词节点、{n_edges} 条共现边、{len(comms)} 个社区（模块度聚类），"
            f"网络密度={nx.density(G):.3f}；核心词（按度中心性）：{', '.join(top_deg)}"
            f"（详见 semantic_network_nodes.csv / semantic_network_edges.csv / semantic_network.png）。"
            f"{fallback_warn}"
            f"⚠ 共现是词袋(bag-of-words)关联、不是语义关系——它忽略词序/否定/语境，"
            f"对分词与 top_words/min_cooccur/min_count 阈值敏感（换阈值网络会变）；"
            f"介数中心性用 1/权重 作边距离（强共现=近），布局种子固定=0 保可复现——"
            f"可用 config text/lang/top_words/min_cooccur/min_count 覆盖。"
        )
        code += [
            "import networkx as nx",
            "from itertools import combinations; from collections import Counter",
            f"docs = df[{text_col!r}].dropna().astype(str).tolist()",
            "# tokenize each doc, keep top words, edge = #docs two words co-occur in",
            "G = nx.Graph()  # nodes=words, weighted by co-occurrence count",
            "print(nx.community.greedy_modularity_communities(G, weight='weight'))",
        ]
    except Exception as err:
        summary.append(f"语义网络失败：{err}")
