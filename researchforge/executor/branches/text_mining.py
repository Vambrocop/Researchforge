"""Branch handlers for the TEXT MINING family — free-text corpus analysis.

Four methods over a free-text column (auto-detected, or config ``text``):

- ``lda_topic_model``   — unsupervised topic discovery (sklearn CountVectorizer with
                          english stop-words + LatentDirichletAllocation). Reports the
                          top words per topic, each document's dominant topic, and the
                          topic sizes; ``n_topics`` via config (default 5). This is the
                          INFERENCE-bearing method (cold-reviewed): LDA is a generative
                          probabilistic bag-of-words model, the topics are a model, not
                          ground truth, and they are sensitive to preprocessing / n_topics.
- ``tfidf_keywords``    — salient-term extraction (sklearn TfidfVectorizer). Reports the
                          top TF-IDF terms overall, and per-group when a low-cardinality
                          grouping column exists (config ``group``).
- ``word_frequency``    — plain term-frequency counts (Counter over the tokenized corpus):
                          the top words by raw frequency with each word's document
                          frequency, a bar chart, and an OPTIONAL word cloud (best-effort).
                          Complements TF-IDF (raw prevalence vs. distinctiveness).
- ``sentiment_analysis``— per-document polarity via an OPTIONAL backend, mirroring the
                          engine's R / optional-lib pattern: try vaderSentiment, else
                          textblob, else nltk's VADER. If NONE is installed it degrades
                          HONESTLY (跳过 + pip hint) rather than shipping a crude hand
                          lexicon and calling it sentiment.

Text-column detection (``_find_text_col``): a string/object column whose non-null
values average ~3+ whitespace tokens (or mean length ~20+ chars) AND is reasonably
high-cardinality. Config ``text`` overrides. If no text column is found every handler
degrades honestly.

Chinese / CJK support: the vectorizer / counting methods detect a CJK corpus (config
``lang`` forces ``zh`` / ``en``) and tokenize it with jieba when the OPTIONAL ``jieba``
package is installed, else an honest character-bigram fallback (⚠ disclosed — words are
adjacent 2-char pairs, cruder than word segmentation). English corpora keep the
byte-identical sklearn english path, so existing behaviour is unchanged. jieba is never
fetched at runtime (offline red-line).

Each handler unpacks ctx into the same local names run_analysis uses and MUTATES
summary/estimates/files/code (never rebinds). See executor/_branch_api.py. This
family file is auto-registered by branches/__init__.py (pkgutil.walk_packages).

RNG note: the LDA fit uses a fixed random_state=0 (disclosed) so topics are
reproducible run-to-run; the ordering / numbering of topics is still arbitrary.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


# ---------------------------------------------------------------------------
# Shared column-role helpers (local to this family).
# ---------------------------------------------------------------------------

# kinds a free-text column can profile as. A free-text column is non-numeric;
# all-unique short strings profile as ``id``, repeated strings as ``categorical``,
# a 2-level string as ``binary``. We accept all of these and then apply the
# token/length heuristic to decide whether the column is genuinely free text.
_TEXT_KINDS = {"categorical", "id", "binary", "unknown"}


def _text_score(s) -> tuple[float, float, int]:
    """(mean whitespace-token count, mean char length, n distinct) for a series.

    Computed on the non-null string-cast values. Used by ``_find_text_col`` to
    rank candidate columns and decide whether a column is genuinely free text.
    """
    nn = s.dropna().astype(str)
    if len(nn) == 0:
        return (0.0, 0.0, 0)
    toks = nn.str.split().map(len)
    mean_tokens = float(toks.mean())
    mean_len = float(nn.str.len().mean())
    n_distinct = int(nn.nunique())
    return (mean_tokens, mean_len, n_distinct)


def _find_text_col(fp, df, cfg) -> str | None:
    """The free-text column: config ``text`` overrides; else the best string column
    whose values average ~3+ whitespace tokens (or ~20+ chars) and that is
    reasonably high-cardinality. Returns None if no column qualifies.

    Numeric / datetime / boolean columns can never be the text column. Among the
    qualifying string columns we pick the one with the most tokens on average (the
    "most free-text-like"); ties broken by mean length then cardinality.
    """
    forced = cfg.get("text")
    if forced is not None and forced in df.columns:
        return forced

    import pandas as pd

    excl = {fp.unit_col, fp.time_col}
    cands: list[tuple[float, float, int, str]] = []
    for c in fp.columns:
        if c.name in excl:
            continue
        if c.kind not in _TEXT_KINDS:
            continue
        s = df[c.name]
        # numeric / datetime / boolean dtypes are never free text (profiler may have
        # mis-bucketed, e.g. all-unique ints as id — re-guard on the dtype here).
        if (
            pd.api.types.is_numeric_dtype(s)
            or pd.api.types.is_bool_dtype(s)
            or pd.api.types.is_datetime64_any_dtype(s)
        ):
            continue
        mean_tokens, mean_len, n_distinct = _text_score(s)
        n_nonnull = int(s.dropna().shape[0])
        if n_nonnull == 0:
            continue
        # free-text gate: multi-word OR long, and not a tiny handful of repeated labels.
        is_texty = (mean_tokens >= 3.0) or (mean_len >= 20.0)
        # cardinality: at least a few distinct values and >= ~30% distinct of non-null
        # (a 2-level categorical with long sentences is unlikely a free-text field).
        high_card = n_distinct >= 4 and (n_distinct / max(1, n_nonnull)) >= 0.3
        if is_texty and high_card:
            cands.append((mean_tokens, mean_len, n_distinct, c.name))
    if not cands:
        return None
    cands.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
    return cands[0][3]


def _group_col(fp, df, used: set[str], max_levels: int = 12) -> str | None:
    """A low-cardinality categorical/binary column usable for per-group breakdowns.

    Excludes the unit/time columns and the text column; lowest-cardinality first.
    """
    excl = {fp.unit_col, fp.time_col} | used
    cands = [
        c.name
        for c in fp.columns
        if c.kind in {"categorical", "binary"}
        and c.name not in excl
        and 2 <= int(df[c.name].nunique()) <= max_levels
    ]
    cands.sort(key=lambda name: int(df[name].nunique()))
    return cands[0] if cands else None


def _docs(df, text_col) -> "tuple[list[str], list[int]]":
    """Non-empty document strings + their original row indices (positional)."""
    docs: list[str] = []
    idx: list[int] = []
    series = df[text_col].astype(str)
    for pos, val in enumerate(series.tolist()):
        v = val.strip()
        if val is None or v == "" or v.lower() == "nan":
            continue
        docs.append(v)
        idx.append(pos)
    return docs, idx


# ---------------------------------------------------------------------------
# Multilingual tokenization (Chinese via an OPTIONAL jieba bridge + degrade).
# ---------------------------------------------------------------------------
# The vectorizer methods default to English tokenization (sklearn stop_words=
# "english", a latin word_pattern). That leaves the vocabulary EMPTY on Chinese
# text (no whitespace tokens, no latin letters) — so 中文 / policy corpora used to
# silently produce nothing. We DETECT a CJK corpus and switch to a Chinese analyzer:
# jieba word segmentation when the OPTIONAL jieba package is installed, else a
# character-bigram fallback (a legitimate, if cruder, Chinese IR tokenization) with
# an honest ⚠. English corpora keep the byte-identical sklearn path (existing
# behaviour / tests unchanged); only CJK corpora take the new branch. jieba is never
# fetched at runtime (offline red-line).

import re as _re

_HAN_RE = _re.compile(r"[一-鿿㐀-䶿]")

# A basic Chinese function-word stop list (extensible). Single characters are dropped
# wholesale by the analyzer, so this targets 2-char function words that would otherwise
# dominate a policy corpus. Not exhaustive — content words (政策/发展/建设…) are kept.
_ZH_STOPWORDS = frozenset({
    "我们", "你们", "他们", "她们", "它们", "自己", "这个", "那个", "这些", "那些",
    "这样", "那样", "这里", "那里", "什么", "怎么", "为什么", "因为", "所以", "但是",
    "而且", "如果", "虽然", "于是", "然后", "已经", "现在", "可以", "没有", "通过",
    "进行", "以及", "或者", "并且", "等等", "之一", "方面", "各种", "一个", "一些",
    "一样", "一直", "一定", "不过", "只是", "还是", "就是", "也是", "都是", "不是",
    "对于", "关于", "根据", "按照", "由于", "从而", "因此", "同时", "此外", "其中",
    "其他", "以上", "以下", "目前", "以来", "起来", "出来", "上来", "下来", "作为",
    "成为", "使得", "能够", "应该", "需要", "必须", "可能", "如何", "这种", "那种",
})


def _cjk_share(docs) -> float:
    """Fraction of non-space characters across the corpus that are CJK ideographs."""
    n_han = 0
    n_char = 0
    for doc in docs:
        for ch in str(doc):
            if ch.isspace():
                continue
            n_char += 1
            if _HAN_RE.match(ch):
                n_han += 1
    return (n_han / n_char) if n_char else 0.0


def _corpus_lang(docs, cfg) -> str:
    """'zh' or 'en' for the corpus. config ``lang`` overrides ('zh'/'en'); else CJK
    when >= 20% of non-space characters are Han ideographs (mixed-but-Chinese policy
    text is common — a modest Han share still means jieba should segment)."""
    forced = str(cfg.get("lang") or "").strip().lower()
    if forced in {"zh", "cn", "chinese", "中文"}:
        return "zh"
    if forced in {"en", "english", "英文", "英语"}:
        return "en"
    return "zh" if _cjk_share(docs) >= 0.20 else "en"


def _jieba_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("jieba") is not None


def _make_zh_analyzer(use_jieba: bool):
    """A callable(text) -> list[str] for sklearn's ``analyzer=`` (and standalone
    counting). jieba word segmentation when available, else overlapping character
    bigrams over Han runs (+ whole latin/number runs). Drops stop-words,
    pure-punctuation tokens, and single characters."""

    def _keep(tok: str) -> bool:
        tok = tok.strip()
        if len(tok) < 2 or tok in _ZH_STOPWORDS:
            return False
        # must contain a CJK ideograph or be alphanumeric (drops 。，、；：""（） etc.)
        return bool(_HAN_RE.search(tok)) or tok.isalnum()

    if use_jieba:
        import jieba

        def analyze(text):
            return [t for t in (w.strip() for w in jieba.lcut(str(text))) if _keep(t)]
    else:

        def analyze(text):
            s = str(text)
            toks: list[str] = []
            for run in _re.findall(r"[一-鿿㐀-䶿]+", s):
                if len(run) >= 2:  # overlapping char bigrams; lone chars are noise
                    toks.extend(run[i:i + 2] for i in range(len(run) - 1))
            toks.extend(_re.findall(r"[a-zA-Z]{2,}|\d+", s))  # keep latin/number runs whole
            return [t for t in toks if _keep(t)]

    return analyze


def _make_text_vectorizer(cls, docs, cfg, *, min_df, max_features=None):
    """Construct a CountVectorizer / TfidfVectorizer appropriate to the corpus
    language. Returns (vectorizer, lang, use_jieba). CJK corpora get a jieba /
    char-bigram analyzer; everything else keeps the byte-identical english path."""
    lang = _corpus_lang(docs, cfg)
    if lang == "zh":
        use_jieba = _jieba_available()
        vect = cls(
            analyzer=_make_zh_analyzer(use_jieba),
            min_df=min_df,
            max_features=max_features,
        )
        return vect, lang, use_jieba
    vect = cls(
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",  # words of >=2 letters
        min_df=min_df,
        max_features=max_features,
    )
    return vect, lang, True


def _tok_note(lang: str, use_jieba: bool) -> str:
    """Human-readable disclosure of which tokenizer was used (for summaries)."""
    if lang != "zh":
        return "英文分词（sklearn，english 停用词）"
    if use_jieba:
        return "中文分词（jieba 词切分）"
    return "中文分词（字符二元组降级——未装 jieba，词≈相邻两字，较粗；装 jieba 更准）"


def _count_terms(docs, lang, use_jieba):
    """(term_freq Counter, doc_freq Counter) over the corpus with the language-
    appropriate tokenizer. English: latin words of >=2 letters minus sklearn english
    stop-words; Chinese: the jieba / char-bigram analyzer."""
    from collections import Counter

    tf: Counter = Counter()
    dfq: Counter = Counter()
    if lang == "zh":
        analyze = _make_zh_analyzer(use_jieba)
    else:
        try:
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as _EN_STOP
        except Exception:
            _EN_STOP = frozenset()

        def analyze(text):
            return [
                w
                for w in _re.findall(r"[a-zA-Z][a-zA-Z]+", str(text).lower())
                if w not in _EN_STOP
            ]

    for doc in docs:
        toks = analyze(doc)
        tf.update(toks)
        dfq.update(set(toks))
    return tf, dfq


# ===========================================================================
# 1. LDA topic model  (INFERENCE-bearing — cold-reviewed)
# ===========================================================================

@register("lda_topic_model")
def _branch_lda_topic_model(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import importlib.util

    if importlib.util.find_spec("sklearn") is None:
        summary.append("LDA 主题模型跳过：需要 scikit-learn（未检测到）。安装：pip install scikit-learn。")
        return

    text_col = _find_text_col(fp, df, cfg)
    if text_col is None:
        summary.append(
            "LDA 主题模型跳过：未找到自由文本列（需要一个多词/长字符串的高基数文本列）。"
            "可用 config text 指定文本列。"
        )
        return

    docs, doc_idx = _docs(df, text_col)
    if len(docs) < 5:
        summary.append(f"LDA 主题模型跳过：有效文档太少（n={len(docs)} < 5）。")
        return

    # n_topics: config else default 5, clamped to [2, n_docs-1].
    try:
        forced_k = cfg.get("n_topics")
        forced_k = int(forced_k) if forced_k is not None else None
    except (TypeError, ValueError):
        forced_k = None
    n_topics = forced_k if (forced_k is not None and forced_k >= 2) else 5
    n_topics = max(2, min(n_topics, len(docs) - 1))

    # min_df: config else default 2 (a term must appear in >=2 docs), clamped so it
    # never exceeds the corpus size (which would empty the vocabulary).
    try:
        forced_mindf = cfg.get("min_df")
        forced_mindf = int(forced_mindf) if forced_mindf is not None else None
    except (TypeError, ValueError):
        forced_mindf = None
    min_df = forced_mindf if (forced_mindf is not None and forced_mindf >= 1) else 2
    min_df = max(1, min(min_df, max(1, len(docs) - 1)))

    try:
        max_features = cfg.get("max_features")
        max_features = int(max_features) if max_features is not None else None
        if max_features is not None and max_features < 1:
            max_features = None
    except (TypeError, ValueError):
        max_features = None

    try:
        import numpy as np
        import pandas as pd
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.feature_extraction.text import CountVectorizer

        vect, lang, use_jieba = _make_text_vectorizer(
            CountVectorizer, docs, cfg, min_df=min_df, max_features=max_features
        )
        try:
            dtm = vect.fit_transform(docs)
        except ValueError:
            # min_df pruned the whole vocabulary — retry with min_df=1.
            vect, lang, use_jieba = _make_text_vectorizer(
                CountVectorizer, docs, cfg, min_df=1, max_features=max_features
            )
            min_df = 1
            try:
                dtm = vect.fit_transform(docs)
            except ValueError:
                summary.append(
                    "LDA 主题模型跳过：去除停用词后词表为空（文档可能太短或全是停用词）。"
                )
                return

        vocab = np.array(vect.get_feature_names_out())
        if len(vocab) < n_topics:
            # cannot have more topics than vocabulary words meaningfully.
            n_topics = max(2, min(n_topics, len(vocab)))
        if len(vocab) < 2:
            summary.append("LDA 主题模型跳过：有效词表少于 2 个词，无法建模主题。")
            return

        lda = LatentDirichletAllocation(
            n_components=n_topics,
            learning_method="batch",
            max_iter=20,
            random_state=0,
        )
        doc_topic = lda.fit_transform(dtm)  # (n_docs, n_topics)

        # --- top words per topic ----------------------------------------------
        top_n = 10
        comps = lda.components_  # (n_topics, vocab) unnormalized topic-word weights
        topic_top_words: list[list[str]] = []
        rows = []
        for t in range(n_topics):
            order = np.argsort(comps[t])[::-1][:top_n]
            words = [str(w) for w in vocab[order]]
            weights = [float(comps[t][i]) for i in order]
            topic_top_words.append(words)
            for rank, (w, wt) in enumerate(zip(words, weights), start=1):
                rows.append({"topic": t, "rank": rank, "word": w, "weight": round(wt, 6)})
        tw_df = pd.DataFrame(rows)
        tw_df.to_csv(d / "lda_top_words.csv", index=False, encoding="utf-8")
        files.append("lda_top_words.csv")

        # --- per-document dominant topic --------------------------------------
        dominant = np.argmax(doc_topic, axis=1)
        dom_conf = doc_topic[np.arange(len(docs)), dominant]
        doc_df = pd.DataFrame({
            "row": doc_idx,
            "dominant_topic": dominant.astype(int),
            "topic_prob": np.round(dom_conf, 6),
        })
        doc_df.to_csv(d / "lda_doc_topics.csv", index=False, encoding="utf-8")
        files.append("lda_doc_topics.csv")

        # --- topic sizes (count of docs whose dominant topic is t) ------------
        sizes = pd.Series(dominant).value_counts().reindex(range(n_topics), fill_value=0)
        size_df = pd.DataFrame({
            "topic": list(range(n_topics)),
            "n_docs_dominant": [int(sizes[t]) for t in range(n_topics)],
            "top_words": ["; ".join(topic_top_words[t][:5]) for t in range(n_topics)],
        })
        size_df.to_csv(d / "lda_topic_sizes.csv", index=False, encoding="utf-8")
        files.append("lda_topic_sizes.csv")

        # --- approximate perplexity (lower is better; descriptive only) -------
        try:
            perplexity = float(lda.perplexity(dtm))
        except Exception:
            perplexity = float("nan")

        # --- PNG: top-words-per-topic bar panels ------------------------------
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            ncol = min(3, n_topics)
            nrow = int(np.ceil(n_topics / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 0.42 * top_n * nrow + 0.6))
            axes = np.atleast_1d(axes).ravel()
            for t in range(n_topics):
                ax = axes[t]
                order = np.argsort(comps[t])[::-1][:top_n]
                words = [str(w) for w in vocab[order]][::-1]
                weights = [float(comps[t][i]) for i in order][::-1]
                ax.barh(range(len(words)), weights, color="#4C72B0")
                ax.set_yticks(range(len(words)))
                ax.set_yticklabels(words, fontsize=8)
                ax.set_title(f"Topic {t} (n={int(sizes[t])})", fontsize=9)
                ax.tick_params(axis="x", labelsize=7)
            for j in range(n_topics, len(axes)):
                axes[j].axis("off")
            fig.suptitle("LDA top words per topic", fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            fig.savefig(d / "lda_top_words.png", dpi=150)
            plt.close(fig)
            files.append("lda_top_words.png")
        except Exception:
            pass

        estimates["n_topics"] = float(n_topics)
        estimates["n_docs"] = float(len(docs))
        estimates["vocab_size"] = float(len(vocab))
        estimates["perplexity"] = round(perplexity, 4) if perplexity == perplexity else float("nan")
        estimates["mean_dominant_topic_prob"] = round(float(np.mean(dom_conf)), 6)
        estimates["min_df"] = float(min_df)

        perp_txt = f"{perplexity:.1f}" if perplexity == perplexity else "N/A"
        rule_txt = (
            f"config 指定 n_topics={n_topics}"
            if (forced_k is not None and forced_k >= 2)
            else f"默认 n_topics={n_topics}"
        )
        preview = " | ".join(
            f"主题{t}: " + ", ".join(topic_top_words[t][:5]) for t in range(min(n_topics, 4))
        )
        fallback_warn = (
            "⚠ 未检测到 jieba，中文按字符二元组降级分词（词≈相邻两字，较粗）——"
            "装 jieba 可得词级主题：pip install jieba。"
            if (lang == "zh" and not use_jieba) else ""
        )
        summary.append(
            f"{entry.method} 完成（LDA 主题模型，文本列={text_col}，{_tok_note(lang, use_jieba)}，"
            f"{len(docs)} 篇文档 × {len(vocab)} 词词表 → {n_topics} 个主题，{rule_txt}，"
            f"min_df={min_df}）：近似困惑度={perp_txt}（描述性，越低越好），各文档主导主题平均概率="
            f"{float(np.mean(dom_conf)):.2f}。主题预览：{preview}"
            f"（详见 lda_top_words.csv / lda_doc_topics.csv / lda_topic_sizes.csv）。{fallback_warn}"
            f"⚠ LDA 是一个生成式词袋(bag-of-words)概率模型——主题是模型的产物、不是客观真相，"
            f"对预处理（停用词/分词/min_df）与 n_topics 的选择高度敏感（换设置主题会变）；"
            f"主题的编号/顺序是任意的，词袋忽略词序与语境（否定、习语会失真）；"
            f"困惑度仅供参考、不等于可解释性；已固定 random_state=0 保证可复现，"
            f"但仍需人工命名/验证主题——可用 config text/n_topics/min_df/max_features 覆盖。"
        )
        vect_line = (
            "import jieba  # 中文语料：用 jieba 分词作 analyzer\n"
            f"vect = CountVectorizer(analyzer=lambda t: [w for w in jieba.lcut(t) if len(w) >= 2], min_df={min_df})"
            if lang == "zh"
            else f"vect = CountVectorizer(stop_words='english', min_df={min_df})"
        )
        code += [
            "from sklearn.feature_extraction.text import CountVectorizer",
            "from sklearn.decomposition import LatentDirichletAllocation",
            f"docs = df[{text_col!r}].dropna().astype(str).tolist()",
            vect_line,
            "dtm = vect.fit_transform(docs); vocab = vect.get_feature_names_out()",
            f"lda = LatentDirichletAllocation(n_components={n_topics}, random_state=0).fit(dtm)",
            "doc_topic = lda.transform(dtm)  # top words: argsort(lda.components_[t])[::-1]",
        ]
    except Exception as err:
        summary.append(f"LDA 主题模型失败：{err}")


# ===========================================================================
# 2. TF-IDF keywords
# ===========================================================================

@register("tfidf_keywords")
def _branch_tfidf_keywords(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import importlib.util

    if importlib.util.find_spec("sklearn") is None:
        summary.append("TF-IDF 关键词跳过：需要 scikit-learn（未检测到）。安装：pip install scikit-learn。")
        return

    text_col = _find_text_col(fp, df, cfg)
    if text_col is None:
        summary.append(
            "TF-IDF 关键词跳过：未找到自由文本列（需要一个多词/长字符串的高基数文本列）。"
            "可用 config text 指定文本列。"
        )
        return

    docs, doc_idx = _docs(df, text_col)
    if len(docs) < 2:
        summary.append(f"TF-IDF 关键词跳过：有效文档太少（n={len(docs)} < 2）。")
        return

    # min_df: config else 1 (TF-IDF tolerates rare terms; default keep all).
    try:
        forced_mindf = cfg.get("min_df")
        forced_mindf = int(forced_mindf) if forced_mindf is not None else None
    except (TypeError, ValueError):
        forced_mindf = None
    min_df = forced_mindf if (forced_mindf is not None and forced_mindf >= 1) else 1
    min_df = max(1, min(min_df, max(1, len(docs))))

    try:
        top_n = cfg.get("top_n")
        top_n = int(top_n) if top_n is not None else 20
        if top_n < 1:
            top_n = 20
    except (TypeError, ValueError):
        top_n = 20

    # optional grouping column for per-group keyword tables.
    group_col = cfg.get("group") if cfg.get("group") in df.columns else _group_col(
        fp, df, {text_col}
    )

    try:
        import numpy as np
        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer

        vect, lang, use_jieba = _make_text_vectorizer(
            TfidfVectorizer, docs, cfg, min_df=min_df
        )
        try:
            tfidf = vect.fit_transform(docs)
        except ValueError:
            vect, lang, use_jieba = _make_text_vectorizer(
                TfidfVectorizer, docs, cfg, min_df=1
            )
            min_df = 1
            try:
                tfidf = vect.fit_transform(docs)
            except ValueError:
                summary.append("TF-IDF 关键词跳过：去除停用词后词表为空。")
                return

        vocab = np.array(vect.get_feature_names_out())
        if len(vocab) == 0:
            summary.append("TF-IDF 关键词跳过：词表为空。")
            return

        # overall salience = mean TF-IDF of each term across documents.
        mean_tfidf = np.asarray(tfidf.mean(axis=0)).ravel()
        order = np.argsort(mean_tfidf)[::-1][:top_n]
        overall_df = pd.DataFrame({
            "rank": range(1, len(order) + 1),
            "term": [str(vocab[i]) for i in order],
            "mean_tfidf": [round(float(mean_tfidf[i]), 6) for i in order],
        })
        overall_df.to_csv(d / "tfidf_top_terms.csv", index=False, encoding="utf-8")
        files.append("tfidf_top_terms.csv")

        # --- per-group top terms (if a grouping column exists) ----------------
        group_rows = []
        groups_seen: set[str] = set()
        if group_col is not None:
            grp_vals = df[group_col].astype(str).tolist()
            grp_for_docs = [grp_vals[p] for p in doc_idx]
            tfidf_arr = tfidf.toarray()
            for g in sorted(set(grp_for_docs)):
                mask = np.array([gg == g for gg in grp_for_docs])
                if not mask.any():
                    continue
                groups_seen.add(g)
                gm = tfidf_arr[mask].mean(axis=0)
                gorder = np.argsort(gm)[::-1][:top_n]
                for rank, i in enumerate(gorder, start=1):
                    if gm[i] <= 0:
                        continue
                    group_rows.append({
                        "group": g,
                        "rank": rank,
                        "term": str(vocab[i]),
                        "mean_tfidf": round(float(gm[i]), 6),
                    })
            if group_rows:
                pd.DataFrame(group_rows).to_csv(
                    d / "tfidf_top_terms_by_group.csv", index=False, encoding="utf-8"
                )
                files.append("tfidf_top_terms_by_group.csv")

        # --- PNG: overall top-terms bar --------------------------------------
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            k = min(top_n, len(order))
            terms = [str(vocab[i]) for i in order][:k][::-1]
            vals = [float(mean_tfidf[i]) for i in order][:k][::-1]
            fig, ax = plt.subplots(figsize=(6.5, 0.34 * k + 1.0))
            ax.barh(range(k), vals, color="#55A868")
            ax.set_yticks(range(k))
            ax.set_yticklabels(terms, fontsize=8)
            ax.set_xlabel("mean TF-IDF")
            ax.set_title(f"Top {k} TF-IDF terms ({text_col})", fontsize=10)
            fig.tight_layout()
            fig.savefig(d / "tfidf_top_terms.png", dpi=150)
            plt.close(fig)
            files.append("tfidf_top_terms.png")
        except Exception:
            pass

        estimates["n_docs"] = float(len(docs))
        estimates["vocab_size"] = float(len(vocab))
        estimates["top_term_tfidf"] = round(float(mean_tfidf[order[0]]), 6)
        estimates["min_df"] = float(min_df)
        estimates["n_groups"] = float(len(groups_seen))

        top_terms_preview = ", ".join(str(vocab[i]) for i in order[:8])
        grp_txt = (
            f"，并按 {group_col} 分组输出各组 Top 词（tfidf_top_terms_by_group.csv）"
            if (group_col is not None and group_rows)
            else "，无低基数分组列可分组"
        )
        fallback_warn = (
            "⚠ 未检测到 jieba，中文按字符二元组降级分词（词≈相邻两字，较粗）——"
            "装 jieba 更准：pip install jieba。"
            if (lang == "zh" and not use_jieba) else ""
        )
        summary.append(
            f"{entry.method} 完成（TF-IDF 关键词，文本列={text_col}，{_tok_note(lang, use_jieba)}，"
            f"{len(docs)} 篇文档 × {len(vocab)} 词词表，min_df={min_df}）：整体 Top 词"
            f"（按平均 TF-IDF）= {top_terms_preview}{grp_txt}（详见 tfidf_top_terms.csv）。{fallback_warn}"
            f"⚠ TF-IDF 是词频统计、不是语义——高分词只反映「在本语料中相对独特且高频」，"
            f"依赖分词与停用词表，对语料构成与文档长度敏感；"
            f"它不理解词义/同义/语境，适合做关键词概览而非主题或情感判断——可用 config text/group/min_df/top_n/lang 覆盖。"
        )
        tfidf_vect_line = (
            "import jieba  # 中文语料：用 jieba 分词作 analyzer\n"
            f"vect = TfidfVectorizer(analyzer=lambda t: [w for w in jieba.lcut(t) if len(w) >= 2], min_df={min_df})"
            if lang == "zh"
            else f"vect = TfidfVectorizer(stop_words='english', min_df={min_df})"
        )
        code += [
            "from sklearn.feature_extraction.text import TfidfVectorizer",
            f"docs = df[{text_col!r}].dropna().astype(str).tolist()",
            tfidf_vect_line,
            "tfidf = vect.fit_transform(docs); vocab = vect.get_feature_names_out()",
            "import numpy as np; mean_tfidf = np.asarray(tfidf.mean(axis=0)).ravel()",
            "top = [vocab[i] for i in np.argsort(mean_tfidf)[::-1][:20]]; print(top)",
        ]
    except Exception as err:
        summary.append(f"TF-IDF 关键词失败：{err}")


# ===========================================================================
# 3. Sentiment analysis  (OPTIONAL backend + HONEST DEGRADE)
# ===========================================================================

def _sentiment_backend():
    """Resolve the best available sentiment backend, mirroring the engine's optional-
    library pattern. Returns (name, scorer) where ``scorer(text) -> dict`` with keys
    ``polarity`` (always) and ``subjectivity`` (textblob only, else NaN). Returns
    (None, None) if NO backend is installed — caller degrades honestly.

    Order: vaderSentiment (purpose-built, no data download) -> textblob (adds
    subjectivity) -> nltk's VADER (needs the vader_lexicon resource).
    """
    import importlib.util

    if importlib.util.find_spec("vaderSentiment") is not None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            an = SentimentIntensityAnalyzer()

            def score(text: str) -> dict:
                return {
                    "polarity": float(an.polarity_scores(text)["compound"]),
                    "subjectivity": float("nan"),
                }

            return "vaderSentiment", score
        except Exception:
            pass

    if importlib.util.find_spec("textblob") is not None:
        try:
            from textblob import TextBlob

            def score(text: str) -> dict:
                tb = TextBlob(text).sentiment
                return {
                    "polarity": float(tb.polarity),
                    "subjectivity": float(tb.subjectivity),
                }

            return "textblob", score
        except Exception:
            pass

    if importlib.util.find_spec("nltk") is not None:
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer  # type: ignore

            try:
                an = SentimentIntensityAnalyzer()
            except LookupError:
                # the vader_lexicon resource is not downloaded — try to fetch it once.
                import nltk

                nltk.download("vader_lexicon", quiet=True)
                an = SentimentIntensityAnalyzer()

            def score(text: str) -> dict:
                return {
                    "polarity": float(an.polarity_scores(text)["compound"]),
                    "subjectivity": float("nan"),
                }

            return "nltk-vader", score
        except Exception:
            pass

    return None, None


@register("sentiment_analysis")
def _branch_sentiment_analysis(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    text_col = _find_text_col(fp, df, cfg)
    if text_col is None:
        summary.append(
            "情感分析跳过：未找到自由文本列（需要一个多词/长字符串的高基数文本列）。"
            "可用 config text 指定文本列。"
        )
        return

    docs, doc_idx = _docs(df, text_col)
    if len(docs) < 2:
        summary.append(f"情感分析跳过：有效文档太少（n={len(docs)} < 2）。")
        return

    backend, scorer = _sentiment_backend()
    if backend is None:
        # HONEST DEGRADE: do NOT ship a crude hand lexicon and pretend it is sentiment.
        summary.append(
            "情感分析跳过：需要 vaderSentiment 或 textblob（未检测到情感后端）。"
            "安装：pip install vaderSentiment（或 pip install textblob），"
            "或 pip install nltk 后 python -m nltk.downloader vader_lexicon。"
            "⚠ 本引擎不提供手搓词典充当情感分数（那会误导）——请装一个经过验证的后端。"
        )
        return

    group_col = cfg.get("group") if cfg.get("group") in df.columns else _group_col(
        fp, df, {text_col}
    )

    try:
        import numpy as np
        import pandas as pd

        scored = [scorer(doc) for doc in docs]
        polarity = np.array([s["polarity"] for s in scored], dtype=float)
        subjectivity = np.array([s["subjectivity"] for s in scored], dtype=float)
        has_subj = bool(np.isfinite(subjectivity).any())

        # label by polarity (vader/textblob compound/polarity in [-1, 1]).
        def lab(p: float) -> str:
            if p > 0.05:
                return "positive"
            if p < -0.05:
                return "negative"
            return "neutral"

        labels = [lab(p) for p in polarity]

        doc_df = pd.DataFrame({
            "row": doc_idx,
            "polarity": np.round(polarity, 6),
            "sentiment": labels,
        })
        if has_subj:
            doc_df["subjectivity"] = np.round(subjectivity, 6)
        doc_df.to_csv(d / "sentiment_per_doc.csv", index=False, encoding="utf-8")
        files.append("sentiment_per_doc.csv")

        # --- distribution summary ---------------------------------------------
        n = len(polarity)
        n_pos = int(sum(1 for x in labels if x == "positive"))
        n_neg = int(sum(1 for x in labels if x == "negative"))
        n_neu = n - n_pos - n_neg
        dist_df = pd.DataFrame({
            "sentiment": ["positive", "neutral", "negative"],
            "count": [n_pos, n_neu, n_neg],
            "proportion": [round(n_pos / n, 4), round(n_neu / n, 4), round(n_neg / n, 4)],
        })
        dist_df.to_csv(d / "sentiment_distribution.csv", index=False, encoding="utf-8")
        files.append("sentiment_distribution.csv")

        # --- per-group means (if a grouping column exists) --------------------
        group_means = None
        if group_col is not None:
            grp_vals = df[group_col].astype(str).tolist()
            grp_for_docs = [grp_vals[p] for p in doc_idx]
            gm = (
                pd.DataFrame({"group": grp_for_docs, "polarity": polarity})
                .groupby("group")["polarity"]
                .agg(["mean", "count"])
                .reset_index()
                .rename(columns={"mean": "mean_polarity", "count": "n_docs"})
            )
            gm["mean_polarity"] = gm["mean_polarity"].round(6)
            gm.to_csv(d / "sentiment_by_group.csv", index=False, encoding="utf-8")
            files.append("sentiment_by_group.csv")
            group_means = gm

        # --- PNG: polarity histogram ------------------------------------------
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(polarity, bins=min(20, max(5, n // 3)), color="#C44E52", alpha=0.85)
            ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
            ax.set_xlabel("polarity (compound, -1..1)")
            ax.set_ylabel("documents")
            ax.set_title(f"Sentiment polarity distribution ({backend})")
            fig.tight_layout()
            fig.savefig(d / "sentiment_polarity_hist.png", dpi=150)
            plt.close(fig)
            files.append("sentiment_polarity_hist.png")
        except Exception:
            pass

        mean_pol = float(np.mean(polarity))
        estimates["mean_polarity"] = round(mean_pol, 6)
        estimates["std_polarity"] = round(float(np.std(polarity, ddof=0)), 6)
        estimates["pct_positive"] = round(n_pos / n, 6)
        estimates["pct_negative"] = round(n_neg / n, 6)
        estimates["n_docs"] = float(n)
        if has_subj:
            estimates["mean_subjectivity"] = round(float(np.nanmean(subjectivity)), 6)

        subj_txt = (
            f"，平均主观性={float(np.nanmean(subjectivity)):.2f}（0=客观,1=主观）"
            if has_subj else ""
        )
        grp_txt = ""
        if group_means is not None and len(group_means):
            pairs = "; ".join(
                f"{r['group']}={r['mean_polarity']:.2f}"
                for _, r in group_means.iterrows()
            )
            grp_txt = f"，按 {group_col} 的均值极性：{pairs}（sentiment_by_group.csv）"
        summary.append(
            f"{entry.method} 完成（情感分析，后端={backend}，文本列={text_col}，{n} 篇文档）："
            f"平均极性={mean_pol:.3f}（>0 偏正、<0 偏负），"
            f"正面 {n_pos}/{n}（{n_pos/n:.0%}）、负面 {n_neg}/{n}（{n_neg/n:.0%}）、"
            f"中性 {n_neu}/{n}{subj_txt}{grp_txt}"
            f"（详见 sentiment_per_doc.csv / sentiment_distribution.csv）。"
            f"⚠ 情感分数依赖所用的词典/模型（此处后端={backend}），是模型的判断、不是客观真相；"
            f"对领域高度敏感（通用词典在专业/讽刺/否定/习语文本上常出错），阈值（±0.05）是约定俗成；"
            f"短文本与跨语言文本尤其不可靠——请把分数当近似信号，重要结论需人工抽样校验。"
            f"可用 config text/group 覆盖。"
        )
        code += [
            f"# backend = {backend!r} (optional: vaderSentiment / textblob / nltk-vader)",
            "from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer",
            "an = SentimentIntensityAnalyzer()",
            f"docs = df[{text_col!r}].dropna().astype(str).tolist()",
            "polarity = [an.polarity_scores(t)['compound'] for t in docs]",
            "import numpy as np; print('mean polarity', np.mean(polarity))",
        ]
    except Exception as err:
        summary.append(f"情感分析失败：{err}")


# ===========================================================================
# 4. Word frequency  (plain term counts + bar chart + OPTIONAL word cloud)
# ===========================================================================

def _try_word_cloud(freqs: dict, out_dir, lang: str) -> bool:
    """Best-effort word cloud from a {term: weight} dict. Requires the OPTIONAL
    ``wordcloud`` package; for CJK it also needs a CJK font (reuses the engine's
    font detection — no font → skip honestly rather than render tofu). Returns True
    iff word_cloud.png was written. Never raises."""
    try:
        import importlib.util

        if importlib.util.find_spec("wordcloud") is None or not freqs:
            return False
        from wordcloud import WordCloud

        font_path = None
        if lang == "zh":
            from matplotlib import font_manager as fm

            from researchforge.executor.run import _detect_cjk_font

            name = _detect_cjk_font()
            if name is None:
                return False  # a Chinese cloud without a CJK font is tofu — skip honestly
            font_path = fm.findfont(name)

        wc = WordCloud(
            width=800, height=500, background_color="white",
            font_path=font_path, max_words=200, prefer_horizontal=0.9,
        )
        wc.generate_from_frequencies({str(k): float(v) for k, v in freqs.items() if v > 0})
        wc.to_file(str(out_dir / "word_cloud.png"))
        return True
    except Exception:
        return False


@register("word_frequency")
def _branch_word_frequency(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    text_col = _find_text_col(fp, df, cfg)
    if text_col is None:
        summary.append(
            "词频统计跳过：未找到自由文本列（需要一个多词/长字符串的高基数文本列）。"
            "可用 config text 指定文本列。"
        )
        return

    docs, _doc_idx = _docs(df, text_col)
    if len(docs) < 2:
        summary.append(f"词频统计跳过：有效文档太少（n={len(docs)} < 2）。")
        return

    try:
        top_n = int(cfg.get("top_n")) if cfg.get("top_n") is not None else 25
        if top_n < 1:
            top_n = 25
    except (TypeError, ValueError):
        top_n = 25
    try:
        min_count = int(cfg.get("min_count")) if cfg.get("min_count") is not None else 1
        if min_count < 1:
            min_count = 1
    except (TypeError, ValueError):
        min_count = 1

    lang = _corpus_lang(docs, cfg)
    use_jieba = _jieba_available() if lang == "zh" else True

    try:
        import pandas as pd

        tf, dfq = _count_terms(docs, lang, use_jieba)
        if not tf:
            summary.append("词频统计跳过：分词后无有效词（文档可能太短、全是停用词或单字）。")
            return

        total_tokens = int(sum(tf.values()))
        vocab_size = len(tf)
        items = [(w, int(c)) for w, c in tf.items() if c >= min_count]
        items.sort(key=lambda kv: (-kv[1], kv[0]))  # count desc, term asc for stable ties
        if not items:
            summary.append(
                f"词频统计跳过：无词达到 min_count={min_count}（可调低 min_count）。"
            )
            return
        top_items = items[:top_n]

        freq_df = pd.DataFrame({
            "rank": range(1, len(top_items) + 1),
            "word": [w for w, _ in top_items],
            "count": [c for _, c in top_items],
            "doc_freq": [int(dfq[w]) for w, _ in top_items],
            "pct_of_tokens": [round(c / total_tokens, 6) for _, c in top_items],
        })
        freq_df.to_csv(d / "word_frequency.csv", index=False, encoding="utf-8")
        files.append("word_frequency.csv")

        # --- PNG: top-words bar chart -----------------------------------------
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            k = len(top_items)
            words = [w for w, _ in top_items][::-1]
            counts = [c for _, c in top_items][::-1]
            fig, ax = plt.subplots(figsize=(6.5, 0.34 * k + 1.0))
            ax.barh(range(k), counts, color="#4C72B0")
            ax.set_yticks(range(k))
            ax.set_yticklabels(words, fontsize=8)
            ax.set_xlabel("frequency")
            ax.set_title(f"Top {k} words by frequency ({text_col})", fontsize=10)
            fig.tight_layout()
            fig.savefig(d / "word_frequency.png", dpi=150)
            plt.close(fig)
            files.append("word_frequency.png")
        except Exception:
            pass

        # --- OPTIONAL word cloud (best-effort) --------------------------------
        made_cloud = _try_word_cloud(dict(items[:200]), d, lang)
        if made_cloud:
            files.append("word_cloud.png")

        top_word, top_count = top_items[0]
        estimates["n_docs"] = float(len(docs))
        estimates["vocab_size"] = float(vocab_size)
        estimates["total_tokens"] = float(total_tokens)
        estimates["top_word_count"] = float(top_count)

        preview = "、".join(f"{w}({c})" for w, c in top_items[:8])
        fallback_warn = (
            "⚠ 未检测到 jieba，中文按字符二元组降级分词（词≈相邻两字，较粗）——"
            "装 jieba 更准：pip install jieba。"
            if (lang == "zh" and not use_jieba) else ""
        )
        cloud_txt = "，并生成词云 word_cloud.png" if made_cloud else ""
        summary.append(
            f"{entry.method} 完成（词频统计，文本列={text_col}，{_tok_note(lang, use_jieba)}，"
            f"{len(docs)} 篇文档，词表 {vocab_size} 词、共 {total_tokens} 词次，"
            f"min_count={min_count}）：高频词 Top{len(top_items)}：{preview}"
            f"（详见 word_frequency.csv，含每词文档频次 doc_freq{cloud_txt}）。{fallback_warn}"
            f"⚠ 词频是最朴素的统计——高频≠重要（功能词/领域惯用语会占榜），结果依赖分词与停用词表；"
            f"它反映「出现多少」而非「区分度」（后者看 TF-IDF）或「语义」——可用 config text/top_n/min_count/lang 覆盖。"
        )
        toks_line = (
            "import jieba  # 中文：jieba 分词\n"
            "toks = [w for doc in docs for w in jieba.lcut(doc) if len(w) >= 2]"
            if lang == "zh"
            else "import re\n"
            "toks = [w for doc in docs for w in re.findall(r'[a-zA-Z]{2,}', doc.lower())]"
        )
        code += [
            "from collections import Counter",
            f"docs = df[{text_col!r}].dropna().astype(str).tolist()",
            toks_line,
            "print(Counter(toks).most_common(25))",
        ]
    except Exception as err:
        summary.append(f"词频统计失败：{err}")
