"""Branch handler for CONTENT ANALYSIS (Wave P5) — text/nlp family.

`content_analysis` — quantitative content analysis with TWO independent capabilities,
whichever the inputs support:

  A. Dictionary-based coding (config ``dictionary`` = {category: [keyword, ...]}): count
     keyword hits per category per document over a free-text column, and report category
     totals, each document's dominant category, and category co-occurrence.
  B. Inter-coder agreement (config ``coder1`` / ``coder2`` = two coder-code columns):
     Cohen's κ between two human coders, with a Landis & Koch strength band.

CRITICAL honesty boundary: the engine does POST-CODING QUANTIFICATION, not automatic
qualitative coding. It never invents categories and never judges meaning — the coding
scheme (dictionary) and the human codes come from the researcher (the same boundary as
three-level grounded-theory coding / NVivo / MaxQDA). With neither a dictionary nor coder
columns, it degrades honestly and says what it needs; it never fabricates categories.

Chinese keyword matching is by raw-substring containment (segmentation-independent, so it
does not depend on jieba); English matching is token-based (a keyword is a whole word),
with multi-word phrases matched as substrings. Deterministic — verified empirically.
Auto-registered by branches/__init__.py.
"""

from __future__ import annotations

import re

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.branches.text_mining import _corpus_lang, _docs, _find_text_col

# Landis & Koch (1977) strength-of-agreement bands for Cohen's κ (a convention, disclosed).
_KAPPA_BANDS: list[tuple[float, str]] = [
    (0.81, "almost perfect 几乎完全一致"),
    (0.61, "substantial 高度一致"),
    (0.41, "moderate 中等一致"),
    (0.21, "fair 一般一致"),
    (0.0, "slight 轻微一致"),
]


def _kappa_band(k: float) -> str:
    if k < 0:
        return "poor 差于随机"
    for thr, label in _KAPPA_BANDS:
        if k >= thr:
            return label
    return "slight 轻微一致"


def _count_kw(raw_lower: str, toks: list[str] | None, kw: str, lang: str) -> int:
    """Occurrences of one keyword in one document. Chinese: raw-substring (no
    segmentation needed). English: whole-token match, or substring for a multi-word
    phrase."""
    k = kw.strip()
    if not k:
        return 0
    if lang == "zh":
        return raw_lower.count(k.lower())
    kl = k.lower()
    if " " in kl:
        return raw_lower.count(kl)
    return (toks or []).count(kl)


def _run_dictionary(ctx: Ctx, dictionary: dict) -> bool:
    """Dictionary-based content analysis. Returns True if it ran (produced output)."""
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    # validate the dictionary shape: {category: [keyword, ...]}
    clean: dict[str, list[str]] = {}
    for cat, kws in dictionary.items():
        if isinstance(kws, (list, tuple)):
            words = [str(w) for w in kws if str(w).strip()]
        elif isinstance(kws, str):
            words = [kws]
        else:
            continue
        if words:
            clean[str(cat)] = words
    if not clean:
        summary.append(
            "内容分析（词典）跳过：config dictionary 需要 {类目: [关键词,...]} 形式的非空映射。"
        )
        return False

    text_col = _find_text_col(fp, df, cfg)
    if text_col is None:
        summary.append(
            "内容分析（词典）跳过：未找到自由文本列（config text 可指定）。"
        )
        return False

    docs, doc_idx = _docs(df, text_col)
    if len(docs) < 2:
        summary.append(f"内容分析（词典）跳过：有效文档太少（n={len(docs)} < 2）。")
        return False

    import pandas as pd

    lang = _corpus_lang(docs, cfg)
    raws = [str(x).lower() for x in docs]
    per_toks = [re.findall(r"[a-z']+", r) for r in raws] if lang != "zh" else [None] * len(raws)

    cats = list(clean.keys())
    counts = []  # n_docs x n_cats
    for r, raw in enumerate(raws):
        row = [
            sum(_count_kw(raw, per_toks[r], kw, lang) for kw in clean[cat])
            for cat in cats
        ]
        counts.append(row)
    import numpy as np

    cmat = np.array(counts, dtype=float)

    # --- category totals ---------------------------------------------------
    totals = cmat.sum(axis=0)
    docs_with = (cmat > 0).sum(axis=0)
    totals_df = pd.DataFrame({
        "category": cats,
        "total_hits": totals.astype(int),
        "n_docs_with_category": docs_with.astype(int),
        "pct_docs": np.round(docs_with / len(docs), 4),
    }).sort_values("total_hits", ascending=False)
    totals_df.to_csv(d / "content_category_totals.csv", index=False, encoding="utf-8")
    files.append("content_category_totals.csv")

    # --- per-doc dominant category ----------------------------------------
    dominant = []
    for r in range(len(docs)):
        row = cmat[r]
        dominant.append(cats[int(np.argmax(row))] if row.sum() > 0 else "(none)")
    doc_rows = {"row": doc_idx, "dominant_category": dominant}
    for j, cat in enumerate(cats):
        doc_rows[cat] = cmat[:, j].astype(int)
    pd.DataFrame(doc_rows).to_csv(
        d / "content_doc_categories.csv", index=False, encoding="utf-8"
    )
    files.append("content_doc_categories.csv")

    # --- category co-occurrence (docs where both categories are present) ---
    from itertools import combinations

    present = [set(cats[j] for j in range(len(cats)) if cmat[r, j] > 0) for r in range(len(docs))]
    cooc_rows = []
    for a, b in combinations(cats, 2):
        n_both = sum(1 for s in present if a in s and b in s)
        if n_both > 0:
            cooc_rows.append({"category1": a, "category2": b, "n_docs_both": n_both})
    if cooc_rows:
        pd.DataFrame(cooc_rows).sort_values("n_docs_both", ascending=False).to_csv(
            d / "content_category_cooccurrence.csv", index=False, encoding="utf-8"
        )
        files.append("content_category_cooccurrence.csv")

    # --- PNG: category totals bar -----------------------------------------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        order = np.argsort(totals)
        fig, ax = plt.subplots(figsize=(6.5, 0.4 * len(cats) + 1.2))
        ax.barh(range(len(cats)), totals[order], color="#4C72B0")
        ax.set_yticks(range(len(cats)))
        ax.set_yticklabels([cats[i] for i in order], fontsize=8)
        ax.set_xlabel("keyword hits")
        ax.set_title(f"Content analysis — category totals ({text_col})", fontsize=10)
        fig.tight_layout()
        fig.savefig(d / "content_category_totals.png", dpi=150)
        plt.close(fig)
        files.append("content_category_totals.png")
    except Exception:
        pass

    top_i = int(np.argmax(totals))
    estimates["n_docs"] = float(len(docs))
    estimates["n_categories"] = float(len(cats))
    estimates["total_hits"] = float(totals.sum())
    estimates["top_category_hits"] = float(totals[top_i])

    preview = "、".join(f"{cats[i]}({int(totals[i])})" for i in np.argsort(totals)[::-1][:5])
    summary.append(
        f"{entry.method} 完成（词典内容分析，文本列={text_col}，{'中文子串匹配' if lang == 'zh' else '英文分词匹配'}，"
        f"{len(docs)} 篇文档 × {len(cats)} 个类目）：类目命中 Top：{preview}"
        f"（详见 content_category_totals.csv / content_doc_categories.csv"
        f"{' / content_category_cooccurrence.csv' if cooc_rows else ''}）。"
        f"⚠ 引擎只做**编码后量化**：类目与关键词由研究者定义（config dictionary），"
        f"引擎不自动发现类目、不替代人工编码（三级编码是研究者判断，NVivo/MaxQDA 的活）；"
        f"命中数完全取决于词典的覆盖度与效度——漏同义词/语境/否定会低估，"
        f"中文按子串匹配（不依赖分词、但可能误配复合词）。可用 config dictionary/text/lang 覆盖。"
    )
    code += [
        "# dictionary content analysis: researcher-defined {category: [keywords]}",
        "dictionary = {'类目A': ['关键词1','关键词2'], ...}  # YOU define this",
        f"docs = df[{text_col!r}].dropna().astype(str).tolist()",
        "hits = {c: sum(d.count(k) for d in docs for k in kws) for c, kws in dictionary.items()}",
        "print(sorted(hits.items(), key=lambda kv: -kv[1]))",
    ]
    return True


def _run_agreement(ctx: Ctx, coder1: str, coder2: str) -> bool:
    """Cohen's κ between two coder columns. Returns True if it ran."""
    df, entry, d = ctx.df, ctx.entry, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    import importlib.util

    if importlib.util.find_spec("sklearn") is None:
        summary.append("内容分析（一致性）跳过：Cohen's κ 需要 scikit-learn（未检测到）。")
        return False
    if coder1 not in df.columns or coder2 not in df.columns:
        summary.append(
            f"内容分析（一致性）跳过：coder 列不存在（coder1={coder1!r}, coder2={coder2!r}）。"
        )
        return False

    import pandas as pd
    from sklearn.metrics import cohen_kappa_score

    pair = df[[coder1, coder2]].dropna()
    if len(pair) < 2:
        summary.append("内容分析（一致性）跳过：两位编码者的共同有效行太少（< 2）。")
        return False

    a = pair[coder1].astype(str)
    b = pair[coder2].astype(str)
    kappa = float(cohen_kappa_score(a, b))
    n_cats = int(pd.concat([a, b]).nunique())
    band = _kappa_band(kappa)

    pd.DataFrame([{
        "coder1": coder1, "coder2": coder2, "n": int(len(pair)),
        "n_categories": n_cats, "cohen_kappa": round(kappa, 6), "agreement_band": band,
    }]).to_csv(d / "content_coder_agreement.csv", index=False, encoding="utf-8")
    files.append("content_coder_agreement.csv")

    estimates["cohen_kappa"] = round(kappa, 6)
    estimates["n_agreement"] = float(len(pair))
    estimates["n_agreement_categories"] = float(n_cats)

    pct_obs = float((a.values == b.values).mean())
    summary.append(
        f"{entry.method}（编码者一致性，{coder1} vs {coder2}，n={len(pair)}，{n_cats} 个类目）："
        f"Cohen's κ={kappa:.3f}（{band}），原始一致率={pct_obs:.1%}。"
        f"⚠ κ 已校正随机一致，强度分档（Landis & Koch：≥.81 几乎完全/≥.61 高度/≥.41 中等/"
        f"≥.21 一般/否则轻微）是约定俗成、非绝对标准；κ 对类目边际分布敏感（极不平衡时会偏低，"
        f"即 kappa 悖论）。可用 config coder1/coder2 指定编码者列。"
    )
    code += [
        "from sklearn.metrics import cohen_kappa_score",
        f"pair = df[[{coder1!r}, {coder2!r}]].dropna()",
        f"kappa = cohen_kappa_score(pair[{coder1!r}].astype(str), pair[{coder2!r}].astype(str))",
        "print('Cohen kappa', kappa)",
    ]
    return True


@register("content_analysis")
def _branch_content_analysis(ctx: Ctx) -> None:
    cfg = ctx.cfg
    summary = ctx.summary

    dictionary = cfg.get("dictionary")
    coder1, coder2 = cfg.get("coder1"), cfg.get("coder2")

    ran = False
    if isinstance(dictionary, dict) and dictionary:
        ran = _run_dictionary(ctx, dictionary) or ran
    if coder1 is not None and coder2 is not None:
        ran = _run_agreement(ctx, str(coder1), str(coder2)) or ran

    if not ran and not summary:
        summary.append(
            "内容分析跳过：需要 ① config dictionary（{类目: [关键词,...]}）做词典内容分析，"
            "或 ② config coder1/coder2（两位编码者的编码列）算 Cohen's κ 一致性。"
            "⚠ 引擎只做编码后量化，不自动发现类目、不替代人工编码——类目与编码须由研究者提供。"
        )
