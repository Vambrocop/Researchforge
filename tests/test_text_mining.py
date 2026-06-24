"""Tests for the TEXT MINING family (pure Python; sentiment via an OPTIONAL backend).

Methods under test:
  - lda_topic_model   — recovers planted topics from a 3-theme synthetic corpus
  - tfidf_keywords    — surfaces the salient theme terms (overall + per-group)
  - sentiment_analysis— orders clearly positive vs negative docs (backend-guarded)

Synthetic corpus: documents about three obvious themes (space/astronomy,
cooking/food, finance/markets). The themes share almost no vocabulary, so a topic
model and TF-IDF should both separate them cleanly. Sentiment docs are written to be
unambiguously positive or negative.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog import Catalog
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


# ---------------------------------------------------------------------------
# Synthetic corpora
# ---------------------------------------------------------------------------

_SPACE = [
    "the telescope observed distant galaxies and bright stars in the night sky",
    "astronauts launched a rocket into orbit around the planet mars",
    "the spacecraft photographed the moon and the rings of saturn clearly",
    "scientists study cosmic radiation from stars galaxies and nebulae",
    "the satellite orbits earth and transmits images of the solar system",
    "a comet streaked across the sky near the orbit of jupiter",
]
_COOK = [
    "she chopped fresh garlic onion and tomato to make a savory pasta sauce",
    "the recipe calls for flour butter sugar and eggs to bake a sweet cake",
    "we grilled chicken with herbs and served it with roasted vegetables",
    "the chef seasoned the soup with salt pepper and fragrant basil leaves",
    "bake the bread dough until golden then spread butter and jam on top",
    "simmer the rice with broth garlic and onion for a tasty risotto dish",
]
_FIN = [
    "the stock market rallied as investors bought shares of major banks",
    "interest rates rose and bond yields climbed across the financial market",
    "the company reported strong quarterly earnings and revenue growth",
    "traders sold equities amid fears of inflation and a market correction",
    "the central bank raised rates to curb inflation and stabilize the economy",
    "investors diversified portfolios across stocks bonds and commodities",
]


def _corpus_df() -> pd.DataFrame:
    rows = []
    # triple each theme's docs: LDA needs enough co-occurrence signal to separate
    # cleanly on a short synthetic corpus (6 docs/theme is too few; 18 is robust).
    for theme, docs in (("space", _SPACE), ("cook", _COOK), ("fin", _FIN)):
        for txt in docs * 3:
            rows.append({"review": txt, "category": theme})
    return pd.DataFrame(rows)


_POS = [
    "this is absolutely wonderful, I love it so much, fantastic and excellent work",
    "amazing experience, truly delightful and a great joy, highly recommended",
    "brilliant, superb, happy and very pleased with this perfect outcome",
    "what a beautiful and lovely day, everything was great and made me smile",
]
_NEG = [
    "this is terrible, I hate it, an awful and horrible disappointing failure",
    "dreadful experience, completely useless and a miserable waste of money",
    "worst ever, disgusting and broken, I am furious and deeply unhappy",
    "a sad and painful disaster, everything went wrong and it was awful",
]


def _sentiment_df() -> pd.DataFrame:
    rows = []
    for txt in _POS:
        rows.append({"comment": txt, "label": "good"})
    for txt in _NEG:
        rows.append({"comment": txt, "label": "bad"})
    return pd.DataFrame(rows)


def _entry(eid: str):
    return Catalog.load().by_id(eid)


# ===========================================================================
# Catalog sanity
# ===========================================================================

@pytest.mark.parametrize("eid", ["lda_topic_model", "tfidf_keywords", "sentiment_analysis"])
def test_catalog_loads(eid):
    e = _entry(eid)
    assert e is not None
    assert e.executor_ref == f"py::{eid}"
    assert isinstance(e.biases, list) and len(e.biases) >= 3
    assert isinstance(e.produces, list) and e.produces
    assert e.params and all(p.name for p in e.params)


# ===========================================================================
# 1. LDA topic model — recovers the planted themes
# ===========================================================================

def test_lda_recovers_topics(tmp_path):
    csv = tmp_path / "corpus.csv"
    _corpus_df().to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("lda_topic_model"), output_root=str(tmp_path / "o"),
                       config={"n_topics": 3, "min_df": 2})
    out = Path(res.output_dir)

    assert (out / "lda_top_words.csv").exists()
    assert (out / "lda_doc_topics.csv").exists()
    assert (out / "lda_topic_sizes.csv").exists()
    assert res.estimates["n_topics"] == 3.0
    assert res.estimates["n_docs"] == 54.0
    assert res.estimates["vocab_size"] > 5

    # Each planted theme should dominate at least one topic's top words. Collect the
    # top words per topic, then check theme marker words appear among them.
    tw = pd.read_csv(out / "lda_top_words.csv")
    topic_words = {t: set(g["word"]) for t, g in tw.groupby("topic")}
    all_top = set().union(*topic_words.values())

    space_markers = {"galaxies", "stars", "orbit", "rocket", "telescope", "spacecraft", "solar"}
    cook_markers = {"garlic", "butter", "bake", "sugar", "onion", "soup", "bread", "sauce"}
    fin_markers = {"market", "stock", "investors", "rates", "inflation", "bank", "bonds"}

    # at least one marker from each theme made it into the top words overall
    assert space_markers & all_top, f"no space markers in top words: {all_top}"
    assert cook_markers & all_top, f"no cooking markers in top words: {all_top}"
    assert fin_markers & all_top, f"no finance markers in top words: {all_top}"

    # at least one topic should be theme-DOMINATED: one theme contributes >=2 marker
    # words to that topic AND more than the other two themes combined (a clean, robust
    # recovery signal — top-N word lists can carry a stray cross-theme term).
    coherent = False
    for words in topic_words.values():
        counts = {
            "space": len(space_markers & words),
            "cook": len(cook_markers & words),
            "fin": len(fin_markers & words),
        }
        top_theme = max(counts, key=counts.get)
        rest = sum(v for k, v in counts.items() if k != top_theme)
        if counts[top_theme] >= 2 and counts[top_theme] > rest:
            coherent = True
            break
    assert coherent, f"no theme-dominated topic recovered: {topic_words}"


def test_lda_default_n_topics(tmp_path):
    """Default n_topics is 5 (clamped to <= n_docs-1)."""
    csv = tmp_path / "corpus.csv"
    _corpus_df().to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("lda_topic_model"), output_root=str(tmp_path / "o"))
    assert res.estimates["n_topics"] == 5.0  # 54 docs -> default 5 stands


def test_lda_config_text_override(tmp_path):
    """config text forces which column is treated as the corpus."""
    df = _corpus_df().rename(columns={"review": "body"})
    csv = tmp_path / "c.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("lda_topic_model"), output_root=str(tmp_path / "o"),
                       config={"text": "body", "n_topics": 3})
    assert "文本列=body" in res.summary
    assert res.estimates["n_topics"] == 3.0


# ===========================================================================
# 2. TF-IDF keywords — surfaces salient terms, per-group
# ===========================================================================

def test_tfidf_surfaces_terms(tmp_path):
    csv = tmp_path / "corpus.csv"
    _corpus_df().to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("tfidf_keywords"), output_root=str(tmp_path / "o"),
                       config={"group": "category"})
    out = Path(res.output_dir)

    assert (out / "tfidf_top_terms.csv").exists()
    assert (out / "tfidf_top_terms_by_group.csv").exists()
    assert res.estimates["n_docs"] == 54.0
    assert res.estimates["n_groups"] == 3.0

    overall = pd.read_csv(out / "tfidf_top_terms.csv")
    terms = set(overall["term"])
    # salient theme terms should appear among the overall top terms
    assert terms & {"market", "garlic", "stars", "orbit", "butter", "investors", "bake"}, terms

    bygrp = pd.read_csv(out / "tfidf_top_terms_by_group.csv")
    # each group's top terms should be dominated by its own theme vocabulary
    space_top = set(bygrp[bygrp["group"] == "space"]["term"])
    cook_top = set(bygrp[bygrp["group"] == "cook"]["term"])
    fin_top = set(bygrp[bygrp["group"] == "fin"]["term"])
    assert space_top & {"stars", "orbit", "galaxies", "rocket", "telescope", "solar", "spacecraft"}
    assert cook_top & {"garlic", "butter", "bake", "onion", "sugar", "bread", "soup", "sauce"}
    assert fin_top & {"market", "investors", "rates", "inflation", "stock", "bank", "bonds"}


def test_tfidf_no_group(tmp_path):
    """No grouping column -> overall only, n_groups == 0, still runs."""
    df = pd.DataFrame({"review": _SPACE + _COOK})
    csv = tmp_path / "c.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("tfidf_keywords"), output_root=str(tmp_path / "o"))
    assert (Path(res.output_dir) / "tfidf_top_terms.csv").exists()
    assert res.estimates["n_groups"] == 0.0


# ===========================================================================
# 3. Sentiment analysis — backend-guarded happy path + honest degrade
# ===========================================================================

def _have_backend() -> bool:
    import importlib.util
    return any(
        importlib.util.find_spec(m) is not None
        for m in ("vaderSentiment", "textblob", "nltk")
    )


def test_sentiment_orders_pos_vs_neg(tmp_path):
    pytest.importorskip("vaderSentiment", reason="needs a sentiment backend")
    csv = tmp_path / "s.csv"
    _sentiment_df().to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("sentiment_analysis"), output_root=str(tmp_path / "o"),
                       config={"group": "label"})
    out = Path(res.output_dir)

    assert (out / "sentiment_per_doc.csv").exists()
    assert (out / "sentiment_distribution.csv").exists()
    assert (out / "sentiment_by_group.csv").exists()
    assert res.estimates["n_docs"] == 8.0

    # clearly positive docs should outscore clearly negative ones on mean polarity.
    grp = pd.read_csv(out / "sentiment_by_group.csv").set_index("group")
    assert grp.loc["good", "mean_polarity"] > grp.loc["bad", "mean_polarity"]
    assert grp.loc["good", "mean_polarity"] > 0
    assert grp.loc["bad", "mean_polarity"] < 0
    # at least one positive and one negative label surfaced
    assert res.estimates["pct_positive"] > 0
    assert res.estimates["pct_negative"] > 0


def test_sentiment_honest_degrade_no_backend(tmp_path, monkeypatch):
    """With NO backend resolvable, sentiment degrades honestly (跳过 + pip hint),
    does not crash and does not fabricate scores."""
    import researchforge.executor.branches.text_mining as tm

    monkeypatch.setattr(tm, "_sentiment_backend", lambda: (None, None))

    csv = tmp_path / "s.csv"
    _sentiment_df().to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("sentiment_analysis"), output_root=str(tmp_path / "o"))

    assert "跳过" in res.summary
    assert "vaderSentiment" in res.summary or "textblob" in res.summary
    # no scores fabricated
    assert "mean_polarity" not in res.estimates
    assert not (Path(res.output_dir) / "sentiment_per_doc.csv").exists()


# ===========================================================================
# 4. Honest-degrade paths shared by all three (no text column)
# ===========================================================================

@pytest.mark.parametrize("eid", ["lda_topic_model", "tfidf_keywords", "sentiment_analysis"])
def test_degrade_no_text_column(tmp_path, eid):
    """A purely numeric dataset has no free-text column -> honest 跳过."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": rng.normal(0, 1, 40),
        "y": rng.normal(0, 1, 40),
        "g": rng.integers(0, 3, 40),
    })
    csv = tmp_path / "nums.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(eid), output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
    assert "未找到自由文本列" in res.summary


def test_lda_degrade_too_few_docs(tmp_path):
    """Fewer than 5 docs -> LDA degrades honestly."""
    df = pd.DataFrame({"review": _SPACE[:3]})
    csv = tmp_path / "few.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("lda_topic_model"), output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
    assert "n_topics" not in res.estimates
