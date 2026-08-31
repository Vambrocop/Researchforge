"""Wave M11 — free-text detection routes to the text-mining family.

Dogfood (new-domain sweep): a product-reviews table surfaced dif_detection / proportional_odds /
fleiss_kappa and NEVER the text-mining family (TF-IDF / sentiment / topics) — a free-text column
profiles as `categorical`/`id`, so nothing downstream knew it was prose. The fix adds an
`is_text` column flag (types.is_text_like), a `has_text` signal, and an nlp-family tilt.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from researchforge.catalog import Catalog
from researchforge.profiler import profile_dataset
from researchforge.profiler.ingest import read_table
from researchforge.profiler.types import is_text_like
from researchforge.recommender import build_plan, select_top
from researchforge.recommender.affinity import data_signals

_REVIEWS = [
    "great product love it works well", "terrible waste of money broke fast",
    "okay average nothing special here", "excellent quality highly recommend buy",
    "poor design cheap material disappointed", "good value for the price satisfied",
    "amazing fast shipping perfect experience", "defective item returned immediately awful",
    "decent but a bit overpriced honestly", "fantastic exceeded all my expectations",
]


def _text_frame(col="comment", n=200, seed=5):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({col: [" ".join(rng.choice(_REVIEWS, 3)) for _ in range(n)]})


def _fp(df, tmp_path):
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    return profile_dataset(csv)


# ── is_text_like unit ─────────────────────────────────────────────────────────────────
def test_is_text_like_true_for_prose():
    assert is_text_like(_text_frame()["comment"]) is True


def test_is_text_like_false_for_category_label():
    rng = np.random.default_rng(1)
    assert is_text_like(pd.Series(rng.choice(["North", "South", "East", "West"], 200))) is False


def test_is_text_like_false_for_numbers():
    assert is_text_like(pd.Series(np.arange(200.0))) is False


# ── profiler flag + signal ────────────────────────────────────────────────────────────
def test_profiler_flags_text_column(tmp_path):
    fp = _fp(_text_frame(), tmp_path)
    assert fp.column("comment").is_text is True
    assert data_signals(fp)["has_text"] is True


def test_region_not_flagged_text(tmp_path):
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"region": rng.choice(["North", "South", "East", "West"], 200),
                       "y": rng.normal(0, 1, 200).round(2)})
    fp = _fp(df, tmp_path)
    assert fp.column("region").is_text is False
    assert data_signals(fp)["has_text"] is False


# ── end-to-end: text-mining leads on free text ────────────────────────────────────────
def test_text_mining_surfaces_on_free_text(tmp_path):
    fp = _fp(_text_frame(), tmp_path)
    cat = Catalog.load()
    plan = build_plan(fp, read_table(fp.path), cat)
    top_ids = [r.entry.id for r in select_top(fp, top=6, catalog=cat, plan=plan)]
    nlp_ids = {e.id for e in cat.all() if e.family == "nlp"}
    assert nlp_ids & set(top_ids), f"a text-mining method should be in top-6, got {top_ids}"
