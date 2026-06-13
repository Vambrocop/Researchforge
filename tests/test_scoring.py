"""Tests for the methodology score card (recommender.scoring)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.profiler import profile_dataset
from researchforge.recommender import recommend, score_method
from researchforge.recommender.rigor import assess_rigor


def _entry(aid: str, family: str) -> AnalysisEntry:
    return AnalysisEntry(
        id=aid, method=aid, domain="x", family=family, goal="explain",
        preconditions=Precondition(min_rows=2),
    )


def _fp(tmp_path: Path):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 1.0, 4.0, 3.0]})
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    return profile_dataset(csv)


def test_score_dimensions_in_range_and_fit_tracks_rigor(tmp_path: Path) -> None:
    fp = _fp(tmp_path)
    e = _entry("ols_regression", "statistics")
    rigor = assess_rigor(fp, e)
    sc = score_method(fp, e, rigor)
    for v in (sc.popularity, sc.publishability, sc.aesthetics, sc.difficulty, sc.fit, sc.novelty, sc.overall):
        assert 0 <= v <= 100
    assert sc.fit == max(0, min(100, rigor.score))  # fit == rigor score


def test_id_override_lifts_novelty_and_publishability(tmp_path: Path) -> None:
    fp = _fp(tmp_path)
    # synthetic_control has id overrides (high novelty + publishability) vs a plain
    # statistics method
    sc_sc = score_method(fp, _entry("synthetic_control", "causal"), assess_rigor(fp, _entry("synthetic_control", "causal")))
    sc_desc = score_method(fp, _entry("descriptive_stats", "statistics"), assess_rigor(fp, _entry("descriptive_stats", "statistics")))
    assert sc_sc.novelty >= 80
    assert sc_sc.publishability >= 80
    assert sc_desc.publishability <= 40  # descriptive override pulls it down
    assert sc_desc.novelty <= 20


def test_recommend_attaches_score(tmp_path: Path) -> None:
    fp = _fp(tmp_path)
    recs = recommend(fp)
    assert recs
    for r in recs:
        assert hasattr(r, "score")
        assert 0 <= r.score.overall <= 100
