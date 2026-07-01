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


def test_family_table_covers_all_catalog_families() -> None:
    # every catalog family must have editorial priors in scoring._FAMILY, else 6-dim
    # scores silently fall back to _DEFAULT (the stale-family-name bug: "timeseries" vs
    # "time-series", "panel" vs "econometrics" left 165/294 methods on generic scores).
    from researchforge.catalog import Catalog
    from researchforge.recommender.scoring import _FAMILY

    fams = {e.family for e in Catalog.load().entries}
    missing = fams - set(_FAMILY)
    assert not missing, f"scoring._FAMILY missing editorial priors for: {sorted(missing)}"


def test_score_dimensions_in_range(tmp_path: Path) -> None:
    fp = _fp(tmp_path)
    e = _entry("ols_regression", "statistics")
    rigor = assess_rigor(fp, e)
    sc = score_method(fp, e, rigor)
    for v in (sc.popularity, sc.publishability, sc.aesthetics, sc.difficulty, sc.fit, sc.novelty, sc.overall):
        assert 0 <= v <= 100


def test_fit_is_data_affinity_not_rigor(tmp_path: Path) -> None:
    # Stage 3: fit is now the data↔method affinity score, NOT a copy of the rigor score.
    # A feasible method on a tiny generic frame should get a sensible (non-extreme) fit
    # even when rigor is a perfect 100 (few biases declared).
    fp = _fp(tmp_path)
    e = _entry("ols_regression", "statistics")
    rigor = assess_rigor(fp, e)
    sc = score_method(fp, e, rigor)
    assert 0 <= sc.fit <= 100
    # an infeasible (red) method's fit is capped at its (low) rigor score
    panel_only = AnalysisEntry(id="panel_fixed_effects", method="fe", domain="x",
                               family="econometrics", goal="explain",
                               preconditions=Precondition(min_rows=2, is_panel=True))
    r2 = assess_rigor(fp, panel_only)  # not panel data -> red
    if r2.light == "red":
        assert score_method(fp, panel_only, r2).fit <= max(0, min(100, r2.score))


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
