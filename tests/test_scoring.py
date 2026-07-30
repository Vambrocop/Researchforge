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


def test_small_data_tilt_demotes_high_capacity(tmp_path: Path) -> None:
    # Wave S: on a small-data regime (few rows, or few rows per predictor) the fit tilt
    # DEMOTES data-hungry learners (with an overfit ⚠) and BOOSTS regularized/Bayesian —
    # the "由简到繁 / start simple" ladder. Outside the regime the tilt is zero.
    import numpy as np

    from researchforge.recommender.affinity import data_signals
    from researchforge.recommender.scoring import _small_data_tilt

    rng = np.random.default_rng(0)
    gbm = _entry("gradient_boosting", "ml")
    reg = _entry("regularized_regression", "regression")

    small = pd.DataFrame({**{f"x{i}": rng.normal(0, 1, 50) for i in range(5)},
                          "y": rng.normal(0, 1, 50)})
    small.to_csv(tmp_path / "s.csv", index=False)
    sig_small = data_signals(profile_dataset(tmp_path / "s.csv"))
    d_gbm, note = _small_data_tilt(gbm, sig_small)
    assert d_gbm < 0 and "过拟合" in note          # high-capacity demoted + disclosed
    assert _small_data_tilt(reg, sig_small)[0] > 0  # regularized boosted

    large = pd.DataFrame({**{f"x{i}": rng.normal(0, 1, 1500) for i in range(5)},
                          "y": rng.normal(0, 1, 1500)})
    large.to_csv(tmp_path / "l.csv", index=False)
    sig_large = data_signals(profile_dataset(tmp_path / "l.csv"))
    assert _small_data_tilt(gbm, sig_large) == (0.0, "")   # no tilt on ample data
    assert _small_data_tilt(reg, sig_large)[0] == 0.0


def test_small_data_advisory_card(tmp_path: Path) -> None:
    # Wave S③: a once-per-dataset 由简到繁 guidance card on small data, empty on ample data.
    import numpy as np

    from researchforge.recommender.scoring import small_data_advisory

    small = pd.DataFrame({**{f"x{i}": np.random.default_rng(0).normal(0, 1, 50) for i in range(5)},
                          "y": np.random.default_rng(1).normal(0, 1, 50)})
    small.to_csv(tmp_path / "s.csv", index=False)
    card = small_data_advisory(profile_dataset(tmp_path / "s.csv"))
    assert "由简到繁" in card and "朴素贝叶斯" in card

    large = pd.DataFrame({**{f"x{i}": np.random.default_rng(2).normal(0, 1, 1500) for i in range(5)},
                          "y": np.random.default_rng(3).normal(0, 1, 1500)})
    large.to_csv(tmp_path / "l.csv", index=False)
    assert small_data_advisory(profile_dataset(tmp_path / "l.csv")) == ""
