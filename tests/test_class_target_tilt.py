"""Wave M5 — classification-target-first tilt (has_class_target signal + scoring tilt).

Deep-dogfood finding (wine): on a CLASSIFICATION dataset (target = cultivar, 13 continuous
features), the study surfaced REGRESSION methods that model an arbitrary feature (alcohol)
because the categorical target is not a continuous-outcome candidate, and the ≤1-per-family
study filter then buried discriminant/manova behind them. The fix boosts methods that model
the target and demotes feature-only regressions — but ONLY when the data has a genuine
multiclass target and NO trustworthy continuous outcome, so it never misfires on ordinary
regression data that merely carries a category column.

These tests pin BOTH directions: the signal fires where it should, stays off where a real
continuous outcome is present (the over-fire guard), and the tilt reorders the menu.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from researchforge.catalog import Catalog
from researchforge.profiler import profile_dataset
from researchforge.recommender.affinity import data_signals
from researchforge.recommender.recommend import recommend
from researchforge.recommender.scoring import _class_target_tilt


def _classification_frame(seed: int = 40, target_name: str = "target") -> pd.DataFrame:
    """Continuous features that separate 3 classes + a class-label-named multiclass target;
    no continuous outcome (wine-shape)."""
    rng = np.random.default_rng(seed)
    n = 180
    y = rng.integers(0, 3, n)
    cols = {f"feat{i}": (rng.normal(0, 1, n) + y * (0.8 if i % 2 == 0 else -0.6)).round(3)
            for i in range(6)}
    cols[target_name] = y
    return pd.DataFrame(cols)


def _fp(df: pd.DataFrame, tmp_path):
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    return profile_dataset(csv)


# ── signal: fires on a genuine multiclass class target ────────────────────────────────
def test_has_class_target_fires_on_multiclass_label(tmp_path):
    fp = _fp(_classification_frame(), tmp_path)
    sig = data_signals(fp)
    assert sig["has_class_target"] is True
    # continuous columns exist but none is a trustworthy outcome (only a low-conf last-numeric)
    assert fp.likely_outcome_confidence in {"", "low"}


def test_has_class_target_recognises_species_name(tmp_path):
    fp = _fp(_classification_frame(seed=7, target_name="species"), tmp_path)
    assert data_signals(fp)["has_class_target"] is True


# ── over-fire guards: does NOT fire when a real continuous outcome is present ──────────
def test_no_class_target_when_named_continuous_outcome_present(tmp_path):
    # price = a real continuous outcome (HIGH-confidence DV name); segment = a class label.
    # The class label must NOT trigger the tilt while a genuine continuous outcome exists.
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(0, 1, (n, 4))
    df = pd.DataFrame(X, columns=[f"feat_{c}" for c in "abcd"])
    df["price"] = (3 * X[:, 0] - 2 * X[:, 1] + rng.normal(0, 1, n)).round(3)
    df["segment"] = rng.integers(0, 3, n)
    fp = _fp(df, tmp_path)
    assert fp.likely_outcome == "price"
    assert data_signals(fp)["has_class_target"] is False


def test_no_class_target_on_plain_regression(tmp_path):
    # continuous 'target' (HIGH-confidence outcome name), no class label at all.
    rng = np.random.default_rng(1)
    n = 200
    X = rng.normal(0, 1, (n, 4))
    df = pd.DataFrame(X, columns=["age", "bmi", "bp", "s1"])
    df["target"] = (2 * X[:, 0] + rng.normal(0, 1, n)).round(3)
    fp = _fp(df, tmp_path)
    assert data_signals(fp)["has_class_target"] is False


def test_no_class_target_on_low_card_grouping_not_named_target(tmp_path):
    # a low-cardinality categorical that is NOT class-label-named (region) alongside a real
    # continuous outcome → not a classification target; the tilt stays off.
    rng = np.random.default_rng(2)
    n = 160
    region = rng.integers(0, 4, n)
    df = pd.DataFrame({
        "region": region,
        "x1": rng.normal(0, 1, n).round(3),
        "response": (rng.normal(0, 1, n) + region * 0.3).round(3),
    })
    fp = _fp(df, tmp_path)
    assert data_signals(fp)["has_class_target"] is False


# ── tilt deltas ───────────────────────────────────────────────────────────────────────
def test_tilt_zero_when_signal_off():
    cat = Catalog.load()
    off = {"has_class_target": False}
    for cid in ("discriminant_analysis", "mixed_effects", "manova", "ols_regression"):
        e = cat.by_id(cid)
        if e is not None:
            assert _class_target_tilt(e, off) == (0.0, "")


def test_tilt_boosts_classifiers_demotes_feature_regressions():
    cat = Catalog.load()
    on = {"has_class_target": True}
    for cid in ("discriminant_analysis", "linear_discriminant", "naive_bayes", "manova"):
        e = cat.by_id(cid)
        if e is not None:
            assert _class_target_tilt(e, on)[0] > 0, cid
    for cid in ("mixed_effects", "glmm", "robust_regression"):
        e = cat.by_id(cid)
        if e is not None:
            delta, note = _class_target_tilt(e, on)
            assert delta < 0 and note, cid  # demoted AND carries an honest disclosure


# ── end-to-end ranking: classifiers outrank feature-only regressions ──────────────────
def test_ranking_classifiers_beat_feature_regressions(tmp_path):
    fp = _fp(_classification_frame(), tmp_path)
    recs = recommend(fp)
    pos = {r.entry.id: i for i, r in enumerate(recs)}
    for good in ("discriminant_analysis", "manova"):
        for bad in ("mixed_effects", "robust_regression"):
            if good in pos and bad in pos:
                assert pos[good] < pos[bad], f"{good} should rank above {bad}"
