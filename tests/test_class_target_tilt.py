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
from researchforge.recommender.affinity import data_signals, is_count_outcome
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
def test_class_target_method_ids_are_live_catalog_ids():
    # guard against dead-id drift: the curated method sets must reference REAL catalog ids
    # (the tilt is silently a no-op for a typo'd id — this is how `multinomial_logistic`, which
    # does not exist, sat in _CLASSIFY_TARGET while the real method `multinomial_logit` got no
    # class-target boost). Families/`regression` are validated separately (they are not ids).
    from researchforge.recommender.scoring import (
        _CLASSIFY_TARGET,
        _COMPARE_BY_TARGET,
        _FEATURE_MODEL,
    )

    ids = {e.id for e in Catalog.load().all()}
    dead = (_CLASSIFY_TARGET | _FEATURE_MODEL | _COMPARE_BY_TARGET) - ids
    assert not dead, f"curated class-target sets reference non-existent catalog ids: {sorted(dead)}"


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


# ── Wave M6: a LOW-confidence positional outcome is not a count outcome ────────────────
def _likert_survey_frame(seed: int = 41) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 260
    latent = rng.normal(0, 1, n)
    cols = {}
    for i, load in enumerate([0.9, 0.8, 0.85, 0.7, 0.75, 0.6], start=1):
        x = load * latent + rng.normal(0, 1, n)
        cols[f"sat_q{i}"] = np.clip(np.round(3 + 1.1 * x), 1, 5).astype(int)
    cols["age"] = rng.integers(18, 66, n)
    return pd.DataFrame(cols)


def test_low_conf_positional_age_is_not_count_outcome(tmp_path):
    # age is the last numeric column → likely_outcome at LOW confidence; it must NOT be read
    # as a count OUTCOME (that mis-routed a Likert survey to NB/Poisson on a rating).
    fp = _fp(_likert_survey_frame(), tmp_path)
    assert fp.likely_outcome == "age" and fp.likely_outcome_confidence == "low"
    age_col = fp.column("age")
    assert age_col.kind == "count"  # profiles as count (integer, many distinct)
    assert is_count_outcome(age_col, fp) is False  # but not a count OUTCOME (low-conf guess)
    assert data_signals(fp)["has_count_outcome"] is False


def test_count_outcome_still_detected_by_name_and_high_conf(tmp_path):
    # a count column named count-ish (visits) stays a count outcome; and a HIGH-confidence
    # DV-named count (y) stays one too — the gate only drops the LOW-confidence positional case.
    rng = np.random.default_rng(4)
    n = 200
    x1 = rng.normal(0, 1, n)
    df = pd.DataFrame({"visits": rng.poisson(np.exp(0.6 + 0.4 * x1)), "x1": x1.round(3),
                       "x2": rng.normal(0, 1, n).round(3)})
    assert data_signals(_fp(df, tmp_path))["has_count_outcome"] is True


def test_likert_survey_demotes_count_models(tmp_path):
    fp = _fp(_likert_survey_frame(), tmp_path)
    recs = recommend(fp)
    pos = {r.entry.id: i for i, r in enumerate(recs)}
    # psychometrics/IRT should top; NB/Poisson must not be in the top handful.
    for cid in ("negative_binomial_regression", "poisson_regression"):
        if cid in pos:
            assert pos[cid] > 20, f"{cid} should be demoted on a Likert survey, got #{pos[cid]}"


# ── Wave M7: finance methods only headline on a financial-asset series ─────────────────
def _sales_ts_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 60
    t = np.arange(n)
    return pd.DataFrame({
        "month": pd.date_range("2020-01-01", periods=n, freq="MS").strftime("%Y-%m"),
        "sales": (200 + 2.5 * t + 8 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 6, n)).round(1),
        "price": (20 - 0.03 * t + rng.normal(0, 0.5, n)).round(2),  # product price, NOT finance
        "promo": rng.binomial(1, 0.3, n),
    })


def _stock_ts_frame() -> pd.DataFrame:
    rng = np.random.default_rng(9)
    n = 200
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n).strftime("%Y-%m-%d"),
        "close": (100 + np.cumsum(rng.normal(0, 1, n))).round(2),
        "volume": rng.integers(1_000_000, 5_000_000, n),
    })


def test_finance_signal_off_on_sales_on_for_stock(tmp_path):
    assert data_signals(_fp(_sales_ts_frame(), tmp_path))["has_finance_signal"] is False
    assert data_signals(_fp(_stock_ts_frame(), tmp_path))["has_finance_signal"] is True


def test_finance_not_false_triggered_by_lookalike_tokens(tmp_path):
    # `livestock`/`disclosure`/`navigation` must not token-match stock/close/nav.
    rng = np.random.default_rng(3)
    n = 60
    df = pd.DataFrame({
        "month": pd.date_range("2020-01-01", periods=n, freq="MS").strftime("%Y-%m"),
        "livestock_count": rng.integers(50, 200, n),
        "disclosure_flag": rng.integers(0, 2, n),
        "navigation_hours": rng.normal(10, 2, n).round(2),
    })
    assert data_signals(_fp(df, tmp_path))["has_finance_signal"] is False


def test_finance_demoted_on_sales_kept_on_stock(tmp_path):
    sales = recommend(_fp(_sales_ts_frame(), tmp_path))
    stock = recommend(_fp(_stock_ts_frame(), tmp_path))
    p_sales = {r.entry.id: i for i, r in enumerate(sales)}
    p_stock = {r.entry.id: i for i, r in enumerate(stock)}
    # on sales the finance methods sink below generic forecasting; on stock they stay near top.
    assert p_sales["value_at_risk"] > p_sales["arima"]
    assert p_sales["value_at_risk"] > 5, "VaR should be demoted on a non-financial series"
    assert p_stock["value_at_risk"] <= 5, "VaR should stay top on a real financial series"
