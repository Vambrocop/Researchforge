"""Model-agnostic explainers: partial dependence, SHAP, ALE.

Planted-signal data: the outcome depends strongly on x1 (and moderately x2) and not
at all on noise features. All three explainers must rank x1 top, and degrade honestly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog import Catalog
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_CAT = Catalog.load()


def _run(csv, aid, tmp_path, config=None):
    fp = profile_dataset(csv)
    return run_analysis(fp, _CAT.by_id(aid), output_root=str(tmp_path / "o"), config=config)


def _signal_df(n=400, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    noise1 = rng.normal(0, 1, n)
    noise2 = rng.normal(0, 1, n)
    # y driven mainly by x1 (strong), some x2; noise features irrelevant
    y = 3.0 * x1 + 1.0 * x2 + rng.normal(0, 0.5, n)
    # put y first so it's the auto outcome (first continuous)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "noise1": noise1, "noise2": noise2})


# --------------------------------------------------------------------------- #
# partial_dependence
# --------------------------------------------------------------------------- #
def test_pdp_ranks_strong_feature_top(tmp_path: Path) -> None:
    csv = tmp_path / "pdp.csv"
    _signal_df().to_csv(csv, index=False)
    res = _run(csv, "partial_dependence", tmp_path)
    e = res.estimates
    assert "pd_range_x1" in e and "pd_range_x2" in e
    assert e["pd_range_x1"] > e["pd_range_x2"]                 # x1 strongest
    assert e["pd_range_x1"] > e.get("pd_range_noise1", 0.0)
    assert (Path(res.output_dir) / "partial_dependence.csv").exists()


def test_pdp_degrades_no_features(tmp_path: Path) -> None:
    csv = tmp_path / "one.csv"
    pd.DataFrame({"y": np.arange(40.0)}).to_csv(csv, index=False)
    res = _run(csv, "partial_dependence", tmp_path)
    assert "跳过" in res.summary
    assert "n_features_explained" not in res.estimates


# --------------------------------------------------------------------------- #
# shap_values
# --------------------------------------------------------------------------- #
def test_shap_ranks_strong_feature_top(tmp_path: Path) -> None:
    pytest.importorskip("shap")
    csv = tmp_path / "shap.csv"
    _signal_df().to_csv(csv, index=False)
    res = _run(csv, "shap_values", tmp_path)
    e = res.estimates
    assert e["mean_abs_shap_x1"] > e["mean_abs_shap_x2"]
    assert e["mean_abs_shap_x1"] > e.get("mean_abs_shap_noise1", 0.0)
    # x1 has a POSITIVE effect (coef +3) -> direction +1 in the saved table
    imp = pd.read_csv(Path(res.output_dir) / "shap_importance.csv").set_index("feature")
    assert imp.loc["x1", "direction"] == 1.0


def test_multiclass_categorical_outcome_skips(tmp_path: Path) -> None:
    # a 3-level categorical outcome via config -> honest skip (positive-class slice
    # would be arbitrary), not a silently-wrong one-vs-rest explanation
    rng = np.random.default_rng(1)
    n = 120
    df = pd.DataFrame({
        "grp": rng.choice(["a", "b", "c"], n),
        "x1": rng.normal(0, 1, n), "x2": rng.normal(0, 1, n), "x3": rng.normal(0, 1, n),
    })
    csv = tmp_path / "mc.csv"
    df.to_csv(csv, index=False)
    res = _run(csv, "partial_dependence", tmp_path, config={"outcome": "grp"})
    assert "跳过" in res.summary and "二值" in res.summary


def test_shap_degrades_few_rows(tmp_path: Path) -> None:
    csv = tmp_path / "small.csv"
    _signal_df(n=10).to_csv(csv, index=False)
    res = _run(csv, "shap_values", tmp_path)
    assert "跳过" in res.summary


# --------------------------------------------------------------------------- #
# accumulated_local_effects
# --------------------------------------------------------------------------- #
def test_ale_ranks_strong_feature_top(tmp_path: Path) -> None:
    csv = tmp_path / "ale.csv"
    _signal_df().to_csv(csv, index=False)
    res = _run(csv, "accumulated_local_effects", tmp_path)
    e = res.estimates
    assert "ale_range_x1" in e
    assert e["ale_range_x1"] > e.get("ale_range_x2", 0.0)
    assert e["ale_range_x1"] > e.get("ale_range_noise1", 0.0)
    assert (Path(res.output_dir) / "accumulated_local_effects.csv").exists()


def test_ale_is_centered(tmp_path: Path) -> None:
    # the centered ALE curve should have values straddling zero (min<=0<=max)
    csv = tmp_path / "ale2.csv"
    _signal_df(seed=3).to_csv(csv, index=False)
    res = _run(csv, "accumulated_local_effects", tmp_path,
               config={"outcome": "y", "features": ["x1"]})
    tab = pd.read_csv(Path(res.output_dir) / "accumulated_local_effects.csv")
    x1 = tab[tab["feature"] == "x1"]["ale"]
    assert x1.min() <= 1e-6 and x1.max() >= -1e-6     # centered around 0
    # monotone-increasing signal (coef +3): ALE should rise with x1
    assert x1.iloc[-1] > x1.iloc[0]


def test_ale_monotone_direction(tmp_path: Path) -> None:
    # negative-coef feature -> ALE should DECREASE across its range
    rng = np.random.default_rng(7)
    n = 400
    x1 = rng.normal(0, 1, n)
    y = -2.5 * x1 + rng.normal(0, 0.5, n)
    csv = tmp_path / "neg.csv"
    pd.DataFrame({"y": y, "x1": x1, "x2": rng.normal(0, 1, n)}).to_csv(csv, index=False)
    res = _run(csv, "accumulated_local_effects", tmp_path,
               config={"outcome": "y", "features": ["x1"]})
    tab = pd.read_csv(Path(res.output_dir) / "accumulated_local_effects.csv")
    x1a = tab[tab["feature"] == "x1"]["ale"]
    assert x1a.iloc[-1] < x1a.iloc[0]                 # decreasing


# --------------------------------------------------------------------------- #
# quantile_intervals
# --------------------------------------------------------------------------- #
def test_quantile_intervals_coverage_near_nominal(tmp_path: Path) -> None:
    csv = tmp_path / "qi.csv"
    _signal_df(n=500).to_csv(csv, index=False)
    res = _run(csv, "quantile_intervals", tmp_path)
    e = res.estimates
    assert e["nominal_coverage"] == 0.9
    assert 0.7 < e["empirical_coverage"] <= 1.0      # roughly calibrated
    assert e["mean_interval_width"] > 0
    assert (Path(res.output_dir) / "quantile_intervals.csv").exists()


def test_quantile_intervals_skips_classification(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    n = 120
    df = pd.DataFrame({"label": rng.integers(0, 2, n), "x1": rng.normal(0, 1, n),
                       "x2": rng.normal(0, 1, n)})
    csv = tmp_path / "clf.csv"
    df.to_csv(csv, index=False)
    res = _run(csv, "quantile_intervals", tmp_path, config={"outcome": "label"})
    assert "跳过" in res.summary


# --------------------------------------------------------------------------- #
# feature_interaction (Friedman's H)
# --------------------------------------------------------------------------- #
def test_feature_interaction_detects_product_term(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 400
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    y = 3.0 * x1 * x2 + 0.5 * x3 + rng.normal(0, 0.3, n)   # x1×x2 pure interaction
    csv = tmp_path / "fi.csv"
    pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3}).to_csv(csv, index=False)
    res = _run(csv, "feature_interaction", tmp_path)
    e = res.estimates
    h_12 = e.get("H_x1__x2", e.get("H_x2__x1", 0.0))
    h_13 = e.get("H_x1__x3", e.get("H_x3__x1", 0.0))
    assert h_12 > h_13                                # interacting pair ranks above noise pair
    assert e["max_H"] > 0.1
    assert (Path(res.output_dir) / "feature_interaction_H.csv").exists()


def test_feature_interaction_low_for_additive(tmp_path: Path) -> None:
    # additive signal -> small interaction
    csv = tmp_path / "add.csv"
    _signal_df(n=400, seed=2).to_csv(csv, index=False)
    res = _run(csv, "feature_interaction", tmp_path)
    assert res.estimates["max_H"] < 0.5              # no dominant interaction


# --------------------------------------------------------------------------- #
# surrogate_model
# --------------------------------------------------------------------------- #
def test_surrogate_fidelity_high_for_simple_signal(tmp_path: Path) -> None:
    csv = tmp_path / "sur.csv"
    _signal_df(n=400).to_csv(csv, index=False)
    res = _run(csv, "surrogate_model", tmp_path, config={"max_depth": 4})
    e = res.estimates
    assert e["fidelity_r2"] > 0.5                     # shallow tree mimics a near-linear GBM
    assert e["surrogate_max_depth"] == 4.0
    assert (Path(res.output_dir) / "surrogate_tree_rules.txt").exists()
    assert (Path(res.output_dir) / "surrogate_importance.csv").exists()


def test_surrogate_classification_fidelity(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    n = 300
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    label = (x1 + 0.3 * rng.normal(0, 1, n) > 0).astype(int)
    csv = tmp_path / "surclf.csv"
    pd.DataFrame({"label": label, "x1": x1, "x2": x2}).to_csv(csv, index=False)
    res = _run(csv, "surrogate_model", tmp_path, config={"outcome": "label"})
    assert "fidelity_accuracy" in res.estimates
    assert res.estimates["fidelity_accuracy"] > 0.7
