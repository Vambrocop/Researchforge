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
