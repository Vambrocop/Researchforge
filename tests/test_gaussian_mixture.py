"""Tests for gaussian_mixture: GMM clustering with BIC model selection.

Known structure: data drawn from 3 well-separated Gaussians ->
  * BIC selects k ~= 3,
  * recovered component weights ~= 1/3 each and means near the (standardized) truth,
  * high silhouette of the hard assignment.
Plus a covariance_type compare + config override (k_range / features), and a degrade
(too few features / rows -> honest skip). Fixed random_state throughout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="gaussian_mixture",
        method="Gaussian mixture model (GMM clustering)",
        domain="machine learning",
        family="mixture",
        goal="explore",
        preconditions=Precondition(min_continuous=2, min_rows=10),
    )


def _three_gaussians(seed: int = 0, per: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    blocks = []
    for cx, cy in centers:
        x = rng.normal(cx, 0.6, per)
        y = rng.normal(cy, 0.6, per)
        blocks.append(np.column_stack([x, y]))
    XY = np.vstack(blocks)
    rng.shuffle(XY)
    return pd.DataFrame({"f1": XY[:, 0], "f2": XY[:, 1]})


def _run(df: pd.DataFrame, tmp_path: Path, config=None):
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    return run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config=config or {})


def test_bic_selects_three_components(tmp_path: Path) -> None:
    df = _three_gaussians()
    res = _run(df, tmp_path)
    assert "完成" in res.summary
    # BIC should land on k = 3 (well-separated, equal-size blobs)
    assert res.estimates["k_selected"] == 3.0
    # weights roughly balanced -> smallest weight well above 0
    assert res.estimates["min_weight"] > 0.2
    # hard assignment is clean -> high silhouette + high membership certainty
    assert res.estimates["silhouette"] > 0.7
    assert res.estimates["mean_max_proba"] > 0.95


def test_components_recover_truth(tmp_path: Path) -> None:
    df = _three_gaussians(seed=1)
    res = _run(df, tmp_path)
    out = Path(res.output_dir)
    comp = pd.read_csv(out / "gmm_components.csv")
    assert len(comp) == 3
    # weights ~= 1/3 each
    assert np.allclose(comp["weight"].to_numpy(), 1 / 3, atol=0.08)
    # The three TRUE centers, standardized, should each be matched by some component
    # mean (means are reported on the standardized scale). Raw centers:
    raw_centers = np.array([[0.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    mu = df[["f1", "f2"]].mean().to_numpy()
    sd = df[["f1", "f2"]].std(ddof=0).to_numpy()
    true_std = (raw_centers - mu) / sd
    got = comp[["mean_f1", "mean_f2"]].to_numpy()
    for tc in true_std:
        dists = np.linalg.norm(got - tc, axis=1)
        assert dists.min() < 0.4  # some recovered mean is near this true center


def test_bic_curve_and_assignments_written(tmp_path: Path) -> None:
    df = _three_gaussians(seed=2)
    res = _run(df, tmp_path)
    out = Path(res.output_dir)
    bic = pd.read_csv(out / "gmm_bic_curve.csv")
    # default k_range 1..6 -> 6 BIC rows for the single "full" covariance_type
    assert set(bic["covariance_type"]) == {"full"}
    assert bic["k"].min() == 1 and bic["k"].max() == 6
    # the selected k must be the BIC argmin
    assert int(bic.loc[bic["bic"].idxmin(), "k"]) == int(res.estimates["k_selected"])
    assign = pd.read_csv(out / "gmm_assignments.csv")
    assert {"row", "component", "max_proba"} <= set(assign.columns)
    assert len(assign) == len(df)


def test_covariance_type_compare(tmp_path: Path) -> None:
    df = _three_gaussians(seed=3)
    res = _run(df, tmp_path, {"covariance_type": ["full", "spherical", "diag"]})
    out = Path(res.output_dir)
    bic = pd.read_csv(out / "gmm_bic_curve.csv")
    assert {"full", "spherical", "diag"} <= set(bic["covariance_type"])
    assert "完成" in res.summary


def test_config_k_range_override(tmp_path: Path) -> None:
    df = _three_gaussians(seed=4)
    res = _run(df, tmp_path, {"k_range": [2, 4], "features": ["f1", "f2"]})
    out = Path(res.output_dir)
    bic = pd.read_csv(out / "gmm_bic_curve.csv")
    assert bic["k"].min() == 2 and bic["k"].max() == 4
    assert 2.0 <= res.estimates["k_selected"] <= 4.0


def test_degrade_one_feature(tmp_path: Path) -> None:
    df = pd.DataFrame({"f1": np.random.default_rng(0).normal(0, 1, 50)})
    res = _run(df, tmp_path)
    assert "跳过" in res.summary
    assert not res.estimates


def test_degrade_too_few_rows(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"f1": rng.normal(0, 1, 6), "f2": rng.normal(0, 1, 6)})
    res = _run(df, tmp_path)
    assert "跳过" in res.summary
    assert not res.estimates
