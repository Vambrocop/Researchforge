"""Tests for gwr: Geographically Weighted Regression (pure-Python, numpy).

Covers: (1) recovery of a KNOWN spatially-varying coefficient gradient + wide
coefficient range vs a single middling global-OLS slope; (2) a stationary
dataset -> narrow local coefficient range; (3) the large-bandwidth -> global-OLS
limit (the riskiest hand-rolled math: at a huge bandwidth GWR must collapse to a
single global OLS slope); (4) honest degrade (no geo / too few rows); (5) config
overrides (fixed bandwidth, kernel).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset
from researchforge.recommender.match import check_preconditions


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="gwr",
        method="Geographically Weighted Regression (GWR)",
        domain="gis",
        family="spatial",
        goal="explain",
        preconditions=Precondition(requires_geo=True, min_continuous=2, min_rows=30),
    )


def _spatially_varying_df(n: int = 120, seed: int = 0) -> pd.DataFrame:
    """y = beta(lon)*x + noise, with beta increasing with longitude.

    The local slope of y on x grows from ~0 at the west edge to ~2 at the east,
    so GWR should recover an east-rising coefficient surface while global OLS
    collapses it to a single middling slope (~1).
    """
    rng = np.random.default_rng(seed)
    lat = rng.uniform(30.0, 40.0, n)
    lon = rng.uniform(-100.0, -90.0, n)
    x = rng.normal(0.0, 1.0, n)
    # beta ranges roughly 0..2 across the longitude span
    beta_local = (lon - lon.min()) / (lon.max() - lon.min()) * 2.0
    y = beta_local * x + rng.normal(0.0, 0.3, n)
    # 'yield' is the first continuous column -> outcome; 'x' the predictor
    return pd.DataFrame({"latitude": lat, "longitude": lon, "yield_": y, "x": x})


def test_gwr_recovers_spatial_gradient(tmp_path: Path) -> None:
    df = _spatially_varying_df(n=130, seed=1)
    csv = tmp_path / "varying.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    assert sum(1 for c in fp.columns if c.kind == "geo") == 2

    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    assert (out / "gwr_coefficients.csv").exists()
    assert (out / "gwr_coefficient_ranges.csv").exists()

    coef = pd.read_csv(out / "gwr_coefficients.csv")
    # local slope of x should rise with longitude (correlate with location)
    r = np.corrcoef(coef["longitude"], coef["beta_x"])[0, 1]
    assert r > 0.5, f"local beta should rise with longitude, corr={r:.2f}"

    # wide local coefficient range (spatial non-stationarity), spanning the truth
    bmin = res.estimates["beta_x_min"]
    bmax = res.estimates["beta_x_max"]
    assert bmax - bmin > 1.0, f"coefficient range too narrow: {bmin:.2f}..{bmax:.2f}"

    # GWR fits the local structure better than global OLS here
    assert res.estimates["mean_local_r2"] >= res.estimates["global_ols_r2"]

    # global OLS gives a single middling slope inside the local range
    ols = res.estimates["beta_x_ols"]
    assert bmin < ols < bmax


def test_gwr_stationary_narrow_range(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    n = 120
    lat = rng.uniform(30.0, 40.0, n)
    lon = rng.uniform(-100.0, -90.0, n)
    x = rng.normal(0.0, 1.0, n)
    # constant slope everywhere -> stationary relationship
    y = 1.5 * x + rng.normal(0.0, 0.3, n)
    df = pd.DataFrame({"latitude": lat, "longitude": lon, "yield_": y, "x": x})
    csv = tmp_path / "stationary.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))

    bmin = res.estimates["beta_x_min"]
    bmax = res.estimates["beta_x_max"]
    # local range much narrower than the spatially-varying case (truth ~constant 1.5)
    assert bmax - bmin < 1.0, f"stationary range too wide: {bmin:.2f}..{bmax:.2f}"
    # local median near the true global slope
    assert abs(res.estimates["beta_x_median"] - 1.5) < 0.5


def test_gwr_large_bw_collapses_to_global_ols(tmp_path: Path) -> None:
    """The critical hand-rolled-math check: a very large fixed bandwidth makes
    every kernel weight ~1, so each local WLS becomes the SAME global OLS — local
    coefficients must collapse to the global OLS coefficient (range -> ~0)."""
    df = _spatially_varying_df(n=110, seed=3)
    csv = tmp_path / "limit.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    # bandwidth far larger than the spatial extent (lon/lat span ~10) -> Gaussian
    # weights ~= 1 for all pairs -> GWR == global OLS at every location.
    res = run_analysis(
        fp,
        _entry(),
        output_root=str(tmp_path / "o"),
        config={"bw": 1.0e6, "kernel": "gaussian"},
    )
    coef = pd.read_csv(Path(res.output_dir) / "gwr_coefficients.csv")
    ols = res.estimates["beta_x_ols"]
    # every local slope equals the global OLS slope (to numerical tolerance)
    assert np.allclose(coef["beta_x"].to_numpy(), ols, atol=1e-3), (
        f"large-bw GWR should equal global OLS; spread="
        f"{coef['beta_x'].max() - coef['beta_x'].min():.2e}"
    )
    assert res.estimates["beta_x_max"] - res.estimates["beta_x_min"] < 1e-3
    # in the global-OLS limit, mean local R^2 ~ global OLS R^2
    assert abs(res.estimates["mean_local_r2"] - res.estimates["global_ols_r2"]) < 0.05


def test_gwr_config_fixed_bandwidth_override(tmp_path: Path) -> None:
    df = _spatially_varying_df(n=120, seed=4)
    csv = tmp_path / "cfg.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(
        fp,
        _entry(),
        output_root=str(tmp_path / "o"),
        config={"bw": 2.5, "kernel": "bisquare"},
    )
    # fixed-bandwidth path reports the bandwidth it was given
    assert "selected_bw" in res.estimates
    assert abs(res.estimates["selected_bw"] - 2.5) < 1e-6
    assert "selected_k" not in res.estimates
    assert (Path(res.output_dir) / "gwr_coefficients.csv").exists()


def test_gwr_config_outcome_predictor_override(tmp_path: Path) -> None:
    """Explicit outcome/predictors override the first-continuous default."""
    df = _spatially_varying_df(n=110, seed=5)
    csv = tmp_path / "ov.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(
        fp,
        _entry(),
        output_root=str(tmp_path / "o"),
        config={"outcome": "yield_", "predictors": ["x"]},
    )
    coef = pd.read_csv(Path(res.output_dir) / "gwr_coefficients.csv")
    assert "beta_x" in coef.columns
    assert "beta_x_median" in res.estimates


def test_gwr_precondition_unmet_no_geo(tmp_path: Path) -> None:
    rng = np.random.default_rng(6)
    df = pd.DataFrame({"a": rng.normal(0, 1, 40), "b": rng.normal(0, 1, 40)})  # no geo
    csv = tmp_path / "nogeo.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)
    assert not ok
    assert any("地理" in u or "经纬度" in u for u in unmet)


def test_gwr_degrade_too_few_rows(tmp_path: Path) -> None:
    """Below the per-window minimum, GWR should honestly skip (no crash)."""
    rng = np.random.default_rng(7)
    n = 12  # well below max(30, 3p)
    df = pd.DataFrame(
        {
            "latitude": rng.uniform(30, 40, n),
            "longitude": rng.uniform(-100, -90, n),
            "yield_": rng.normal(0, 1, n),
            "x": rng.normal(0, 1, n),
        }
    )
    csv = tmp_path / "few.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    # no coefficients produced; honest Chinese skip message present
    # (RunResult.summary is a newline-joined string, not a list)
    assert not (Path(res.output_dir) / "gwr_coefficients.csv").exists()
    assert "GWR 失败" in res.summary
