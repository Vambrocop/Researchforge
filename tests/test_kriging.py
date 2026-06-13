"""Tests for kriging: geo gate + gstat ordinary kriging (skips without R/gstat)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import rbridge, run_analysis
from researchforge.profiler import profile_dataset
from researchforge.recommender.match import check_preconditions


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="kriging",
        method="Ordinary kriging (geostatistical interpolation)",
        domain="gis",
        family="spatial",
        goal="predict",
        preconditions=Precondition(requires_geo=True, min_continuous=1, min_rows=30),
    )


def test_kriging_interpolates(tmp_path: Path) -> None:
    if not (rbridge.r_available() and rbridge.r_package_available("gstat")):
        pytest.skip("R gstat package not available")

    rng = np.random.default_rng(0)
    n = 100
    lat = rng.uniform(30, 40, n)
    lon = rng.uniform(-100, -90, n)
    # a smooth, spatially-autocorrelated bump (no dominant linear trend) — the
    # stationary-ish setting ordinary kriging is designed for
    z = 10.0 * np.exp(-(((lat - 35) ** 2 + (lon + 95) ** 2)) / 6.0) + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"latitude": lat, "longitude": lon, "soil_c": np.round(z, 3)})
    csv = tmp_path / "soil.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)
    grid = pd.read_csv(out / "kriged_surface.csv")

    assert set(["prediction", "kriging_variance"]).issubset(grid.columns)
    assert len(grid) == 40 * 40
    assert (grid["kriging_variance"] >= -1e-9).all()  # variance is non-negative
    assert res.estimates["loo_rmse"] < 0.5 * (z.max() - z.min())  # decent cross-validation fit


def test_kriging_precondition_unmet(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(0, 1, 40), "b": rng.normal(0, 1, 40)})  # no geo cols
    csv = tmp_path / "nogeo.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)

    assert not ok
    assert any("地理" in u or "经纬度" in u for u in unmet)
