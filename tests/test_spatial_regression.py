"""Tests for spatial_regression: geo gate + SAR/SEM via R (skips without R)."""

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
        id="spatial_regression",
        method="Spatial regression (SAR / SEM)",
        domain="gis",
        family="spatial",
        goal="explain",
        preconditions=Precondition(requires_geo=True, min_continuous=2, min_rows=25),
    )


def test_spatial_regression(tmp_path: Path) -> None:
    if not (
        rbridge.r_available()
        and rbridge.r_package_available("spdep")
        and rbridge.r_package_available("spatialreg")
    ):
        pytest.skip("R spdep/spatialreg not available")

    rng = np.random.default_rng(0)
    n = 120
    lat = rng.uniform(30, 40, n)
    lon = rng.uniform(-100, -90, n)
    x = rng.normal(0, 1, n)
    # spatial trend (drives residual autocorrelation) + a real covariate effect
    y = 0.5 * lat + 1.2 * x + rng.normal(0, 1, n)
    df = pd.DataFrame({"latitude": lat, "longitude": lon, "y": y, "x": x})
    csv = tmp_path / "sar.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    assert (out / "spatial_coefficients.csv").exists()
    assert (out / "diagnostics.txt").exists()
    # spatial dependence detected; covariate effect recovered with correct sign
    # (estimates["x"] is the SEM marginal coef or the SAR total impact)
    assert res.estimates["x"] > 0.5
    assert res.estimates["sar_rho"] > 0.1
    assert res.estimates["resid_moran_p"] < 0.05


def test_spatial_regression_precondition_unmet(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"y": rng.normal(0, 1, 40), "x": rng.normal(0, 1, 40)})  # no geo
    csv = tmp_path / "nogeo.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)

    assert not ok
    assert any("地理" in u or "经纬度" in u for u in unmet)
