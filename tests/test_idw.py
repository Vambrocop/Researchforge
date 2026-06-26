"""Tests for idw_interpolation: geo gate + IDW surface + LOO cross-validation."""

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
        id="idw_interpolation",
        method="IDW spatial interpolation",
        domain="soil",
        family="spatial",
        goal="predict",
        preconditions=Precondition(requires_geo=True, min_continuous=1, min_rows=15),
    )


def test_idw_interpolates_smooth_field(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 120
    lat = rng.uniform(30, 40, n)
    lon = rng.uniform(-100, -90, n)
    # a smooth spatial field -> IDW should reconstruct it with low LOO error
    ph = 6.0 + 0.1 * (lat - 35) + 0.05 * (lon + 95) + rng.normal(0, 0.05, n)
    df = pd.DataFrame({"latitude": lat, "longitude": lon, "soil_ph": ph})
    csv = tmp_path / "soil.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)
    surf = pd.read_csv(out / "idw_surface.csv")

    assert "soil_ph_idw" in surf.columns
    assert len(surf) == 60 * 60  # full grid written
    # IDW never extrapolates beyond the observed range
    assert surf["soil_ph_idw"].min() >= ph.min() - 1e-6
    assert surf["soil_ph_idw"].max() <= ph.max() + 1e-6
    # smooth field -> small cross-validation error
    assert res.estimates["loo_rmse"] < 0.5


def test_idw_precondition_unmet(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(0, 1, 30), "b": rng.normal(0, 1, 30)})  # no geo cols
    csv = tmp_path / "nogeo.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)

    assert not ok
    assert any("地理" in u or "经纬度" in u for u in unmet)
