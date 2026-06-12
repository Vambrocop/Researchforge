"""Tests for moran_i: geo gate + spatial autocorrelation with permutation p."""

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
        id="moran_i",
        method="Moran's I (spatial autocorrelation)",
        domain="gis",
        family="spatial",
        goal="explain",
        preconditions=Precondition(requires_geo=True, min_continuous=1, min_rows=20),
    )


def test_moran_i_detects_clustering(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 120
    lat = rng.uniform(30, 40, n)
    lon = rng.uniform(-100, -90, n)
    # value is a smooth function of position -> strong positive spatial autocorr
    value = 2.0 * lat + 1.5 * lon + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"latitude": lat, "longitude": lon, "yield": value})
    csv = tmp_path / "spatial.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    assert sum(1 for c in fp.columns if c.kind == "geo") == 2

    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    assert (out / "moran.txt").exists()
    assert res.estimates["moran_i"] > res.estimates["expected_i"]
    assert res.estimates["moran_i"] > 0.1  # clear positive clustering
    assert res.estimates["p_value"] < 0.05  # significant


def test_moran_i_null_is_not_significant(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n = 120
    df = pd.DataFrame(
        {
            "latitude": rng.uniform(30, 40, n),
            "longitude": rng.uniform(-100, -90, n),
            "yield": rng.normal(0, 1, n),  # value independent of location
        }
    )
    csv = tmp_path / "random.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert res.estimates["p_value"] > 0.05  # no spurious autocorrelation


def test_moran_i_precondition_unmet(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"a": rng.normal(0, 1, 30), "b": rng.normal(0, 1, 30)})  # no geo cols
    csv = tmp_path / "nogeo.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)

    assert not ok
    assert any("地理" in u or "经纬度" in u for u in unmet)
