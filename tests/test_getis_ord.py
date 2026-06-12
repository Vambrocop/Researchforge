"""Tests for getis_ord_gi: geo gate + Gi* hotspot detection."""

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
        id="getis_ord_gi",
        method="Getis-Ord Gi* (hotspot analysis)",
        domain="gis",
        family="spatial",
        goal="explain",
        preconditions=Precondition(requires_geo=True, min_continuous=1, min_rows=20),
    )


def test_getis_ord_finds_hotspot(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 150
    lat = rng.uniform(30, 40, n)
    lon = rng.uniform(-100, -90, n)
    # inject a high-value cluster near (39, -91): those points get a big bump
    near = (lat > 38.5) & (lon > -91.5)
    value = rng.normal(0, 1, n) + near * 8.0
    df = pd.DataFrame({"latitude": lat, "longitude": lon, "pollution": value})
    csv = tmp_path / "spatial.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)
    tab = pd.read_csv(out / "getis_ord.csv")

    assert "gi_star" in tab.columns
    assert res.estimates["n_hotspots"] >= 1  # the injected cluster is detected
    assert res.estimates["max_gi"] > 1.96
    # hotspots should sit inside the injected high-value region
    hot = tab[tab["class"] == "hotspot"]
    assert (hot["latitude"] > 38.0).mean() > 0.5


def test_getis_ord_small_n_no_divzero(tmp_path: Path) -> None:
    # n just above the n<10 floor: k=min(8,n-2) must keep the Gi* variance > 0
    rng = np.random.default_rng(7)
    n = 12
    df = pd.DataFrame(
        {
            "latitude": rng.uniform(30, 40, n),
            "longitude": rng.uniform(-100, -90, n),
            "z": rng.normal(0, 1, n),
        }
    )
    csv = tmp_path / "small.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    tab = pd.read_csv(Path(res.output_dir) / "getis_ord.csv")
    assert np.isfinite(tab["gi_star"]).all()  # no inf/NaN from a zero denominator


def test_getis_ord_precondition_unmet(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(0, 1, 30), "b": rng.normal(0, 1, 30)})  # no geo cols
    csv = tmp_path / "nogeo.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)

    assert not ok
    assert any("地理" in u or "经纬度" in u for u in unmet)
