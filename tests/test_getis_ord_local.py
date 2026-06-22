"""Tests for getis_ord (Gi* local hotspots + global G).

NOTE: the id is `getis_ord` (this family's pure-Python Gi* + global-G method).
A separate, pre-existing `tests/test_getis_ord.py` covers the older `getis_ord_gi`
analysis — this file is deliberately named differently so neither clobbers the
other. A planted high-value cluster should be flagged as a hotspot and drive a
significant global G.
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
        id="getis_ord",
        method="Getis-Ord Gi* hotspots + global G",
        domain="gis",
        family="spatial",
        goal="explain",
        preconditions=Precondition(requires_geo=True, min_continuous=1, min_rows=20),
    )


def test_getis_ord_finds_hotspot_and_global_g(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 150
    lat = rng.uniform(30, 40, n)
    lon = rng.uniform(-100, -90, n)
    # inject a high-value cluster near (39, -91). pollution is NON-NEGATIVE (a
    # concentration), which is the domain global G requires; the background base 5
    # keeps it positive while the +8 cluster drives a significant global G.
    near = (lat > 38.5) & (lon > -91.5)
    value = np.clip(5.0 + rng.normal(0, 1, n) + near * 8.0, 0.0, None)
    df = pd.DataFrame({"latitude": lat, "longitude": lon, "pollution": value})
    csv = tmp_path / "spatial.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)
    tab = pd.read_csv(out / "getis_ord_local.csv")

    assert {"x", "y", "value", "gi_z", "class"} <= set(tab.columns)
    assert res.estimates["n_hotspots"] >= 1
    assert res.estimates["max_gi_z"] > 1.96
    assert res.estimates["n"] == float(n)
    # high-value clustering -> significant positive global G
    assert np.isfinite(res.estimates["global_g_z"])
    assert res.estimates["global_g_z"] > 1.96
    assert res.estimates["global_g_p"] < 0.05
    # the detected hotspots sit inside the injected high-value region
    hot = tab[tab["class"] == "hotspot"]
    assert (hot["y"] > 38.0).mean() > 0.5
    assert (out / "getis_ord_map.png").exists()


def test_getis_ord_small_n_finite(tmp_path: Path) -> None:
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
    tab = pd.read_csv(Path(res.output_dir) / "getis_ord_local.csv")
    assert np.isfinite(tab["gi_z"]).all()  # no inf/NaN from a zero denominator


def test_getis_ord_precondition_unmet(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(0, 1, 30), "b": rng.normal(0, 1, 30)})
    csv = tmp_path / "nogeo.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)
    assert not ok
    assert any("地理" in u or "经纬度" in u for u in unmet)
