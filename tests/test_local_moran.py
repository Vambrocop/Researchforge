"""Tests for local_moran (LISA): geo gate + per-location cluster classification."""

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
        id="local_moran",
        method="Local Moran's I (LISA)",
        domain="gis",
        family="spatial",
        goal="explain",
        preconditions=Precondition(requires_geo=True, min_continuous=1, min_rows=20),
    )


def test_local_moran_finds_hh_cluster(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 150
    lat = rng.uniform(30, 40, n)
    lon = rng.uniform(-100, -90, n)
    # high-value cluster in one corner -> a high-high (HH) LISA cluster there
    near = (lat > 38.5) & (lon > -91.5)
    value = rng.normal(0, 1, n) + near * 8.0
    df = pd.DataFrame({"latitude": lat, "longitude": lon, "income": value})
    csv = tmp_path / "spatial.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)
    tab = pd.read_csv(out / "lisa.csv")

    assert set(["local_I", "p_value", "cluster"]).issubset(tab.columns)
    assert res.estimates["n_HH"] >= 1  # the injected high cluster shows up as HH
    hh = tab[tab["cluster"] == "HH"]
    assert (hh["latitude"] > 38.0).mean() > 0.5  # HH points sit in the injected region


def test_local_moran_precondition_unmet(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(0, 1, 30), "b": rng.normal(0, 1, 30)})  # no geo cols
    csv = tmp_path / "nogeo.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)

    assert not ok
    assert any("地理" in u or "经纬度" in u for u in unmet)
