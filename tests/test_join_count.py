"""Tests for join_count: BB/WW/BW join counts for a binary spatial field.

A spatially clumped binary field (1s concentrated in one corner) should produce
more BB (1-1) joins than free-sampling expects -> positive BB z. A spatially
random binary field should not.
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
        id="join_count",
        method="Join-count statistics (binary spatial autocorrelation)",
        domain="gis",
        family="spatial",
        goal="explain",
        preconditions=Precondition(requires_geo=True, min_categorical_cols=1, min_rows=20),
    )


def test_join_count_detects_like_clustering(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    # regular-ish grid of points so neighbours are well defined
    gx, gy = np.meshgrid(np.linspace(0, 11, 12), np.linspace(0, 11, 12))
    lon = gx.ravel() + rng.normal(0, 0.05, gx.size)
    lat = gy.ravel() + rng.normal(0, 0.05, gy.size)
    # binary "presence": 1 in the lower-left quadrant (spatial clump), else 0
    present = ((gx.ravel() < 5.5) & (gy.ravel() < 5.5)).astype(int)
    df = pd.DataFrame({"longitude": lon, "latitude": lat, "present": present})
    csv = tmp_path / "clumped.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)
    tab = pd.read_csv(out / "join_count.csv")

    assert {"join_type", "observed", "expected", "z", "p_value"} <= set(tab.columns)
    assert res.estimates["bb_count"] > res.estimates["bb_expected"]
    assert res.estimates["bb_z"] > 1.96  # significant like-clustering
    assert res.estimates["bb_p"] < 0.05
    assert (out / "join_count.png").exists()


def test_join_count_random_binary_not_significant(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    gx, gy = np.meshgrid(np.linspace(0, 11, 12), np.linspace(0, 11, 12))
    lon = gx.ravel() + rng.normal(0, 0.05, gx.size)
    lat = gy.ravel() + rng.normal(0, 0.05, gy.size)
    present = rng.integers(0, 2, gx.size)  # spatially random labels
    df = pd.DataFrame({"longitude": lon, "latitude": lat, "present": present})
    csv = tmp_path / "random.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    # random labels -> BB z should not be a strong positive
    assert res.estimates["bb_z"] < 1.96


def test_join_count_non_binary_degrades(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    n = 40
    df = pd.DataFrame(
        {
            "longitude": rng.uniform(0, 10, n),
            "latitude": rng.uniform(0, 10, n),
            "cont": rng.normal(0, 1, n),  # continuous, not binary
        }
    )
    csv = tmp_path / "cont.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    # no binary attribute -> honest skip, no crash, no counts
    assert "跳过" in res.summary
    assert "bb_count" not in res.estimates


def test_join_count_precondition_unmet(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"a": rng.normal(0, 1, 30), "b": rng.normal(0, 1, 30)})
    csv = tmp_path / "nogeo.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)
    assert not ok
    assert any("地理" in u or "经纬度" in u for u in unmet)
