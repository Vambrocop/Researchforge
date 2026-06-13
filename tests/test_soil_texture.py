"""Tests for soil_texture: USDA texture classification + the classify helper."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.executor.run import _usda_texture
from researchforge.profiler import profile_dataset
from researchforge.recommender.match import check_preconditions


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="soil_texture",
        method="USDA soil texture classification (质地三角)",
        domain="soil",
        family="soil",
        goal="describe",
        preconditions=Precondition(requires_soil_texture=True, min_rows=1),
    )


def test_usda_texture_reference_points() -> None:
    # canonical reference points on the USDA triangle (sand, silt, clay)
    assert _usda_texture(92, 5, 3) == "sand"
    assert _usda_texture(80, 12, 8) == "loamy sand"
    assert _usda_texture(65, 25, 10) == "sandy loam"
    assert _usda_texture(40, 40, 20) == "loam"
    assert _usda_texture(20, 60, 20) == "silt loam"
    assert _usda_texture(10, 85, 5) == "silt"
    assert _usda_texture(50, 15, 35) == "sandy clay"
    assert _usda_texture(30, 30, 40) == "clay"
    assert _usda_texture(10, 50, 40) == "silty clay"
    assert _usda_texture(55, 20, 25) == "sandy clay loam"
    assert _usda_texture(33, 32, 35) == "clay loam"
    assert _usda_texture(8, 57, 35) == "silty clay loam"


def test_usda_texture_is_complete() -> None:
    # every valid integer (sand,silt,clay) summing to 100 must map to a class
    # (brute-force completeness guard — no fall-through to "unclassified")
    bad = [
        (100 - clay - silt, silt, clay)
        for clay in range(101)
        for silt in range(101 - clay + 1)
        if (100 - clay - silt) >= 0
        and _usda_texture(100 - clay - silt, silt, clay) == "unclassified"
    ]
    assert bad == []


def test_soil_texture_executor(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "sand_pct": [92.0, 40.0, 10.0, 30.0, 50.0],
            "silt_pct": [5.0, 40.0, 85.0, 30.0, 15.0],
            "clay_pct": [3.0, 20.0, 5.0, 40.0, 35.0],
        }
    )
    csv = tmp_path / "soil.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)
    tab = pd.read_csv(out / "soil_texture.csv")

    assert "usda_texture" in tab.columns
    assert list(tab["usda_texture"]) == ["sand", "loam", "silt", "clay", "sandy clay"]
    assert res.estimates["n_samples"] == 5


def test_soil_texture_precondition_unmet(tmp_path: Path) -> None:
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})  # no sand/silt/clay cols
    csv = tmp_path / "nope.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)

    assert not ok
    assert any("sand" in u or "黏粒" in u for u in unmet)
