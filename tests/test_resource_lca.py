"""Tests for FOOTPRINT ANALYSIS (LCA-style carbon/water/material footprint) -
resource family / sustainability.

footprint = sum(quantity x factor); contribution (hotspot) shares; intensity per
functional unit. Known-value cases are hand-computed in the docstrings; the
honest-degrade path asserts the Chinese "跳过" message and no crash.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="footprint_analysis",
        method="Footprint analysis",
        domain="sustainability",
        family="resource",
        goal="describe",
        preconditions=Precondition(min_rows=1),
    )


def _csv(tmp_path: Path, name: str, df: pd.DataFrame) -> Path:
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


# --------------------------------------------------------------------------- #
# (a) known quantities x factors via config factors -> total + shares
# --------------------------------------------------------------------------- #
def test_footprint_config_factors_known_values(tmp_path: Path) -> None:
    """electricity=[100,200] (sum 300), fuel=[50,50] (sum 100);
       factors {electricity:0.5, fuel:2.0}:
         footprint_electricity = 300*0.5 = 150
         footprint_fuel        = 100*2.0 = 200
         total                 = 350
         share_electricity = 150/350 = 0.428571..., share_fuel = 200/350 = 0.571429...
         top_contributor_share = 0.571429 (fuel is the hotspot)."""
    csv = _csv(tmp_path, "fp.csv", pd.DataFrame({
        "electricity": [100.0, 200.0],
        "fuel": [50.0, 50.0],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"factors": {"electricity": 0.5, "fuel": 2.0}})
    e = res.estimates
    assert math.isclose(e["total_footprint"], 350.0, abs_tol=1e-9)
    assert math.isclose(e["footprint__electricity"], 150.0, abs_tol=1e-9)
    assert math.isclose(e["footprint__fuel"], 200.0, abs_tol=1e-9)
    assert math.isclose(e["top_contributor_share"], 200.0 / 350.0, abs_tol=1e-6)  # estimate is 8dp-rounded
    assert math.isclose(e["n_activities"], 2.0, abs_tol=1e-9)
    # no functional unit given -> intensity is NaN
    assert math.isnan(e["intensity_per_unit"])
    out = Path(res.output_dir)
    assert (out / "footprint_contribution.csv").exists()
    tbl = pd.read_csv(out / "footprint_contribution.csv")
    assert {"activity", "quantity", "factor", "footprint", "share_pct"}.issubset(tbl.columns)
    # sorted descending by footprint (fuel first)
    assert tbl["footprint"].iloc[0] >= tbl["footprint"].iloc[-1]
    assert tbl["activity"].iloc[0] == "fuel"


# --------------------------------------------------------------------------- #
# (b) intensity per functional unit
# --------------------------------------------------------------------------- #
def test_footprint_intensity_per_functional_unit(tmp_path: Path) -> None:
    """Same factors; functional unit column units=[10,40] (sum 50):
         total footprint = 350 (as above)
         intensity = 350 / 50 = 7.0 per functional unit."""
    csv = _csv(tmp_path, "fp.csv", pd.DataFrame({
        "electricity": [100.0, 200.0],
        "fuel": [50.0, 50.0],
        "units": [10.0, 40.0],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"factors": {"electricity": 0.5, "fuel": 2.0},
                               "functional_unit": "units"})
    e = res.estimates
    assert math.isclose(e["total_footprint"], 350.0, abs_tol=1e-9)
    assert math.isclose(e["intensity_per_unit"], 7.0, abs_tol=1e-9)


# --------------------------------------------------------------------------- #
# (b2) long-format: activity / quantity / factor columns
# --------------------------------------------------------------------------- #
def test_footprint_long_format(tmp_path: Path) -> None:
    """Long table: activity rows with own quantity & factor.
         row a: 10 * 3 = 30 ; row b: 5 * 4 = 20 ; total = 50."""
    csv = _csv(tmp_path, "long.csv", pd.DataFrame({
        "act": ["a", "b"],
        "qty": [10.0, 5.0],
        "fac": [3.0, 4.0],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"activity": "act", "quantity": "qty", "factor": "fac"})
    e = res.estimates
    assert math.isclose(e["total_footprint"], 50.0, abs_tol=1e-9)
    assert math.isclose(e["footprint__a"], 30.0, abs_tol=1e-9)
    assert math.isclose(e["footprint__b"], 20.0, abs_tol=1e-9)
    assert math.isclose(e["top_contributor_share"], 30.0 / 50.0, abs_tol=1e-9)


# --------------------------------------------------------------------------- #
# (b3) factor_column x quantity column (per-row)
# --------------------------------------------------------------------------- #
def test_footprint_factor_column(tmp_path: Path) -> None:
    """Per-row factor column times a quantity column:
         row0: 10 * 2 = 20 ; row1: 20 * 1 = 20 ; total = 40."""
    csv = _csv(tmp_path, "fc.csv", pd.DataFrame({
        "amount": [10.0, 20.0],
        "ef": [2.0, 1.0],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"quantity": "amount", "factor_column": "ef"})
    e = res.estimates
    assert math.isclose(e["total_footprint"], 40.0, abs_tol=1e-9)
    assert math.isclose(e["n_activities"], 2.0, abs_tol=1e-9)


# --------------------------------------------------------------------------- #
# (c) degrade with no factors
# --------------------------------------------------------------------------- #
def test_footprint_degrade_no_factors(tmp_path: Path) -> None:
    """Numeric data but NO factor source -> honest 跳过 (factors never fabricated)."""
    csv = _csv(tmp_path, "x.csv", pd.DataFrame({
        "electricity": [100.0, 200.0],
        "fuel": [50.0, 50.0],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
    assert "total_footprint" not in res.estimates


def test_footprint_degrade_factors_no_match(tmp_path: Path) -> None:
    """factors dict naming columns that don't exist -> honest 跳过."""
    csv = _csv(tmp_path, "x.csv", pd.DataFrame({"electricity": [100.0, 200.0]}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"factors": {"nonexistent": 1.0}})
    assert "跳过" in res.summary
    assert "total_footprint" not in res.estimates
