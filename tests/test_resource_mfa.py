"""Tests for MATERIAL FLOW ANALYSIS (MFA) — resource family / sustainability.

Mass-balance accounting of physical resource flows: inputs = outputs + net stock
addition. Known-value cases are hand-computed in the docstrings; the honest-degrade
path asserts the Chinese "跳过" message and no crash.
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
        id="material_flow_analysis",
        method="Material Flow Analysis",
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
# (a) known inputs/outputs by config -> totals hand-computed
# --------------------------------------------------------------------------- #
def test_mfa_config_known_values(tmp_path: Path) -> None:
    """extraction=[100,50], imports=[30,20] (inputs);
       emission=[40,10], waste=[20,30] (outputs).
         total_input  = (100+50)+(30+20) = 150+50 = 200  (= throughput)
         total_output = (40+10)+(20+30)  = 50+50  = 100
         net_stock_addition = 200-100 = 100
         balance_ratio = 100/200 = 0.5"""
    csv = _csv(tmp_path, "mfa.csv", pd.DataFrame({
        "extraction": [100.0, 50.0],
        "imports": [30.0, 20.0],
        "emission": [40.0, 10.0],
        "waste": [20.0, 30.0],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"input_flows": ["extraction", "imports"],
                               "output_flows": ["emission", "waste"]})
    e = res.estimates
    assert math.isclose(e["total_input"], 200.0, abs_tol=1e-9)
    assert math.isclose(e["total_output"], 100.0, abs_tol=1e-9)
    assert math.isclose(e["net_stock_addition"], 100.0, abs_tol=1e-9)
    assert math.isclose(e["throughput"], 200.0, abs_tol=1e-9)
    assert math.isclose(e["balance_ratio"], 0.5, abs_tol=1e-9)
    assert e["n_input_flows"] == 2.0
    assert e["n_output_flows"] == 2.0
    # no product/recycled flag -> NaN
    assert math.isnan(e["resource_efficiency"])
    assert math.isnan(e["recycling_rate"])
    out = Path(res.output_dir)
    assert (out / "mfa_flows.csv").exists()
    tbl = pd.read_csv(out / "mfa_flows.csv")
    assert {"flow", "type", "value", "share_pct"}.issubset(tbl.columns)
    # extraction share of inputs = 150/200 = 75%
    ex = tbl[tbl["flow"] == "extraction"].iloc[0]
    assert ex["type"] == "input"
    assert math.isclose(float(ex["share_pct"]), 75.0, abs_tol=1e-4)


# --------------------------------------------------------------------------- #
# (b) name-based classification picks the right columns
# --------------------------------------------------------------------------- #
def test_mfa_name_based_classification(tmp_path: Path) -> None:
    """water_intake=[60,40] ('intake' -> input), water_discharge=[30,20]
       ('discharge' -> output).
         total_input  = 100, total_output = 50, NAS = 50."""
    csv = _csv(tmp_path, "water.csv", pd.DataFrame({
        "water_intake": [60.0, 40.0],
        "water_discharge": [30.0, 20.0],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    e = res.estimates
    assert math.isclose(e["total_input"], 100.0, abs_tol=1e-9)
    assert math.isclose(e["total_output"], 50.0, abs_tol=1e-9)
    assert math.isclose(e["net_stock_addition"], 50.0, abs_tol=1e-9)
    assert e["n_input_flows"] == 1.0
    assert e["n_output_flows"] == 1.0


# --------------------------------------------------------------------------- #
# (c) recycling_rate = recycled / total_input, hand-computed
# --------------------------------------------------------------------------- #
def test_mfa_recycling_rate(tmp_path: Path) -> None:
    """virgin_supply=[80,20] ('supply' -> input), recycled_use=[40,60]
       ('use' -> input), product_out=[50,50] ('output'/'product' -> output).
         total_input = (80+20)+(40+60) = 100+100 = 200
         recycled_flow=recycled_use -> recycling_rate = 100/200 = 0.5."""
    csv = _csv(tmp_path, "rec.csv", pd.DataFrame({
        "virgin_supply": [80.0, 20.0],
        "recycled_use": [40.0, 60.0],
        "product_out": [50.0, 50.0],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"recycled_flow": "recycled_use",
                               "product_flow": "product_out"})
    e = res.estimates
    assert math.isclose(e["total_input"], 200.0, abs_tol=1e-9)
    assert math.isclose(e["recycling_rate"], 0.5, abs_tol=1e-9)
    # product_flow -> resource efficiency = product_out total / total_input = 100/200 = 0.5
    assert math.isclose(e["resource_efficiency"], 0.5, abs_tol=1e-9)
    assert "再生" in res.summary or "recycling" in res.summary.lower()


# --------------------------------------------------------------------------- #
# (d) honest degrade when no output flow identifiable
# --------------------------------------------------------------------------- #
def test_mfa_degrade_no_output(tmp_path: Path) -> None:
    """Two input-named columns, no output -> cannot balance -> honest 跳过, no crash."""
    csv = _csv(tmp_path, "noout.csv", pd.DataFrame({
        "extraction": [10.0, 20.0],
        "import_qty": [5.0, 5.0],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
    assert "total_input" not in res.estimates


def test_mfa_degrade_no_numeric(tmp_path: Path) -> None:
    """No numeric flow column -> honest 跳过, no crash."""
    csv = _csv(tmp_path, "txt.csv", pd.DataFrame({"label": ["a", "b", "c"]}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
    assert "total_input" not in res.estimates


# --------------------------------------------------------------------------- #
# NaN handling: blank flow cells treated as 0 (sparse-table convention)
# --------------------------------------------------------------------------- #
def test_mfa_nan_treated_as_zero(tmp_path: Path) -> None:
    """input_flow=[10, NaN] -> sum 10; output_flow=[NaN, 4] -> sum 4.
       NAS = 10 - 4 = 6 (NaN cells counted as 0, not dropped)."""
    csv = _csv(tmp_path, "nan.csv", pd.DataFrame({
        "input_supply": [10.0, float("nan")],
        "output_waste": [float("nan"), 4.0],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"input_flows": ["input_supply"],
                               "output_flows": ["output_waste"]})
    e = res.estimates
    assert math.isclose(e["total_input"], 10.0, abs_tol=1e-9)
    assert math.isclose(e["total_output"], 4.0, abs_tol=1e-9)
    assert math.isclose(e["net_stock_addition"], 6.0, abs_tol=1e-9)
