"""Tests for LIFE-CYCLE COST (LCC / total cost of ownership) - techno_economic family.

LCC = capex + sum opex_t/(1+r)^t - salvage/(1+r)^N ;
EAC = LCC x CRF , CRF = r(1+r)^N/((1+r)^N-1) (r=0 -> 1/N).
Known-value cases are hand-computed in the docstrings; the honest-degrade path
asserts the Chinese "跳过" message and no crash.
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
        id="life_cycle_cost",
        method="Life-cycle cost (LCC)",
        domain="techno_economic",
        family="techno_economic",
        goal="describe",
        preconditions=Precondition(min_rows=1),
    )


def _csv(tmp_path: Path, name: str, df: pd.DataFrame) -> Path:
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


# --------------------------------------------------------------------------- #
# (a) known capex + opex + rate -> LCC and EAC hand-computed
# --------------------------------------------------------------------------- #
def test_lcc_known_value(tmp_path: Path) -> None:
    """capex=1000, opex=100/yr for 3 yr, r=0.10, no salvage:
         LCC = 1000 + 100/1.1 + 100/1.21 + 100/1.331
             = 1000 + 90.90909 + 82.64463 + 75.13148 = 1248.68520
         CRF = 0.1*1.1^3 / (1.1^3 - 1) = 0.1331/0.331 = 0.40211480
         EAC = LCC*CRF = 1248.68520 * 0.40211480 = 502.11416..."""
    csv = _csv(tmp_path, "lcc.csv", pd.DataFrame({"opex": [100.0, 100.0, 100.0]}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"capex": 1000, "opex": "opex", "rate": 0.10})
    e = res.estimates
    assert math.isclose(e["lcc"], 1248.68520, rel_tol=0, abs_tol=1e-3)
    assert math.isclose(e["capex"], 1000.0, abs_tol=1e-9)
    assert math.isclose(e["n_periods"], 3.0, abs_tol=1e-9)
    assert math.isclose(e["rate"], 0.10, abs_tol=1e-12)
    # opex_pv = LCC - capex (no salvage) = 248.68520
    assert math.isclose(e["opex_pv"], 248.68520, abs_tol=1e-3)
    # EAC = LCC * CRF
    crf = 0.1 * 1.1 ** 3 / (1.1 ** 3 - 1)
    assert math.isclose(e["eac"], 1248.68520 * crf, rel_tol=0, abs_tol=1e-3)
    out = Path(res.output_dir)
    assert (out / "lcc_breakdown.csv").exists()
    tbl = pd.read_csv(out / "lcc_breakdown.csv")
    assert {"phase", "present_value", "share_pct"}.issubset(tbl.columns)


def test_lcc_from_cost_cashflow(tmp_path: Path) -> None:
    """cost cashflow column: row 0 = capex (1000), rows 1..3 = opex (100 each).
       Same as the known-value case -> LCC = 1248.68520."""
    csv = _csv(tmp_path, "cf.csv", pd.DataFrame({
        "cost": [1000.0, 100.0, 100.0, 100.0],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"cost": "cost", "rate": 0.10})
    e = res.estimates
    assert math.isclose(e["capex"], 1000.0, abs_tol=1e-9)
    assert math.isclose(e["lcc"], 1248.68520, rel_tol=0, abs_tol=1e-3)
    assert math.isclose(e["n_periods"], 3.0, abs_tol=1e-9)


# --------------------------------------------------------------------------- #
# (b) salvage reduces LCC
# --------------------------------------------------------------------------- #
def test_lcc_salvage_reduces(tmp_path: Path) -> None:
    """Same as (a) but salvage=200 at end of life N=3:
         salvage_pv = 200/1.331 = 150.26296
         LCC = 1248.68520 - 150.26296 = 1098.42224 (lower than no-salvage 1248.68520)."""
    csv = _csv(tmp_path, "lcc.csv", pd.DataFrame({"opex": [100.0, 100.0, 100.0]}))
    fp = profile_dataset(csv)
    base = run_analysis(fp, _entry(), output_root=str(tmp_path / "o0"),
                        config={"capex": 1000, "opex": "opex", "rate": 0.10})
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o1"),
                       config={"capex": 1000, "opex": "opex", "rate": 0.10, "salvage": 200})
    e = res.estimates
    assert math.isclose(e["salvage_pv"], 200.0 / 1.331, rel_tol=0, abs_tol=1e-3)
    assert math.isclose(e["lcc"], 1248.68520 - 200.0 / 1.331, rel_tol=0, abs_tol=1e-3)
    # salvage strictly reduces LCC
    assert res.estimates["lcc"] < base.estimates["lcc"]


# --------------------------------------------------------------------------- #
# (c) rate = 0 -> CRF = 1/N guard
# --------------------------------------------------------------------------- #
def test_lcc_zero_rate_crf_guard(tmp_path: Path) -> None:
    """r=0: no discounting -> LCC = 1000 + 100*3 = 1300 (no salvage);
       CRF guard = 1/N = 1/3 -> EAC = 1300/3 = 433.3333..."""
    csv = _csv(tmp_path, "lcc.csv", pd.DataFrame({"opex": [100.0, 100.0, 100.0]}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"capex": 1000, "opex": "opex", "rate": 0.0})
    e = res.estimates
    assert math.isclose(e["lcc"], 1300.0, abs_tol=1e-9)
    assert math.isclose(e["eac"], 1300.0 / 3.0, rel_tol=0, abs_tol=1e-6)


def test_lcc_annual_opex_scalar(tmp_path: Path) -> None:
    """Scalar annual_opex=100 repeated for life=3, capex=1000, r=0.10 -> LCC=1248.68520."""
    csv = _csv(tmp_path, "x.csv", pd.DataFrame({"dummy": [1.0]}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"capex": 1000, "annual_opex": 100, "life": 3, "rate": 0.10})
    e = res.estimates
    assert math.isclose(e["lcc"], 1248.68520, rel_tol=0, abs_tol=1e-3)
    assert math.isclose(e["n_periods"], 3.0, abs_tol=1e-9)


# --------------------------------------------------------------------------- #
# (d) honest degrade: no capex and no opex source
# --------------------------------------------------------------------------- #
def test_lcc_degrade_no_cost_source(tmp_path: Path) -> None:
    """No capex / opex / cost / annual_opex and no capex column -> honest 跳过."""
    csv = _csv(tmp_path, "txt.csv", pd.DataFrame({"label": ["a", "b"]}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
    assert "lcc" not in res.estimates
