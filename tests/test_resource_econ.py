"""Tests for the RESOURCE-ECONOMICS bridge (cost <-> physical/resource units),
the decision-economics piece of the Energy-Water-Food nexus toolkit:
cost_effectiveness_analysis (CEA / ICER) and marginal_abatement_cost (MACC).

Every small case is HAND-COMPUTED in the test docstring. Honest-degrade paths
assert the Chinese "跳过" message and no crash. Mirrors test_techno_economic.py.

NOTE (extended dominance, CEA): extended (weak) dominance is iterative — removing
one option changes another's ICER — so we implement the standard textbook loop
(recompute consecutive ICERs after each removal until they are non-decreasing).
test_cea_extended_dominance below exercises a case where this matters.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry(eid: str, method: str) -> AnalysisEntry:
    return AnalysisEntry(
        id=eid,
        method=method,
        domain="economics",
        family="resource",
        goal="evaluate",
        preconditions=Precondition(min_rows=2),
    )


def _csv(tmp_path: Path, name: str, df: pd.DataFrame) -> Path:
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


# --------------------------------------------------------------------------- #
# 1) cost_effectiveness_analysis  (CEA / ICER)
# --------------------------------------------------------------------------- #
def test_cea_strong_dominance_flagged(tmp_path: Path) -> None:
    """Options A(cost=10, effect=1), B(cost=20, effect=1), C(cost=15, effect=2).
       B is STRONGLY dominated by A (A is cheaper AND not-less effective: same
       effect, lower cost). A and C survive on the frontier.
       ACER: A=10/1=10, B=20/1=20, C=15/2=7.5 -> best_acer=7.5."""
    csv = _csv(tmp_path, "cea.csv", pd.DataFrame({
        "option": ["A", "B", "C"],
        "cost": [10.0, 20.0, 15.0],
        "effect": [1.0, 1.0, 2.0],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("cost_effectiveness_analysis", "CEA"),
                       output_root=str(tmp_path / "o"),
                       config={"cost": "cost", "effect": "effect", "option": "option"})
    e = res.estimates
    assert e["n_options"] == 3.0
    assert e["n_dominated"] >= 1.0
    assert math.isclose(e["best_acer"], 7.5, abs_tol=1e-9)
    out = Path(res.output_dir)
    tbl = pd.read_csv(out / "cost_effectiveness.csv")
    statuses = dict(zip(tbl["option"].astype(str), tbl["status"]))
    assert statuses["B"] == "dominated"
    assert statuses["A"] == "frontier"
    assert statuses["C"] == "frontier"


def test_cea_icer_between_frontier_options(tmp_path: Path) -> None:
    """Two frontier options A(cost=10, effect=1), B(cost=30, effect=3).
       ICER(B vs A) = (30-10)/(3-1) = 20/2 = 10.0.  A is the baseline (ICER nan)."""
    csv = _csv(tmp_path, "cea.csv", pd.DataFrame({
        "option": ["A", "B"],
        "cost": [10.0, 30.0],
        "effect": [1.0, 3.0],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("cost_effectiveness_analysis", "CEA"),
                       output_root=str(tmp_path / "o"),
                       config={"cost": "cost", "effect": "effect", "option": "option"})
    assert math.isclose(res.estimates["icer__B"], 10.0, abs_tol=1e-9)
    # A is the least-effective frontier option -> no incremental ICER reported
    assert "icer__A" not in res.estimates
    assert res.estimates["n_dominated"] == 0.0


def test_cea_extended_dominance(tmp_path: Path) -> None:
    """Extended (weak) dominance, hand-computed.
       A(cost=10, effect=2), B(cost=20, effect=3), C(cost=22, effect=4).
       Sorted by effect: A, B, C. Consecutive ICERs:
         A->B = (20-10)/(3-2) = 10
         B->C = (22-20)/(4-3) =  2
       ICERs are NOT non-decreasing (10 then 2) -> B is EXTENDED-dominated.
       Remove B and recompute: A->C = (22-10)/(4-2) = 6 (single segment, monotone).
       Frontier = {A, C}; A baseline; ICER(C vs A) = 6.0; B = extended_dominated."""
    csv = _csv(tmp_path, "cea.csv", pd.DataFrame({
        "option": ["A", "B", "C"],
        "cost": [10.0, 20.0, 22.0],
        "effect": [2.0, 3.0, 4.0],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("cost_effectiveness_analysis", "CEA"),
                       output_root=str(tmp_path / "o"),
                       config={"cost": "cost", "effect": "effect", "option": "option"})
    out = Path(res.output_dir)
    tbl = pd.read_csv(out / "cost_effectiveness.csv")
    statuses = dict(zip(tbl["option"].astype(str), tbl["status"]))
    assert statuses["B"] == "extended_dominated"
    assert statuses["A"] == "frontier"
    assert statuses["C"] == "frontier"
    # ICER on the recomputed frontier A->C = 6.0
    assert math.isclose(res.estimates["icer__C"], 6.0, abs_tol=1e-9)
    # B is not on the frontier -> no incremental ICER for it
    assert "icer__B" not in res.estimates


def test_cea_wtp_picks_optimal_and_nmb(tmp_path: Path) -> None:
    """A(cost=10, effect=1), B(cost=30, effect=3). ICER(B vs A)=10.
       WTP=15 per unit effect: 10 <= 15, so the more-effective B is optimal.
       NMB = wtp*effect - cost: NMB_A = 15*1-10 = 5; NMB_B = 15*3-30 = 15 (>0)."""
    csv = _csv(tmp_path, "cea.csv", pd.DataFrame({
        "option": ["A", "B"],
        "cost": [10.0, 30.0],
        "effect": [1.0, 3.0],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("cost_effectiveness_analysis", "CEA"),
                       output_root=str(tmp_path / "o"),
                       config={"cost": "cost", "effect": "effect",
                               "option": "option", "wtp": 15})
    e = res.estimates
    assert math.isclose(e["nmb__A"], 5.0, abs_tol=1e-9)
    assert math.isclose(e["nmb__B"], 15.0, abs_tol=1e-9)
    # optimal option = B -> its effect/cost recorded
    assert math.isclose(e["optimal_effect"], 3.0, abs_tol=1e-9)
    assert math.isclose(e["optimal_cost"], 30.0, abs_tol=1e-9)


def test_cea_low_wtp_picks_baseline(tmp_path: Path) -> None:
    """Same options; WTP=5 < ICER(B vs A)=10, so the incremental step to B is NOT
       worth it -> optimal stays the baseline A (effect=1, cost=10)."""
    csv = _csv(tmp_path, "cea.csv", pd.DataFrame({
        "option": ["A", "B"],
        "cost": [10.0, 30.0],
        "effect": [1.0, 3.0],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("cost_effectiveness_analysis", "CEA"),
                       output_root=str(tmp_path / "o"),
                       config={"cost": "cost", "effect": "effect",
                               "option": "option", "wtp": 5})
    e = res.estimates
    assert math.isclose(e["optimal_effect"], 1.0, abs_tol=1e-9)
    assert math.isclose(e["optimal_cost"], 10.0, abs_tol=1e-9)


def test_cea_degrade_one_numeric(tmp_path: Path) -> None:
    """Only a label + a single numeric column -> can't resolve a separate effect
       column -> honest 跳过, no crash, no estimates written."""
    csv = _csv(tmp_path, "cea.csv", pd.DataFrame({
        "option": ["A", "B", "C"],
        "cost": [10.0, 20.0, 30.0],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("cost_effectiveness_analysis", "CEA"),
                       output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
    assert "n_options" not in res.estimates


# --------------------------------------------------------------------------- #
# 2) marginal_abatement_cost  (MACC)
# --------------------------------------------------------------------------- #
def test_macc_sorted_ascending_by_mac(tmp_path: Path) -> None:
    """Measures M1(cost=100, ab=10)->MAC 10, M2(cost=20, ab=10)->MAC 2,
       M3(cost=60, ab=10)->MAC 6. Ranked ascending by MAC: M2(2) < M3(6) < M1(10).
       Cumulative abatement along the curve: 10, 20, 30. total_abatement=30."""
    csv = _csv(tmp_path, "macc.csv", pd.DataFrame({
        "measure": ["M1", "M2", "M3"],
        "cost": [100.0, 20.0, 60.0],
        "abatement": [10.0, 10.0, 10.0],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("marginal_abatement_cost", "MACC"),
                       output_root=str(tmp_path / "o"),
                       config={"cost": "cost", "abatement": "abatement",
                               "measure": "measure"})
    out = Path(res.output_dir)
    tbl = pd.read_csv(out / "macc_curve.csv")
    assert tbl["measure"].astype(str).tolist() == ["M2", "M3", "M1"]
    # MAC values in order
    assert [round(v, 6) for v in tbl["mac"].tolist()] == [2.0, 6.0, 10.0]
    # cumulative abatement is non-decreasing and ends at the total
    assert tbl["cumulative_abatement"].tolist() == [10.0, 20.0, 30.0]
    assert math.isclose(res.estimates["total_abatement"], 30.0, abs_tol=1e-9)
    assert math.isclose(res.estimates["mac__M2"], 2.0, abs_tol=1e-9)


def test_macc_no_regret_flagged(tmp_path: Path) -> None:
    """A negative-cost measure is 'no-regret'. Mneg(cost=-50, ab=10)->MAC -5,
       Mpos(cost=40, ab=10)->MAC 4. n_no_regret=1; Mneg flagged no_regret=True.
       total_cost = -50 + 40 = -10."""
    csv = _csv(tmp_path, "macc.csv", pd.DataFrame({
        "measure": ["Mneg", "Mpos"],
        "cost": [-50.0, 40.0],
        "abatement": [10.0, 10.0],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("marginal_abatement_cost", "MACC"),
                       output_root=str(tmp_path / "o"),
                       config={"cost": "cost", "abatement": "abatement",
                               "measure": "measure"})
    assert res.estimates["n_no_regret"] == 1.0
    assert math.isclose(res.estimates["total_cost"], -10.0, abs_tol=1e-9)
    assert math.isclose(res.estimates["mac__Mneg"], -5.0, abs_tol=1e-9)
    out = Path(res.output_dir)
    tbl = pd.read_csv(out / "macc_curve.csv")
    nr = dict(zip(tbl["measure"].astype(str), tbl["no_regret"]))
    assert bool(nr["Mneg"]) is True
    assert bool(nr["Mpos"]) is False


def test_macc_abatement_below_price(tmp_path: Path) -> None:
    """MAC values 2, 6, 10 (abatements 10, 10, 10). carbon_price=6:
       measures with MAC<=6 are MAC=2 and MAC=6 -> abatement_below_price = 10+10 = 20.
       cost_below_price = 20 + 60 = 80 (the costs of those two measures)."""
    csv = _csv(tmp_path, "macc.csv", pd.DataFrame({
        "measure": ["M1", "M2", "M3"],
        "cost": [100.0, 20.0, 60.0],   # MAC 10, 2, 6
        "abatement": [10.0, 10.0, 10.0],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("marginal_abatement_cost", "MACC"),
                       output_root=str(tmp_path / "o"),
                       config={"cost": "cost", "abatement": "abatement",
                               "measure": "measure", "carbon_price": 6})
    assert math.isclose(res.estimates["abatement_below_price"], 20.0, abs_tol=1e-9)
    assert math.isclose(res.estimates["cost_below_price"], 80.0, abs_tol=1e-9)


def test_macc_drops_nonpositive_abatement(tmp_path: Path) -> None:
    """A measure with abatement <= 0 has no definable per-unit cost -> dropped with
       disclosure. M1(ab=10), M2(ab=0), M3(ab=10): only M1, M3 survive (n_measures=2)."""
    csv = _csv(tmp_path, "macc.csv", pd.DataFrame({
        "measure": ["M1", "M2", "M3"],
        "cost": [20.0, 5.0, 60.0],
        "abatement": [10.0, 0.0, 10.0],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("marginal_abatement_cost", "MACC"),
                       output_root=str(tmp_path / "o"),
                       config={"cost": "cost", "abatement": "abatement",
                               "measure": "measure"})
    assert res.estimates["n_measures"] == 2.0
    assert "mac__M2" not in res.estimates
    assert "剔除" in res.summary  # disclosure of the dropped measure


def test_macc_degrade_no_abatement_col(tmp_path: Path) -> None:
    """Only a label + a single numeric (cost) column -> no separate abatement
       column to resolve -> honest 跳过, no crash, no estimates written."""
    csv = _csv(tmp_path, "macc.csv", pd.DataFrame({
        "measure": ["M1", "M2", "M3"],
        "cost": [10.0, 20.0, 30.0],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("marginal_abatement_cost", "MACC"),
                       output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
    assert "n_measures" not in res.estimates
