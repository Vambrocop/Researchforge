"""Tests for the OPERATIONS-RESEARCH family (classic OR / management science):
eoq_inventory, queue_mmc, newsvendor.

Known-value cases are hand-computed in the docstrings; honest-degrade paths assert
the Chinese "跳过" message and no crash.
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
        domain="operations",
        family="operations_research",
        goal="optimize",
        preconditions=Precondition(min_rows=1),
    )


def _csv(tmp_path: Path, name: str, df: pd.DataFrame) -> Path:
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


# --------------------------------------------------------------------------- #
# 1) eoq_inventory
# --------------------------------------------------------------------------- #
def test_eoq_known_value(tmp_path: Path) -> None:
    """D=1000, S=50, H=2 -> Q* = sqrt(2*1000*50/2) = sqrt(50000) = 223.6068;
       n_orders = D/Q* = 1000/223.6068 = 4.4721;
       total ordering+holding cost = (D/Q*)S + (Q*/2)H = 223.607 + 223.607 = 447.21
       (at the EOQ ordering cost == holding cost)."""
    csv = _csv(tmp_path, "x.csv", pd.DataFrame({"dummy": [1.0, 2.0]}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("eoq_inventory", "EOQ"),
                       output_root=str(tmp_path / "o"),
                       config={"demand": 1000, "order_cost": 50, "holding_cost": 2})
    e = res.estimates
    assert math.isclose(e["eoq"], 223.6068, rel_tol=0, abs_tol=1e-3)
    assert math.isclose(e["n_orders"], 4.47214, rel_tol=0, abs_tol=1e-4)
    assert math.isclose(e["ordering_cost"], e["holding_cost"], rel_tol=0, abs_tol=1e-4)
    assert math.isclose(e["total_cost"], math.sqrt(2 * 1000 * 50 * 2), abs_tol=1e-2)
    out = Path(res.output_dir)
    assert (out / "eoq_cost_curve.csv").exists()


def test_eoq_reorder_point_lead_time(tmp_path: Path) -> None:
    """Lead-time demand: D=1000/yr, lead_time=0.1 yr -> d_L = 1000*0.1 = 100.
       No service_level/demand_sd -> no safety stock -> ROP = 100."""
    csv = _csv(tmp_path, "x.csv", pd.DataFrame({"dummy": [1.0, 2.0]}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("eoq_inventory", "EOQ"),
                       output_root=str(tmp_path / "o"),
                       config={"demand": 1000, "order_cost": 50, "holding_cost": 2,
                               "lead_time": 0.1})
    assert math.isclose(res.estimates["reorder_point"], 100.0, abs_tol=1e-6)
    assert math.isclose(res.estimates["safety_stock"], 0.0, abs_tol=1e-9)


def test_eoq_safety_stock(tmp_path: Path) -> None:
    """Service level 0.95 -> z = Phi^-1(0.95) = 1.6449; demand_sd=10 over lead time
       -> safety stock = 1.6449*10 = 16.449; ROP = d_L + safety = 100 + 16.449."""
    csv = _csv(tmp_path, "x.csv", pd.DataFrame({"dummy": [1.0, 2.0]}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("eoq_inventory", "EOQ"),
                       output_root=str(tmp_path / "o"),
                       config={"demand": 1000, "order_cost": 50, "holding_cost": 2,
                               "lead_time": 0.1, "service_level": 0.95, "demand_sd": 10})
    e = res.estimates
    assert math.isclose(e["z_value"], 1.6449, rel_tol=0, abs_tol=1e-3)
    assert math.isclose(e["safety_stock"], 1.6449 * 10.0, rel_tol=0, abs_tol=1e-2)
    assert math.isclose(e["reorder_point"], 100.0 + 1.6449 * 10.0, rel_tol=0, abs_tol=1e-2)


def test_eoq_purchase_cost(tmp_path: Path) -> None:
    """With unit_price=5: purchase cost = D*c = 1000*5 = 5000 added to total."""
    csv = _csv(tmp_path, "x.csv", pd.DataFrame({"dummy": [1.0, 2.0]}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("eoq_inventory", "EOQ"),
                       output_root=str(tmp_path / "o"),
                       config={"demand": 1000, "order_cost": 50, "holding_cost": 2,
                               "unit_price": 5})
    e = res.estimates
    assert math.isclose(e["purchase_cost"], 5000.0, abs_tol=1e-6)
    assert math.isclose(e["total_cost"], math.sqrt(2 * 1000 * 50 * 2) + 5000.0, abs_tol=1e-2)


def test_eoq_from_columns(tmp_path: Path) -> None:
    """Parameters read from same-named columns (first row)."""
    csv = _csv(tmp_path, "params.csv", pd.DataFrame({
        "demand": [1000.0], "order_cost": [50.0], "holding_cost": [2.0]}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("eoq_inventory", "EOQ"),
                       output_root=str(tmp_path / "o"))
    assert math.isclose(res.estimates["eoq"], 223.6068, abs_tol=1e-3)


def test_eoq_degrade_missing(tmp_path: Path) -> None:
    """No D/S/H -> honest 跳过, no crash."""
    csv = _csv(tmp_path, "x.csv", pd.DataFrame({"label": ["a", "b"]}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("eoq_inventory", "EOQ"),
                       output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
    assert "eoq" not in res.estimates


# --------------------------------------------------------------------------- #
# 2) queue_mmc
# --------------------------------------------------------------------------- #
def test_mm1_known_value(tmp_path: Path) -> None:
    """M/M/1 with lambda=0.5, mu=1, c=1: a=0.5, rho=0.5.
       P0 = 1/(1 + a/(1-rho)) = 1/(1 + 0.5/0.5) = 0.5.
       P_wait = (a/(1-rho))*P0 = 1*0.5 = 0.5.
       Lq = P_wait*rho/(1-rho) = 0.5*0.5/0.5 = 0.5.
       Wq = Lq/lambda = 0.5/0.5 = 1.0.
       L  = Lq + a = 1.0;  W = Wq + 1/mu = 2.0."""
    csv = _csv(tmp_path, "x.csv", pd.DataFrame({"dummy": [1.0, 2.0]}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("queue_mmc", "M/M/c"),
                       output_root=str(tmp_path / "o"),
                       config={"lambda": 0.5, "mu": 1.0, "servers": 1})
    e = res.estimates
    assert math.isclose(e["rho"], 0.5, abs_tol=1e-9)
    assert math.isclose(e["p0"], 0.5, abs_tol=1e-9)
    assert math.isclose(e["prob_wait"], 0.5, abs_tol=1e-9)
    assert math.isclose(e["lq"], 0.5, abs_tol=1e-9)
    assert math.isclose(e["wq"], 1.0, abs_tol=1e-9)
    assert math.isclose(e["l"], 1.0, abs_tol=1e-9)
    assert math.isclose(e["w"], 2.0, abs_tol=1e-9)
    out = Path(res.output_dir)
    assert (out / "queue_state_dist.csv").exists()


def test_mmc_multi_server(tmp_path: Path) -> None:
    """M/M/2 with lambda=1, mu=1, c=2: a=1, rho=0.5.
       P0 = 1/( (a^0/0! + a^1/1!) + a^2/(2!*(1-rho)) )
          = 1/( (1 + 1) + 1/(2*0.5) ) = 1/(2 + 1) = 1/3.
       P_wait = a^2/(2!*(1-rho)) * P0 = 1 * (1/3) = 1/3.
       Lq = P_wait*rho/(1-rho) = (1/3)*0.5/0.5 = 1/3.
       L  = Lq + a = 1/3 + 1 = 4/3."""
    csv = _csv(tmp_path, "x.csv", pd.DataFrame({"dummy": [1.0, 2.0]}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("queue_mmc", "M/M/c"),
                       output_root=str(tmp_path / "o"),
                       config={"lambda": 1.0, "mu": 1.0, "servers": 2})
    e = res.estimates
    assert math.isclose(e["rho"], 0.5, abs_tol=1e-9)
    assert math.isclose(e["p0"], 1.0 / 3.0, abs_tol=1e-5)
    assert math.isclose(e["prob_wait"], 1.0 / 3.0, abs_tol=1e-5)
    assert math.isclose(e["lq"], 1.0 / 3.0, abs_tol=1e-5)
    assert math.isclose(e["l"], 4.0 / 3.0, abs_tol=1e-5)


def test_mmc_little_law(tmp_path: Path) -> None:
    """Little's law sanity: L = lambda * W and Lq = lambda * Wq."""
    csv = _csv(tmp_path, "x.csv", pd.DataFrame({"dummy": [1.0, 2.0]}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("queue_mmc", "M/M/c"),
                       output_root=str(tmp_path / "o"),
                       config={"lambda": 1.5, "mu": 1.0, "servers": 2})
    e = res.estimates
    assert math.isclose(e["l"], 1.5 * e["w"], rel_tol=1e-6)
    assert math.isclose(e["lq"], 1.5 * e["wq"], rel_tol=1e-6)


def test_mmc_degrade_unstable(tmp_path: Path) -> None:
    """rho >= 1 -> unstable -> honest 跳过, no estimates, no crash.
       lambda=2, mu=1, c=1 -> rho = 2 >= 1."""
    csv = _csv(tmp_path, "x.csv", pd.DataFrame({"dummy": [1.0, 2.0]}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("queue_mmc", "M/M/c"),
                       output_root=str(tmp_path / "o"),
                       config={"lambda": 2.0, "mu": 1.0, "servers": 1})
    assert "跳过" in res.summary
    assert "不稳定" in res.summary
    assert "rho" not in res.estimates


def test_mmc_degrade_missing(tmp_path: Path) -> None:
    """No lambda/mu -> honest 跳过."""
    csv = _csv(tmp_path, "x.csv", pd.DataFrame({"label": ["a", "b"]}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("queue_mmc", "M/M/c"),
                       output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
    assert "rho" not in res.estimates


# --------------------------------------------------------------------------- #
# 3) newsvendor
# --------------------------------------------------------------------------- #
def test_newsvendor_known_value(tmp_path: Path) -> None:
    """Cu=7, Co=3 -> CR = 7/(7+3) = 0.7; z = Phi^-1(0.7) = 0.5244.
       Normal demand mu_d=50, sigma_d=20 -> Q* = 50 + 0.5244*20 = 60.49."""
    csv = _csv(tmp_path, "x.csv", pd.DataFrame({"dummy": [1.0, 2.0]}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("newsvendor", "Newsvendor"),
                       output_root=str(tmp_path / "o"),
                       config={"underage_cost": 7, "overage_cost": 3,
                               "demand_mean": 50, "demand_sd": 20})
    e = res.estimates
    assert math.isclose(e["critical_ratio"], 0.7, abs_tol=1e-9)
    assert math.isclose(e["z_value"], 0.524401, rel_tol=0, abs_tol=1e-3)
    assert math.isclose(e["q_star"], 50.0 + 0.524401 * 20.0, rel_tol=0, abs_tol=1e-2)
    out = Path(res.output_dir)
    assert (out / "newsvendor_curve.csv").exists()


def test_newsvendor_price_cost_salvage(tmp_path: Path) -> None:
    """Cu = price - cost = 10 - 4 = 6; Co = cost - salvage = 4 - 1 = 3.
       CR = 6/(6+3) = 0.6667. Expected profit reported (price & cost given)."""
    csv = _csv(tmp_path, "x.csv", pd.DataFrame({"dummy": [1.0, 2.0]}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("newsvendor", "Newsvendor"),
                       output_root=str(tmp_path / "o"),
                       config={"price": 10, "cost": 4, "salvage": 1,
                               "demand_mean": 100, "demand_sd": 30})
    e = res.estimates
    assert math.isclose(e["underage_cost"], 6.0, abs_tol=1e-9)
    assert math.isclose(e["overage_cost"], 3.0, abs_tol=1e-9)
    assert math.isclose(e["critical_ratio"], 6.0 / 9.0, abs_tol=1e-5)
    assert not math.isnan(e["expected_profit"])


def test_newsvendor_empirical(tmp_path: Path) -> None:
    """Empirical demand column: Q* = CR-quantile of the sample. CR=0.5 -> median.
       demand = 10..100 (10 values); 0.5-quantile (linear interp) = 55."""
    demand = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    csv = _csv(tmp_path, "d.csv", pd.DataFrame({"demand": demand}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("newsvendor", "Newsvendor"),
                       output_root=str(tmp_path / "o"),
                       config={"underage_cost": 5, "overage_cost": 5,
                               "demand_col": "demand"})
    e = res.estimates
    assert math.isclose(e["critical_ratio"], 0.5, abs_tol=1e-9)
    # numpy linear-interp 0.5-quantile of 10..100 == 55
    assert math.isclose(e["q_star"], 55.0, abs_tol=1e-6)
    # empirical mode -> z N/A
    assert math.isnan(e["z_value"])


def test_newsvendor_degrade_missing_costs(tmp_path: Path) -> None:
    """No Cu/Co and no price/cost -> honest 跳过."""
    csv = _csv(tmp_path, "x.csv", pd.DataFrame({"dummy": [1.0, 2.0]}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("newsvendor", "Newsvendor"),
                       output_root=str(tmp_path / "o"),
                       config={"demand_mean": 50, "demand_sd": 20})
    assert "跳过" in res.summary
    assert "q_star" not in res.estimates


def test_newsvendor_degrade_missing_demand(tmp_path: Path) -> None:
    """Costs given but no demand model (no mean/sd, no column) -> honest 跳过."""
    csv = _csv(tmp_path, "x.csv", pd.DataFrame({"label": ["a", "b"]}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("newsvendor", "Newsvendor"),
                       output_root=str(tmp_path / "o"),
                       config={"underage_cost": 7, "overage_cost": 3})
    assert "跳过" in res.summary
    assert "q_star" not in res.estimates
