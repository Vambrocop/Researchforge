"""Tests for the TECHNO-ECONOMIC ANALYSIS family (engineering economics /
project appraisal): npv_irr, cost_benefit, breakeven_analysis,
sensitivity_tornado, monte_carlo_cashflow, lcoe.

Known-value cases are hand-computed in the docstrings of each test; honest-degrade
paths assert the Chinese "跳过" message and no crash.
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
# 1) npv_irr
# --------------------------------------------------------------------------- #
def test_npv_known_value(tmp_path: Path) -> None:
    """NPV of [-100, 50, 60, 70] at r=0.10:
       -100 + 50/1.1 + 60/1.1^2 + 70/1.1^3 = 47.6333..."""
    csv = _csv(tmp_path, "cf.csv", pd.DataFrame({"net_cf": [-100.0, 50.0, 60.0, 70.0]}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("npv_irr", "NPV/IRR"),
                       output_root=str(tmp_path / "o"),
                       config={"cashflow": "net_cf", "rate": 0.10})
    assert math.isclose(res.estimates["npv"], 47.6333, rel_tol=0, abs_tol=1e-3)
    assert res.estimates["discount_rate"] == 0.10
    out = Path(res.output_dir)
    assert (out / "npv_schedule.csv").exists()


def test_irr_known_value(tmp_path: Path) -> None:
    """IRR of [-100, 60, 60] solves -100 + 60/(1+r) + 60/(1+r)^2 = 0 -> r ~= 0.1306."""
    csv = _csv(tmp_path, "cf.csv", pd.DataFrame({"net_cf": [-100.0, 60.0, 60.0]}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("npv_irr", "NPV/IRR"),
                       output_root=str(tmp_path / "o"),
                       config={"cashflow": "net_cf", "rate": 0.10})
    assert math.isclose(res.estimates["irr"], 0.13066, rel_tol=0, abs_tol=1e-3)
    # NPV>0 at r=0.10 since IRR>0.10 -> accept
    assert res.estimates["npv"] > 0


def test_npv_from_cost_revenue_columns(tmp_path: Path) -> None:
    """net = revenue - cost: [-100, 50, 60, 70] reconstructed from two columns."""
    csv = _csv(tmp_path, "cr.csv", pd.DataFrame({
        "cost": [100.0, 0.0, 0.0, 0.0],
        "revenue": [0.0, 50.0, 60.0, 70.0],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("npv_irr", "NPV/IRR"),
                       output_root=str(tmp_path / "o"),
                       config={"cost": "cost", "revenue": "revenue", "rate": 0.10})
    assert math.isclose(res.estimates["npv"], 47.6333, rel_tol=0, abs_tol=1e-3)


def test_npv_payback_periods(tmp_path: Path) -> None:
    """[-100, 40, 40, 40]: simple payback between period 2 and 3.
       cum: -100, -60, -20, 20 -> crosses 0 at 2 + 20/40 = 2.5."""
    csv = _csv(tmp_path, "cf.csv", pd.DataFrame({"net_cf": [-100.0, 40.0, 40.0, 40.0]}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("npv_irr", "NPV/IRR"),
                       output_root=str(tmp_path / "o"),
                       config={"cashflow": "net_cf", "rate": 0.10})
    assert math.isclose(res.estimates["simple_payback"], 2.5, abs_tol=1e-6)
    # discounted payback is later than simple payback (or never within horizon)
    dpb = res.estimates["discounted_payback"]
    assert math.isnan(dpb) or dpb >= res.estimates["simple_payback"]


def test_irr_no_sign_change_honest(tmp_path: Path) -> None:
    """All-positive cash flows -> no investment structure -> IRR not applicable,
    reported honestly (NaN), no crash."""
    csv = _csv(tmp_path, "cf.csv", pd.DataFrame({"net_cf": [10.0, 20.0, 30.0]}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("npv_irr", "NPV/IRR"),
                       output_root=str(tmp_path / "o"),
                       config={"cashflow": "net_cf"})
    assert math.isnan(res.estimates["irr"])
    # NPV still computed (all positive -> positive NPV)
    assert res.estimates["npv"] > 0
    assert "不适用" in res.summary or "无实根" in res.summary


def test_npv_degrade_no_numeric(tmp_path: Path) -> None:
    """No numeric column -> honest 跳过, no crash."""
    csv = _csv(tmp_path, "txt.csv", pd.DataFrame({"label": ["a", "b", "c"]}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("npv_irr", "NPV/IRR"),
                       output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
    assert "npv" not in res.estimates


# --------------------------------------------------------------------------- #
# 2) cost_benefit
# --------------------------------------------------------------------------- #
def test_bcr_known_value(tmp_path: Path) -> None:
    """cost=[100,10,10], benefit=[0,80,80] at r=0.10:
       PV_cost = 100 + 10/1.1 + 10/1.21 = 117.3554
       PV_ben  = 0 + 80/1.1 + 80/1.21 = 138.8430
       BCR = 1.18311..."""
    csv = _csv(tmp_path, "cb.csv", pd.DataFrame({
        "cost": [100.0, 10.0, 10.0],
        "benefit": [0.0, 80.0, 80.0],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("cost_benefit", "Cost-Benefit"),
                       output_root=str(tmp_path / "o"),
                       config={"cost": "cost", "benefit": "benefit", "rate": 0.10})
    assert math.isclose(res.estimates["bcr"], 1.18311, rel_tol=0, abs_tol=1e-4)
    assert res.estimates["net_present_benefit"] > 0  # BCR>1 -> NPB>0
    assert "接受" in res.summary


def test_bcr_degrade_negative_pv_cost(tmp_path: Path) -> None:
    """Non-positive PV(cost) -> BCR undefined -> honest 跳过."""
    csv = _csv(tmp_path, "cb.csv", pd.DataFrame({
        "cost": [-5.0, -5.0],   # negative costs -> PV(cost) < 0
        "benefit": [10.0, 10.0],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("cost_benefit", "Cost-Benefit"),
                       output_root=str(tmp_path / "o"),
                       config={"cost": "cost", "benefit": "benefit"})
    assert "跳过" in res.summary
    assert "bcr" not in res.estimates


# --------------------------------------------------------------------------- #
# 3) breakeven_analysis
# --------------------------------------------------------------------------- #
def test_breakeven_known_value(tmp_path: Path) -> None:
    """fixed=1000, var=4, price=10 -> cm=6, breakeven=1000/6=166.667 units,
       breakeven revenue=1666.67. At units=300: MOS=133.33 (44.4%)."""
    csv = _csv(tmp_path, "x.csv", pd.DataFrame({"dummy": [1.0, 2.0]}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("breakeven_analysis", "Break-even"),
                       output_root=str(tmp_path / "o"),
                       config={"fixed_cost": 1000, "var_cost": 4, "price": 10, "units": 300})
    e = res.estimates
    assert math.isclose(e["contribution_margin"], 6.0, abs_tol=1e-9)
    assert math.isclose(e["breakeven_units"], 1000.0 / 6.0, abs_tol=1e-6)
    assert math.isclose(e["breakeven_revenue"], 10.0 * 1000.0 / 6.0, abs_tol=1e-4)
    assert math.isclose(e["margin_of_safety_units"], 300.0 - 1000.0 / 6.0, abs_tol=1e-4)
    assert math.isclose(e["contribution_margin_ratio"], 0.6, abs_tol=1e-9)
    # break-even price at 300 units = var + fixed/units = 4 + 1000/300 = 7.3333
    assert math.isclose(e["breakeven_price"], 4.0 + 1000.0 / 300.0, abs_tol=1e-4)


def test_breakeven_degrade_nonpositive_margin(tmp_path: Path) -> None:
    """price <= var_cost -> contribution margin <= 0 -> no break-even -> honest 跳过."""
    csv = _csv(tmp_path, "x.csv", pd.DataFrame({"dummy": [1.0, 2.0]}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("breakeven_analysis", "Break-even"),
                       output_root=str(tmp_path / "o"),
                       config={"fixed_cost": 1000, "var_cost": 12, "price": 10})
    assert "跳过" in res.summary
    assert "breakeven_units" not in res.estimates


def test_breakeven_degrade_missing_params(tmp_path: Path) -> None:
    """No fixed/var/price scalars and no matching columns -> honest 跳过."""
    csv = _csv(tmp_path, "x.csv", pd.DataFrame({"dummy": [1.0, 2.0]}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("breakeven_analysis", "Break-even"),
                       output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary


# --------------------------------------------------------------------------- #
# 4) sensitivity_tornado
# --------------------------------------------------------------------------- #
def test_sensitivity_ranks_and_outputs(tmp_path: Path) -> None:
    """Tornado: base NPV matches npv_irr; sensitivity.csv sorted by swing desc."""
    csv = _csv(tmp_path, "cf.csv", pd.DataFrame({"net_cf": [-100.0, 50.0, 60.0, 70.0]}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("sensitivity_tornado", "Tornado"),
                       output_root=str(tmp_path / "o"),
                       config={"cashflow": "net_cf", "rate": 0.10, "sensitivity_pct": 20})
    assert math.isclose(res.estimates["base_npv"], 47.6333, abs_tol=1e-3)
    out = Path(res.output_dir)
    tbl = pd.read_csv(out / "sensitivity.csv")
    assert {"parameter", "low_npv", "high_npv", "swing"}.issubset(tbl.columns)
    # sorted descending by swing
    sw = tbl["swing"].to_numpy()
    assert all(sw[i] >= sw[i + 1] for i in range(len(sw) - 1))
    # a parameter row exists for the discount rate and for each non-zero cash flow
    assert "discount_rate" in set(tbl["parameter"])
    assert any(p.startswith("cashflow_t") for p in tbl["parameter"])


def test_sensitivity_degrade(tmp_path: Path) -> None:
    csv = _csv(tmp_path, "txt.csv", pd.DataFrame({"label": ["a", "b"]}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("sensitivity_tornado", "Tornado"),
                       output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary


# --------------------------------------------------------------------------- #
# 5) monte_carlo_cashflow
# --------------------------------------------------------------------------- #
def test_monte_carlo_reproducible_and_directional(tmp_path: Path) -> None:
    """Fixed seed -> reproducible. A marginal project (small positive base NPV
    ~= +11.9 for [-100,45,45,45] at r=0.10) with cv=0.30 should show a non-trivial
    probability of loss (some draws push NPV below 0)."""
    csv = _csv(tmp_path, "cf.csv", pd.DataFrame({"net_cf": [-100.0, 45.0, 45.0, 45.0]}))
    fp = profile_dataset(csv)
    cfg = {"cashflow": "net_cf", "rate": 0.10, "n_sim": 20000,
           "seed": 7, "cv": 0.30, "var_pct": 5}
    r1 = run_analysis(fp, _entry("monte_carlo_cashflow", "Monte-Carlo"),
                      output_root=str(tmp_path / "o1"), config=cfg)
    r2 = run_analysis(fp, _entry("monte_carlo_cashflow", "Monte-Carlo"),
                      output_root=str(tmp_path / "o2"), config=cfg)
    # reproducible with a fixed seed
    assert r1.estimates["mc_mean_npv"] == r2.estimates["mc_mean_npv"]
    assert r1.estimates["prob_loss"] == r2.estimates["prob_loss"]
    # P(loss) is a probability and non-trivial here
    assert 0.0 <= r1.estimates["prob_loss"] <= 1.0
    assert 0.05 < r1.estimates["prob_loss"] < 0.95
    # mean of simulated NPV ~ deterministic base NPV (symmetric noise, unbiased)
    assert math.isclose(r1.estimates["mc_mean_npv"], r1.estimates["base_npv"],
                        rel_tol=0, abs_tol=2.0)
    out = Path(r1.output_dir)
    assert (out / "monte_carlo_npv.csv").exists()


def test_monte_carlo_loss_prob_increases_with_cv(tmp_path: Path) -> None:
    """Directional: for a project with POSITIVE base NPV, more uncertainty
    (higher cv) -> wider NPV distribution -> more mass below 0 -> higher P(loss).
    Sanity check on the distributional model. Base NPV[-100,45,45,45]@0.10 ~= +11.9."""
    csv = _csv(tmp_path, "cf.csv", pd.DataFrame({"net_cf": [-100.0, 45.0, 45.0, 45.0]}))
    fp = profile_dataset(csv)
    base = {"cashflow": "net_cf", "rate": 0.10, "n_sim": 20000, "seed": 7}
    lo = run_analysis(fp, _entry("monte_carlo_cashflow", "Monte-Carlo"),
                      output_root=str(tmp_path / "lo"), config={**base, "cv": 0.10})
    hi = run_analysis(fp, _entry("monte_carlo_cashflow", "Monte-Carlo"),
                      output_root=str(tmp_path / "hi"), config={**base, "cv": 0.40})
    assert hi.estimates["prob_loss"] > lo.estimates["prob_loss"]
    # higher cv -> wider NPV distribution
    assert hi.estimates["mc_sd_npv"] > lo.estimates["mc_sd_npv"]


def test_monte_carlo_degrade(tmp_path: Path) -> None:
    csv = _csv(tmp_path, "txt.csv", pd.DataFrame({"label": ["a", "b"]}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("monte_carlo_cashflow", "Monte-Carlo"),
                       output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary


# --------------------------------------------------------------------------- #
# 6) lcoe
# --------------------------------------------------------------------------- #
def test_lcoe_known_value(tmp_path: Path) -> None:
    """cost=[1000,100,100], output=[0,500,500] at r=0.10:
       PV_cost = 1000 + 100/1.1 + 100/1.21 = 1173.5537
       PV_out  = 0 + 500/1.1 + 500/1.21 = 867.7686
       LCOE = 1.35238..."""
    csv = _csv(tmp_path, "le.csv", pd.DataFrame({
        "capex_opex": [1000.0, 100.0, 100.0],
        "kwh": [0.0, 500.0, 500.0],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("lcoe", "LCOE"),
                       output_root=str(tmp_path / "o"),
                       config={"cost": "capex_opex", "output": "kwh", "rate": 0.10})
    assert math.isclose(res.estimates["lcoe"], 1.35238, rel_tol=0, abs_tol=1e-4)
    assert res.estimates["pv_output"] > 0


def test_lcoe_degrade_nonpositive_output(tmp_path: Path) -> None:
    """Zero output everywhere -> PV(output)=0 -> LCOE undefined -> honest 跳过."""
    csv = _csv(tmp_path, "le.csv", pd.DataFrame({
        "cost": [1000.0, 100.0],
        "kwh": [0.0, 0.0],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("lcoe", "LCOE"),
                       output_root=str(tmp_path / "o"),
                       config={"cost": "cost", "output": "kwh"})
    assert "跳过" in res.summary
    assert "lcoe" not in res.estimates
