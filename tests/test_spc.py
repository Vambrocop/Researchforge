"""Tests for the STATISTICAL PROCESS CONTROL / quality family (family=spc):
control_chart, process_capability, gage_rr.

Known-value cases are hand-computed in each test's docstring; honest-degrade paths
assert the Chinese "跳过" message and no crash. Mirrors tests/test_techno_economic.py:
write CSV -> profile_dataset -> AnalysisEntry/Precondition -> run_analysis ->
assert on res.estimates / res.summary.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry(eid: str, method: str, goal: str = "describe") -> AnalysisEntry:
    return AnalysisEntry(
        id=eid,
        method=method,
        domain="quality",
        family="spc",
        goal=goal,
        preconditions=Precondition(min_rows=1),
    )


def _csv(tmp_path: Path, name: str, df: pd.DataFrame) -> Path:
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


# --------------------------------------------------------------------------- #
# 1) control_chart
# --------------------------------------------------------------------------- #
def test_control_chart_imr_spike_detected(tmp_path: Path) -> None:
    """I-MR chart on stable data (~10) with one obvious spike (100) -> at least one
    out-of-control point. With no equal-size subgroup column the engine uses I-MR;
    control limits = x-bar +/- 2.66*MR-bar and the spike is far beyond UCL."""
    vals = [10.0, 10.2, 9.8, 10.1, 9.9, 10.0, 10.3, 9.7, 100.0, 10.1, 9.9, 10.0]
    csv = _csv(tmp_path, "m.csv", pd.DataFrame({"measure": vals}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("control_chart", "Control Chart"),
                       output_root=str(tmp_path / "o"),
                       config={"measurement": "measure"})
    e = res.estimates
    assert e["n_out_of_control"] >= 1
    assert e["ucl"] > e["center_line"] > e["lcl"]
    assert e["sigma_hat"] > 0
    out = Path(res.output_dir)
    assert (out / "control_chart_points.csv").exists()
    pts = pd.read_csv(out / "control_chart_points.csv")
    # the spike row (value 100) must be flagged out of control
    spike = pts.loc[pts["value"] == 100.0]
    assert bool(spike["out_of_control"].iloc[0])


def test_control_chart_xbar_r_subgroups(tmp_path: Path) -> None:
    """X-bar & R chart: 5 subgroups of constant size n=3 (centered ~20, tight) plus
    one subgroup shifted to ~40 -> at least one out-of-control subgroup mean.
    n=3 -> A2=1.023, D4=2.574, d2=1.693 (hard-coded constants)."""
    # deterministic small jitter per subgroup, tight spread (range 0.4 in every group)
    data = {"sg": [], "y": []}
    base = {0: 20.0, 1: 20.1, 2: 19.9, 3: 20.2, 4: 40.0}  # subgroup 4 is shifted high
    jit = [-0.2, 0.0, 0.2]
    for g in range(5):
        for j in jit:
            data["sg"].append(g)
            data["y"].append(base[g] + j)
    csv = _csv(tmp_path, "xb.csv", pd.DataFrame(data))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("control_chart", "Control Chart"),
                       output_root=str(tmp_path / "o"),
                       config={"measurement": "y", "subgroup": "sg"})
    e = res.estimates
    assert e["subgroup_size"] == 3.0
    assert e["n_subgroups"] == 5.0
    assert e["n_out_of_control"] >= 1
    assert "X-bar & R" in res.summary
    # sigma_hat = R_bar / d2; R within each subgroup = 0.4 for groups 0..4 -> R_bar=0.4,
    # d2(n=3)=1.693 -> sigma_hat ~= 0.2363
    assert math.isclose(e["r_bar"], 0.4, abs_tol=1e-6)
    assert math.isclose(e["sigma_hat"], 0.4 / 1.693, rel_tol=0, abs_tol=1e-4)


def test_control_chart_degrade_no_numeric(tmp_path: Path) -> None:
    """No numeric measurement column -> honest 跳过, no crash."""
    csv = _csv(tmp_path, "txt.csv", pd.DataFrame({"label": ["a", "b", "c"]}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("control_chart", "Control Chart"),
                       output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
    assert "center_line" not in res.estimates


# --------------------------------------------------------------------------- #
# 2) process_capability
# --------------------------------------------------------------------------- #
def test_capability_centered_cp_cpk_one(tmp_path: Path) -> None:
    """Data standardized to mean=0, sample SD=1; USL=+3, LSL=-3 ->
       Cp = (3-(-3))/(6*1) = 1.0 ; Cpk = min((3-0)/3, (0-(-3))/3) = 1.0.
       Pp/Ppk use the overall sample SD (=1 here, no subgroups) -> also 1.0.
       ppm out of spec ~ 2 * (1 - Phi(3)) * 1e6 ~= 2700 ppm."""
    raw = np.linspace(-2, 2, 50)
    z = (raw - raw.mean()) / raw.std(ddof=1)  # mean exactly 0, sample SD exactly 1
    csv = _csv(tmp_path, "cap.csv", pd.DataFrame({"x": z}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("process_capability", "Process Capability", "evaluate"),
                       output_root=str(tmp_path / "o"),
                       config={"measurement": "x", "lsl": -3.0, "usl": 3.0})
    e = res.estimates
    assert math.isclose(e["cp"], 1.0, rel_tol=0, abs_tol=1e-6)
    assert math.isclose(e["cpk"], 1.0, rel_tol=0, abs_tol=1e-6)
    assert math.isclose(e["pp"], 1.0, rel_tol=0, abs_tol=1e-6)
    assert math.isclose(e["ppk"], 1.0, rel_tol=0, abs_tol=1e-6)
    # normal-model ppm out of spec ~ 2700 (2*(1-Phi(3))*1e6)
    assert math.isclose(e["ppm_out"], 2700.0, rel_tol=0, abs_tol=120.0)
    out = Path(res.output_dir)
    assert (out / "process_capability.csv").exists()


def test_capability_one_sided_usl(tmp_path: Path) -> None:
    """One-sided spec (only USL): Cp undefined (NaN), Cpk = Cpu = (USL-xbar)/(3*sigma).
       mean=0, SD=1, USL=+3 -> Cpk = 3/3 = 1.0; Cpl/Cp are NaN."""
    raw = np.linspace(-2, 2, 50)
    z = (raw - raw.mean()) / raw.std(ddof=1)
    csv = _csv(tmp_path, "cap.csv", pd.DataFrame({"x": z}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("process_capability", "Process Capability", "evaluate"),
                       output_root=str(tmp_path / "o"),
                       config={"measurement": "x", "usl": 3.0})
    e = res.estimates
    assert math.isnan(e["cp"])
    assert math.isclose(e["cpk"], 1.0, rel_tol=0, abs_tol=1e-6)
    assert math.isclose(e["cpu"], 1.0, rel_tol=0, abs_tol=1e-6)
    assert math.isnan(e["cpl"])


def test_capability_degrade_no_spec(tmp_path: Path) -> None:
    """No spec limits provided -> honest 跳过 with the required message."""
    raw = np.linspace(-2, 2, 30)
    csv = _csv(tmp_path, "cap.csv", pd.DataFrame({"x": raw}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("process_capability", "Process Capability", "evaluate"),
                       output_root=str(tmp_path / "o"),
                       config={"measurement": "x"})
    assert "跳过" in res.summary
    assert "规格限" in res.summary
    assert "cpk" not in res.estimates


# --------------------------------------------------------------------------- #
# 3) gage_rr
# --------------------------------------------------------------------------- #
def _gage_rr_dataset(part_means, n_operators=3, n_rep=3, op_bias=None, noise=0.05,
                     seed=42) -> pd.DataFrame:
    """Balanced crossed parts x operators x replicates design.

    measurement = part_mean + operator_bias + small noise. Widely-separated part
    means with tiny noise -> part variance >> gauge variance -> small %GRR, large ndc.
    """
    rng = np.random.default_rng(seed)
    if op_bias is None:
        op_bias = [0.0] * n_operators
    rows = []
    for pi, pm in enumerate(part_means):
        for oi in range(n_operators):
            for _ in range(n_rep):
                y = pm + op_bias[oi] + rng.normal(0.0, noise)
                rows.append({"part": pi + 1, "operator": f"op{oi + 1}", "y": y})
    return pd.DataFrame(rows)


def test_gage_rr_good_system(tmp_path: Path) -> None:
    """Part variance >> gauge variance: 10 widely-separated parts (means 10,20,...,100),
    3 operators with tiny bias, tiny measurement noise -> small %GRR (<10%) and ndc>=5
    (the gauge resolves the parts well)."""
    parts = [10.0 * k for k in range(1, 11)]  # 10..100, huge part-to-part spread
    df = _gage_rr_dataset(parts, n_operators=3, n_rep=3,
                          op_bias=[0.0, 0.05, -0.05], noise=0.05, seed=7)
    csv = _csv(tmp_path, "grr.csv", df)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("gage_rr", "Gage R&R", "evaluate"),
                       output_root=str(tmp_path / "o"),
                       config={"measurement": "y", "part": "part", "operator": "operator"})
    e = res.estimates
    assert e["n_parts"] == 10.0
    assert e["n_operators"] == 3.0
    assert e["n_replicates"] == 3.0
    # an excellent gauge: %GRR (study variation) is small and ndc is large
    assert e["pct_gagerr_studyvar"] < 10.0
    assert e["ndc"] >= 5
    # part-to-part variance dominates the total
    assert e["var_part"] > e["var_gagerr"]
    assert math.isclose(
        e["pct_part"] ** 2 + e["pct_gagerr_studyvar"] ** 2, 100.0 ** 2, rel_tol=0.02
    )  # %SV are ratios of SDs -> components add in quadrature to 100%
    out = Path(res.output_dir)
    assert (out / "gage_rr_components.csv").exists()
    assert "ANOVA" in res.summary


def test_gage_rr_degrade_no_operator(tmp_path: Path) -> None:
    """Only a part column, no operator column -> cannot run crossed design -> 跳过."""
    df = pd.DataFrame({"part": [1, 1, 2, 2, 3, 3], "y": [10.0, 10.1, 20.0, 20.1, 30.0, 30.1]})
    csv = _csv(tmp_path, "grr.csv", df)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("gage_rr", "Gage R&R", "evaluate"),
                       output_root=str(tmp_path / "o"),
                       config={"measurement": "y", "part": "part"})
    assert "跳过" in res.summary
    assert "ndc" not in res.estimates


def test_gage_rr_degrade_unbalanced(tmp_path: Path) -> None:
    """Unbalanced / no-replicate crossed design -> honest 跳过 (cannot split error)."""
    # each part x operator cell has exactly 1 reading -> repeatability not separable
    rows = []
    for p in (1, 2, 3):
        for o in ("a", "b"):
            rows.append({"part": p, "operator": o, "y": 10.0 * p + (0.1 if o == "a" else -0.1)})
    csv = _csv(tmp_path, "grr.csv", pd.DataFrame(rows))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("gage_rr", "Gage R&R", "evaluate"),
                       output_root=str(tmp_path / "o"),
                       config={"measurement": "y", "part": "part", "operator": "operator"})
    assert "跳过" in res.summary
    assert "ndc" not in res.estimates
