"""Hydrology family: mann_kendall_trend (+ Sen slope), flow_duration_curve, idf_curve.

Known-value cases: a linear series recovers Sen slope = the true slope and a
significant increasing trend; the FDC exceedance percentiles are ordered; the IDF
fit recovers the Talbot parameters of synthetic data and degrades without the
duration/intensity columns.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry(eid: str, method: str) -> AnalysisEntry:
    return AnalysisEntry(id=eid, method=method, domain="environmental",
                         family="hydrology", goal="relate",
                         preconditions=Precondition(min_rows=1))


def _csv(tmp_path: Path, name: str, df: pd.DataFrame) -> Path:
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


# --------------------------------------------------------------------------- #
# mann_kendall_trend
# --------------------------------------------------------------------------- #
def test_mann_kendall_increasing_recovers_sen_slope(tmp_path: Path) -> None:
    # strictly increasing linear series flow = 2*t + 5 -> Sen slope = 2, sig increasing
    t = np.arange(30, dtype=float)
    df = pd.DataFrame({"flow": 2.0 * t + 5.0})
    res = run_analysis(profile_dataset(_csv(tmp_path, "mk.csv", df)),
                       _entry("mann_kendall_trend", "Mann-Kendall"),
                       output_root=str(tmp_path / "o"), config={"value": "flow"})
    e = res.estimates
    assert e["mk_z"] > 0 and e["mk_p"] < 0.05            # significant increasing trend
    assert math.isclose(e["sen_slope"], 2.0, rel_tol=1e-6)
    assert e["sen_slope_low"] <= 2.0 <= e["sen_slope_high"]
    assert "增" in res.summary or "increasing" in res.summary.lower() or "趋势" in res.summary


def test_mann_kendall_noisy_ci_brackets_slope(tmp_path: Path) -> None:
    # noisy linear series exercises the Gilbert rank-CI index arithmetic (the clean
    # linear case is degenerate — all pairwise slopes equal). CI must strictly bracket.
    rng = np.random.default_rng(7)
    t = np.arange(40, dtype=float)
    df = pd.DataFrame({"flow": 1.5 * t + rng.normal(0, 4.0, t.size)})
    res = run_analysis(profile_dataset(_csv(tmp_path, "noisy.csv", df)),
                       _entry("mann_kendall_trend", "Mann-Kendall"),
                       output_root=str(tmp_path / "o"), config={"value": "flow"})
    e = res.estimates
    assert e["sen_slope_low"] < e["sen_slope"] < e["sen_slope_high"]
    assert e["mk_p"] < 0.05  # clear upward trend survives the noise


def test_mann_kendall_flat_no_trend(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"flow": rng.normal(10.0, 1.0, 40)})  # no trend
    res = run_analysis(profile_dataset(_csv(tmp_path, "flat.csv", df)),
                       _entry("mann_kendall_trend", "Mann-Kendall"),
                       output_root=str(tmp_path / "o"), config={"value": "flow"})
    assert res.estimates["mk_p"] > 0.05  # not a significant trend


# --------------------------------------------------------------------------- #
# flow_duration_curve
# --------------------------------------------------------------------------- #
def test_flow_duration_percentiles_ordered(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"flow": rng.gamma(2.0, 50.0, 365)})  # a year of daily flows
    res = run_analysis(profile_dataset(_csv(tmp_path, "fdc.csv", df)),
                       _entry("flow_duration_curve", "FDC"),
                       output_root=str(tmp_path / "o"), config={"value": "flow"})
    e = res.estimates
    # Qx = flow exceeded x% of the time -> Q5 (high) > Q50 (median) > Q95 (low)
    assert e["q5"] > e["q50"] > e["q95"]
    assert math.isclose(e["q50"], float(np.median(df["flow"])), rel_tol=0.05)
    assert 0 < e["low_flow_index"] < 1  # Q90/Q50


# --------------------------------------------------------------------------- #
# idf_curve
# --------------------------------------------------------------------------- #
def test_idf_recovers_talbot_params(tmp_path: Path) -> None:
    # synthetic Talbot i = a/(d+b)^c with a=1000, b=10, c=0.8
    d = np.array([5, 10, 15, 30, 45, 60, 90, 120], dtype=float)
    a, b, c = 1000.0, 10.0, 0.8
    intensity = a / (d + b) ** c
    df = pd.DataFrame({"duration": d, "intensity": intensity})
    res = run_analysis(profile_dataset(_csv(tmp_path, "idf.csv", df)),
                       _entry("idf_curve", "IDF"), output_root=str(tmp_path / "o"),
                       config={"duration": "duration", "intensity": "intensity"})
    e = res.estimates
    assert e["idf_r2"] > 0.99                       # near-perfect fit of noiseless data
    assert math.isclose(e["idf_c"], 0.8, rel_tol=0.1)


def test_idf_degrades_without_duration_intensity(tmp_path: Path) -> None:
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 3.0, 4.0, 5.0]})
    res = run_analysis(profile_dataset(_csv(tmp_path, "no_idf.csv", df)),
                       _entry("idf_curve", "IDF"), output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
    assert "idf_a" not in res.estimates
