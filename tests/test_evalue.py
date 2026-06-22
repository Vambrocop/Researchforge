"""Tests for the VanderWeele & Ding (2017) E-value sensitivity branch.

Known checks:
  - hand cross-check of E = RR + sqrt(RR*(RR-1)) on the helper;
  - strong effect -> large E-value; near-null -> E-value ≈ 1;
  - binary-outcome (logistic OR -> RR) and continuous-outcome (SMD d -> RR) paths;
  - degrade on missing columns; config plumbing.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.executor.branches.sensitivity import _evalue_from_rr, _evalue_for_ci
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="evalue", method="E-value (unmeasured confounding)",
        domain="epidemiology", family="causal", goal="explain",
        preconditions=Precondition(requires_treatment=True, min_rows=10),
    )


def test_evalue_formula_hand_values() -> None:
    # RR = 2 -> E = 2 + sqrt(2*1) = 2 + sqrt(2) ≈ 3.41421356
    assert math.isclose(_evalue_from_rr(2.0), 2.0 + math.sqrt(2.0), rel_tol=1e-9)
    # RR = 1 (null) -> E = 1 exactly
    assert math.isclose(_evalue_from_rr(1.0), 1.0, abs_tol=1e-12)
    # RR = 3 -> E = 3 + sqrt(3*2) = 3 + sqrt(6) ≈ 5.449489743
    assert math.isclose(_evalue_from_rr(3.0), 3.0 + math.sqrt(6.0), rel_tol=1e-9)
    # protective RR = 0.5 -> invert to 2 -> same as RR=2
    assert math.isclose(_evalue_from_rr(0.5), _evalue_from_rr(2.0), rel_tol=1e-9)


def test_evalue_ci_crosses_null_is_one() -> None:
    # estimate > 1 but lower CI <= 1 -> the CI E-value is exactly 1
    assert _evalue_for_ci(1.5, 0.8, 2.5) == 1.0
    # estimate > 1, lower CI > 1 -> use the lower limit
    assert math.isclose(_evalue_for_ci(2.0, 1.5, 2.6), _evalue_from_rr(1.5), rel_tol=1e-9)


def test_evalue_strong_continuous_effect_large(tmp_path: Path) -> None:
    # large standardized mean difference -> large RR -> large E-value
    rng = np.random.default_rng(0)
    n = 400
    expo = rng.integers(0, 2, n)
    y = 2.0 * expo + rng.normal(0, 1, n)  # ~2 SD shift -> strong
    csv = tmp_path / "strong.csv"
    pd.DataFrame({"y": y.astype(float), "expo": expo}).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "y", "exposure": "expo"})
    assert "完成" in res.summary
    assert res.estimates["evalue_point"] > 2.0   # robust
    assert res.estimates["evalue_ci"] > 1.5
    # cross-check the reported point E-value against the formula on the reported RR
    rr = res.estimates["rr_used"]
    assert math.isclose(res.estimates["evalue_point"], _evalue_from_rr(rr), rel_tol=1e-3)


def test_evalue_near_null_is_about_one(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n = 400
    expo = rng.integers(0, 2, n)
    y = 0.02 * expo + rng.normal(0, 1, n)  # negligible effect
    csv = tmp_path / "null.csv"
    pd.DataFrame({"y": y, "expo": expo}).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "y", "exposure": "expo"})
    assert "完成" in res.summary
    assert res.estimates["evalue_point"] < 1.4          # point E-value near 1
    assert math.isclose(res.estimates["evalue_ci"], 1.0, abs_tol=1e-6)  # CI crosses null


def test_evalue_binary_outcome_logistic_path(tmp_path: Path) -> None:
    # binary outcome strongly driven by exposure -> OR>>1 -> E-value > 1
    rng = np.random.default_rng(2)
    n = 500
    expo = rng.integers(0, 2, n)
    lin = -0.5 + 1.8 * expo
    p = 1.0 / (1.0 + np.exp(-lin))
    y = (rng.uniform(size=n) < p).astype(int)
    csv = tmp_path / "bin.csv"
    pd.DataFrame({"y": y, "expo": expo}).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "y", "exposure": "expo"})
    assert "完成" in res.summary
    assert res.estimates["or_point"] > 1.5
    assert res.estimates["evalue_point"] > 1.3
    rr = res.estimates["rr_used"]
    assert math.isclose(res.estimates["evalue_point"], _evalue_from_rr(rr), rel_tol=1e-3)


def test_evalue_needs_two_columns(tmp_path: Path) -> None:
    df = pd.DataFrame({"y": np.random.default_rng(3).normal(0, 1, 20)})  # only one column
    csv = tmp_path / "x.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"outcome": "y"})
    assert "E-value 失败" in res.summary
