"""Tests for the POLICY-EVALUATION family — PMC index model (Wave P4).

Deterministic arithmetic: policies × researcher-coded secondary indicators (0/1) →
first-level variables (mean per group) → PMC index (sum) → ratio + rating band → 3D
surface. Tests lock the arithmetic against hand-computed values and the honest degrade
paths, plus config overrides (groups / indicators / policy) and Chinese labels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog import Catalog
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry():
    return Catalog.load().by_id("pmc_index")


def _pmc_df() -> pd.DataFrame:
    # 3 first-level variables (X1/X2/X3) x 2 secondary indicators each, 4 policies.
    # Hand-computed PMC: A=3.0(perfect) B=0(low) C=2.0(acceptable) D=2.5(good).
    return pd.DataFrame({
        "政策": ["A完美", "B较低", "C中等", "D优秀"],
        "X1_a": [1, 0, 1, 1], "X1_b": [1, 0, 0, 1],
        "X2_a": [1, 0, 1, 1], "X2_b": [1, 0, 1, 0],
        "X3_a": [1, 0, 0, 1], "X3_b": [1, 0, 1, 1],
    })


def test_catalog_loads():
    e = _entry()
    assert e is not None
    assert e.executor_ref == "py::pmc_index"
    assert isinstance(e.biases, list) and len(e.biases) >= 4
    assert isinstance(e.produces, list) and e.produces
    assert e.params and all(p.name for p in e.params)


def test_pmc_scores_and_ratings(tmp_path):
    csv = tmp_path / "p.csv"
    _pmc_df().to_csv(csv, index=False, encoding="utf-8")
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    assert (out / "pmc_scores.csv").exists()
    assert (out / "pmc_variable_means.csv").exists()
    assert res.estimates["n_policies"] == 4.0
    assert res.estimates["n_variables"] == 3.0
    assert res.estimates["n_indicators"] == 6.0
    assert res.estimates["max_PMC"] == 3.0
    assert res.estimates["min_PMC"] == 0.0

    sc = pd.read_csv(out / "pmc_scores.csv").set_index("policy")
    assert {"X_X1", "X_X2", "X_X3", "PMC", "PMC_ratio", "rating"} <= set(sc.columns)
    # hand-computed indices
    assert sc.loc["A完美", "PMC"] == 3.0
    assert sc.loc["B较低", "PMC"] == 0.0
    assert sc.loc["C中等", "PMC"] == 2.0
    assert sc.loc["D优秀", "PMC"] == 2.5
    # rating bands (Ruiz Estrada convention)
    assert "完美" in sc.loc["A完美", "rating"]
    assert "较低" in sc.loc["B较低", "rating"]
    assert "可接受" in sc.loc["C中等", "rating"]
    assert "优秀" in sc.loc["D优秀", "rating"]
    # honest-quantification disclosure present
    assert "编码后量化" in res.summary
    assert "不自动" in res.summary or "研究者" in res.summary


def test_pmc_surface_pngs_produced(tmp_path):
    csv = tmp_path / "p.csv"
    _pmc_df().to_csv(csv, index=False, encoding="utf-8")
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    # 3 variables >= 2 -> a surface per policy (4 policies, cap 6)
    surfaces = [f for f in res.files if f.startswith("pmc_surface_") and f.endswith(".png")]
    assert len(surfaces) == 4
    assert "pmc_variable_means.png" in res.files


def test_pmc_config_groups_override(tmp_path):
    """config groups regroups the SAME indicators into different first-level variables."""
    csv = tmp_path / "p.csv"
    _pmc_df().to_csv(csv, index=False, encoding="utf-8")
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={
        "groups": {"治理": ["X1_a", "X2_a", "X3_a"], "保障": ["X1_b", "X2_b", "X3_b"]},
    })
    assert res.estimates["n_variables"] == 2.0
    assert "config groups" in res.summary
    sc = pd.read_csv(Path(res.output_dir) / "pmc_scores.csv").set_index("policy")
    assert {"X_治理", "X_保障"} <= set(sc.columns)
    # A: 治理=(1+1+1)/3=1, 保障=(1+1+1)/3=1 -> PMC=2 (max for 2 vars)
    assert sc.loc["A完美", "PMC"] == 2.0
    assert sc.loc["A完美", "X_治理"] == 1.0


def test_pmc_indicators_override_and_normalization(tmp_path):
    """config indicators forces the indicator set; values outside [0,1] are min-max
    normalized per column and disclosed."""
    df = pd.DataFrame({
        "policy": ["p1", "p2", "p3"],
        "s1": [1, 3, 5],   # 1..5 Likert -> normalized to 0/.5/1
        "s2": [5, 3, 1],
        "noise": [100, 200, 300],  # excluded (not in indicators)
    })
    csv = tmp_path / "n.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"indicators": ["s1", "s2"], "policy": "policy"})
    assert res.estimates["n_indicators"] == 2.0
    assert "归一" in res.summary  # normalization disclosed
    sc = pd.read_csv(Path(res.output_dir) / "pmc_scores.csv").set_index("policy")
    # each indicator its own variable (no prefix); p1: s1->0, s2->1 -> PMC=1.0
    assert np.isclose(sc.loc["p1", "PMC"], 1.0)
    assert np.isclose(sc.loc["p2", "PMC"], 1.0)  # 0.5 + 0.5


def test_pmc_single_policy(tmp_path):
    """A single policy still scores (PMC works on n=1)."""
    df = pd.DataFrame({"X1_a": [1], "X1_b": [1], "X2_a": [0], "X2_b": [1]})
    csv = tmp_path / "one.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert res.estimates["n_policies"] == 1.0
    sc = pd.read_csv(Path(res.output_dir) / "pmc_scores.csv")
    # X1=(1+1)/2=1, X2=(0+1)/2=0.5 -> PMC=1.5 of 2
    assert np.isclose(sc.iloc[0]["PMC"], 1.5)


def test_pmc_degrade_too_few_indicators(tmp_path):
    """Fewer than 2 secondary-indicator columns -> honest 跳过."""
    df = pd.DataFrame({"name": ["a", "b", "c"], "only_one": [1, 0, 1], "big": [10, 20, 30]})
    csv = tmp_path / "few.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
    assert "n_policies" not in res.estimates
