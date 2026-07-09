"""Tests for dominance_analysis (Budescu general dominance).

Builds data with a KNOWN importance ordering (x1 >> x2 >> x3) under mild predictor
correlation, then asserts (a) the recovered dominance ranking matches, (b) the
per-predictor general dominances SUM to the full-model R² (the decomposition is
exact), and (c) honest degrade with <2 predictors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="dominance_analysis",
        method="Dominance analysis (Budescu general dominance)",
        domain="statistics",
        family="statistics",
        goal="explain",
        preconditions=Precondition(min_continuous=1, min_numeric_cols=3, min_rows=20),
    )


def _make_df(seed: int = 0, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, n)  # shared factor -> mild correlation among predictors
    x1 = z + rng.normal(0, 1, n)
    x2 = 0.5 * z + rng.normal(0, 1, n)
    x3 = 0.3 * z + rng.normal(0, 1, n)
    # known ordering: x1 (3) >> x2 (1) >> x3 (0.2)
    y = 3.0 * x1 + 1.0 * x2 + 0.2 * x3 + rng.normal(0, 1.0, n)
    # outcome must be the first continuous column (engine convention) -> y first
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})


def test_dominance_recovers_ordering_and_sums_to_r2(tmp_path: Path) -> None:
    df = _make_df()
    csv = tmp_path / "dom.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    assert (out / "dominance.csv").exists()
    assert (out / "dominance.png").exists()

    tab = pd.read_csv(out / "dominance.csv")
    # ranking: x1 > x2 > x3
    dom = dict(zip(tab["predictor"], tab["general_dominance"]))
    assert dom["x1"] > dom["x2"] > dom["x3"]
    rank = dict(zip(tab["predictor"], tab["rank"]))
    assert rank["x1"] == 1 and rank["x2"] == 2 and rank["x3"] == 3

    # decomposition is exact: general dominances sum to the full-model R²
    est = res.estimates
    assert est["n_predictors"] == 3.0
    assert abs(est["dominance_sum_check"] - est["model_r2"]) < 1e-6
    assert abs(float(tab["general_dominance"].sum()) - est["model_r2"]) < 1e-6
    # share-of-R² adds to ~100%
    assert abs(float(tab["pct_of_R2"].sum()) - 100.0) < 0.5


def test_dominance_resolver_picks_high_confidence_outcome_not_first(tmp_path: Path) -> None:
    """A high-confidence-named outcome ('target') placed AFTER the predictor columns
    must still be resolved as the outcome (shared resolve_outcome, not raw cont[0])."""
    df = _make_df(seed=9).rename(columns={"y": "target"})
    df = df[["x1", "x2", "x3", "target"]]  # predictors first, outcome last
    csv = tmp_path / "resolver.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    assert fp.likely_outcome == "target" and fp.likely_outcome_confidence == "high"
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)
    tab = pd.read_csv(out / "dominance.csv")
    # predictors must be {x1, x2, x3} -> 'target' was resolved as outcome, not a predictor
    assert set(tab["predictor"]) == {"x1", "x2", "x3"}
    dom = dict(zip(tab["predictor"], tab["general_dominance"]))
    assert dom["x1"] > dom["x2"] > dom["x3"]


def test_dominance_degrades_with_one_predictor(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n = 50
    x = rng.normal(0, 1, n)
    df = pd.DataFrame({"y": 2 * x + rng.normal(0, 1, n), "x": x})  # only 1 predictor
    csv = tmp_path / "one.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    # honest degrade: a Chinese skip message, no dominance.csv, no crash
    assert "优势分析跳过" in res.summary
    assert not (Path(res.output_dir) / "dominance.csv").exists()
