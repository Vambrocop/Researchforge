"""Tests for entropy_weight: Shannon-entropy objective weighting + ranking.

Hand-computed checks on tiny matrices: a zero-dispersion column gets weight 0,
weights normalise to 1, the dominant alternative ranks first, and the < 2 criteria
gate produces an honest skip.

NOTE: criteria columns must look "continuous" to the profiler. An all-distinct
all-whole-number column is flagged as an id (the profiler "id trap"), so every test
matrix below carries a non-integer value (e.g. a .5) to stay continuous.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset
from researchforge.recommender.match import check_preconditions


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="entropy_weight",
        method="Entropy-weight objective evaluation (熵权法)",
        domain="evaluation",
        family="mcda",
        goal="describe",
        preconditions=Precondition(min_continuous=2, min_rows=3),
    )


def test_entropy_zero_dispersion_column_gets_zero_weight(tmp_path: Path) -> None:
    # 'flat' is constant -> after min-max it is 0.5 everywhere -> proportions uniform
    # -> entropy e=1 -> diversity d=0 -> weight 0. 'spread' carries all the weight.
    df = pd.DataFrame(
        {
            "variety": ["A", "B", "C", "D"],
            "spread": [10.5, 7.0, 4.0, 1.0],
            "flat": [5.0, 5.0, 5.0, 5.0],
        }
    )
    csv = tmp_path / "ew.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    w = pd.read_csv(out / "weights.csv").set_index("criterion")["entropy_weight"]
    assert abs(w.sum() - 1.0) < 1e-6  # weights normalise to 1
    assert abs(w["flat"] - 0.0) < 1e-6  # zero-dispersion -> zero weight
    assert abs(w["spread"] - 1.0) < 1e-6  # the only informative criterion gets it all


def test_entropy_weight_matches_hand_computation(tmp_path: Path) -> None:
    # Two criteria; raw columns already lie in [0,1] (with a .5 so they read continuous),
    # so min-max leaves them unchanged. Hand-compute the entropy weights and assert match.
    df = pd.DataFrame(
        {
            "alt": ["A", "B", "C"],
            "c1": [1.0, 0.5, 0.0],   # min-max -> [1, 0.5, 0]
            "c2": [1.0, 0.0, 0.5],   # min-max -> [1, 0, 0.5]
        }
    )
    csv = tmp_path / "ew2.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    # Hand computation: m=3, k=1/ln(3); d_j = 1 - e_j on p = z/Σz.
    def diversity(z):
        z = np.asarray(z, float)
        p = z / z.sum()
        k = 1.0 / math.log(3)
        e = -k * sum(pi * math.log(pi) for pi in p if pi > 0)
        return 1.0 - e

    d1 = diversity([1.0, 0.5, 0.0])
    d2 = diversity([1.0, 0.0, 0.5])
    # By symmetry these two columns have identical dispersion -> equal weights 0.5/0.5.
    exp_w1 = d1 / (d1 + d2)
    exp_w2 = d2 / (d1 + d2)

    w = pd.read_csv(out / "weights.csv").set_index("criterion")["entropy_weight"]
    assert abs(w["c1"] - round(exp_w1, 4)) < 2e-3
    assert abs(w["c2"] - round(exp_w2, 4)) < 2e-3
    assert abs(w["c1"] - 0.5) < 2e-3  # symmetric columns -> equal weight


def test_entropy_dominant_alternative_ranks_first(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "variety": ["A", "B", "C", "D"],
            "yield_t": [10.2, 8.1, 6.3, 4.5],
            "protein": [9.1, 7.2, 5.3, 3.4],
        }
    )
    csv = tmp_path / "dom.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)
    scores = pd.read_csv(out / "entropy_scores.csv")
    assert scores.sort_values("rank").iloc[0]["alternative"] == "A"


def test_entropy_cost_criteria_config_flips_direction(tmp_path: Path) -> None:
    # After flipping 'cost', both criteria normalise to [0, 0.5, 1] (C best on both),
    # so C must rank #1. Without the flip A would dominate -> proves the flip ran.
    df = pd.DataFrame(
        {
            "alt": ["A", "B", "C"],
            "benefit": [1.0, 2.0, 3.5],
            "cost": [9.5, 5.0, 1.0],  # cost: C smallest -> best on this criterion
        }
    )
    csv = tmp_path / "cost.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"), config={"cost_criteria": ["cost"]}
    )
    out = Path(res.output_dir)
    scores = pd.read_csv(out / "entropy_scores.csv")
    assert scores.sort_values("rank").iloc[0]["alternative"] == "C"


def test_entropy_precondition_unmet(tmp_path: Path) -> None:
    df = pd.DataFrame({"only_one": [1.1, 2.2, 3.3, 4.4]})  # < 2 numeric criteria
    csv = tmp_path / "one.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)
    assert not ok
    assert any("连续" in u for u in unmet)
