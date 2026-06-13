"""Tests for topsis: criteria gate + entropy-weighted ranking of alternatives."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset
from researchforge.recommender.match import check_preconditions


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="topsis",
        method="Entropy-weighted TOPSIS (comprehensive evaluation)",
        domain="evaluation",
        family="mcda",
        goal="describe",
        preconditions=Precondition(min_continuous=2, min_rows=3),
    )


def test_topsis_ranks_dominant_alternative_first(tmp_path: Path) -> None:
    # variety A dominates on every (benefit) indicator -> should rank #1
    df = pd.DataFrame(
        {
            "variety": ["A", "B", "C", "D"],
            "yield_t": [10.2, 8.1, 6.3, 4.5],
            "protein": [9.1, 7.2, 5.3, 3.4],
            "resistance": [8.5, 6.6, 4.7, 2.8],
        }
    )
    csv = tmp_path / "varieties.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    scores = pd.read_csv(out / "topsis_scores.csv")
    weights = pd.read_csv(out / "weights.csv")

    assert set(["alternative", "score", "rank"]).issubset(scores.columns)
    top = scores.sort_values("rank").iloc[0]
    assert top["alternative"] == "A"  # the dominant alternative wins
    assert 0.0 <= scores["score"].max() <= 1.0
    assert abs(weights["entropy_weight"].sum() - 1.0) < 1e-6  # weights normalise to 1


def test_topsis_precondition_unmet(tmp_path: Path) -> None:
    df = pd.DataFrame({"only_one": [1.1, 2.2, 3.3, 4.4]})  # < 2 numeric criteria
    csv = tmp_path / "one.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)

    assert not ok
    assert any("连续" in u for u in unmet)
