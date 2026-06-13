"""Tests for critic: criteria gate + CRITIC objective weighting and ranking."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset
from researchforge.recommender.match import check_preconditions


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="critic",
        method="CRITIC objective-weight evaluation",
        domain="evaluation",
        family="mcda",
        goal="describe",
        preconditions=Precondition(min_continuous=2, min_rows=3),
    )


def test_critic_ranks_dominant_first(tmp_path: Path) -> None:
    # A dominates on every indicator, but the indicators are NOT perfectly
    # correlated (others reorder) -> conflict > 0, a non-degenerate CRITIC case.
    df = pd.DataFrame(
        {
            "variety": ["A", "B", "C", "D", "E"],
            "yield_t": [10.2, 7.5, 8.1, 5.0, 6.3],
            "protein": [9.1, 6.0, 5.3, 7.2, 4.0],
            "resistance": [8.5, 4.7, 6.6, 5.5, 7.0],
        }
    )
    csv = tmp_path / "varieties.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    scores = pd.read_csv(out / "critic_scores.csv")
    weights = pd.read_csv(out / "weights.csv")
    assert scores.sort_values("rank").iloc[0]["alternative"] == "A"  # dominant alternative wins
    assert abs(weights["critic_weight"].sum() - 1.0) < 1e-6  # weights normalise to 1


def test_critic_precondition_unmet(tmp_path: Path) -> None:
    df = pd.DataFrame({"only_one": [1.1, 2.2, 3.3, 4.4]})  # < 2 numeric criteria
    csv = tmp_path / "one.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)

    assert not ok
    assert any("连续" in u for u in unmet)
