"""Tests for membership_function and grey_relational MCDA methods."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry(eid: str, method: str) -> AnalysisEntry:
    return AnalysisEntry(
        id=eid,
        method=method,
        domain="evaluation",
        family="mcda",
        goal="describe",
        preconditions=Precondition(min_continuous=2, min_rows=3),
    )


def _varieties(tmp_path: Path) -> Path:
    # variety A dominates on every benefit indicator -> should rank #1
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
    return csv


def test_membership_function_ranks_dominant_first(tmp_path: Path) -> None:
    fp = profile_dataset(_varieties(tmp_path))
    res = run_analysis(
        fp, _entry("membership_function", "Membership-function"), output_root=str(tmp_path / "o")
    )
    out = Path(res.output_dir)
    scores = pd.read_csv(out / "membership_scores.csv")
    assert (out / "membership_matrix.csv").exists()
    assert scores.sort_values("rank").iloc[0]["alternative"] == "A"
    assert 0.0 <= scores["membership_score"].max() <= 1.0


def test_grey_relational_ranks_dominant_first(tmp_path: Path) -> None:
    fp = profile_dataset(_varieties(tmp_path))
    res = run_analysis(
        fp, _entry("grey_relational", "Grey relational analysis"), output_root=str(tmp_path / "o")
    )
    out = Path(res.output_dir)
    grades = pd.read_csv(out / "grey_relational.csv")
    assert set(["alternative", "relational_grade", "rank"]).issubset(grades.columns)
    assert grades.sort_values("rank").iloc[0]["alternative"] == "A"
    assert (grades["relational_grade"] > 0).all()
