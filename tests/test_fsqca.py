"""Tests for fsqca: configurational gate + R/QCA solution (skips without R)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis, rbridge
from researchforge.profiler import profile_dataset
from researchforge.recommender.match import check_preconditions


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="fsqca",
        method="Fuzzy-set Qualitative Comparative Analysis (fsQCA)",
        domain="social science",
        family="configurational",
        goal="explain",
        preconditions=Precondition(min_continuous=3, min_rows=12),
    )


def test_fsqca_solution(tmp_path: Path) -> None:
    if not (rbridge.r_available() and rbridge.r_package_available("QCA")):
        pytest.skip("R QCA package not available")

    rng = np.random.default_rng(0)
    n = 80
    a = rng.uniform(0, 1, n)
    b = rng.uniform(0, 1, n)
    c = rng.uniform(0, 1, n)
    # outcome high when (A AND B) OR C -> a recoverable sufficiency structure
    y = np.clip(np.maximum(np.minimum(a, b), c) + rng.normal(0, 0.04, n), 0, 1)
    df = pd.DataFrame({"Y": y, "A": a, "B": b, "C": c})
    csv = tmp_path / "qca.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    assert (out / "fsqca_solution.csv").exists()
    sol = pd.read_csv(out / "fsqca_solution.csv")
    assert len(sol) >= 1
    assert res.estimates["min_consistency"] > 0.7  # sufficiency consistency is high
    assert (sol["consistency"] <= 1.0).all()


def test_fsqca_precondition_unmet(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"y": rng.normal(0, 1, 30), "x": rng.normal(0, 1, 30)})  # only 2 continuous
    csv = tmp_path / "two.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)

    assert not ok
    assert any("连续" in u for u in unmet)
