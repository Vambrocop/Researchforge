"""Tests for csqca: configurational gate + crisp-set QCA via R (skips without R)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import rbridge, run_analysis
from researchforge.profiler import profile_dataset
from researchforge.recommender.match import check_preconditions


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="csqca",
        method="Crisp-set QCA (csQCA)",
        domain="social science",
        family="configurational",
        goal="explain",
        preconditions=Precondition(min_continuous=3, min_rows=12),
    )


def test_csqca_solution(tmp_path: Path) -> None:
    if not (rbridge.r_available() and rbridge.r_package_available("QCA")):
        pytest.skip("R QCA package not available")

    rng = np.random.default_rng(0)
    n = 90
    a = rng.uniform(0, 1, n)
    b = rng.uniform(0, 1, n)
    c = rng.uniform(0, 1, n)
    # outcome high when (A AND B) OR C -> a recoverable crisp sufficiency structure
    y = np.clip(np.maximum(np.minimum(a, b), c) + rng.normal(0, 0.03, n), 0, 1)
    df = pd.DataFrame({"Y": y, "A": a, "B": b, "C": c})
    csv = tmp_path / "csqca.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    assert (out / "csqca_solution.csv").exists()
    sol = pd.read_csv(out / "csqca_solution.csv")
    assert len(sol) >= 1
    assert (sol["consistency"] <= 1.0 + 1e-9).all()
    assert res.estimates["min_consistency"] > 0.6


def test_csqca_precondition_unmet(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"y": rng.normal(0, 1, 30), "x": rng.normal(0, 1, 30)})  # only 2 continuous
    csv = tmp_path / "two.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)

    assert not ok
    assert any("连续" in u for u in unmet)
