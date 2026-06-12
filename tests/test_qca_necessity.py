"""Tests for qca_necessity: configurational gate + R/QCA superSubset (skips no R)."""

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
        id="qca_necessity",
        method="QCA necessity analysis (superSubset)",
        domain="social science",
        family="configurational",
        goal="explain",
        preconditions=Precondition(min_continuous=3, min_rows=12),
    )


def test_qca_necessity(tmp_path: Path) -> None:
    if not (rbridge.r_available() and rbridge.r_package_available("QCA")):
        pytest.skip("R QCA package not available")

    rng = np.random.default_rng(0)
    n = 80
    c = rng.uniform(0, 1, n)
    # outcome can be high only when C is high -> C is necessary for Y
    y = np.clip(c * rng.uniform(0.3, 1, n), 0, 1)
    a = rng.uniform(0, 1, n)
    b = rng.uniform(0, 1, n)
    df = pd.DataFrame({"Y": y, "A": a, "B": b, "C": c})
    csv = tmp_path / "nec.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    assert (out / "necessity.csv").exists()
    nec = pd.read_csv(out / "necessity.csv")
    assert len(nec) >= 1
    assert res.estimates["max_inclN"] > 0.85  # a strong necessary condition exists
    assert (nec["consistency_inclN"] <= 1.0).all()


def test_qca_necessity_precondition_unmet(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"y": rng.normal(0, 1, 30), "x": rng.normal(0, 1, 30)})  # 2 continuous
    csv = tmp_path / "two.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)

    assert not ok
    assert any("连续" in u for u in unmet)
