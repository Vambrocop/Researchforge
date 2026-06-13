"""Tests for the panel / clustered fsQCA (SetMethods) executor branch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis, rbridge
from researchforge.profiler import profile_dataset

_HAS_SM = (
    rbridge.r_available()
    and rbridge.r_package_available("SetMethods")
    and rbridge.r_package_available("QCA")
)


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="panel_qca", method="Panel fsQCA", domain="social science",
        family="configurational", goal="explain",
        preconditions=Precondition(is_panel=True, min_continuous=3, min_rows=20),
    )


def _panel(tmp_path: Path) -> Path:
    rng = np.random.default_rng(2)
    rows = []
    for i in range(12):
        for p in range(1, 7):
            A, B, C = rng.uniform(0, 10), rng.uniform(0, 10), rng.uniform(0, 10)
            Y = max(A, B) * 0.8 + rng.uniform(0, 2)
            rows.append({"unit": f"u{i}", "period": p, "A": round(A, 2),
                         "B": round(B, 2), "C": round(C, 2), "Y": round(Y, 2)})
    df = pd.DataFrame(rows)
    csv = tmp_path / "pq.csv"
    df.to_csv(csv, index=False)
    return csv


@pytest.mark.skipif(not _HAS_SM, reason="R SetMethods/QCA not available")
def test_panel_qca_decomposition(tmp_path: Path) -> None:
    fp = profile_dataset(_panel(tmp_path))
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"outcome": "Y", "conditions": ["A", "B", "C"]},
    )
    assert "SetMethods" in res.summary
    assert res.estimates["n_terms"] >= 1
    assert 0 <= res.estimates["max_pooled_consistency"] <= 1
    # distances are non-negative consistency gaps (between/within -> pooled)
    assert res.estimates["max_dist_between"] >= 0
    assert res.estimates["max_dist_within"] >= 0


def test_panel_qca_needs_panel(tmp_path: Path) -> None:
    # cross-sectional data (no unit/time) -> honest failure, no R call
    df = pd.DataFrame({"A": [1.0, 2, 3, 4], "B": [4.0, 3, 2, 1], "Y": [1.0, 2, 2, 1]})
    csv = tmp_path / "x.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "面板 QCA 失败" in res.summary
