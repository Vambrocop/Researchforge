"""Tests for the changes-in-changes (qte::CiC) executor branch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis, rbridge
from researchforge.profiler import profile_dataset

_HAS_QTE = rbridge.r_available() and rbridge.r_package_available("qte")


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="changes_in_changes", method="Changes-in-changes", domain="economics",
        family="causal", goal="explain",
        preconditions=Precondition(requires_treatment=True, requires_time=True, min_rows=100),
    )


def _data(tmp_path: Path) -> Path:
    rng = np.random.default_rng(1)
    n = 500
    rows = []
    for grp, period, eff in [(0, 1, 0), (0, 2, 0), (1, 1, 0), (1, 2, 2.0)]:
        for _ in range(n):
            rows.append({"y": rng.normal(5 + 0.5 * period, 1.5) + eff, "treat": grp, "period": period})
    df = pd.DataFrame(rows)
    csv = tmp_path / "cic.csv"
    df.to_csv(csv, index=False)
    return csv


@pytest.mark.skipif(not _HAS_QTE, reason="R qte not available")
def test_cic_recovers_att(tmp_path: Path) -> None:
    fp = profile_dataset(_data(tmp_path))
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"outcome": "y", "treatment": "treat", "time": "period"},
    )
    assert "qte" in res.summary
    # true ATT = 2.0; CiC ATT CI should cover it
    assert res.estimates["att_lb"] <= 2.0 <= res.estimates["att_ub"]
    # constant effect -> QTE roughly flat near 2
    assert abs(res.estimates["qte_min"] - 2.0) < 0.8
    assert abs(res.estimates["qte_max"] - 2.0) < 0.8


def test_cic_no_time_degrades(tmp_path: Path) -> None:
    # no time column -> honest failure (or qte-missing message)
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"y": rng.normal(0, 1, 120), "treat": rng.integers(0, 2, 120), "z": rng.normal(0, 1, 120)})
    csv = tmp_path / "x.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "Changes-in-changes" in res.summary and ("失败" in res.summary or "qte" in res.summary)
