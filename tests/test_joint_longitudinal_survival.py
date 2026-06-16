"""Tests for the joint longitudinal–survival model (JM) executor branch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis, rbridge
from researchforge.profiler import profile_dataset

_HAS_JM = rbridge.r_available() and rbridge.r_package_available("JM")


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="joint_longitudinal_survival", method="Joint model", domain="statistics",
        family="survival", goal="explain",
        preconditions=Precondition(is_panel=True, requires_binary_outcome=True, min_rows=60),
    )


_DEMO = Path(__file__).resolve().parent.parent / "data" / "demo_joint.csv"


@pytest.mark.skipif(not _HAS_JM, reason="R JM not available")
@pytest.mark.skipif(not _DEMO.exists(), reason="demo_joint.csv (aids export) not present")
def test_joint_model_association(tmp_path: Path) -> None:
    # canonical aids data (CD4 marker + survival): higher CD4 -> lower hazard
    fp = profile_dataset(_DEMO)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"id": "patient", "time": "obstime", "marker": "CD4",
                "event_time": "Time", "event": "death", "covariates": ["drug"]},
    )
    assert "R/JM" in res.summary
    assert res.estimates["association"] < 0  # higher CD4 -> lower event hazard
    assert res.estimates["association_p"] < 0.05
    assert res.estimates["n_events"] == 188


def test_joint_model_needs_structure(tmp_path: Path) -> None:
    # flat cross-sectional data with no marker/event structure -> honest failure
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"a": rng.normal(0, 1, 80), "b": rng.normal(0, 1, 80)})
    csv = tmp_path / "x.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "联合模型失败" in res.summary or "联合模型需要" in res.summary
