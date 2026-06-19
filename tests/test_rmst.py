"""Tests for rmst: between-group RMST difference sign + detection, tau reporting.

Synthetic structure: two groups with different survival (group 1 lives longer).
We assert RMST(group 1) > RMST(group 0), the reported difference has the right
sign, and the z-test detects it. Plus tau is reported and an honest-skip test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="rmst",
        method="Restricted Mean Survival Time (RMST)",
        domain="statistics",
        family="survival",
        goal="explain",
        preconditions=Precondition(requires_binary_outcome=True, min_continuous=1, min_rows=20),
    )


def _two_group_data(n=300, seed=0) -> pd.DataFrame:
    """group 0: shorter survival (scale 8); group 1: longer survival (scale 16)."""
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 2, n)
    scale = np.where(g == 1, 16.0, 8.0)
    t = rng.exponential(scale=scale)
    cens = rng.exponential(scale=40.0, size=n)
    time = np.minimum(t, cens)
    event = (t <= cens).astype(int)
    return pd.DataFrame({
        "duration": np.round(time, 4),
        "event": event,
        "arm": g.astype(int),
    })


def test_rmst_diff_sign_and_detection(tmp_path: Path) -> None:
    pytest.importorskip("lifelines")
    df = _two_group_data()
    csv = tmp_path / "rmst.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    assert (out / "rmst_curves.png").exists()
    assert (out / "rmst_table.csv").exists()
    assert "tau" in res.estimates and res.estimates["tau"] > 0

    # group 1 (scale 16) lives longer -> larger RMST than group 0 (scale 8)
    assert res.estimates["rmst_1"] > res.estimates["rmst_0"]
    # the reported difference (arm=1 - arm=0) is positive and detected
    assert res.estimates["rmst_diff"] > 0
    assert res.estimates["rmst_diff_p"] < 0.05
    # CI should exclude 0 when the difference is significant
    assert res.estimates["rmst_diff_ci_low"] > 0

    tab = pd.read_csv(out / "rmst_table.csv")
    assert len(tab) == 2
    assert (tab["RMST"] > 0).all()
    assert (tab["SE"] >= 0).all()


def test_rmst_tau_override_and_clamp(tmp_path: Path) -> None:
    pytest.importorskip("lifelines")
    df = _two_group_data(seed=7)
    csv = tmp_path / "rmst.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    # ask for a small tau within support -> respected exactly
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"duration": "duration", "event": "event",
                               "group": "arm", "tau": 5.0})
    assert abs(res.estimates["tau"] - 5.0) < 1e-6
    # RMST restricted to [0,5] cannot exceed tau
    assert res.estimates["rmst_1"] <= 5.0 + 1e-6
    assert res.estimates["rmst_0"] <= 5.0 + 1e-6


def test_rmst_missing_cols_skips(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(0, 1, 30), "b": rng.normal(0, 1, 30)})  # no event
    csv = tmp_path / "plain.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "RMST 失败" in res.summary
    assert "rmst_diff" not in res.estimates
