"""Tests for survival_analysis: event/duration gate + KM + Cox (skips no lifelines)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset
from researchforge.recommender.match import check_preconditions


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="survival_analysis",
        method="Survival analysis (Kaplan-Meier + Cox PH)",
        domain="statistics",
        family="survival",
        goal="explain",
        preconditions=Precondition(requires_binary_outcome=True, min_continuous=1, min_rows=15),
    )


def test_survival_km_and_cox(tmp_path: Path) -> None:
    pytest.importorskip("lifelines")
    rng = np.random.default_rng(0)
    n = 220
    x = rng.normal(0, 1, n)
    grp = rng.integers(0, 2, n)
    # higher x and grp=1 -> shorter survival -> hazard ratio > 1
    dur = rng.exponential(scale=np.exp(-0.5 * x - 0.4 * grp)) * 10
    event = (rng.uniform(0, 1, n) < 0.8).astype(int)  # 80% events, 20% censored
    df = pd.DataFrame({"time": np.round(dur, 3), "event": event, "x": np.round(x, 3), "grp": grp})
    csv = tmp_path / "surv.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    assert (out / "km_curve.png").exists()
    assert (out / "cox_hazard_ratios.csv").exists()
    assert res.estimates["median_survival"] > 0
    assert res.estimates["n_events"] >= 1
    assert res.estimates["HR_x"] > 1.0  # hazard rises with x (recovered)


def test_survival_precondition_unmet(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(0, 1, 20), "b": rng.normal(0, 1, 20)})  # no binary event
    csv = tmp_path / "cont.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)

    assert not ok
    assert any("二值" in u for u in unmet)
