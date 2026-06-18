"""Tests for the event-study (dynamic DiD) executor branch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="event_study", method="Event-study (dynamic DiD)", domain="economics",
        family="causal", goal="explain",
        preconditions=Precondition(is_panel=True, requires_treatment=True, min_rows=40),
    )


def test_event_study_recovers_dynamic_effect(tmp_path: Path) -> None:
    # 15 units treated at t=5, 15 never-treated controls; post effect ~3 on top of unit/time FE
    rng = np.random.default_rng(0)
    rows = []
    for u in range(30):
        ufe = rng.normal(0, 1.0)
        treated_unit = u < 15
        for t in range(10):
            post = int(treated_unit and t >= 5)
            rows.append({"unit": u, "year": t, "treat": post,
                         "y": ufe + 0.1 * t + 3.0 * post + rng.normal(0, 0.5)})
    csv = tmp_path / "p.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"unit": "unit", "time": "year", "treatment": "treat", "outcome": "y"})
    assert "完成" in res.summary
    assert res.estimates["n_treated_units"] == 15
    assert res.estimates["att_post_mean"] > 1.5    # true post effect ~3 (some attenuation OK)


def test_event_study_all_treated_staggered_fails(tmp_path: Path) -> None:
    # every unit eventually treated + staggered onset, NO never-treated -> under-identified -> honest fail
    rng = np.random.default_rng(2)
    rows = []
    for u in range(20):
        onset = 3 + (u % 5)  # staggered, all treated
        for t in range(10):
            rows.append({"unit": u, "year": t, "treat": int(t >= onset), "y": rng.normal(0, 1)})
    csv = tmp_path / "p.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"unit": "unit", "time": "year", "treatment": "treat", "outcome": "y"})
    assert "事件研究失败" in res.summary


def test_event_study_no_lead_period_fails(tmp_path: Path) -> None:
    # treated from t=0 (no pre-treatment lead) -> no valid reference -> honest fail (not a lag baseline)
    rng = np.random.default_rng(3)
    rows = []
    for u in range(20):
        treated_unit = u < 10
        for t in range(8):
            rows.append({"unit": u, "year": t, "treat": int(treated_unit), "y": rng.normal(0, 1)})
    csv = tmp_path / "p.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"unit": "unit", "time": "year", "treatment": "treat", "outcome": "y"})
    assert "事件研究失败" in res.summary


def test_event_study_needs_panel(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"y": rng.normal(0, 1, 20), "treat": [0, 1] * 10})  # no unit/time
    csv = tmp_path / "x.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"treatment": "treat", "outcome": "y"})
    assert "事件研究失败" in res.summary
