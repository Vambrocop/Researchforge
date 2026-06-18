"""Tests for the staggered DiD (Sun-Abraham interaction-weighted) executor branch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="staggered_did", method="Staggered DiD (Sun-Abraham)", domain="economics",
        family="causal", goal="explain",
        preconditions=Precondition(is_panel=True, requires_treatment=True, min_rows=60),
    )


def test_staggered_did_recovers_heterogeneous_att(tmp_path: Path) -> None:
    # 3 cohorts (onset 4/6/8) with HETEROGENEOUS constant post effects 2/3/4 + a never-treated group.
    # Pooled TWFE is biased here; Sun-Abraham IW should recover an overall ATT around the cohort mix (~3).
    rng = np.random.default_rng(0)
    eff = {4: 2.0, 6: 3.0, 8: 4.0}
    rows = []
    uid = 0
    for onset in [4, 6, 8, None]:          # None = never-treated control group
        for _ in range(12):                # 12 units per group -> 48 units x 12 periods = 576 rows
            ufe = rng.normal(0, 1.0)
            for t in range(12):
                treat = 0 if onset is None else int(t >= onset)
                e_eff = 0.0 if onset is None else eff[onset] * treat
                y = ufe + 0.2 * t + e_eff + rng.normal(0, 0.4)
                rows.append({"unit": uid, "year": t, "treat": treat, "y": y})
            uid += 1
    csv = tmp_path / "p.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"unit": "unit", "time": "year", "treatment": "treat", "outcome": "y"})
    assert "完成" in res.summary
    assert res.estimates["n_cohorts"] == 3
    assert res.estimates["n_treated_units"] == 36
    assert res.estimates["n_never_treated"] == 12
    assert abs(res.estimates["att_overall"] - 3.0) < 1.0     # cohort-share-weighted truth ~3
    assert res.estimates["att_overall_se"] > 0
    assert res.estimates["pretrend_violation"] == 0.0        # effects are post-only -> clean pre-trends


def test_staggered_did_no_never_treated_fails(tmp_path: Path) -> None:
    # every unit eventually treated (staggered) -> no clean control -> honest fail toward R did
    rng = np.random.default_rng(2)
    rows = []
    for u in range(20):
        onset = 3 + (u % 5)
        for t in range(12):
            rows.append({"unit": u, "year": t, "treat": int(t >= onset), "y": rng.normal(0, 1)})
    csv = tmp_path / "p.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"unit": "unit", "time": "year", "treatment": "treat", "outcome": "y"})
    assert "交错DiD失败" in res.summary
    assert "从未处理" in res.summary


def test_staggered_did_needs_panel(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"y": rng.normal(0, 1, 30), "treat": [0, 1] * 15})  # no unit/time
    csv = tmp_path / "x.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"treatment": "treat", "outcome": "y"})
    assert "交错DiD失败" in res.summary
