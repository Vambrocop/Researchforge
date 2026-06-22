"""Tests for time_varying_cox (CoxTimeVaryingFitter on counting-process long data).

Synthetic structure: each subject is split into 2 counting-process intervals
[0, s) and [s, stop). A time-varying covariate `tv` switches value at the split;
a baseline covariate `x` is constant within subject. We bake in a strong positive
log-hazard effect for `tv` (higher tv -> higher hazard -> HR > 1) by making the
event more likely on intervals with high tv. We assert the recovered HR direction,
the products, and the estimates keys. Plus honest-skip tests for non-counting-process
data and missing columns.
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
        id="time_varying_cox",
        method="Time-varying Cox PH (counting-process covariates)",
        domain="statistics",
        family="survival",
        goal="explain",
        preconditions=Precondition(
            is_panel=True, requires_binary_outcome=True, min_continuous=1, min_rows=30
        ),
    )


def _counting_process_data(n_subjects=180, beta_tv=1.2, seed=0) -> pd.DataFrame:
    """Build counting-process long data with a known time-varying effect.

    Each subject gets two intervals: [0, 1) with tv=0 and [1, T) with tv switched
    to a subject-specific level. The event hazard rises with the active tv value,
    so the recovered HR for tv should exceed 1.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for sid in range(n_subjects):
        x = float(rng.normal(0, 1))            # baseline covariate (constant)
        tv_level = float(rng.integers(0, 2))   # 0 or 1 in the second interval
        # second-interval hazard increases with tv_level (and a bit with x)
        lin = beta_tv * tv_level + 0.4 * x
        # probability of the event occurring in the second interval
        p_event = 1.0 / (1.0 + np.exp(-(lin - 0.3)))
        ev2 = int(rng.uniform() < p_event)
        stop2 = float(1.0 + rng.uniform(0.5, 2.0))
        # interval 1: [0,1), tv=0, no event
        rows.append({"subject_id": sid, "start": 0.0, "stop": 1.0,
                     "event": 0, "tv": 0.0, "x": round(x, 4)})
        # interval 2: [1, stop2), tv=tv_level, event = ev2
        rows.append({"subject_id": sid, "start": 1.0, "stop": round(stop2, 4),
                     "event": ev2, "tv": tv_level, "x": round(x, 4)})
    return pd.DataFrame(rows)


def test_time_varying_cox_recovers_effect(tmp_path: Path) -> None:
    pytest.importorskip("lifelines")
    df = _counting_process_data()
    csv = tmp_path / "ctv.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    # pass roles explicitly so the test does not depend on profiler heuristics
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"id": "subject_id", "start": "start", "stop": "stop",
                "event": "event", "covariates": ["tv", "x"]},
    )
    out = Path(res.output_dir)

    assert (out / "tv_cox_hazard_ratios.csv").exists()
    assert (out / "tv_cox_forest.png").exists()

    # estimates keys per the spec
    for k in ("loglik", "n_covariates", "n_intervals", "max_abs_hr", "n_subjects"):
        assert k in res.estimates, f"missing estimate {k}"
    assert res.estimates["n_subjects"] == 180.0
    assert res.estimates["n_intervals"] == 360.0
    assert res.estimates["n_covariates"] == 2.0

    # the time-varying covariate has a strong positive effect -> HR > 1
    assert res.estimates["HR_tv"] > 1.0

    tab = pd.read_csv(out / "tv_cox_hazard_ratios.csv", index_col=0)
    assert "hazard_ratio" in tab.columns and "p_value" in tab.columns
    assert "完成" in res.summary


def test_time_varying_cox_not_long_format_skips(tmp_path: Path) -> None:
    """A flat (one-row-per-subject) frame is not counting-process -> honest skip."""
    pytest.importorskip("lifelines")
    rng = np.random.default_rng(2)
    n = 60
    df = pd.DataFrame({
        "subject_id": np.arange(n),
        "start": np.zeros(n),
        "stop": rng.uniform(1, 5, n).round(3),
        "event": (rng.uniform(0, 1, n) < 0.5).astype(int),
        "tv": rng.normal(0, 1, n).round(3),
    })
    csv = tmp_path / "flat.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"id": "subject_id", "start": "start", "stop": "stop",
                "event": "event", "covariates": ["tv"]},
    )
    assert "时变 Cox 失败" in res.summary
    assert "loglik" not in res.estimates


def test_time_varying_cox_missing_cols_skips(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"a": rng.normal(0, 1, 40), "b": rng.normal(0, 1, 40)})
    csv = tmp_path / "plain.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "时变 Cox 失败" in res.summary
    assert "loglik" not in res.estimates
