"""Tests for stratified_cox (CoxPHFitter with strata: per-stratum baseline hazards).

Synthetic structure: two strata with DIFFERENT baseline hazard scales but a COMMON
covariate effect for x (higher x -> shorter survival -> HR > 1). The stratified Cox
should recover HR_x > 1, report 2 strata, and emit the table + forest plot. Plus an
honest-skip test for missing event column.
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
        id="stratified_cox",
        method="Stratified Cox PH (per-stratum baseline hazards)",
        domain="statistics",
        family="survival",
        goal="explain",
        preconditions=Precondition(
            requires_binary_outcome=True, requires_group=True, min_continuous=1, min_rows=30
        ),
    )


def _stratified_data(n_per=140, beta=0.8, seed=0) -> pd.DataFrame:
    """Two strata, different baseline scales, common covariate effect for x.

    Stratum 0 baseline scale 12, stratum 1 baseline scale 4 (different baseline
    hazard). In both, hazard rises with x via exp(beta*x) -> survival ~ Exp scaled
    by exp(-beta*x). Common beta across strata -> HR_x = exp(beta) > 1.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for s, base_scale in ((0, 12.0), (1, 4.0)):
        x = rng.normal(0, 1, n_per)
        scale = base_scale * np.exp(-beta * x)          # common covariate effect
        t = rng.exponential(scale=scale)
        c = np.quantile(t, 0.85)                          # ~15% administrative censoring
        time = np.minimum(t, c)
        event = (t <= c).astype(int)
        frames.append(pd.DataFrame({
            "time": np.round(time, 4),
            "event": event,
            "x": np.round(x, 4),
            "site": s,
        }))
    df = pd.concat(frames, ignore_index=True)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def test_stratified_cox_recovers_effect(tmp_path: Path) -> None:
    pytest.importorskip("lifelines")
    df = _stratified_data()
    csv = tmp_path / "strat.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"duration": "time", "event": "event", "strata": "site", "covariates": ["x"]},
    )
    out = Path(res.output_dir)

    assert (out / "stratified_cox_hazard_ratios.csv").exists()
    assert (out / "stratified_cox_forest.png").exists()

    for k in ("loglik", "concordance", "n_strata", "n_covariates", "max_abs_hr"):
        assert k in res.estimates, f"missing estimate {k}"
    assert res.estimates["n_strata"] == 2.0
    assert res.estimates["n_covariates"] == 1.0
    assert res.estimates["HR_x"] > 1.0      # common covariate effect recovered
    assert 0.0 <= res.estimates["concordance"] <= 1.0

    tab = pd.read_csv(out / "stratified_cox_hazard_ratios.csv", index_col=0)
    xrow = tab.loc["x"]
    assert xrow["HR_lower95"] < xrow["hazard_ratio"] < xrow["HR_upper95"]
    assert "完成" in res.summary
    assert "分层" in res.summary


def test_stratified_cox_missing_event_skips(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(0, 1, 40), "b": rng.normal(0, 1, 40)})  # no event
    csv = tmp_path / "plain.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "分层 Cox 失败" in res.summary
    assert "loglik" not in res.estimates
