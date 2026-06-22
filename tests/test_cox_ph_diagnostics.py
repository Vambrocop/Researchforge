"""Tests for cox_ph_diagnostics (proportional-hazards test via Schoenfeld residuals).

Synthetic structure: a deliberately NON-proportional covariate `x_np` whose effect
turns on only after a time point (its hazard contribution changes over follow-up),
plus a well-behaved proportional covariate `x_p` with a constant effect. The
Schoenfeld PH test should flag x_np (low p, violates_PH) and not x_p. Plus an
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
        id="cox_ph_diagnostics",
        method="Cox proportional-hazards diagnostics (Schoenfeld residuals)",
        domain="statistics",
        family="survival",
        goal="explain",
        preconditions=Precondition(requires_binary_outcome=True, min_continuous=1, min_rows=30),
    )


def _nonproportional_data(n=1000, seed=1) -> pd.DataFrame:
    """One non-PH covariate + one PH covariate, via a TWO-PHASE piecewise-exponential.

    The hazard is piecewise-constant in time with a breakpoint tau:
      - x_p enters BOTH phases with the same log-hazard coefficient -> proportional.
      - x_np's coefficient CROSSES at tau (strong positive before tau, strong negative
        after), so its hazard ratio changes with time -> non-proportional, which makes
        the scaled-Schoenfeld residual of x_np trend with time (low PH p-value).
    Memoryless property makes the construction exact: draw phase-1 time from Exp(h1);
    if it survives past tau, draw the remainder from Exp(h2). Censor at the 90th pct.
    """
    rng = np.random.default_rng(seed)
    x_p = rng.normal(0, 1, n)
    x_np = rng.normal(0, 1, n)

    tau = 0.4
    lin_p = 0.5 * x_p                       # proportional (both phases share this)
    h1 = np.exp(lin_p + 0.9 * x_np)         # phase 1 (t < tau): x_np HR = exp(+0.9)
    h2 = np.exp(lin_p + 0.0 * x_np)         # phase 2 (t >= tau): x_np effect vanishes

    t1 = rng.exponential(scale=1.0 / h1)
    t = np.where(t1 < tau, t1, tau + rng.exponential(scale=1.0 / h2))
    t = np.clip(t, 1e-4, None)

    c = np.quantile(t, 0.90)
    time = np.minimum(t, c)
    event = (t <= c).astype(int)
    return pd.DataFrame({
        "time": np.round(time, 4),
        "event": event,
        "x_p": np.round(x_p, 4),
        "x_np": np.round(x_np, 4),
    })


def test_ph_diagnostics_flags_nonproportional(tmp_path: Path) -> None:
    pytest.importorskip("lifelines")
    df = _nonproportional_data()
    csv = tmp_path / "ph.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"duration": "time", "event": "event", "covariates": ["x_p", "x_np"]},
    )
    out = Path(res.output_dir)

    assert (out / "ph_test_table.csv").exists()
    assert (out / "ph_test_bar.png").exists()

    for k in ("global_ph_p", "n_violations", "min_ph_p", "n_covariates", "concordance"):
        assert k in res.estimates, f"missing estimate {k}"
    assert res.estimates["n_covariates"] == 2.0
    assert 0.0 <= res.estimates["min_ph_p"] <= 1.0

    tab = pd.read_csv(out / "ph_test_table.csv")
    assert set(["covariate", "ph_test_statistic", "ph_p_value", "violates_PH"]).issubset(tab.columns)
    # the non-proportional covariate should have a smaller PH p than the proportional one
    p_np = float(tab.loc[tab["covariate"] == "x_np", "ph_p_value"].iloc[0])
    p_p = float(tab.loc[tab["covariate"] == "x_p", "ph_p_value"].iloc[0])
    assert p_np < p_p
    # at least the non-proportional covariate is flagged
    assert res.estimates["n_violations"] >= 1
    assert "完成" in res.summary


def test_ph_diagnostics_missing_event_skips(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(0, 1, 40), "b": rng.normal(0, 1, 40)})  # no event
    csv = tmp_path / "plain.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "比例风险诊断失败" in res.summary
    assert "min_ph_p" not in res.estimates
