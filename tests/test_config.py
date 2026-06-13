"""Tests for the user-config override mechanism (run_analysis(config=...))."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _ols_entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="ols_regression",
        method="OLS regression",
        domain="statistics",
        family="statistics",
        goal="explain",
        preconditions=Precondition(min_continuous=2, min_rows=20),
    )


def test_config_outcome_and_predictors_override(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 150
    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    y = 2.0 * a + rng.normal(0, 0.3, n)
    # column order makes 'a' the first continuous -> the DEFAULT outcome
    df = pd.DataFrame({"a": a, "b": b, "y": y})
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)

    # default: outcome = first continuous ('a'); 'a' is the DV, not a predictor key
    res0 = run_analysis(fp, _ols_entry(), output_root=str(tmp_path / "o0"))
    assert "a" not in res0.estimates

    # override: outcome='y', predictors=['a'] -> regresses y ~ a, slope ~ 2
    res1 = run_analysis(
        fp,
        _ols_entry(),
        output_root=str(tmp_path / "o1"),
        config={"outcome": "y", "predictors": ["a"]},
    )
    assert abs(res1.estimates["a"] - 2.0) < 0.3


def test_config_none_is_default(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"y": 1.5 * rng.normal(0, 1, 60), "x": rng.normal(0, 1, 60)})
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    # config=None must behave exactly like no config (no crash, runs)
    res = run_analysis(fp, _ols_entry(), output_root=str(tmp_path / "o"), config=None)
    assert res.summary
