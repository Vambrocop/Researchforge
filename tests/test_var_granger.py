"""Tests for var_granger: multi-series gate + VAR + Granger causality direction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset
from researchforge.recommender.match import check_preconditions


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="var_granger",
        method="VAR + Granger causality",
        domain="economics",
        family="time-series",
        goal="explain",
        preconditions=Precondition(min_continuous=2, min_rows=30),
    )


def test_var_granger_direction(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 200
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + rng.normal(0, 1)
        y[t] = 0.4 * y[t - 1] + 0.6 * x[t - 1] + rng.normal(0, 1)  # x Granger-causes y
    df = pd.DataFrame({"x": np.round(x, 4), "y": np.round(y, 4)})
    csv = tmp_path / "var.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    pmat = pd.read_csv(Path(res.output_dir) / "granger_pvalues.csv", index_col=0)

    assert res.estimates["selected_lag"] >= 1
    # x -> y significant; y -> x not (directionality recovered)
    assert pmat.loc["x", "y"] < 0.05
    assert pmat.loc["y", "x"] > 0.05


def test_var_granger_precondition_unmet(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(0, 1, 40), "g": ["a", "b"] * 20})  # only 1 continuous
    csv = tmp_path / "one.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)

    assert not ok
    assert any("连续" in u for u in unmet)
