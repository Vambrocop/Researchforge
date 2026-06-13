"""Tests for efa: continuous gate + factor recovery (sklearn FA + varimax)."""

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
        id="efa",
        method="Exploratory factor analysis (EFA)",
        domain="statistics",
        family="statistics",
        goal="explore",
        preconditions=Precondition(min_continuous=3, min_rows=50),
    )


def test_efa_recovers_two_factors(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 300
    f1 = rng.normal(0, 1, n)
    f2 = rng.normal(0, 1, n)
    df = pd.DataFrame(
        {
            "v1": 0.8 * f1 + rng.normal(0, 0.5, n),
            "v2": 0.75 * f1 + rng.normal(0, 0.5, n),
            "v3": 0.7 * f1 + rng.normal(0, 0.5, n),
            "v4": 0.8 * f2 + rng.normal(0, 0.5, n),
            "v5": 0.75 * f2 + rng.normal(0, 0.5, n),
            "v6": 0.7 * f2 + rng.normal(0, 0.5, n),
        }
    )
    csv = tmp_path / "efa.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    load = pd.read_csv(Path(res.output_dir) / "loadings.csv", index_col=0)

    assert res.estimates["n_factors"] == 2  # Kaiser recovers the 2 factors
    assert load.shape == (6, 2)
    # v1-v3 share one dominant factor, v4-v6 the other (factor order/sign arbitrary)
    dom = load.abs().idxmax(axis=1)
    assert dom["v1"] == dom["v2"] == dom["v3"]
    assert dom["v4"] == dom["v5"] == dom["v6"]
    assert dom["v1"] != dom["v4"]


def test_efa_precondition_unmet(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(0, 1, 60), "g": ["a", "b"] * 30})  # only 1 continuous
    csv = tmp_path / "one.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)

    assert not ok
    assert any("连续" in u for u in unmet)
