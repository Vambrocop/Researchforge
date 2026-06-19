"""Tests for the cointegration / VECM executor branch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="cointegration_vecm", method="Cointegration + VECM", domain="economics",
        family="time-series", goal="explain",
        preconditions=Precondition(min_continuous=2, min_rows=30),
    )


def test_cointegration_detects_longrun_relation(tmp_path: Path) -> None:
    # two I(1) series sharing a long-run equilibrium: y2 = 0.7*y1 + stationary noise
    rng = np.random.default_rng(0)
    n = 250
    y1 = np.cumsum(rng.normal(0, 1, n))             # random walk -> I(1)
    y2 = 0.7 * y1 + rng.normal(0, 0.5, n)           # cointegrated with y1
    df = pd.DataFrame({"t": np.arange(n), "price": y1, "cost": y2})
    csv = tmp_path / "c.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"series": ["price", "cost"]})
    assert "完成" in res.summary
    assert res.estimates["n_coint_relations"] >= 1          # Johansen finds the relation
    assert res.estimates["levels_nonstationary"] >= 1       # levels are I(1)
    assert res.estimates["ect_adf_pvalue"] < 0.10           # error-correction term is mean-reverting


def test_cointegration_independent_walks_none(tmp_path: Path) -> None:
    # two INDEPENDENT random walks -> no long-run equilibrium -> r = 0
    rng = np.random.default_rng(3)
    n = 250
    a = np.cumsum(rng.normal(0, 1, n))
    b = np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame({"t": np.arange(n), "a": a, "b": b})
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"series": ["a", "b"]})
    assert "完成" in res.summary
    assert res.estimates["n_coint_relations"] == 0
    assert "未检出协整" in res.summary


def test_cointegration_needs_two_series(tmp_path: Path) -> None:
    df = pd.DataFrame({"t": np.arange(40), "x": np.cumsum(np.random.default_rng(1).normal(0, 1, 40))})
    csv = tmp_path / "x.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "协整/VECM 失败" in res.summary
