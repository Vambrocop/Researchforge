"""Tests for the causal forest / CATE (econml) executor branch."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_HAS_ECONML = importlib.util.find_spec("econml") is not None


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="causal_forest", method="Causal forest", domain="economics", family="causal",
        goal="explain", preconditions=Precondition(requires_treatment=True, min_rows=200),
    )


def _data(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    n = 1500
    x0, x1, x2 = rng.normal(0, 1, n), rng.normal(0, 1, n), rng.normal(0, 1, n)
    d = (x0 + rng.normal(0, 1, n) > 0).astype(int)
    tau = 1.0 + 1.5 * x0  # heterogeneous effect driven by x0
    y = tau * d + x0 + 0.5 * x1 + rng.normal(0, 1, n)
    df = pd.DataFrame({"x0": x0, "x1": x1, "x2": x2, "treat": d, "y": y})
    csv = tmp_path / "cf.csv"
    df.to_csv(csv, index=False)
    return csv


@pytest.mark.skipif(not _HAS_ECONML, reason="econml not available")
def test_causal_forest_detects_heterogeneity(tmp_path: Path) -> None:
    fp = profile_dataset(_data(tmp_path))
    out = str(tmp_path / "o")
    res = run_analysis(fp, _entry(), output_root=out, config={"treatment": "treat", "outcome": "y"})
    assert "econml" in res.summary
    # overall ATE ~ 1.0 (E[1+1.5*x0] = 1)
    assert res.estimates["ate_lb"] <= 1.0 <= res.estimates["ate_ub"]
    # real heterogeneity present (CATE spread well above 0)
    assert res.estimates["cate_sd"] > 0.4
    # x0 is the true effect-modifier -> top heterogeneity driver
    drivers = pd.read_csv(Path(res.output_dir) / "heterogeneity_drivers.csv")
    assert drivers.iloc[0]["modifier"] == "x0"


def test_causal_forest_no_treatment_degrades(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"y": rng.normal(0, 1, 250), "a": rng.normal(0, 1, 250), "b": rng.normal(0, 1, 250)})
    csv = tmp_path / "x.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "因果森林失败" in res.summary and ("处理变量" in res.summary or "econml" in res.summary)
