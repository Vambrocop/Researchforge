"""Tests for the synthetic control (pysyncon) executor branch."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_HAS_PYSYNCON = importlib.util.find_spec("pysyncon") is not None


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="synthetic_control", method="Synthetic control", domain="economics",
        family="causal", goal="explain",
        preconditions=Precondition(is_panel=True, requires_treatment=True, requires_time=True, min_rows=30),
    )


def _panel(tmp_path: Path) -> Path:
    rng = np.random.default_rng(42)
    units = [f"region{i}" for i in range(12)]
    years = list(range(2000, 2016))
    uf = {u: rng.normal(0, 2) for u in units}
    fac = {y: 0.3 * (y - 2000) + rng.normal(0, 0.4) for y in years}
    rows = []
    for u in units:
        load = rng.uniform(0.5, 1.5)
        for y in years:
            base = 10 + uf[u] + load * fac[y] + rng.normal(0, 0.3)
            eff = 4.0 if (u == "region0" and y >= 2010) else 0.0
            treat = 1 if (u == "region0" and y >= 2010) else 0
            rows.append({"region": u, "year": y, "gdp": round(base + eff, 3), "treat": treat})
    df = pd.DataFrame(rows)
    csv = tmp_path / "synth.csv"
    df.to_csv(csv, index=False)
    return csv


@pytest.mark.skipif(not _HAS_PYSYNCON, reason="pysyncon not available")
def test_synthetic_control_recovers_positive_effect(tmp_path: Path) -> None:
    fp = profile_dataset(_panel(tmp_path))
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "pysyncon" in res.summary
    # true effect is +4; synthetic control should recover a clearly positive ATT
    assert res.estimates["att"] > 1.5
    assert res.estimates["pre_rmspe"] >= 0
    assert res.estimates["n_donors_used"] >= 1


@pytest.mark.skipif(not _HAS_PYSYNCON, reason="pysyncon not available")
def test_synthetic_control_config_treated_unit(tmp_path: Path) -> None:
    # drop the treatment column so detection relies on config instead
    df = pd.read_csv(_panel(tmp_path)).drop(columns=["treat"])
    csv = tmp_path / "notreat.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"treated_unit": "region0", "treatment_time": 2010, "outcome": "gdp"},
    )
    assert "pysyncon" in res.summary
    assert res.estimates["att"] > 1.5


@pytest.mark.skipif(not _HAS_PYSYNCON, reason="pysyncon not available")
def test_synthetic_control_excludes_other_treated_donors(tmp_path: Path) -> None:
    # staggered adoption: region0 treated @2010, region1 treated @2012.
    # region1 must be dropped from region0's donor pool (contamination bias).
    rng = np.random.default_rng(7)
    units = [f"region{i}" for i in range(12)]
    years = list(range(2000, 2016))
    uf = {u: rng.normal(0, 2) for u in units}
    fac = {y: 0.3 * (y - 2000) + rng.normal(0, 0.4) for y in years}
    rows = []
    for u in units:
        load = rng.uniform(0.5, 1.5)
        for y in years:
            base = 10 + uf[u] + load * fac[y] + rng.normal(0, 0.3)
            eff = 4.0 if (u == "region0" and y >= 2010) else (5.0 if (u == "region1" and y >= 2012) else 0.0)
            treat = 1 if ((u == "region0" and y >= 2010) or (u == "region1" and y >= 2012)) else 0
            rows.append({"region": u, "year": y, "gdp": round(base + eff, 3), "treat": treat})
    csv = tmp_path / "stag.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "pysyncon" in res.summary
    assert "被处理单位" in res.summary  # multi-treated disclosure fired
    # region1 (also treated) must NOT appear among donor weights
    wcsv = Path(res.output_dir) / "donor_weights.csv"
    if wcsv.exists():
        donors = set(pd.read_csv(wcsv)["donor"].astype(str))
        assert "region1" not in donors


def test_synthetic_control_needs_treated_unit(tmp_path: Path) -> None:
    # panel with no treatment column and no config -> honest failure (no R/pysyncon call)
    df = pd.read_csv(_panel(tmp_path)).drop(columns=["treat"])
    csv = tmp_path / "notreat.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "合成控制失败" in res.summary
