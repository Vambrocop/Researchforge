"""Tests for dea: input/output gate + CCR/BCC efficiency frontier."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset
from researchforge.recommender.match import check_preconditions


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="dea",
        method="Data Envelopment Analysis (DEA, CCR/BCC)",
        domain="economics",
        family="efficiency",
        goal="explain",
        preconditions=Precondition(min_continuous=2, min_rows=4),
    )


def test_dea_identifies_efficient_dmu(tmp_path: Path) -> None:
    # F1: highest output with the lowest inputs -> must be CCR-efficient (theta=1)
    df = pd.DataFrame(
        {
            "farm": ["F1", "F2", "F3", "F4", "F5"],
            "output_yield": [10.2, 8.1, 6.3, 9.4, 5.5],  # first continuous = output
            "input_land": [2.1, 3.2, 4.3, 5.4, 4.1],
            "input_labor": [1.1, 2.2, 3.3, 4.4, 3.2],
        }
    )
    csv = tmp_path / "farms.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    eff = pd.read_csv(Path(res.output_dir) / "dea_efficiency.csv")

    assert set(["DMU", "CCR_efficiency", "BCC_efficiency", "scale_efficiency"]).issubset(eff.columns)
    # F1 dominates -> efficient; all CCR scores within (0, 1]; at least one frontier DMU
    f1 = eff[eff["DMU"] == "F1"].iloc[0]
    assert f1["CCR_efficiency"] >= 0.999
    assert (eff["CCR_efficiency"] <= 1.0 + 1e-6).all()
    assert res.estimates["n_ccr_efficient"] >= 1


def test_dea_precondition_unmet(tmp_path: Path) -> None:
    df = pd.DataFrame({"only_output": [1.1, 2.2, 3.3, 4.4]})  # < 2 numeric columns
    csv = tmp_path / "one.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)

    assert not ok
    assert any("连续" in u for u in unmet)
