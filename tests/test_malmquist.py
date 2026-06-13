"""Tests for malmquist: panel gate + TFP-change index and EC×TC decomposition."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset
from researchforge.recommender.match import check_preconditions


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="malmquist",
        method="Malmquist productivity index (TFP change)",
        domain="economics",
        family="efficiency",
        goal="explain",
        preconditions=Precondition(is_panel=True, min_continuous=2, min_rows=6),
    )


def _panel(tmp_path: Path, growth: float) -> Path:
    base = {"A": (2.1, 1.1, 10.2), "B": (3.2, 2.1, 12.3), "C": (4.1, 3.2, 13.1), "D": (5.3, 4.1, 14.2)}
    rows = []
    for farm, (land, labor, yld) in base.items():
        rows.append({"farm": farm, "year": 2020, "yield": yld, "land": land, "labor": labor})
    for farm, (land, labor, yld) in base.items():
        # same inputs, output scaled by `growth` -> frontier shift (technical change)
        rows.append(
            {"farm": farm, "year": 2021, "yield": round(yld * growth, 2), "land": land, "labor": labor}
        )
    csv = tmp_path / f"panel_{growth}.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def test_malmquist_detects_tfp_growth(tmp_path: Path) -> None:
    fp = profile_dataset(_panel(tmp_path, growth=1.3))  # 30% output gain at same inputs
    assert fp.is_panel and fp.unit_col == "farm" and fp.time_col == "year"

    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    tab = pd.read_csv(Path(res.output_dir) / "malmquist.csv")

    assert set(["malmquist_tfp", "efficiency_change", "technical_change"]).issubset(tab.columns)
    assert res.estimates["mean_malmquist_tfp"] > 1.1  # clear TFP growth
    assert res.estimates["mean_technical_change"] > 1.05  # driven by frontier shift


def test_malmquist_no_change_is_unity(tmp_path: Path) -> None:
    fp = profile_dataset(_panel(tmp_path, growth=1.0))  # identical periods
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert abs(res.estimates["mean_malmquist_tfp"] - 1.0) < 0.03  # no productivity change


def test_malmquist_precondition_unmet(tmp_path: Path) -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"y": rng.normal(0, 1, 20), "x": rng.normal(0, 1, 20)})  # not panel
    csv = tmp_path / "flat.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)

    assert not ok
    assert any("面板" in u for u in unmet)
