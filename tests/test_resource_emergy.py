"""Emergy analysis (Odum): U=R+N+F, EYR=U/F, ELR=(N+F)/R, ESI=EYR/ELR.

Known-value cases use config transformities/categories so the arithmetic is exact;
a library-match case and an honest-degrade case round it out.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(id="emergy_analysis", method="Emergy analysis",
                         domain="sustainability", family="resource", goal="evaluate",
                         preconditions=Precondition(min_rows=1))


def _csv(tmp_path: Path, name: str, df: pd.DataFrame) -> Path:
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


def test_emergy_indicators_known_values(tmp_path: Path) -> None:
    """quantities sun=100, fuel=50, electricity=50; transformities all 1.0;
    categories R/N/F -> R=100, N=50, F=50, U=200; EYR=U/F=4, ELR=(N+F)/R=1,
    ESI=EYR/ELR=4, %renewable=R/U=0.5."""
    df = pd.DataFrame({"sun": [50.0, 50.0], "fuel": [25.0, 25.0],
                       "elec": [25.0, 25.0]})
    res = run_analysis(profile_dataset(_csv(tmp_path, "em.csv", df)), _entry(),
                       output_root=str(tmp_path / "o"),
                       config={"transformities": {"sun": 1.0, "fuel": 1.0, "elec": 1.0},
                               "categories": {"sun": "R", "fuel": "N", "elec": "F"}})
    e = res.estimates
    assert math.isclose(e["total_emergy_U"], 200.0, abs_tol=1e-6)
    assert math.isclose(e["emergy_R"], 100.0, abs_tol=1e-6)
    assert math.isclose(e["eyr"], 4.0, abs_tol=1e-6)
    assert math.isclose(e["elr"], 1.0, abs_tol=1e-6)
    assert math.isclose(e["esi"], 4.0, abs_tol=1e-6)
    assert math.isclose(e["pct_renewable"], 0.5, abs_tol=1e-6)
    assert e["n_inputs"] == 3.0


def test_emergy_library_match(tmp_path: Path) -> None:
    """A column named 'solar' / 'wind' resolves a transformity from the built-in
    public library (no config) -> a positive total emergy + baseline disclosed."""
    df = pd.DataFrame({"solar": [1000.0, 1000.0], "wind": [500.0, 500.0]})
    res = run_analysis(profile_dataset(_csv(tmp_path, "lib.csv", df)), _entry(),
                       output_root=str(tmp_path / "o"))
    assert "跳过" not in res.summary
    assert res.estimates["total_emergy_U"] > 0
    assert "基线" in res.summary  # baseline disclosed


def test_emergy_degrades_without_transformities(tmp_path: Path) -> None:
    df = pd.DataFrame({"widget_a": [1.0, 2.0], "gizmo_b": [3.0, 4.0]})
    res = run_analysis(profile_dataset(_csv(tmp_path, "x.csv", df)), _entry(),
                       output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
    assert "total_emergy_U" not in res.estimates
