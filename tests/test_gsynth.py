"""Tests for the generalized synthetic control (gsynth) executor branch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis, rbridge
from researchforge.profiler import profile_dataset

_HAS_GSYNTH = rbridge.r_available() and rbridge.r_package_available("gsynth")
_DEMO = Path(__file__).resolve().parent.parent / "data" / "demo_gsynth.csv"


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="gsynth", method="Generalized synthetic control", domain="economics",
        family="causal", goal="explain",
        preconditions=Precondition(is_panel=True, requires_treatment=True, requires_time=True, min_rows=100),
    )


@pytest.mark.skipif(not _HAS_GSYNTH, reason="R gsynth not available")
@pytest.mark.skipif(not _DEMO.exists(), reason="demo_gsynth.csv (simdata export) not present")
def test_gsynth_recovers_att(tmp_path: Path) -> None:
    fp = profile_dataset(_DEMO)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"outcome": "Y", "treatment": "D"},
    )
    assert "gsynth" in res.summary
    # simdata's true ATT is ~4.6 (textbook gsynth result at the default seed)
    assert 3.0 < res.estimates["att"] < 7.0
    assert res.estimates["att_lb"] <= res.estimates["att"] <= res.estimates["att_ub"]
    assert res.estimates["n_treated_units"] == 5.0
    assert res.estimates["n_factors"] >= 1.0  # CV recovers latent factor structure


def test_gsynth_needs_panel(tmp_path: Path) -> None:
    # flat cross-sectional data -> honest failure (no R call)
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"y": rng.normal(0, 1, 120), "d": rng.integers(0, 2, 120)})
    csv = tmp_path / "x.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "广义合成控制失败" in res.summary or "广义合成控制需要" in res.summary
