"""Tests for mediation: continuous gate + ACME/ADE decomposition (statsmodels)."""

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
        id="mediation",
        method="Mediation analysis (X to M to Y)",
        domain="statistics",
        family="causal",
        goal="explain",
        preconditions=Precondition(min_continuous=3, min_rows=30),
    )


def test_mediation_recovers_indirect_effect(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 400
    x = rng.normal(0, 1, n)
    m = 0.6 * x + rng.normal(0, 1, n)  # path a = 0.6
    y = 0.5 * m + 0.2 * x + rng.normal(0, 1, n)  # path b = 0.5, direct = 0.2; indirect = 0.30
    # column order -> Y first, then X, then M
    df = pd.DataFrame({"Y": y, "X": x, "M": m})
    csv = tmp_path / "med.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    assert (out / "mediation_summary.csv").exists()
    # indirect (ACME) ~ 0.30, direct (ADE) ~ 0.20 (Monte-Carlo, allow tolerance)
    assert abs(res.estimates["indirect_effect_ACME"] - 0.30) < 0.15
    assert abs(res.estimates["direct_effect_ADE"] - 0.20) < 0.15
    assert 0.0 < res.estimates["prop_mediated"] < 1.0


def test_mediation_precondition_unmet(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"y": rng.normal(0, 1, 40), "x": rng.normal(0, 1, 40)})  # only 2 continuous
    csv = tmp_path / "two.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)

    assert not ok
    assert any("连续" in u for u in unmet)
