"""Tests for random_effects: panel gate + RE/FE + Hausman (skips without linearmodels)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset
from researchforge.recommender.match import check_preconditions


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="random_effects",
        method="Random-effects panel model (+ Hausman test)",
        domain="economics",
        family="econometrics",
        goal="explain",
        preconditions=Precondition(is_panel=True, min_continuous=2, min_rows=12),
    )


def test_random_effects_recovers_slope_and_hausman(tmp_path: Path) -> None:
    pytest.importorskip("linearmodels")
    rng = np.random.default_rng(0)
    rows = []
    for u in range(30):
        alpha = rng.normal(0, 1)  # unit effect, INDEPENDENT of x -> RE consistent
        for t in range(5):
            x = rng.normal(0, 1)
            y = 1.5 * x + alpha + rng.normal(0, 0.5)
            rows.append({"firm": f"u{u}", "year": 2015 + t, "y": round(y, 4), "x": round(x, 4)})
    csv = tmp_path / "panel.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    fp = profile_dataset(csv)
    assert fp.is_panel
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    assert (out / "fe_re_coefficients.csv").exists()
    assert abs(res.estimates["x"] - 1.5) < 0.4  # slope recovered
    assert "hausman_p" in res.estimates
    assert res.estimates["hausman_p"] > 0.05  # effect independent of x -> RE not rejected


def test_random_effects_precondition_unmet(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"y": rng.normal(0, 1, 30), "x": rng.normal(0, 1, 30)})  # not panel
    csv = tmp_path / "flat.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)

    assert not ok
    assert any("面板" in u for u in unmet)
