"""Tests for dynamic_panel_gmm: panel gate + Arellano-Bond GMM (skips without R/plm)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import rbridge, run_analysis
from researchforge.profiler import profile_dataset
from researchforge.recommender.match import check_preconditions


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="dynamic_panel_gmm",
        method="Dynamic panel GMM (Arellano-Bond)",
        domain="economics",
        family="econometrics",
        goal="explain",
        preconditions=Precondition(is_panel=True, min_continuous=2, min_rows=30),
    )


def test_dynamic_panel_gmm(tmp_path: Path) -> None:
    if not (rbridge.r_available() and rbridge.r_package_available("plm")):
        pytest.skip("R plm package not available")

    rng = np.random.default_rng(0)
    rows = []
    for u in range(60):
        a = rng.normal(0, 1)
        ylag = rng.normal(0, 1)
        for t in range(8):
            x = rng.normal(0, 1)
            y = 0.4 * ylag + 0.6 * x + a + rng.normal(0, 0.5)  # AR(1) coef 0.4
            rows.append({"firm": f"f{u}", "year": 2010 + t, "y": round(y, 4), "x": round(x, 4)})
            ylag = y
    csv = tmp_path / "dyn.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    assert (out / "gmm_coefficients.csv").exists()
    # lagged-dependent coefficient (persistence) recovered near 0.4
    assert abs(res.estimates["persistence_lag_coef"] - 0.4) < 0.2
    assert abs(res.estimates["x"] - 0.6) < 0.2  # covariate effect recovered
    assert res.estimates["ar2_p"] > 0.05  # no second-order serial correlation -> GMM valid


def test_dynamic_panel_gmm_precondition_unmet(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"y": rng.normal(0, 1, 30), "x": rng.normal(0, 1, 30)})  # not panel
    csv = tmp_path / "flat.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)

    assert not ok
    assert any("面板" in u for u in unmet)
