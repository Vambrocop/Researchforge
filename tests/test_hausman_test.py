"""Tests for hausman_test: formal Hausman FE-vs-RE — panel gate + rejects RE when the
entity effect correlates with x, accepts RE when independent. Skips without linearmodels."""

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
        id="hausman_test",
        method="Hausman specification test (FE vs RE)",
        domain="economics",
        family="econometrics",
        goal="explain",
        preconditions=Precondition(is_panel=True, min_continuous=2, min_rows=12),
    )


def _panel(seed: int, correlated: bool, beta: float = 1.5):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(30):
        alpha = rng.normal(0, 1)  # unit effect
        for t in range(5):
            # correlated=True -> x depends on alpha -> RE inconsistent -> reject
            x = (0.8 * alpha if correlated else 0.0) + rng.normal(0, 1)
            y = beta * x + alpha + rng.normal(0, 0.5)
            rows.append({"firm": f"u{u}", "year": 2015 + t, "y": round(y, 4), "x": round(x, 4)})
    return pd.DataFrame(rows)


def _cfg():
    return {"unit": "firm", "time": "year", "outcome": "y", "predictors": ["x"]}


def test_hausman_rejects_re_when_correlated(tmp_path: Path) -> None:
    pytest.importorskip("linearmodels")
    csv = tmp_path / "panel.csv"
    _panel(0, correlated=True).to_csv(csv, index=False)

    fp = profile_dataset(csv)
    assert fp.is_panel
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config=_cfg())
    out = Path(res.output_dir)

    assert (out / "hausman_fe_re_coefficients.csv").exists()
    assert "hausman_chi2" in res.estimates
    assert "hausman_df" in res.estimates
    assert res.estimates["hausman_df"] == 1.0
    assert res.estimates["n_coefs_compared"] == 1.0
    # entity effect correlates with x -> Hausman rejects RE
    assert res.estimates["hausman_p"] < 0.05
    # FE coefficient recovers the true within slope
    assert abs(res.estimates["x_FE"] - 1.5) < 0.4


def test_hausman_accepts_re_when_independent(tmp_path: Path) -> None:
    pytest.importorskip("linearmodels")
    csv = tmp_path / "panel.csv"
    _panel(7, correlated=False).to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config=_cfg())
    # effect independent of x -> RE not rejected
    assert res.estimates["hausman_p"] > 0.05


def test_hausman_not_panel_skips(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"y": rng.normal(0, 1, 30), "x": rng.normal(0, 1, 30)})
    csv = tmp_path / "flat.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)
    assert not ok
    assert any("面板" in u for u in unmet)
