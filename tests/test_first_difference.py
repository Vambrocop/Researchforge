"""Tests for first_difference: FD estimator — panel gate + within beta recovery,
T=2 FD==FE coincidence, time-invariant predictor dropped. Skips without linearmodels."""

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
        id="first_difference",
        method="First-difference (FD) panel estimator",
        domain="economics",
        family="econometrics",
        goal="explain",
        preconditions=Precondition(is_panel=True, min_continuous=2, min_rows=12),
    )


def _panel(seed: int, n_periods: int = 5, beta: float = 1.5):
    """Entity effect correlated with x (FD removes it, still recovers beta)."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(30):
        alpha = rng.normal(0, 1)
        for t in range(n_periods):
            x = 0.8 * alpha + rng.normal(0, 1)
            y = beta * x + alpha + rng.normal(0, 0.5)
            rows.append({"firm": f"u{u}", "year": 2015 + t, "y": round(y, 4), "x": round(x, 4)})
    return pd.DataFrame(rows)


def _cfg(predictors=None):
    return {
        "unit": "firm", "time": "year", "outcome": "y",
        "predictors": predictors or ["x"],
    }


def test_first_difference_recovers_beta(tmp_path: Path) -> None:
    pytest.importorskip("linearmodels")
    csv = tmp_path / "panel.csv"
    _panel(0).to_csv(csv, index=False)

    fp = profile_dataset(csv)
    assert fp.is_panel
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config=_cfg())
    out = Path(res.output_dir)

    assert (out / "first_difference_coefficients.csv").exists()
    assert abs(res.estimates["x"] - 1.5) < 0.4  # FD recovers the within slope
    assert res.estimates["n_predictors"] == 1.0
    assert res.estimates["n_entities"] == 30.0
    assert "rsquared" in res.estimates
    assert "n_obs_differenced" in res.estimates
    # 30 entities * (5 periods - 1 difference) = 120 differenced obs
    assert res.estimates["n_obs_differenced"] == 120.0


def test_first_difference_t2_matches_fe(tmp_path: Path) -> None:
    """At T=2, FD and FE (within) estimates coincide — check the CSV carries the FE
    contrast and the two are within numerical tolerance."""
    pytest.importorskip("linearmodels")
    csv = tmp_path / "panel.csv"
    _panel(3, n_periods=2).to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config=_cfg())
    tab = pd.read_csv(Path(res.output_dir) / "first_difference_coefficients.csv")
    assert "FE_within_coef" in tab.columns
    row = tab[tab["term"] == "x"].iloc[0]
    assert abs(row["FD_coef"] - row["FE_within_coef"]) < 1e-3


def test_first_difference_drops_time_invariant(tmp_path: Path) -> None:
    """A time-invariant predictor differences out to zero and must be dropped, while
    the time-varying one is still estimated."""
    pytest.importorskip("linearmodels")
    df = _panel(5)
    # add a unit-constant (time-invariant) column
    const_map = {u: i * 0.3 for i, u in enumerate(df["firm"].unique())}
    df["z_const"] = df["firm"].map(const_map)
    csv = tmp_path / "panel.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config=_cfg(predictors=["x", "z_const"]),
    )
    # only the time-varying x survives
    assert res.estimates["n_predictors"] == 1.0
    assert "x" in res.estimates
    assert "z_const" not in res.estimates


def test_first_difference_not_panel_skips(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"y": rng.normal(0, 1, 30), "x": rng.normal(0, 1, 30)})
    csv = tmp_path / "flat.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)
    assert not ok
    assert any("面板" in u for u in unmet)
