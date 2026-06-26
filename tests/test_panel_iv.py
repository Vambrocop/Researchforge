"""Tests for panel_iv: system_gmm (Blundell-Bond) + hausman_taylor.

Both delegate to R plm — the recovery tests skip cleanly when R/plm is unavailable.
A degrade test (R forced absent) runs everywhere and asserts an honest Chinese skip
with no crash and no fabricated estimates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import rbridge, run_analysis
from researchforge.profiler import profile_dataset


def _has_plm() -> bool:
    return rbridge.r_available() and rbridge.r_package_available("plm")


def _entry(eid: str, method: str) -> AnalysisEntry:
    return AnalysisEntry(
        id=eid, method=method, domain="economics", family="econometrics",
        goal="explain", preconditions=Precondition(is_panel=True, min_continuous=1, min_rows=12),
    )


def _ar1_panel(seed: int = 0, rho: float = 0.5, n_units: int = 40, n_t: int = 7):
    """Dynamic AR(1) panel with unit effects: y_it = rho*y_{i,t-1} + b*x_it + a_i + e."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        a = rng.normal(0, 1)
        ylag = a + rng.normal(0, 1)
        for t in range(n_t):
            x = rng.normal(0, 1)
            y = rho * ylag + 0.8 * x + a + rng.normal(0, 0.5)
            rows.append({"firm": f"u{u}", "year": 2010 + t, "y": round(y, 4), "x": round(x, 4)})
            ylag = y
    return pd.DataFrame(rows)


def _ht_panel(seed: int = 1, n_units: int = 50, n_t: int = 5):
    """Panel with a TIME-INVARIANT regressor z (constant within unit) + time-varying x."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        a = rng.normal(0, 1)
        z = round(rng.normal(0, 1), 4)            # time-invariant (e.g. schooling)
        for t in range(n_t):
            x = rng.normal(0, 1)                  # time-varying
            y = 1.0 * x + 1.5 * z + a + rng.normal(0, 0.5)
            rows.append({"person": f"p{u}", "wave": t, "y": round(y, 4),
                         "x": round(x, 4), "z": z})
    return pd.DataFrame(rows)


# ── system_gmm ───────────────────────────────────────────────────────────────
def test_system_gmm_recovers_persistence(tmp_path: Path) -> None:
    if not _has_plm():
        pytest.skip("R/plm not available")
    csv = tmp_path / "p.csv"
    _ar1_panel(rho=0.5).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    assert fp.is_panel
    res = run_analysis(fp, _entry("system_gmm", "System GMM"), output_root=str(tmp_path / "o"),
                       config={"unit": "firm", "time": "year", "outcome": "y", "predictors": ["x"]})
    out = Path(res.output_dir)
    assert (out / "system_gmm_coefficients.csv").exists()
    assert "persistence_lag_coef" in res.estimates
    # recovered persistence in a plausible neighbourhood of the true rho=0.5
    assert 0.1 < res.estimates["persistence_lag_coef"] < 0.9
    assert "ar2_p" in res.estimates and "sargan_p" in res.estimates


def test_system_gmm_degrades_without_r(monkeypatch, tmp_path: Path) -> None:
    csv = tmp_path / "p.csv"
    _ar1_panel().to_csv(csv, index=False)
    fp = profile_dataset(csv)
    monkeypatch.setattr(rbridge, "r_available", lambda: False)
    res = run_analysis(fp, _entry("system_gmm", "System GMM"), output_root=str(tmp_path / "o"),
                       config={"unit": "firm", "time": "year", "outcome": "y", "predictors": ["x"]})
    assert "persistence_lag_coef" not in res.estimates
    assert "系统 GMM" in res.summary and ("plm" in res.summary or "dynamic_panel_gmm" in res.summary)


# ── hausman_taylor ───────────────────────────────────────────────────────────
def test_hausman_taylor_estimates_time_invariant(tmp_path: Path) -> None:
    if not _has_plm():
        pytest.skip("R/plm not available")
    csv = tmp_path / "h.csv"
    _ht_panel().to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry("hausman_taylor", "Hausman-Taylor"), output_root=str(tmp_path / "o"),
        config={"unit": "person", "time": "wave", "outcome": "y",
                "predictors": ["x", "z"], "endogenous": ["x"]},
    )
    out = Path(res.output_dir)
    assert (out / "hausman_taylor_coefficients.csv").exists()
    # z is the time-invariant regressor HT exists to estimate; it should be recovered near 1.5
    assert "z" in res.estimates
    assert abs(res.estimates["z"] - 1.5) < 0.6
    assert res.estimates["n_time_invariant"] == 1.0


def test_hausman_taylor_skips_without_time_invariant(tmp_path: Path) -> None:
    # all regressors vary within unit -> HT has nothing to add -> honest skip
    rng = np.random.default_rng(2)
    rows = []
    for u in range(20):
        a = rng.normal(0, 1)
        for t in range(4):
            rows.append({"person": f"p{u}", "wave": t,
                         "y": rng.normal(0, 1) + a, "x1": rng.normal(0, 1), "x2": rng.normal(0, 1)})
    csv = tmp_path / "n.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry("hausman_taylor", "Hausman-Taylor"), output_root=str(tmp_path / "o"),
        config={"unit": "person", "time": "wave", "outcome": "y", "predictors": ["x1", "x2"]},
    )
    assert "n_time_invariant" not in res.estimates
    assert "Hausman-Taylor 跳过" in res.summary


def test_hausman_taylor_degrades_without_r(monkeypatch, tmp_path: Path) -> None:
    csv = tmp_path / "h.csv"
    _ht_panel().to_csv(csv, index=False)
    fp = profile_dataset(csv)
    monkeypatch.setattr(rbridge, "r_available", lambda: False)
    res = run_analysis(
        fp, _entry("hausman_taylor", "Hausman-Taylor"), output_root=str(tmp_path / "o"),
        config={"unit": "person", "time": "wave", "outcome": "y",
                "predictors": ["x", "z"], "endogenous": ["x"]},
    )
    assert "n_time_invariant" not in res.estimates
    assert "Hausman-Taylor" in res.summary and ("plm" in res.summary or "random_effects" in res.summary)
