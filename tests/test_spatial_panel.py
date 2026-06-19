"""Tests for spatial_panel: panel+geo gate + splm SAR/SEM/SDM (skips/degrades
without R/splm). The R-bridge path is OPTIONAL + graceful-degrade per CLAUDE.md,
so the precondition / honest-degrade tests always run; the empirical recovery
test only runs when R + splm are installed.

Empirical check: a synthetic spatial panel with a KNOWN positive spatial-lag
coefficient (rho = 0.5) is generated via the SAR reduced form
    y_t = (I - rho W)^{-1} (X_t b + mu + eps_t)
on a k-NN W, and we assert the splm within-FE estimate recovers a positive rho
in roughly the right neighbourhood.
"""

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
        id="spatial_panel",
        method="Spatial panel econometrics (SAR / SEM / SDM, fixed effects)",
        domain="economics",
        family="econometrics",
        goal="explain",
        preconditions=Precondition(
            is_panel=True, requires_geo=True, min_continuous=2, min_rows=30
        ),
    )


def _knn_W(coords: np.ndarray, k: int) -> np.ndarray:
    """Row-standardised k-nearest-neighbour weights (matches the R helper's W)."""
    n = coords.shape[0]
    W = np.zeros((n, n))
    for i in range(n):
        dist = np.sum((coords - coords[i]) ** 2, axis=1)
        dist[i] = np.inf
        nn = np.argsort(dist)[:k]
        W[i, nn] = 1.0
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return W / rs


def _make_spatial_panel(rho: float = 0.5, n_units: int = 30, n_time: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 10, size=(n_units, 2))  # (lon, lat) per unit
    k = 4
    W = _knn_W(coords, k)
    A_inv = np.linalg.inv(np.eye(n_units) - rho * W)  # (I - rho W)^{-1}
    mu = rng.normal(0, 1.0, size=n_units)  # unit fixed effects
    b = 0.8  # covariate effect
    rows = []
    for t in range(n_time):
        x = rng.normal(0, 1, size=n_units)
        eps = rng.normal(0, 0.3, size=n_units)
        y = A_inv @ (b * x + mu + eps)  # SAR reduced form
        for i in range(n_units):
            rows.append(
                {
                    "region": f"u{i:02d}",
                    "year": 2010 + t,
                    "lon": round(float(coords[i, 0]), 5),
                    "lat": round(float(coords[i, 1]), 5),
                    "y": round(float(y[i]), 5),
                    "x": round(float(x[i]), 5),
                }
            )
    return pd.DataFrame(rows)


def test_spatial_panel_recovers_positive_rho(tmp_path: Path) -> None:
    if not (rbridge.r_available() and rbridge.r_package_available("splm")):
        pytest.skip("R splm package not available")

    df = _make_spatial_panel(rho=0.5)
    csv = tmp_path / "spml.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    assert fp.is_panel  # profiler must see the panel structure
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    assert (out / "spatial_panel_coefficients.csv").exists()
    # known positive spatial-lag coefficient is recovered (sign + roughly)
    rho_hat = res.estimates.get("spatial_coef")
    assert rho_hat is not None
    assert rho_hat > 0.15, f"expected positive rho near 0.5, got {rho_hat}"
    assert abs(rho_hat - 0.5) < 0.35, f"rho off: {rho_hat}"
    # impacts (direct/indirect/total spillovers) are produced for the lag model
    assert (out / "spatial_panel_impacts.csv").exists()


def test_spatial_panel_sdm_runs(tmp_path: Path) -> None:
    """Spatial-Durbin variant (config model=sdm) also fits + reports impacts."""
    if not (rbridge.r_available() and rbridge.r_package_available("splm")):
        pytest.skip("R splm package not available")

    df = _make_spatial_panel(rho=0.4, seed=1)
    csv = tmp_path / "spml_sdm.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"model": "sdm"})
    out = Path(res.output_dir)
    assert (out / "spatial_panel_coefficients.csv").exists()
    assert "spatial_coef" in res.estimates
    # SDM also yields direct/indirect/total impacts (Wx Durbin terms folded in)
    assert (out / "spatial_panel_impacts.csv").exists()
    assert any(kk.endswith("_total") for kk in res.estimates)


def test_spatial_panel_degrades_without_splm(tmp_path: Path) -> None:
    """Without R/splm the handler must degrade honestly (no crash) and point the
    user at spatial_regression / panel_fixed_effects. If splm IS installed this
    asserts the success path produced a spatial coefficient instead."""
    df = _make_spatial_panel(rho=0.5)
    csv = tmp_path / "spml.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))

    if not (rbridge.r_available() and rbridge.r_package_available("splm")):
        assert "splm" in res.summary
        assert ("panel_fixed_effects" in res.summary) or ("spatial_regression" in res.summary)
    else:
        assert "spatial_coef" in res.estimates


def test_spatial_panel_precondition_unmet_not_panel(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    # cross-section with geo but no panel structure -> precondition fails
    df = pd.DataFrame(
        {
            "lon": rng.uniform(0, 10, 40),
            "lat": rng.uniform(0, 10, 40),
            "y": rng.normal(0, 1, 40),
            "x": rng.normal(0, 1, 40),
        }
    )
    csv = tmp_path / "flat.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)
    assert not ok
    assert any("面板" in u or "panel" in u.lower() for u in unmet)


def test_spatial_panel_precondition_no_geo(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    rows = []
    for u in range(10):
        for t in range(5):
            rows.append(
                {"region": f"u{u}", "year": 2010 + t,
                 "y": rng.normal(0, 1), "x": rng.normal(0, 1)}
            )
    df = pd.DataFrame(rows)  # panel but no lon/lat
    csv = tmp_path / "nogeo.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)
    assert not ok  # requires_geo unmet
