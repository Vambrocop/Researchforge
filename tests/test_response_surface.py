"""Tests for the response-surface methodology (RSM, second-order) branch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="response_surface", method="Response surface methodology",
        domain="experimental design", family="experimental_design", goal="explain",
        preconditions=Precondition(min_continuous=3, min_rows=9),
    )


def _ccd_grid(centers, half_range, n_levels=5):
    """A small central-composite-like full grid over two factors."""
    x1c, x2c = centers
    h1, h2 = half_range
    g1 = np.linspace(x1c - h1, x1c + h1, n_levels)
    g2 = np.linspace(x2c - h2, x2c + h2, n_levels)
    pts = [(a, b) for a in g1 for b in g2]
    return pts


def test_rsm_recovers_known_maximum(tmp_path: Path) -> None:
    # True surface: y = 100 - 2(x1-5)^2 - 3(x2-8)^2  → unique MAXIMUM at (5, 8), y=100.
    rng = np.random.default_rng(0)
    rows = []
    for (x1, x2) in _ccd_grid((5.0, 8.0), (3.0, 3.0), n_levels=5):
        y = 100.0 - 2.0 * (x1 - 5.0) ** 2 - 3.0 * (x2 - 8.0) ** 2 + rng.normal(0, 0.05)
        rows.append({"resp": y, "x1": x1, "x2": x2})
    csv = tmp_path / "rsm.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "resp", "factors": ["x1", "x2"]})
    assert "完成" in res.summary
    assert res.estimates["n_factors"] == 2
    assert res.estimates["r_squared"] > 0.99
    # stationary point recovers the planted optimum
    assert abs(res.estimates["stationary_x1"] - 5.0) < 0.1
    assert abs(res.estimates["stationary_x2"] - 8.0) < 0.1
    assert res.estimates["stationary_in_region"] == 1.0
    # both Hessian eigenvalues negative → maximum
    assert res.estimates["hessian_eig1"] < 0
    assert res.estimates["hessian_eig2"] < 0
    assert "maximum" in res.summary or "极大" in res.summary
    assert "rsm_stationary_point.csv" in set(res.files)
    assert "rsm_coefficients.csv" in set(res.files)


def test_rsm_recovers_known_minimum(tmp_path: Path) -> None:
    # y = 10 + 1.5(x1-2)^2 + 2.5(x2+1)^2  → unique MINIMUM at (2, -1).
    rng = np.random.default_rng(1)
    rows = []
    for (x1, x2) in _ccd_grid((2.0, -1.0), (2.0, 2.0), n_levels=5):
        y = 10.0 + 1.5 * (x1 - 2.0) ** 2 + 2.5 * (x2 + 1.0) ** 2 + rng.normal(0, 0.05)
        rows.append({"resp": y, "x1": x1, "x2": x2})
    csv = tmp_path / "rsm.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "resp", "factors": ["x1", "x2"]})
    assert "完成" in res.summary
    assert abs(res.estimates["stationary_x1"] - 2.0) < 0.1
    assert abs(res.estimates["stationary_x2"] + 1.0) < 0.1
    assert res.estimates["hessian_eig1"] > 0
    assert res.estimates["hessian_eig2"] > 0
    assert "minimum" in res.summary or "极小" in res.summary


def test_rsm_detects_saddle(tmp_path: Path) -> None:
    # y = (x1-3)^2 - (x2-4)^2  → SADDLE at (3, 4) (mixed Hessian eigenvalues).
    rng = np.random.default_rng(2)
    rows = []
    for (x1, x2) in _ccd_grid((3.0, 4.0), (2.5, 2.5), n_levels=5):
        y = (x1 - 3.0) ** 2 - (x2 - 4.0) ** 2 + rng.normal(0, 0.05)
        rows.append({"resp": y, "x1": x1, "x2": x2})
    csv = tmp_path / "rsm.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "resp", "factors": ["x1", "x2"]})
    assert "完成" in res.summary
    eigs = [res.estimates["hessian_eig1"], res.estimates["hessian_eig2"]]
    assert min(eigs) < 0 < max(eigs)   # mixed signs → saddle
    assert "saddle" in res.summary or "鞍点" in res.summary


def test_rsm_needs_two_factors(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"resp": rng.normal(0, 1, 20), "x1": rng.normal(0, 1, 20)})
    csv = tmp_path / "one.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "resp", "factors": ["x1"]})
    assert "RSM 失败" in res.summary
    assert "stationary_x1" not in res.estimates
