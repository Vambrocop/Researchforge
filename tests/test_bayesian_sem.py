"""Tests for bayesian_sem: single-factor Bayesian CFA via PyMC (recovery of known
loadings up to the sign convention) + honest degrade without PyMC + too-few skip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="bayesian_sem", method="Bayesian CFA", domain="statistics", family="sem",
        goal="explain", preconditions=Precondition(min_continuous=3, min_rows=30),
    )


def _one_factor_data(seed: int = 0, n: int = 150, loadings=(0.9, 0.8, 0.7, 0.6)):
    """y_j = lam_j * f + noise, f ~ N(0,1) — single latent factor with KNOWN loadings."""
    rng = np.random.default_rng(seed)
    f = rng.normal(0, 1, n)
    cols = {}
    for j, lam in enumerate(loadings):
        resid = float(np.sqrt(max(1e-6, 1.0 - lam ** 2)))  # unit-variance indicators
        cols[f"q{j + 1}"] = lam * f + rng.normal(0, resid, n)
    return pd.DataFrame(cols)


def test_bayesian_sem_recovers_loadings(tmp_path: Path) -> None:
    pytest.importorskip("pymc")
    csv = tmp_path / "cfa.csv"
    _one_factor_data().to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"draws": 400, "tune": 400, "chains": 2})
    out = Path(res.output_dir)
    assert (out / "bayesian_sem_loadings.csv").exists()
    # loadings recovered up to the sign convention (all flipped to be positive)
    for j, lam_true in zip(range(1, 5), (0.9, 0.8, 0.7, 0.6)):
        got = res.estimates[f"lam_q{j}"]
        assert got > 0  # sign-fixed positive
        assert abs(got - lam_true) < 0.25
    # construct reliability omega in (0,1) and high for these strong loadings
    assert 0.6 < res.estimates["omega_reliability"] < 1.0
    assert res.estimates["n_indicators"] == 4.0
    assert res.estimates["max_rhat"] < 1.15


def test_bayesian_sem_degrades_without_pymc(monkeypatch, tmp_path: Path) -> None:
    csv = tmp_path / "cfa.csv"
    _one_factor_data().to_csv(csv, index=False)
    fp = profile_dataset(csv)

    import researchforge.executor.branches.bayesian_mcmc as bm
    monkeypatch.setattr(bm, "_have_pymc", lambda: False)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "omega_reliability" not in res.estimates
    assert "跳过" in res.summary and ("pymc" in res.summary or "sem" in res.summary)


def test_bayesian_sem_too_few_indicators(tmp_path: Path) -> None:
    pytest.importorskip("pymc")
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(0, 1, 60), "b": rng.normal(0, 1, 60)})
    csv = tmp_path / "x.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "omega_reliability" not in res.estimates
    assert "贝叶斯 SEM 跳过" in res.summary
