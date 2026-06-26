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


def _two_factor_data(seed: int = 0, n: int = 300, rho: float = 0.5):
    """2 correlated factors (true corr=rho); F1→q1..q3, F2→q4..q6, known loadings."""
    rng = np.random.default_rng(seed)
    Lf = np.linalg.cholesky([[1.0, rho], [rho, 1.0]])
    F = rng.normal(0, 1, (n, 2)) @ Lf.T
    lam = [0.9, 0.8, 0.7, 0.85, 0.75, 0.65]
    fac = [0, 0, 0, 1, 1, 1]
    cols = {f"q{j + 1}": lam[j] * F[:, fac[j]] + rng.normal(0, np.sqrt(1 - lam[j] ** 2), n)
            for j in range(6)}
    return pd.DataFrame(cols)


def test_bayesian_sem_multifactor_recovers_factor_correlation(tmp_path: Path) -> None:
    pytest.importorskip("pymc")
    csv = tmp_path / "mf.csv"
    _two_factor_data(rho=0.5).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    spec = "F1 =~ q1 + q2 + q3\nF2 =~ q4 + q5 + q6"
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"model_spec": spec, "draws": 400, "tune": 400, "chains": 2})
    out = Path(res.output_dir)
    assert res.estimates["n_factors"] == 2.0
    # the inter-factor correlation (the structural association) is recovered near 0.5
    assert "corr_F1_F2" in res.estimates
    assert abs(res.estimates["corr_F1_F2"] - 0.5) < 0.3
    # anchor (first) loading of each factor is positive (sign-identified)
    assert res.estimates["lam_q1"] > 0 and res.estimates["lam_q4"] > 0
    assert "omega_F1" in res.estimates and "omega_F2" in res.estimates
    assert (out / "bayesian_sem_factor_corr.csv").exists()
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
