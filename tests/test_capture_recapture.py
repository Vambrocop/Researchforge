"""Tests for capture_recapture: closed-population abundance N̂ from a capture-history
matrix. Simulate a closed population of known N with K occasions and constant capture
prob p, build the matrix of individuals caught >= once, and check the estimators recover
N within tolerance. Covers Schnabel/M0 (K>=3), Chapman (K=2), and an honest skip.
Pure numpy/scipy — no optional backends, so no importorskip needed."""

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
        id="capture_recapture",
        method="Capture-recapture closed-population abundance (N̂)",
        domain="ecology",
        family="ecology",
        goal="estimate",
        preconditions=Precondition(min_rows=8),
    )


def _simulate(N: int, K: int, p: float, seed: int = 0) -> pd.DataFrame:
    """Closed population of N individuals, K occasions, constant capture prob p.
    Returns the capture-history of the individuals caught at least once (0/1 columns
    occ1..occK), the standard capture-recapture input."""
    rng = np.random.default_rng(seed)
    hist = (rng.random((N, K)) < p).astype(int)  # row=individual, col=occasion
    hist = hist[hist.sum(axis=1) > 0]             # caught at least once
    cols = [f"occ{t + 1}" for t in range(K)]
    return pd.DataFrame(hist, columns=cols)


def test_schnabel_m0_recover_known_N(tmp_path: Path) -> None:
    """K>=3 closed population: Schnabel + M0 N̂ within ~25% of the true N=300, and the
    observed distinct count is strictly below N."""
    N_true, K, p = 300, 5, 0.3
    df = _simulate(N_true, K, p, seed=1)
    csv = tmp_path / "ch.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    assert (out / "capture_recapture_occasions.csv").exists()
    assert res.estimates["n_occasions"] == float(K)
    # observed distinct individuals strictly below the true population
    assert res.estimates["n_observed"] < N_true

    assert "schnabel_N" in res.estimates
    assert abs(res.estimates["schnabel_N"] - N_true) <= 0.25 * N_true
    assert "schnabel_N_ci_low" in res.estimates and "schnabel_N_ci_high" in res.estimates

    assert "m0_mle_N" in res.estimates
    assert abs(res.estimates["m0_mle_N"] - N_true) <= 0.25 * N_true
    assert 0.0 < res.estimates["capture_prob"] < 1.0
    assert "闭合种群" in res.summary and "⚠" in res.summary


def test_chapman_two_occasions(tmp_path: Path) -> None:
    """K=2: Chapman bias-corrected Lincoln-Petersen recovers N within ~25%, with SE/CI
    reported and observed n < N."""
    N_true, K, p = 300, 2, 0.45
    df = _simulate(N_true, K, p, seed=2)
    csv = tmp_path / "ch2.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o2"))

    assert res.estimates["n_occasions"] == 2.0
    assert res.estimates["n_observed"] < N_true
    assert "petersen_chapman_N" in res.estimates
    assert abs(res.estimates["petersen_chapman_N"] - N_true) <= 0.25 * N_true
    for k in ("petersen_chapman_N_se", "petersen_chapman_N_ci_low",
              "petersen_chapman_N_ci_high"):
        assert k in res.estimates
    # Chapman branch must NOT emit a Schnabel estimate
    assert "schnabel_N" not in res.estimates


def test_too_few_occasions_skips(tmp_path: Path) -> None:
    """Only one 0/1 occasion column -> honest skip (no crash, Chinese msg, no N̂)."""
    df = pd.DataFrame({
        "occ1": [1] * 10,
        "weight": np.linspace(1.0, 5.0, 10),  # continuous, not a 0/1 occasion
    })
    csv = tmp_path / "few.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o3"))

    assert "petersen_chapman_N" not in res.estimates
    assert "schnabel_N" not in res.estimates
    assert "跳过" in res.summary


def test_precondition_min_rows(tmp_path: Path) -> None:
    """min_rows=8 gate: a 5-row dataset is rejected by check_preconditions."""
    df = _simulate(300, 3, 0.3, seed=3).head(5)
    csv = tmp_path / "small.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)
    assert not ok
    assert any("行" in u for u in unmet)
