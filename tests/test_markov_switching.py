"""Tests for the markov_switching regime model executor branch."""

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
        id="markov_switching", method="Markov-switching regime model",
        domain="economics", family="time-series", goal="explain",
        preconditions=Precondition(is_timeseries=True, min_rows=40),
    )


def _two_regime_series(seed: int = 0, seg: int = 60, reps: int = 3):
    """Alternating segments: low-mean/low-var vs high-mean/high-var (known 2-regime structure)."""
    rng = np.random.default_rng(seed)
    parts = []
    truth = []
    for _r in range(reps):
        parts.append(rng.normal(0.0, 0.5, seg))   # regime 0: low mean, low var
        truth.append(np.zeros(seg, dtype=int))
        parts.append(rng.normal(5.0, 1.5, seg))   # regime 1: high mean, high var
        truth.append(np.ones(seg, dtype=int))
    return np.concatenate(parts), np.concatenate(truth)


def test_markov_recovers_two_regimes(tmp_path: Path) -> None:
    y, truth = _two_regime_series(seed=0)
    n = len(y)
    df = pd.DataFrame({"t": np.arange(n), "y": np.round(y, 4)})
    csv = tmp_path / "ms.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"column": "y", "k_regimes": 2})
    assert "完成" in res.summary
    assert res.estimates["k_regimes"] == 2.0
    m0 = res.estimates["regime0_mean"]
    m1 = res.estimates["regime1_mean"]
    # regimes ordered by mean ascending -> m0 < m1, and the two means are clearly distinct
    assert m0 < m1
    assert (m1 - m0) > 2.0
    # recovered means near the truth (0 and 5)
    assert abs(m0 - 0.0) < 1.5
    assert abs(m1 - 5.0) < 2.0
    # self-stay probabilities sensible (regimes are persistent, segments of length 60)
    assert res.estimates["regime0_p_stay"] > 0.5
    assert res.estimates["regime1_p_stay"] > 0.5

    # transition matrix CSV rows ~ sum to 1
    P = pd.read_csv(Path(res.output_dir) / "transition_matrix.csv", index_col=0)
    assert np.allclose(P.sum(axis=1).to_numpy(), 1.0, atol=1e-6)

    # smoothed probs track the true high-mean segments: where P(regime1) is high should align
    # with the true regime-1 indices (robust: AUC-like agreement via correlation of P with truth).
    probs = pd.read_csv(Path(res.output_dir) / "regime_probabilities.csv")
    p1 = probs["P_regime1"].to_numpy()
    # probabilities may drop the first `order` obs (order=0 here -> full length)
    assert len(p1) == n
    rho = np.corrcoef(p1, truth.astype(float))[0, 1]
    assert rho > 0.7  # smoothed prob of high regime correlates strongly with the truth


def test_markov_precondition_too_short(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n = 25
    df = pd.DataFrame({"t": np.arange(n), "y": rng.normal(0, 1, n)})
    csv = tmp_path / "short.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    ok, _ = check_preconditions(fp, _entry().preconditions)
    assert not ok  # min_rows=40 not met
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"column": "y"})
    assert "失败" in res.summary
