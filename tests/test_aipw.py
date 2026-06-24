"""Tests for aipw — augmented inverse-probability weighting (doubly-robust ATE).

Design (mirrors the IPW test, but checks the doubly-robust guarantee):
- A CONFOUNDED binary-treatment design with a KNOWN constant treatment effect tau:
      e(X)  = logit^{-1}(0.7*x1 + 0.5*x2)        (treatment depends on confounders)
      Y     = 1 + tau*T + 1.2*x1 + 0.8*x2 + noise
  With a constant effect ATE = ATT = tau. AIPW should recover ~tau, and on this
  confounded design AIPW must beat the naive difference-in-means (which is biased
  upward by the positive confounding).
- Degrade paths: missing covariates/outcome, too few rows, a too-small arm.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="aipw", method="Augmented inverse-probability weighting", domain="economics",
        family="causal", goal="explain",
        preconditions=Precondition(requires_treatment=True, min_rows=30),
    )


def _confounded(n: int = 800, tau: float = 2.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(0, 1, n), rng.normal(0, 1, n)
    ps = 1.0 / (1.0 + np.exp(-(0.7 * x1 + 0.5 * x2)))  # treatment depends on confounders
    t = (rng.uniform(size=n) < ps).astype(int)
    y = 1.0 + tau * t + 1.2 * x1 + 0.8 * x2 + rng.normal(0, 1, n)
    return pd.DataFrame({"y": y, "treat": t, "x1": x1, "x2": x2})


def test_catalog_loads_aipw() -> None:
    from researchforge.catalog import Catalog

    entry = Catalog.load().by_id("aipw")
    assert entry is not None
    assert entry.family == "causal"
    assert entry.executor_ref == "py::aipw"
    # biases / produces must be lists (folded strings would crash Catalog.load).
    assert isinstance(entry.biases, list) and len(entry.biases) >= 3
    assert isinstance(entry.produces, list) and entry.produces
    assert {"treatment", "outcome", "covariates"} <= {p.name for p in entry.params}


def test_aipw_recovers_ate_and_beats_naive(tmp_path: Path) -> None:
    tau = 2.0
    df = _confounded(n=800, tau=tau, seed=0)
    csv = tmp_path / "c.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"treatment": "treat", "outcome": "y", "covariates": ["x1", "x2"]},
    )
    assert "完成" in res.summary
    ate = res.estimates["ate"]
    naive = res.estimates["naive_diff"]
    # AIPW recovers ~tau (constant effect -> ATE = ATT).
    assert abs(ate - tau) < 0.5
    # Influence-function SE and 95% CI are well-formed.
    assert res.estimates["se"] > 0
    assert res.estimates["ci_low"] < ate < res.estimates["ci_high"]
    # On the confounded design AIPW must beat the (upward-biased) naive diff-in-means.
    assert abs(ate - tau) < abs(naive - tau)
    # Naive is biased upward here (positive confounding) — sanity-check the design.
    assert naive > tau


def test_aipw_se_is_influence_function(tmp_path: Path) -> None:
    """The reported SE equals sd(per-unit AIPW score, ddof=1)/sqrt(n)."""
    df = _confounded(n=600, tau=1.5, seed=3)
    csv = tmp_path / "c.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"treatment": "treat", "outcome": "y", "covariates": ["x1", "x2"]},
    )
    out = Path(res.output_dir)
    scores = pd.read_csv(out / "aipw_scores.csv")["aipw_score"].to_numpy()
    expect_ate = float(np.mean(scores))
    expect_se = float(np.std(scores, ddof=1) / np.sqrt(len(scores)))
    assert abs(res.estimates["ate"] - expect_ate) < 1e-4
    assert abs(res.estimates["se"] - expect_se) < 1e-4


def test_aipw_needs_treatment_and_covariates(tmp_path: Path) -> None:
    df = pd.DataFrame({"y": np.random.default_rng(1).normal(0, 1, 40)})
    csv = tmp_path / "x.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"outcome": "y"})
    assert "跳过" in res.summary
    assert "ate" not in res.estimates


def test_aipw_too_few_rows(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    n = 20
    df = pd.DataFrame({
        "y": rng.normal(0, 1, n),
        "treat": rng.integers(0, 2, n),
        "x1": rng.normal(0, 1, n),
    })
    csv = tmp_path / "tiny.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"treatment": "treat", "outcome": "y", "covariates": ["x1"]},
    )
    assert "跳过" in res.summary
    assert "ate" not in res.estimates


def test_aipw_arm_too_small(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    n = 60
    t = np.zeros(n, dtype=int)
    t[:3] = 1  # only 3 treated -> arm too small (needs >=10)
    df = pd.DataFrame({
        "y": rng.normal(0, 1, n),
        "treat": t,
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(0, 1, n),
    })
    csv = tmp_path / "arm.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"treatment": "treat", "outcome": "y", "covariates": ["x1", "x2"]},
    )
    assert "跳过" in res.summary
    assert "ate" not in res.estimates
