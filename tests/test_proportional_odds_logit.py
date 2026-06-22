"""Tests for proportional_odds_logit: ordinal-outcome gate + proportional-odds fit.

(The basic `ordered_logit` lives in branches/statistics.py and has its own test inline
above historically; this module covers the richer odds-ratio version registered as
`proportional_odds_logit` in branches/ordinal_regression.py.)
"""

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
        id="proportional_odds_logit",
        method="Proportional-odds ordinal logistic regression",
        domain="statistics",
        family="statistics",
        goal="explain",
        preconditions=Precondition(requires_ordinal=True, min_rows=30),
    )


def _proportional_data(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Known proportional-odds DGP: latent y* = 1.2*x1 - 0.7*x2 + logistic noise,
    cut into 4 ordered levels. Slopes act equally on every cut (proportional)."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    latent = 1.2 * x1 - 0.7 * x2 + rng.logistic(0, 1, n)
    sat = np.digitize(latent, [-1.0, 0.0, 1.0]) + 1  # 4 ordered levels 1..4
    return pd.DataFrame({"sat": sat.astype(int), "x1": x1, "x2": x2})


def test_proportional_odds_logit_executor(tmp_path: Path) -> None:
    df = _proportional_data()
    csv = tmp_path / "likert.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    assert (out / "coefficients.csv").exists()
    assert (out / "summary.txt").exists()
    # recovered slope signs match the data-generating process
    assert res.estimates["x1"] > 0
    assert res.estimates["x2"] < 0
    # estimates contract
    for k in ("loglik", "llr_p", "pseudo_r2", "n", "n_thresholds", "max_abs_or_log"):
        assert k in res.estimates
    assert res.estimates["n_thresholds"] == 3.0  # K-1 with 4 levels
    assert res.estimates["pseudo_r2"] > 0
    assert res.estimates["llr_p"] < 0.05  # predictors jointly significant


def test_proportional_odds_logit_config_outcome(tmp_path: Path) -> None:
    """config outcome/predictors override is honoured."""
    df = _proportional_data(seed=3)
    csv = tmp_path / "likert2.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"outcome": "sat", "predictors": ["x1"]},
    )
    assert "x1" in res.estimates
    assert "x2" not in res.estimates  # excluded by config


def test_proportional_odds_logit_precondition_unmet(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    # all-continuous data: no 3-10 level ordinal column to serve as outcome
    df = pd.DataFrame({"y": rng.normal(0, 1, 50), "x": rng.normal(0, 1, 50)})
    csv = tmp_path / "cont.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)

    assert not ok
    assert any("有序" in u for u in unmet)
