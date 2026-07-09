"""Tests for ordered_probit: ordinal-outcome gate + latent-normal fit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="ordered_probit",
        method="Ordered probit regression",
        domain="statistics",
        family="statistics",
        goal="explain",
        preconditions=Precondition(requires_ordinal=True, min_rows=30),
    )


def _probit_data(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Known latent-normal DGP: y* = 1.0*x1 - 0.6*x2 + N(0,1), cut into 4 levels."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    latent = 1.0 * x1 - 0.6 * x2 + rng.normal(0, 1, n)
    sat = np.digitize(latent, [-1.0, 0.0, 1.0]) + 1  # 4 ordered levels 1..4
    return pd.DataFrame({"sat": sat.astype(int), "x1": x1, "x2": x2})


def test_ordered_probit_executor(tmp_path: Path) -> None:
    df = _probit_data()
    csv = tmp_path / "likert.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    assert (out / "coefficients.csv").exists()
    assert (out / "summary.txt").exists()
    # recovered slope signs match the latent DGP (per-predictor keys are namespaced
    # "coef_<name>" so a predictor named e.g. "n"/"loglik" can't collide with the
    # bookkeeping keys below)
    assert res.estimates["coef_x1"] > 0
    assert res.estimates["coef_x2"] < 0
    # estimates contract
    for k in ("loglik", "llr_p", "pseudo_r2", "n", "n_thresholds", "max_abs_coef"):
        assert k in res.estimates
    assert res.estimates["n_thresholds"] == 3.0
    assert res.estimates["pseudo_r2"] > 0
    assert res.estimates["llr_p"] < 0.05


def test_ordered_probit_predictor_named_n_no_key_collision(tmp_path: Path) -> None:
    """A predictor literally named 'n' must not clobber the bookkeeping
    estimates['n'] (sample size) — its coefficient lands in estimates['coef_n']."""
    df = _probit_data(seed=5).rename(columns={"x1": "n"})
    csv = tmp_path / "likert_n.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"outcome": "sat", "predictors": ["n", "x2"]},
    )
    assert res.estimates["n"] == float(len(df))  # bookkeeping sample size, not clobbered
    assert "coef_n" in res.estimates
    assert res.estimates["coef_n"] > 0  # same DGP sign as x1 (renamed to n)


def test_ordered_probit_too_few_levels_degrades(tmp_path: Path) -> None:
    """Binary outcome (2 levels) is not ordinal-with-3+-levels -> honest skip, no crash."""
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "y2": rng.integers(0, 2, 60),  # only 2 levels
        "x1": rng.normal(0, 1, 60),
    })
    csv = tmp_path / "bin.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    # force the binary column as outcome via config; resolver should reject <3 levels
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"), config={"outcome": "y2"}
    )
    assert "跳过" in res.summary  # RunResult.summary is the joined string
    assert "coef_x1" not in res.estimates  # nothing fit
