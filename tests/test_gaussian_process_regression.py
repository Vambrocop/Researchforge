"""Tests for gaussian_process_regression: a GP recovers a smooth nonlinear signal
with decent HELD-OUT R², its 95% predictive interval is sanely calibrated, and it
writes a predictions CSV. Plus an honest skip on too-few rows. Seeded RNG; generous
tolerances (GP fit + a random split). scikit-learn is always installed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="gaussian_process_regression",
        method="Gaussian-process regression (Bayesian nonparametric)",
        domain="statistics",
        family="bayesian",
        goal="predict",
        preconditions=Precondition(min_continuous=2, min_rows=20),
    )


def _smooth_nonlinear(seed: int = 0, n: int = 120):
    """y is a smooth nonlinear function of one x (sine + quadratic) plus small noise —
    exactly what an RBF Gaussian process should fit well out-of-sample."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-3.0, 3.0, n))
    y = np.sin(1.5 * x) + 0.3 * x**2 + rng.normal(0, 0.15, n)
    # outcome first so the regression convention picks y as the outcome
    return pd.DataFrame({"y": np.round(y, 5), "x": np.round(x, 5)})


def test_gpr_fits_smooth_signal_and_calibrated(tmp_path: Path) -> None:
    csv = tmp_path / "smooth.csv"
    _smooth_nonlinear().to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"outcome": "y", "predictors": ["x"], "random_state": 0},
    )
    out = Path(res.output_dir)

    # predictions CSV exists with the expected columns
    pred_path = out / "gpr_predictions.csv"
    assert pred_path.exists()
    pdf = pd.read_csv(pred_path)
    for col in ("actual", "pred_mean", "lower_95", "upper_95"):
        assert col in pdf.columns

    est = res.estimates
    # held-out R² is decent on a clean smooth signal
    assert "r2_heldout" in est
    assert est["r2_heldout"] > 0.5
    # 95% predictive-interval coverage in a sane range (generous: split is small)
    assert 0.7 <= est["coverage_95"] <= 1.0
    # learned hyper-parameters + bookkeeping are present as plain floats
    assert est["lengthscale"] > 0
    assert "log_marginal_likelihood" in est
    assert est["n_predictors"] == 1.0
    assert est["n"] == 120.0
    # every estimate is a plain float (RED LINE)
    assert all(isinstance(v, float) for v in est.values())


def test_gpr_resolver_picks_high_confidence_outcome_not_first(tmp_path: Path) -> None:
    """A high-confidence-named outcome ('target') placed AFTER a decoy continuous
    column must still be resolved as the outcome (shared resolve_outcome, not raw
    cont[0]) — no config override this time, so the resolver alone must fire."""
    df = _smooth_nonlinear(seed=9).rename(columns={"y": "target"})
    decoy = pd.Series(np.random.default_rng(10).normal(0, 1, len(df)), name="decoy")
    df = pd.concat([decoy, df[["target", "x"]]], axis=1)
    csv = tmp_path / "resolver.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    assert fp.likely_outcome == "target" and fp.likely_outcome_confidence == "high"
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    est = res.estimates
    assert "r2_heldout" in est
    # only true if 'target' (the smooth nonlinear signal) was modeled, not 'decoy'
    assert est["r2_heldout"] > 0.3


def test_gpr_skips_too_few_rows(tmp_path: Path) -> None:
    """Fewer than 20 rows → honest Chinese skip, no crash, no estimates."""
    rng = np.random.default_rng(1)
    x = np.linspace(0, 1, 12)
    df = pd.DataFrame({"y": np.round(x**2 + rng.normal(0, 0.05, 12), 5),
                       "x": np.round(x, 5)})
    csv = tmp_path / "tiny.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"outcome": "y", "predictors": ["x"]},
    )
    assert "r2_heldout" not in res.estimates
    assert "跳过" in res.summary
