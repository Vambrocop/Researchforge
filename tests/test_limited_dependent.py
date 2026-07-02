"""Tests for the limited-dependent-variable (LDV) family — Tobit / truncated /
Heckman. Each method gets (a) a recovery test against a known truth (with the
naive-OLS-biased contrast where relevant) and (b) a skip-honestly path. MLE
recovery is noisy: tolerances are generous (absolute ~0.2-0.4), all RNGs seeded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry(eid: str, method: str) -> AnalysisEntry:
    return AnalysisEntry(
        id=eid,
        method=method,
        domain="economics",
        family="econometrics",
        goal="explain",
        preconditions=Precondition(min_continuous=2, min_rows=20),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tobit
# ─────────────────────────────────────────────────────────────────────────────
def _tobit_data(seed: int = 0, b1: float = 2.0, b2: float = -1.0, n: int = 500):
    """y* = 1 + b1*x1 + b2*x2 + N(0,1), left-censored at 0 (y = max(y*, 0))."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    ystar = 1.0 + b1 * x1 + b2 * x2 + rng.normal(0, 1.0, n)
    y = np.maximum(ystar, 0.0)
    # outcome must be the FIRST continuous column for the auto-resolver
    return pd.DataFrame({"y": np.round(y, 4), "x1": np.round(x1, 4), "x2": np.round(x2, 4)})


def test_tobit_recovers_beta_and_beats_naive_ols(tmp_path: Path) -> None:
    csv = tmp_path / "tobit.csv"
    _tobit_data().to_csv(csv, index=False)
    fp = profile_dataset(csv)

    res = run_analysis(
        fp, _entry("tobit_regression", "Tobit (Type-I censored normal regression)"),
        output_root=str(tmp_path / "o"),
        config={"outcome": "y", "predictors": ["x1", "x2"],
                "censoring": "left", "censor_value": 0.0},
    )
    out = Path(res.output_dir)
    assert (out / "tobit_coefficients.csv").exists()
    e = res.estimates

    # Tobit recovers the latent-index betas
    assert abs(e["x1"] - 2.0) < 0.3
    assert abs(e["x2"] - (-1.0)) < 0.3
    # required diagnostic keys present
    for k in ("x1_se", "x1_ci_low", "x1_ci_high", "sigma", "loglik",
              "n_censored", "n_uncensored", "scale_factor_Ey"):
        assert k in e, f"missing estimate key {k}"
    assert e["n_censored"] >= 3
    assert "converged" in e

    # Naive OLS on censored y is attenuated toward 0 vs the truth; Tobit is closer.
    tab = pd.read_csv(out / "tobit_coefficients.csv").set_index("term")
    naive_x1 = float(tab.loc["x1", "naive_ols_coef"])
    assert abs(naive_x1) < abs(e["x1"])  # OLS shrunk toward zero
    assert abs(e["x1"] - 2.0) < abs(naive_x1 - 2.0)  # Tobit closer to truth


def _tobit_data_shifted(seed: int, mu: float, b1: float, sigma: float,
                        censor: float, n: int, side: str) -> pd.DataFrame:
    """y* = mu + b1*x1 + N(0,sigma), censored at a NON-zero threshold `censor`.
    Small b1 keeps X*beta tightly concentrated around its mean so the plug-in
    scale factor Phi((mean(Xb)-c)/sigma) tracks the population/empirical
    uncensored fraction closely (used to sanity-check the fix numerically)."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    ystar = mu + b1 * x1 + rng.normal(0, sigma, n)
    y = np.maximum(ystar, censor) if side == "left" else np.minimum(ystar, censor)
    return pd.DataFrame({"y": np.round(y, 6), "x1": np.round(x1, 6)})


def test_tobit_scale_factor_uses_censor_point_left(tmp_path: Path) -> None:
    """Regression test for the marginal-on-E[y] scale factor bug: the correct
    formula for LEFT-censoring is Phi((x̄β − c)/σ), not the old Phi(x̄β/σ) which
    silently assumed c=0. Threshold here is c=3 (far from 0) so the two
    formulas diverge sharply."""
    from scipy.stats import norm

    censor = 3.0
    df = _tobit_data_shifted(seed=7, mu=4.0, b1=0.4, sigma=1.0, censor=censor,
                             n=800, side="left")
    csv = tmp_path / "tobit_shift.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)

    res = run_analysis(
        fp, _entry("tobit_regression", "Tobit (Type-I censored normal regression)"),
        output_root=str(tmp_path / "o"),
        config={"outcome": "y", "predictors": ["x1"],
                "censoring": "left", "censor_value": censor},
    )
    e = res.estimates
    out = Path(res.output_dir)
    tab = pd.read_csv(out / "tobit_coefficients.csv").set_index("term")

    # Empirical uncensored fraction of the sample (proxy for the true scale).
    emp_frac = e["n_uncensored"] / e["n_obs"]
    scale = e["scale_factor_Ey"]
    assert abs(scale - emp_frac) < 0.05, (
        f"scale_factor_Ey={scale} should track the empirical uncensored "
        f"fraction {emp_frac} (within 0.05)"
    )

    # The OLD (buggy) formula dropped the censor point: Phi(x̄β/σ). At c=3
    # (far from 0) this is a very different number — make sure we're NOT it.
    mu_bar_est = float(tab.loc["const", "tobit_coef"] + tab.loc["x1", "tobit_coef"] * df["x1"].mean())
    sigma_est = e["sigma"]
    old_wrong = float(norm.cdf(mu_bar_est / sigma_est))
    assert abs(scale - old_wrong) > 0.05, (
        "scale_factor_Ey matches the old Phi(x̄β/σ) formula that ignores the "
        "censor point — the fix did not take effect"
    )

    # Sanity: scale_factor_Ey exactly reproduces the corrected closed form.
    correct = float(norm.cdf((mu_bar_est - censor) / sigma_est))
    assert abs(scale - correct) < 1e-3


def test_tobit_scale_factor_uses_censor_point_right(tmp_path: Path) -> None:
    """Same regression test for RIGHT-censoring: correct formula is
    Phi((c − x̄β)/σ)."""
    from scipy.stats import norm

    censor = -3.0
    df = _tobit_data_shifted(seed=7, mu=-4.0, b1=0.4, sigma=1.0, censor=censor,
                             n=800, side="right")
    csv = tmp_path / "tobit_shift_right.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)

    res = run_analysis(
        fp, _entry("tobit_regression", "Tobit (Type-I censored normal regression)"),
        output_root=str(tmp_path / "o"),
        config={"outcome": "y", "predictors": ["x1"],
                "censoring": "right", "censor_value": censor},
    )
    e = res.estimates
    out = Path(res.output_dir)
    tab = pd.read_csv(out / "tobit_coefficients.csv").set_index("term")

    emp_frac = e["n_uncensored"] / e["n_obs"]
    scale = e["scale_factor_Ey"]
    assert abs(scale - emp_frac) < 0.05

    mu_bar_est = float(tab.loc["const", "tobit_coef"] + tab.loc["x1", "tobit_coef"] * df["x1"].mean())
    sigma_est = e["sigma"]
    old_wrong = float(norm.cdf(mu_bar_est / sigma_est))
    assert abs(scale - old_wrong) > 0.05

    correct = float(norm.cdf((censor - mu_bar_est) / sigma_est))
    assert abs(scale - correct) < 1e-3


def test_tobit_skips_without_censoring_mass(tmp_path: Path) -> None:
    """Uncensored continuous outcome -> no limit mass -> honest skip to OLS."""
    rng = np.random.default_rng(3)
    n = 200
    x1 = rng.normal(0, 1, n)
    y = 1.0 + 2.0 * x1 + rng.normal(0, 1, n)  # no censoring
    df = pd.DataFrame({"y": np.round(y, 4), "x1": np.round(x1, 4),
                       "x2": np.round(rng.normal(0, 1, n), 4)})
    csv = tmp_path / "nocens.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)

    res = run_analysis(
        fp, _entry("tobit_regression", "Tobit (Type-I censored normal regression)"),
        output_root=str(tmp_path / "o"),
        config={"outcome": "y", "predictors": ["x1", "x2"]},
    )
    assert "x1" not in res.estimates
    assert "跳过" in res.summary and "删失" in res.summary


# ─────────────────────────────────────────────────────────────────────────────
# Truncated regression
# ─────────────────────────────────────────────────────────────────────────────
def _truncated_data(seed: int = 1, b1: float = 2.0, b2: float = -1.0, n_latent: int = 4000):
    """y* = 1 + b1*x1 + b2*x2 + N(0,1); keep ONLY y* > 0 (lower truncation at 0).
    The sample beyond the threshold is absent (truncation, not censoring)."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n_latent)
    x2 = rng.normal(0, 1, n_latent)
    ystar = 1.0 + b1 * x1 + b2 * x2 + rng.normal(0, 1.0, n_latent)
    keep = ystar > 0.0
    return pd.DataFrame({
        "y": np.round(ystar[keep], 4),
        "x1": np.round(x1[keep], 4),
        "x2": np.round(x2[keep], 4),
    })


def test_truncated_recovers_beta_and_beats_naive_ols(tmp_path: Path) -> None:
    df = _truncated_data()
    assert len(df) >= 50
    csv = tmp_path / "trunc.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)

    res = run_analysis(
        fp, _entry("truncated_regression", "Truncated-normal regression (MLE)"),
        output_root=str(tmp_path / "o"),
        config={"outcome": "y", "predictors": ["x1", "x2"],
                "truncation": "lower", "trunc_value": 0.0},
    )
    out = Path(res.output_dir)
    assert (out / "truncated_coefficients.csv").exists()
    e = res.estimates

    assert abs(e["x1"] - 2.0) < 0.3
    assert abs(e["x2"] - (-1.0)) < 0.3
    for k in ("x1_se", "x1_ci_low", "x1_ci_high", "sigma", "loglik",
              "trunc_value", "n_obs"):
        assert k in e
    assert "converged" in e

    # naive OLS on the truncated sample is attenuated; truncated MLE is closer
    naive_x1 = e["x1_naive_ols"]
    assert abs(e["x1"] - 2.0) < abs(naive_x1 - 2.0)


def test_truncated_too_few_rows_skips(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    n = 12
    df = pd.DataFrame({
        "y": np.round(rng.normal(5, 1, n), 4),
        "x1": np.round(rng.normal(0, 1, n), 4),
        "x2": np.round(rng.normal(0, 1, n), 4),
    })
    csv = tmp_path / "tiny.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry("truncated_regression", "Truncated-normal regression (MLE)"),
        output_root=str(tmp_path / "o"),
        config={"outcome": "y", "predictors": ["x1", "x2"]},
    )
    assert "x1" not in res.estimates
    assert "跳过" in res.summary


# ─────────────────────────────────────────────────────────────────────────────
# Heckman two-step
# ─────────────────────────────────────────────────────────────────────────────
def _heckman_data(seed: int = 2, b1: float = 1.0, rho: float = 0.7, n: int = 3000):
    """Correlated selection & outcome errors so naive OLS on the selected rows is
    biased. Selection eq uses x1 + z (z = exclusion restriction). Outcome eq uses x1.
    rho>0 correlates the two errors -> selection bias."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    z = rng.normal(0, 1, n)  # exclusion: enters selection only
    # bivariate-normal errors with correlation rho
    e_sel = rng.normal(0, 1, n)
    e_out = rho * e_sel + np.sqrt(1 - rho ** 2) * rng.normal(0, 1, n)

    sel_index = 0.3 + 0.9 * x1 + 1.1 * z + e_sel
    selected = (sel_index > 0).astype(int)
    y_star = 2.0 + b1 * x1 + e_out
    y = np.where(selected == 1, y_star, np.nan)  # observed only when selected
    return pd.DataFrame({
        "wage": np.round(y, 4),          # outcome (first continuous), NaN when not selected
        "x1": np.round(x1, 4),
        "z": np.round(z, 4),
        "work": selected,                # binary selection indicator
    })


def test_heckman_corrects_bias_and_flags_selection(tmp_path: Path) -> None:
    df = _heckman_data()
    csv = tmp_path / "heck.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)

    res = run_analysis(
        fp, _entry("heckman_selection", "Heckman two-step sample-selection model (heckit)"),
        output_root=str(tmp_path / "o"),
        config={"outcome": "wage", "predictors": ["x1"],
                "selection": "work", "exclusion": ["z"]},
    )
    out = Path(res.output_dir)
    assert (out / "heckman_coefficients.csv").exists()
    e = res.estimates

    # selection-bias test should fire (rho=0.7 != 0)
    assert "lambda_coef" in e and "lambda_p" in e
    assert e["lambda_p"] < 0.05, "selection-bias (lambda) test should be significant"
    assert "rho" in e and e["rho"] > 0  # recovered positive correlation
    assert e["n_selected"] > 0 and e["n_total"] >= e["n_selected"]

    # Heckman-corrected x1 closer to truth (1.0) than naive OLS on selected rows
    naive_x1 = e["x1_naive_ols"]
    assert abs(e["x1"] - 1.0) < abs(naive_x1 - 1.0) + 0.05
    assert abs(e["x1"] - 1.0) < 0.4


def test_heckman_derives_selection_from_missingness(tmp_path: Path) -> None:
    """No explicit selection col given -> derive from outcome NaN pattern; still
    produces the IMR coefficient + correction."""
    df = _heckman_data(seed=4).drop(columns=["work"])  # drop the binary indicator
    csv = tmp_path / "heck_miss.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)

    res = run_analysis(
        fp, _entry("heckman_selection", "Heckman two-step sample-selection model (heckit)"),
        output_root=str(tmp_path / "o"),
        config={"outcome": "wage", "predictors": ["x1"], "exclusion": ["z"]},
    )
    e = res.estimates
    assert "lambda_coef" in e
    assert "x1" in e
    assert (Path(res.output_dir) / "heckman_coefficients.csv").exists()


def test_heckman_skips_no_selection_variation(tmp_path: Path) -> None:
    """All rows selected (no variation) -> probit unidentified -> honest skip."""
    rng = np.random.default_rng(6)
    n = 100
    df = pd.DataFrame({
        "wage": np.round(rng.normal(5, 1, n), 4),
        "x1": np.round(rng.normal(0, 1, n), 4),
        "work": np.ones(n, dtype=int),  # everyone selected
    })
    csv = tmp_path / "novar.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry("heckman_selection", "Heckman two-step sample-selection model (heckit)"),
        output_root=str(tmp_path / "o"),
        config={"outcome": "wage", "predictors": ["x1"], "selection": "work"},
    )
    assert "lambda_coef" not in res.estimates
    assert "跳过" in res.summary
