"""Tests for the `pls_regression` executor branch (Partial Least Squares).

Known structure: a continuous outcome driven by a single LATENT factor that loads on
a block of strongly collinear predictors (the rest are noise). PLS should predict well
(high cross-validated R-squared with few components) and the VIP scores should flag the
driving (latent-loaded) predictors over the noise ones. Plus degrade checks (no
continuous outcome / too few predictors / too few rows -> honest skip) and a
config-override check.

The catalog yaml exists (ordination.yaml) but the AnalysisEntry is built inline so the
test is self-contained.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog import AnalysisEntry
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_ENTRY = AnalysisEntry(
    id="pls_regression",
    method="Partial least squares regression (PLS)",
    domain="statistics",
    family="ml",
    goal="predict",
    preconditions={"min_continuous": 3, "min_rows": 10},
)


def _latent_csv(tmp_path: Path, n: int = 120) -> Path:
    """y driven by ONE latent factor that loads on a collinear predictor block.

    x1..x4 are noisy copies of a latent z (so highly collinear); x5..x8 are pure
    noise. y = 3*z + small noise. The driving predictors are x1..x4 -> they should
    carry the high VIP scores.
    """
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, n)
    # collinear block loaded on the latent factor
    x1 = z + rng.normal(0, 0.15, n)
    x2 = z + rng.normal(0, 0.15, n)
    x3 = z + rng.normal(0, 0.15, n)
    x4 = z + rng.normal(0, 0.15, n)
    # noise predictors (no relation to y)
    x5 = rng.normal(0, 1, n)
    x6 = rng.normal(0, 1, n)
    x7 = rng.normal(0, 1, n)
    x8 = rng.normal(0, 1, n)
    y = 3.0 * z + rng.normal(0, 0.3, n)
    df = pd.DataFrame({
        "y": y,
        "x1": x1, "x2": x2, "x3": x3, "x4": x4,
        "x5": x5, "x6": x6, "x7": x7, "x8": x8,
    })
    csv = tmp_path / "pls_latent.csv"
    df.to_csv(csv, index=False)
    return csv


def test_pls_predicts_well(tmp_path):
    csv = _latent_csv(tmp_path)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "out"),
                       config={"outcome": "y"})
    out = Path(res.output_dir)

    assert (out / "pls_cv_r2.csv").exists(), "pls_cv_r2.csv missing"
    assert (out / "pls_vip.csv").exists(), "pls_vip.csv missing"
    assert (out / "pls_component_variance.csv").exists()
    assert (out / "pls_coefficients.csv").exists()

    for k in ("cv_r2", "insample_r2", "n_components", "n_vip_above_1"):
        assert k in res.estimates, f"{k} missing from estimates"

    # latent signal is strong -> PLS predicts well out of sample.
    assert res.estimates["cv_r2"] > 0.85, res.estimates
    # one latent dimension suffices.
    assert res.estimates["n_components"] <= 3, res.estimates


def test_pls_vip_flags_driving_predictors(tmp_path):
    csv = _latent_csv(tmp_path)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "out"),
                       config={"outcome": "y"})
    out = Path(res.output_dir)

    vip = pd.read_csv(out / "pls_vip.csv").set_index("predictor")["VIP"]
    driving = ["x1", "x2", "x3", "x4"]
    noise = ["x5", "x6", "x7", "x8"]
    # all driving predictors have higher VIP than every noise predictor.
    assert vip[driving].min() > vip[noise].max(), vip.to_dict()
    # driving predictors are flagged important (VIP > 1).
    assert (vip[driving] > 1.0).all(), vip[driving].to_dict()

    # VIP normalization sanity: mean of squared VIP ~ 1 (Wold's identity).
    msq = float((vip ** 2).mean())
    assert abs(msq - 1.0) < 0.05, f"mean(VIP^2) should be ~1, got {msq}"


def test_pls_no_continuous_outcome_skips(tmp_path):
    """With no continuous column the branch has no outcome -> honest skip."""
    rng = np.random.default_rng(1)
    n = 40
    df = pd.DataFrame({
        "cat": rng.choice(["a", "b", "c"], n),
        "flag": rng.integers(0, 2, n),
    })
    csv = tmp_path / "no_cont.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "out"))
    assert "cv_r2" not in res.estimates
    assert "跳过" in res.summary


def test_pls_too_few_predictors_skips(tmp_path):
    rng = np.random.default_rng(2)
    n = 40
    df = pd.DataFrame({"y": rng.normal(0, 1, n)})
    csv = tmp_path / "one_col.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "out"))
    assert "cv_r2" not in res.estimates
    assert "跳过" in res.summary


def test_pls_too_few_rows_skips(tmp_path):
    rng = np.random.default_rng(3)
    n = 6
    df = pd.DataFrame({
        "y": rng.normal(0, 1, n),
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(0, 1, n),
        "x3": rng.normal(0, 1, n),
    })
    csv = tmp_path / "few_rows.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "out"))
    assert "cv_r2" not in res.estimates
    assert "跳过" in res.summary


def test_pls_resolver_picks_named_outcome_not_first(tmp_path):
    """A decoy continuous column ('other_metric', unrelated to the latent signal) is
    placed BEFORE 'y' — the shared resolver must still pick 'y', not cont[0]."""
    rng = np.random.default_rng(9)
    n = 120
    z = rng.normal(0, 1, n)
    x1 = z + rng.normal(0, 0.15, n)
    x2 = z + rng.normal(0, 0.15, n)
    x3 = z + rng.normal(0, 0.15, n)
    x4 = z + rng.normal(0, 0.15, n)
    y = 3.0 * z + rng.normal(0, 0.3, n)
    other_metric = rng.normal(0, 1, n)  # decoy, unrelated, FIRST column
    df = pd.DataFrame({
        "other_metric": other_metric, "x1": x1, "x2": x2, "x3": x3, "x4": x4, "y": y,
    })
    csv = tmp_path / "resolver.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    assert fp.likely_outcome == "y" and fp.likely_outcome_confidence == "high"
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "out"))  # no config outcome
    # latent signal is strong -> high cv_r2; a wrong (positional) pick of the unrelated
    # other_metric column as outcome would produce near-zero cv_r2.
    assert res.estimates.get("cv_r2", 0) > 0.85, res.estimates


def test_pls_config_override(tmp_path):
    """config predictors restricts the predictor set; n_components fixes k."""
    csv = _latent_csv(tmp_path)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _ENTRY, output_root=str(tmp_path / "out"),
        config={"outcome": "y", "predictors": ["x1", "x2", "x3"], "n_components": 1},
    )
    out = Path(res.output_dir)
    assert (out / "pls_vip.csv").exists()
    assert res.estimates["n_components"] == 1
    vip = pd.read_csv(out / "pls_vip.csv")
    assert set(vip["predictor"]) == {"x1", "x2", "x3"}
