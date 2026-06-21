"""Tests for mahalanobis_outliers: robust (MCD) multivariate outlier detection.

Known structure: a clean multivariate-normal cloud + a few injected FAR outliers ->
  * the injected points are flagged (robust d^2 > chi-square cutoff),
  * few clean points are flagged,
  * the ROBUST covariance ~= the clean truth while the CLASSICAL covariance is inflated
    by the outliers (generalized-variance ratio classical/robust > 1).
Plus alpha config override + a degrade (too few features / rows, n <= 2p). Fixed seed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="mahalanobis_outliers",
        method="Robust Mahalanobis outlier detection (MCD)",
        domain="statistics",
        family="mixture",
        goal="explore",
        preconditions=Precondition(min_continuous=2, min_rows=10),
    )


def _clean_plus_outliers(seed: int = 0, n_clean: int = 300, n_out: int = 10):
    rng = np.random.default_rng(seed)
    # clean cloud: correlated bivariate normal, sd ~ 1
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    clean = rng.multivariate_normal([0.0, 0.0], cov, n_clean)
    # injected outliers: far from the cloud
    outliers = rng.normal([15.0, -15.0], 0.5, (n_out, 2))
    XY = np.vstack([clean, outliers])
    df = pd.DataFrame({"f1": XY[:, 0], "f2": XY[:, 1]})
    out_idx = list(range(n_clean, n_clean + n_out))  # last rows are the injected ones
    return df, out_idx, cov


def _run(df: pd.DataFrame, tmp_path: Path, config=None):
    tmp_path.mkdir(parents=True, exist_ok=True)  # callers may pass a nested subdir
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    return run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config=config or {})


def test_flags_injected_outliers(tmp_path: Path) -> None:
    df, out_idx, _ = _clean_plus_outliers(seed=0)
    res = _run(df, tmp_path)
    assert "完成" in res.summary
    out = Path(res.output_dir)
    dist = pd.read_csv(out / "mahalanobis_distances.csv")
    flagged = set(dist.loc[dist["outlier"], "row"])
    # every injected outlier is caught
    assert set(out_idx) <= flagged
    # few clean points flagged (chi-square 0.975 -> ~2.5% false-positive expected;
    # allow a generous bound)
    clean_flagged = len(flagged - set(out_idx))
    assert clean_flagged <= 0.06 * 300


def test_robust_cov_near_truth_classical_inflated(tmp_path: Path) -> None:
    df, out_idx, true_cov = _clean_plus_outliers(seed=1)
    res = _run(df, tmp_path)
    out = Path(res.output_dir)
    rob = pd.read_csv(out / "robust_covariance.csv", index_col=0).to_numpy()
    cla = pd.read_csv(out / "classical_covariance.csv", index_col=0).to_numpy()
    # robust covariance is close to the clean truth (MCD ignores the outliers).
    # MCD applies a consistency correction; the structure (variances ~1, corr ~0.5)
    # should be recovered to within a loose tolerance.
    assert np.linalg.norm(rob - true_cov) < 0.6
    # classical covariance is badly inflated by the far outliers -> much larger det
    assert np.linalg.det(cla) > np.linalg.det(rob) * 3
    # the engine reports this inflation ratio
    assert res.estimates["gen_var_ratio_classical_over_robust"] > 3.0


def test_cutoff_is_chi_square(tmp_path: Path) -> None:
    df, _, _ = _clean_plus_outliers(seed=2)
    res = _run(df, tmp_path)
    # default alpha 0.975, p = 2 features
    expected = float(stats.chi2.ppf(0.975, df=2))
    assert abs(res.estimates["chi2_cutoff"] - expected) < 1e-9
    out = Path(res.output_dir)
    assert (out / "mahalanobis_distances.png").exists()


def test_alpha_config_override(tmp_path: Path) -> None:
    df, out_idx, _ = _clean_plus_outliers(seed=3)
    # a stricter alpha raises the cutoff -> fewer flags than the default
    res_default = _run(df, tmp_path, {"features": ["f1", "f2"]})
    res_strict = _run(df, tmp_path / "strict", {"alpha": 0.999, "features": ["f1", "f2"]})
    expected = float(stats.chi2.ppf(0.999, df=2))
    assert abs(res_strict.estimates["chi2_cutoff"] - expected) < 1e-9
    assert res_strict.estimates["chi2_cutoff"] > res_default.estimates["chi2_cutoff"]
    # injected far outliers are still caught even at the strict cutoff
    out = Path(res_strict.output_dir)
    dist = pd.read_csv(out / "mahalanobis_distances.csv")
    flagged = set(dist.loc[dist["outlier"], "row"])
    assert set(out_idx) <= flagged


def test_degrade_one_feature(tmp_path: Path) -> None:
    df = pd.DataFrame({"f1": np.random.default_rng(0).normal(0, 1, 50)})
    res = _run(df, tmp_path)
    assert "跳过" in res.summary
    assert not res.estimates


def test_degrade_too_few_rows(tmp_path: Path) -> None:
    # n must exceed ~2*p; p=2 needs n>=10 here, give 8 -> skip
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"f1": rng.normal(0, 1, 8), "f2": rng.normal(0, 1, 8)})
    res = _run(df, tmp_path)
    assert "跳过" in res.summary
    assert not res.estimates
