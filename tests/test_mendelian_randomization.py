"""Tests for mendelian_randomization: two-sample summary-data MR (IVW + MR-Egger +
weighted median). With directional pleiotropy, IVW is biased and MR-Egger's intercept
detects it while its slope corrects toward the truth; honest skip when columns absent."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="mendelian_randomization", method="Mendelian randomization", domain="epidemiology",
        family="causal", goal="explain", preconditions=Precondition(min_continuous=4, min_rows=3),
    )


def _mr_data(seed: int = 0, K: int = 40, theta: float = 0.4, pleio_mean: float = 0.03):
    rng = np.random.default_rng(seed)
    bx = rng.uniform(0.1, 0.6, K)
    sex = rng.uniform(0.02, 0.05, K)
    sey = rng.uniform(0.02, 0.05, K)
    pleio = rng.normal(pleio_mean, 0.01, K)          # directional pleiotropy
    by = theta * bx + pleio + rng.normal(0, sey)
    return pd.DataFrame({"beta_exposure": bx, "se_exposure": sex,
                         "beta_outcome": by, "se_outcome": sey})


def test_mr_estimators_recover_and_detect_pleiotropy(tmp_path: Path) -> None:
    csv = tmp_path / "mr.csv"
    _mr_data(theta=0.4, pleio_mean=0.03).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)
    assert (out / "mr_instruments.csv").exists()
    e = res.estimates
    assert e["n_instruments"] == 40.0
    # all three estimators land in a plausible neighbourhood of the true 0.4
    for k in ("ivw_estimate", "egger_slope", "weighted_median"):
        assert 0.2 < e[k] < 0.7, (k, e[k])
    # MR-Egger intercept detects the injected directional pleiotropy
    assert e["egger_intercept"] > 0
    # Egger slope corrects toward truth: closer to 0.4 than the pleiotropy-biased IVW
    assert abs(e["egger_slope"] - 0.4) <= abs(e["ivw_estimate"] - 0.4) + 1e-9
    assert "cochran_q" in e and "weighted_median_se" in e


def test_mr_egger_handles_mixed_sign_betas(tmp_path: Path) -> None:
    # Real GWAS betas are mixed-sign (coded-allele dependent). MR-Egger must orient each
    # variant to a positive exposure effect; without orientation its slope/intercept break.
    rng = np.random.default_rng(3)
    K, theta, pleio = 50, 0.4, 0.03
    mag = rng.uniform(0.15, 0.6, K)
    sign = rng.choice([-1.0, 1.0], K)
    bx = sign * mag                                  # MIXED-SIGN exposure effects
    sex = rng.uniform(0.02, 0.05, K)
    sey = rng.uniform(0.02, 0.05, K)
    by = theta * bx + sign * pleio + rng.normal(0, sey)   # pleiotropy on the oriented scale
    df = pd.DataFrame({"beta_exposure": bx, "se_exposure": sex,
                       "beta_outcome": by, "se_outcome": sey})
    csv = tmp_path / "mrs.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    e = res.estimates
    # with orientation, Egger slope recovers theta and the intercept recovers the
    # oriented pleiotropy (~0.03 > 0) — both would be wrong on raw mixed-sign betas.
    assert abs(e["egger_slope"] - 0.4) < 0.2, e["egger_slope"]
    assert e["egger_intercept"] > 0
    assert abs(e["ivw_estimate"] - 0.4) < 0.25


def test_mr_config_override_columns(tmp_path: Path) -> None:
    df = _mr_data().rename(columns={"beta_exposure": "bx", "se_exposure": "sx",
                                    "beta_outcome": "by", "se_outcome": "sy"})
    csv = tmp_path / "mr2.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"beta_exposure": "bx", "se_exposure": "sx",
                               "beta_outcome": "by", "se_outcome": "sy"})
    assert "ivw_estimate" in res.estimates


def test_mr_skips_without_summary_columns(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(0, 1, 20), "b": rng.normal(0, 1, 20)})
    csv = tmp_path / "x.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "ivw_estimate" not in res.estimates
    assert "孟德尔随机化跳过" in res.summary
