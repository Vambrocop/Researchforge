"""Tests for zero_inflated_negbin: overdispersed excess-zero counts + degrade.

Synthetic data = structural-zero mixture + an OVERDISPERSED count process
(negative-binomial-like via a Gamma-mixed Poisson). ZINB should fit, estimate
a positive dispersion alpha, and recover a positive x1 count-coefficient sign.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _make_entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="zero_inflated_negbin",
        method="Zero-inflated negative binomial (excess zeros + overdispersion)",
        domain="statistics",
        family="statistics",
        goal="explain",
        preconditions=Precondition(requires_count_outcome=True, min_rows=40),
    )


def _overdispersed_excess_zero(seed: int = 0, n: int = 800) -> pd.DataFrame:
    """Structural zeros + Gamma-mixed Poisson (overdispersion) with x1 effect."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    mu = np.exp(0.4 + 0.5 * x1 - 0.2 * x2)
    # Gamma mixing => negative-binomial counts (variance > mean = overdispersion)
    gamma_noise = rng.gamma(shape=1.0, scale=1.0, size=n)  # mean 1, adds variance
    counts = rng.poisson(mu * gamma_noise)
    structural = rng.random(n) < 0.40
    counts = np.where(structural, 0, counts).astype(int)
    return pd.DataFrame({"claims": counts, "x1": x1, "x2": x2})


def test_zinb_fits_alpha_positive(tmp_path: Path) -> None:
    df = _overdispersed_excess_zero(seed=1)
    csv = tmp_path / "zinb.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    assert any(c.kind == "count" for c in fp.columns), (
        f"'claims' not detected as count: {[(c.name, c.kind) for c in fp.columns]}"
    )

    res = run_analysis(fp, _make_entry(), output_root=str(tmp_path / "out"))
    out = Path(res.output_dir)

    assert "失败" not in res.summary, f"ZINB reported failure: {res.summary}"
    assert (out / "summary.txt").exists()
    assert (out / "count_coefficients.csv").exists()
    assert (out / "inflation_coefficients.csv").exists()

    # dispersion alpha must be present and positive (overdispersion confirmed)
    assert "alpha" in res.estimates, f"alpha missing: {res.estimates}"
    assert res.estimates["alpha"] > 0, f"alpha should be >0: {res.estimates['alpha']}"

    # count-coefficient sign recovery (true x1 effect is +0.5)
    assert "x1" in res.estimates, f"x1 missing: {res.estimates}"
    assert res.estimates["x1"] > 0, f"x1 count-coef should be positive: {res.estimates['x1']}"

    # AIC trio recorded
    for k in ("aic_zinb", "aic_zip", "aic_poisson"):
        assert k in res.estimates, f"{k} missing from estimates"


def test_zinb_disclosure_present(tmp_path: Path) -> None:
    df = _overdispersed_excess_zero(seed=2)
    csv = tmp_path / "zinb2.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _make_entry(), output_root=str(tmp_path / "out"))
    assert "⚠" in res.summary
    assert "α" in res.summary  # alpha / overdispersion is discussed


def test_zinb_config_predictor_override(tmp_path: Path) -> None:
    df = _overdispersed_excess_zero(seed=3)
    csv = tmp_path / "zinb3.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp,
        _make_entry(),
        output_root=str(tmp_path / "out"),
        config={"outcome": "claims", "predictors": ["x1"]},
    )
    assert "失败" not in res.summary, res.summary
    assert "x1" in res.estimates


def test_zinb_degrades_without_count(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    n = 80
    df = pd.DataFrame({"y": rng.normal(5, 1, n), "x1": rng.normal(0, 1, n)})
    csv = tmp_path / "nocount.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _make_entry(), output_root=str(tmp_path / "out"))
    assert "未找到计数型结果变量" in res.summary


def test_zinb_degrades_too_few_rows(tmp_path: Path) -> None:
    rng = np.random.default_rng(10)
    n = 15
    df = pd.DataFrame(
        {"claims": rng.poisson(2, n).astype(int), "x1": rng.normal(0, 1, n)}
    )
    csv = tmp_path / "fewrows.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _make_entry(), output_root=str(tmp_path / "out"))
    assert "失败" in res.summary
    assert "行数不足" in res.summary
