"""Tests for tweedie_glm: compound Poisson-Gamma (zeros + positive continuous).

Synthetic data = a compound Poisson-Gamma outcome (a Poisson number of Gamma
jumps, summed): exactly zero when the Poisson count is 0, continuous positive
otherwise — the canonical Tweedie 1<p<2 case. The positive part's mean rises
with x1, so the log-link coefficient on x1 should be positive.
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
        id="tweedie_glm",
        method="Tweedie GLM (semi-continuous nonnegative outcome)",
        domain="statistics",
        family="statistics",
        goal="explain",
        preconditions=Precondition(requires_count_outcome=True, min_rows=30),
    )


def _compound_poisson_gamma(seed: int = 0, n: int = 600) -> pd.DataFrame:
    """Compound Poisson-Gamma: y = sum of N~Pois(lam) Gamma jumps; x1 raises mean."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    lam = np.exp(0.1 + 0.5 * x1 - 0.2 * x2)  # Poisson frequency rises with x1
    y = np.empty(n, dtype=float)
    for i in range(n):
        k = rng.poisson(lam[i])
        y[i] = rng.gamma(shape=2.0, scale=1.5, size=k).sum() if k > 0 else 0.0
    return pd.DataFrame({"cost": y, "x1": x1, "x2": x2})


def test_tweedie_fits_and_recovers_sign(tmp_path: Path) -> None:
    df = _compound_poisson_gamma(seed=1)
    # ensure the data really is semi-continuous (a zero mass + positive part)
    assert (df["cost"] == 0).mean() > 0.05, "expected a meaningful zero mass"
    assert (df["cost"] > 0).mean() > 0.3, "expected a continuous positive part"

    csv = tmp_path / "tweedie.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _make_entry(), output_root=str(tmp_path / "out"))
    out = Path(res.output_dir)

    assert "失败" not in res.summary, f"Tweedie reported failure: {res.summary}"
    assert (out / "summary.txt").exists()
    assert (out / "coefficients.csv").exists()
    assert (out / "report.md").exists()

    # log-link coefficient on x1 should be positive (true frequency effect +0.5)
    assert "x1" in res.estimates, f"x1 missing: {res.estimates}"
    assert res.estimates["x1"] > 0, f"x1 coef should be positive: {res.estimates['x1']}"

    # reported diagnostics
    assert "deviance" in res.estimates
    assert res.estimates["var_power"] == 1.5  # default
    assert res.estimates["pct_zeros_observed"] > 0


def test_tweedie_disclosure_present(tmp_path: Path) -> None:
    df = _compound_poisson_gamma(seed=2)
    csv = tmp_path / "tweedie2.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _make_entry(), output_root=str(tmp_path / "out"))
    assert "⚠" in res.summary
    assert "var_power" in res.summary


def test_tweedie_config_var_power_override(tmp_path: Path) -> None:
    df = _compound_poisson_gamma(seed=3)
    csv = tmp_path / "tweedie3.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp,
        _make_entry(),
        output_root=str(tmp_path / "out"),
        config={"outcome": "cost", "predictors": ["x1"], "var_power": 1.3},
    )
    assert "失败" not in res.summary, res.summary
    assert res.estimates["var_power"] == 1.3
    assert "x1" in res.estimates


def test_tweedie_degrades_without_numeric_outcome(tmp_path: Path) -> None:
    """No continuous/count numeric outcome -> honest skip."""
    n = 60
    df = pd.DataFrame(
        {
            "label": ["a", "b", "c"] * (n // 3),  # categorical text, not numeric
            "tag": ["x", "y"] * (n // 2),
        }
    )
    csv = tmp_path / "nonumeric.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _make_entry(), output_root=str(tmp_path / "out"))
    assert "未找到非负数值结果变量" in res.summary


def test_tweedie_degrades_too_few_rows(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    n = 12
    df = pd.DataFrame(
        {"cost": np.abs(rng.normal(3, 1, n)), "x1": rng.normal(0, 1, n)}
    )
    csv = tmp_path / "fewrows.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _make_entry(), output_root=str(tmp_path / "out"))
    assert "失败" in res.summary
    assert "行数不足" in res.summary
