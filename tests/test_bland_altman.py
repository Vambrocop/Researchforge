"""Tests for bland_altman — Bland & Altman (1986) method-comparison agreement.

Known-value cross-checks:
  * a constant known bias between two methods -> bias recovered, LoA = bias +/- 1.96 SD;
  * identical methods -> bias ~ 0 and tight LoA;
  * a constructed proportional-bias case -> significant regression slope flagged;
  * the agreement-not-correlation disclosure is present;
  * config method1/method2 override and the <2-continuous honest skip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog import AnalysisEntry
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_ENTRY = AnalysisEntry(
    id="bland_altman",
    method="Bland-Altman method-comparison agreement",
    domain="statistics",
    family="agreement",
    goal="describe",
    preconditions={"min_continuous": 2, "min_rows": 10},
)


def test_constant_bias_recovered(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n = 200
    true = rng.normal(100, 15, n)
    bias_true = 4.0
    m1 = true + bias_true + rng.normal(0, 2, n)
    m2 = true + rng.normal(0, 2, n)
    df = pd.DataFrame({"method_a": m1, "method_b": m2})
    csv = tmp_path / "bias.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))

    diff = m1 - m2
    bias = diff.mean()
    sd = diff.std(ddof=1)
    # bias recovered (close to the planted 4.0, and to the empirical mean diff)
    assert abs(res.estimates["bias"] - bias) < 1e-6
    assert abs(res.estimates["bias"] - bias_true) < 1.0
    # LoA = bias +/- 1.96 SD (exact)
    assert abs(res.estimates["loa_upper"] - (bias + 1.96 * sd)) < 1e-6
    assert abs(res.estimates["loa_lower"] - (bias - 1.96 * sd)) < 1e-6
    # roughly 95% of points within the limits
    assert 90.0 <= res.estimates["pct_within_loa"] <= 100.0


def test_identical_methods_zero_bias(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(50, 10, 100)
    df = pd.DataFrame({"m1": x, "m2": x.copy()})
    csv = tmp_path / "same.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))
    assert abs(res.estimates["bias"]) < 1e-9
    assert abs(res.estimates["sd_diff"]) < 1e-9
    assert abs(res.estimates["loa_upper"]) < 1e-9
    assert abs(res.estimates["loa_lower"]) < 1e-9


def test_proportional_bias_flagged(tmp_path: Path) -> None:
    # Difference grows with magnitude -> significant slope of diff on mean.
    rng = np.random.default_rng(5)
    n = 250
    true = rng.uniform(10, 100, n)
    # m1 systematically over-reads in proportion to the value
    m1 = true * 1.10 + rng.normal(0, 1, n)
    m2 = true + rng.normal(0, 1, n)
    df = pd.DataFrame({"device_x": m1, "device_y": m2})
    csv = tmp_path / "prop.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))
    assert res.estimates["proportional_bias_p"] < 0.05
    assert res.estimates["proportional_bias_slope"] > 0.0
    assert "比例偏差" in res.summary


def test_agreement_not_correlation_disclosure(tmp_path: Path) -> None:
    rng = np.random.default_rng(6)
    n = 80
    true = rng.normal(0, 1, n)
    m1 = true + 3.0  # large constant bias but near-perfect correlation
    m2 = true
    df = pd.DataFrame({"a": m1, "b": m2})
    csv = tmp_path / "corr.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))
    # high correlation but a clear systematic bias — the core ⚠
    assert res.estimates["pearson_r"] > 0.95
    assert abs(res.estimates["bias"] - 3.0) < 0.2
    assert "一致性" in res.summary and "相关" in res.summary


def test_config_override_and_products(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    n = 60
    true = rng.normal(20, 4, n)
    df = pd.DataFrame(
        {
            "noise_col": rng.normal(0, 1, n),
            "scan_old": true + 2.0 + rng.normal(0, 1, n),
            "scan_new": true + rng.normal(0, 1, n),
        }
    )
    csv = tmp_path / "cfg.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp,
        _ENTRY,
        output_root=str(tmp_path / "o"),
        config={"method1": "scan_old", "method2": "scan_new"},
    )
    assert abs(res.estimates["bias"] - 2.0) < 1.0
    out = Path(res.output_dir)
    assert (out / "bland_altman_estimates.csv").exists()
    assert (out / "bland_altman_pairs.csv").exists()
    # bias 95% CI present and brackets the point estimate
    assert res.estimates["bias_ci_low"] < res.estimates["bias"] < res.estimates["bias_ci_high"]


def test_too_few_continuous_skips(tmp_path: Path) -> None:
    # one genuine continuous column (non-integer floats) + a categorical -> skip
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"only_one": rng.normal(0, 1, 30), "label": ["a", "b", "c"] * 10})
    csv = tmp_path / "one.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))
    assert "bias" not in res.estimates
    assert "跳过" in res.summary
    assert (Path(res.output_dir) / "report.md").exists()
