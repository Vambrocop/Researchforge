"""Tests for goodness_of_fit — KS / Anderson-Darling / Cramer-von Mises / Shapiro.

Cross-checks:
  * normal data vs norm -> all tests present, Shapiro p large (not rejected);
  * non-normal (uniform) data vs norm -> Shapiro rejects;
  * KS statistic recomputed independently with fitted params matches the engine;
  * dist=expon path runs (AD present for expon);
  * positive-support dist on data with <=0 values honest-skips;
  * config column/dist override; too-few-rows honest skip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog import AnalysisEntry
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_ENTRY = AnalysisEntry(
    id="goodness_of_fit",
    method="Goodness-of-fit tests (KS / Anderson-Darling / Cramer-von Mises / Shapiro-Wilk)",
    domain="statistics",
    family="distribution",
    goal="describe",
    preconditions={"min_numeric_cols": 1, "min_rows": 8},
)


def _run(df: pd.DataFrame, tmp_path: Path, config=None):
    csv = tmp_path / "g.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    return run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"), config=config)


def test_normal_vs_norm_not_rejected(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(0, 1, 400)})
    res = _run(df, tmp_path)
    for k in ("ks_stat", "ks_p", "ad_stat", "cvm_stat", "cvm_p", "shapiro_p"):
        assert k in res.estimates
    # genuine normal data: Shapiro should not reject
    assert res.estimates["shapiro_p"] > 0.05
    out = Path(res.output_dir)
    assert (out / "goodness_of_fit.csv").exists()
    tbl = pd.read_csv(out / "goodness_of_fit.csv")
    # all four tests for the normal case
    assert any("Shapiro" in t for t in tbl["test"])
    assert any("Anderson" in t for t in tbl["test"])


def test_uniform_vs_norm_rejected(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.uniform(-1, 1, 800)})
    res = _run(df, tmp_path)
    # uniform is clearly non-normal at n=800 -> Shapiro rejects
    assert res.estimates["shapiro_p"] < 0.05
    assert "不" in res.summary  # verdict mentions does-not-fit


def test_ks_stat_independent_recompute(tmp_path: Path) -> None:
    from scipy import stats

    rng = np.random.default_rng(2)
    x = rng.normal(5, 2, 300)
    df = pd.DataFrame({"x": x})
    res = _run(df, tmp_path)
    params = stats.norm.fit(x)
    ks_ref = float(stats.kstest(x, stats.norm.cdf, args=params).statistic)
    assert abs(res.estimates["ks_stat"] - ks_ref) < 1e-4


def test_expon_dist_path(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"x": rng.exponential(2.0, 400)})
    res = _run(df, tmp_path, config={"dist": "expon"})
    out = Path(res.output_dir)
    tbl = pd.read_csv(out / "goodness_of_fit.csv")
    # Anderson-Darling supports expon; Shapiro should NOT appear (not normality)
    assert any("Anderson" in t for t in tbl["test"])
    assert not any("Shapiro" in t for t in tbl["test"])
    assert np.isnan(res.estimates["shapiro_p"])


def test_positive_dist_on_negative_skips(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"x": rng.normal(0, 3, 200)})  # has negatives
    res = _run(df, tmp_path, config={"dist": "gamma"})
    assert "跳过" in res.summary
    assert "ks_stat" not in res.estimates


def test_config_column_override(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    df = pd.DataFrame({
        "junk": rng.uniform(0, 1, 200),
        "target": rng.normal(0, 1, 200),
    })
    res = _run(df, tmp_path, config={"column": "target", "dist": "norm"})
    assert "target" in res.summary
    assert (Path(res.output_dir) / "goodness_of_fit.csv").exists()


def test_too_few_rows_skips(tmp_path: Path) -> None:
    df = pd.DataFrame({"x": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]})  # n=6 < 8
    res = _run(df, tmp_path)
    assert "ks_stat" not in res.estimates
    assert "跳过" in res.summary
    assert (Path(res.output_dir) / "report.md").exists()


def test_weibull_ad_critical_value_uses_confidence_level_table(tmp_path: Path) -> None:
    """Regression for the AD 5% critical-value lookup bug: scipy's anderson() reports
    weibull_min's table as CONFIDENCE levels (1-alpha) in [0.5..0.995], not the percent
    convention used by norm/expon/logistic/gumbel_r. The 5% critical value is the 0.95
    entry (~0.75-0.76), not the nearest-to-0.05 entry (~0.342, the 50th-percentile crit)."""
    from scipy import stats

    rng = np.random.default_rng(7)
    x = stats.weibull_min.rvs(2.0, size=300, random_state=rng)
    df = pd.DataFrame({"x": x})
    res = _run(df, tmp_path, config={"dist": "weibull_min"})
    out = Path(res.output_dir)
    tbl = pd.read_csv(out / "goodness_of_fit.csv")
    ad_row = tbl[tbl["test"].str.contains("Anderson")].iloc[0]
    crit5 = float(ad_row["critical_5pct"])
    # correct 5% critical value (the 0.95 confidence-level entry) is ~0.75-0.76;
    # the old bug picked the 0.5 entry (~0.342) instead.
    assert 0.7 < crit5 < 0.8
    assert abs(crit5 - 0.342) > 0.1


def test_norm_ad_critical_value_still_uses_percent_table(tmp_path: Path) -> None:
    """Regression: the norm/expon/logistic/gumbel_r percent-table path (target=5.0) must
    be unaffected by the weibull_min confidence-level disambiguation."""
    rng = np.random.default_rng(8)
    df = pd.DataFrame({"x": rng.normal(0, 1, 400)})
    res = _run(df, tmp_path)
    out = Path(res.output_dir)
    tbl = pd.read_csv(out / "goodness_of_fit.csv")
    ad_row = tbl[tbl["test"].str.contains("Anderson")].iloc[0]
    crit5 = float(ad_row["critical_5pct"])
    # scipy's norm 5% critical value (percent table index for 5.0) is ~0.75
    assert 0.7 < crit5 < 0.8
