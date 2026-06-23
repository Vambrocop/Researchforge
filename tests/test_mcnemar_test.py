"""Tests for mcnemar_test — McNemar's test for paired binary data.

Cross-checks:
  * the paired 2x2 (b, c discordant counts) is built correctly from the raw rows;
  * exact-binomial p (small discordant) matches an independent binom.test recompute;
  * continuity-corrected chi-square (large discordant) matches an independent
    (|b-c|-1)^2/(b+c) recompute;
  * a clear before/after shift is detected (small p, b>>c); a symmetric shift is not;
  * non-binary column and too-few-columns honest skips.
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
        id="mcnemar_test", method="McNemar's test (paired binary)",
        domain="statistics", family="categorical_tests", goal="explain",
        preconditions=Precondition(min_categorical_cols=2, min_rows=4),
    )


def _df_paired(a: int, b: int, c: int, d: int) -> pd.DataFrame:
    """Paired 2x2 [[a,b],[c,d]] with rows=before(0/1), cols=after(0/1):
    a = (0,0), b = (0,1) [0->1], c = (1,0) [1->0], d = (1,1).
    Larger label (1) is coded positive so the branch's coding lines up."""
    rows = []
    rows += [{"before": 0, "after": 0}] * a
    rows += [{"before": 0, "after": 1}] * b
    rows += [{"before": 1, "after": 0}] * c
    rows += [{"before": 1, "after": 1}] * d
    return pd.DataFrame(rows)


def _exact_binom_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p = binomial test of b successes in b+c at 0.5."""
    return float(stats.binomtest(int(b), int(b + c), 0.5, alternative="two-sided").pvalue)


def _cc_chi2(b: int, c: int) -> float:
    return (abs(b - c) - 1) ** 2 / (b + c)


def test_paired_table_and_exact_p(tmp_path: Path) -> None:
    # small discordant total (b+c=14 < 25) -> exact binomial branch.
    a, b, c, d = 30, 12, 2, 30
    df = _df_paired(a, b, c, d)
    csv = tmp_path / "t.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"var1": "before", "var2": "after"})
    assert res.estimates["b"] == float(b)
    assert res.estimates["c"] == float(c)
    assert res.estimates["n_discordant"] == float(b + c)
    assert res.estimates["n_pairs"] == float(a + b + c + d)
    assert abs(res.estimates["p_value"] - _exact_binom_p(b, c)) < 1e-4
    assert "精确二项检验" in res.summary
    assert (Path(res.output_dir) / "mcnemar_paired_table.csv").exists()


def test_continuity_corrected_chi2(tmp_path: Path) -> None:
    # large discordant total (b+c=55 >= 25) -> continuity-corrected chi-square.
    a, b, c, d = 40, 40, 15, 40
    df = _df_paired(a, b, c, d)
    csv = tmp_path / "cc.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"var1": "before", "var2": "after"})
    assert abs(res.estimates["statistic"] - _cc_chi2(b, c)) < 1e-3
    assert abs(res.estimates["p_value"] - float(stats.chi2.sf(_cc_chi2(b, c), 1))) < 1e-4
    assert "连续性校正" in res.summary


def test_clear_shift_detected(tmp_path: Path) -> None:
    # strong asymmetric shift 0->1 (b>>c) -> small p, b/c large.
    a, b, c, d = 20, 35, 3, 20
    df = _df_paired(a, b, c, d)
    csv = tmp_path / "shift.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"var1": "before", "var2": "after"})
    assert res.estimates["p_value"] < 0.001
    assert res.estimates["b"] > res.estimates["c"]
    assert "显著改变" in res.summary


def test_symmetric_no_shift(tmp_path: Path) -> None:
    # b ~ c -> no marginal change -> large p.
    a, b, c, d = 30, 20, 20, 30
    df = _df_paired(a, b, c, d)
    csv = tmp_path / "sym.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"var1": "before", "var2": "after"})
    assert res.estimates["p_value"] > 0.5
    assert "未见配对边际分布改变" in res.summary


def test_non_binary_column_skips(tmp_path: Path) -> None:
    # one column has 3 levels -> not paired binary -> honest skip.
    df = pd.DataFrame({
        "before": [0, 1, 2] * 10,
        "after": [0, 1] * 15,
    })
    csv = tmp_path / "nb.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"var1": "before", "var2": "after"})
    assert "statistic" not in res.estimates
    assert "跳过" in res.summary


def test_too_few_columns_skips(tmp_path: Path) -> None:
    df = pd.DataFrame({"bin": [0, 1] * 15, "cont": np.arange(30.0)})
    csv = tmp_path / "few.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "statistic" not in res.estimates
    assert "跳过" in res.summary
    assert (Path(res.output_dir) / "report.md").exists()
