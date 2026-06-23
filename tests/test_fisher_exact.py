"""Tests for fisher_exact — Fisher's exact test (2x2 exact; r×c honest degrade).

Cross-checks:
  * 2x2 odds ratio + exact two-sided p match scipy.stats.fisher_exact exactly;
  * OR direction is correct (>1 when the diagonal dominates);
  * a strong sparse 2x2 association is detected (small p); a balanced 2x2 is not;
  * r×c degrades honestly: OR is NaN, a note is emitted, p is finite and the
    Monte-Carlo / asymptotic-chi-square fallback is recorded;
  * too-few-columns honest skip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="fisher_exact", method="Fisher's exact test",
        domain="statistics", family="categorical_tests", goal="explain",
        preconditions=Precondition(min_categorical_cols=2, min_rows=4),
    )


def _df_from_table(tab: np.ndarray, rlabels, clabels) -> pd.DataFrame:
    rows = []
    for i, rl in enumerate(rlabels):
        for j, cl in enumerate(clabels):
            rows += [{"row": rl, "col": cl}] * int(tab[i, j])
    return pd.DataFrame(rows)


def test_2x2_matches_scipy(tmp_path: Path) -> None:
    # sparse 2x2 with diagonal dominance -> OR > 1.
    tab = np.array([[8, 2], [1, 9]])
    rl, cl = ["A", "B"], ["x", "y"]
    df = _df_from_table(tab, rl, cl)
    csv = tmp_path / "t.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"var1": "row", "var2": "col"})
    or_ref, p_ref = fisher_exact(tab, alternative="two-sided")
    assert abs(res.estimates["odds_ratio"] - float(or_ref)) < 1e-3
    assert abs(res.estimates["p_value"] - float(p_ref)) < 1e-5
    assert res.estimates["odds_ratio"] > 1.0  # direction correct
    assert res.estimates["table_rows"] == 2.0
    assert res.estimates["table_cols"] == 2.0
    assert res.estimates["n"] == float(tab.sum())
    assert (Path(res.output_dir) / "fisher_table.csv").exists()


def test_strong_sparse_assoc_detected(tmp_path: Path) -> None:
    tab = np.array([[15, 1], [1, 15]])
    rl, cl = ["A", "B"], ["x", "y"]
    df = _df_from_table(tab, rl, cl)
    csv = tmp_path / "eff.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"var1": "row", "var2": "col"})
    assert res.estimates["p_value"] < 0.001
    assert res.estimates["odds_ratio"] > 5.0
    assert "显著相关" in res.summary


def test_balanced_2x2_not_detected(tmp_path: Path) -> None:
    tab = np.array([[10, 10], [10, 10]])
    rl, cl = ["A", "B"], ["x", "y"]
    df = _df_from_table(tab, rl, cl)
    csv = tmp_path / "null.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"var1": "row", "var2": "col"})
    assert res.estimates["p_value"] > 0.5
    assert abs(res.estimates["odds_ratio"] - 1.0) < 1e-6


def test_rxc_degrades_honestly(tmp_path: Path) -> None:
    # 3x3 -> scipy fisher_exact cannot run; OR must be NaN, fallback p finite,
    # and the summary must disclose the degrade.
    tab = np.array([[30, 5, 5], [5, 30, 5], [5, 5, 30]])
    rl, cl = ["r0", "r1", "r2"], ["c0", "c1", "c2"]
    df = _df_from_table(tab, rl, cl)
    csv = tmp_path / "rxc.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"var1": "row", "var2": "col"})
    assert np.isnan(res.estimates["odds_ratio"])  # OR undefined for r×c
    assert "p_chi2_asymptotic" in res.estimates
    assert np.isfinite(res.estimates["p_value"])
    assert res.estimates["p_value"] < 0.01  # strong assoc still detected
    assert res.estimates["table_rows"] == 3.0 and res.estimates["table_cols"] == 3.0
    assert "2×2" in res.summary  # honest note about the 2x2-only limitation


def test_too_few_columns_skips(tmp_path: Path) -> None:
    df = pd.DataFrame({"only": ["a", "b"] * 15, "cont": np.arange(30.0)})
    csv = tmp_path / "few.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "odds_ratio" not in res.estimates
    assert "跳过" in res.summary
    assert (Path(res.output_dir) / "report.md").exists()
