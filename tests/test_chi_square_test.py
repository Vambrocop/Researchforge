"""Tests for chi_square_test — Pearson chi-square independence + goodness-of-fit.

Cross-checks:
  * chi2/df/p on a known table match scipy.stats.chi2_contingency(correction=False);
  * bias-corrected Cramer's V matches an independent Bergsma recompute;
  * a strongly-associated table rejects independence (small p, moderate V); a
    no-association control does not (large p, V ~ 0);
  * goodness-of-fit mode against uniform and config `expected`;
  * 2x2 Yates correction surfaced; sparse-cell flag; too-few-columns honest skip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="chi_square_test", method="Chi-square test (independence + goodness-of-fit)",
        domain="statistics", family="categorical_tests", goal="explain",
        preconditions=Precondition(min_categorical_cols=1, min_rows=4),
    )


def _df_from_table(tab: np.ndarray, rlabels, clabels) -> pd.DataFrame:
    rows = []
    for i, rl in enumerate(rlabels):
        for j, cl in enumerate(clabels):
            rows += [{"row": rl, "col": cl}] * int(tab[i, j])
    return pd.DataFrame(rows)


def _cramers_v_biascorr(tab: np.ndarray) -> float:
    chi2, _, _, _ = chi2_contingency(tab, correction=False)
    n = tab.sum()
    r, c = tab.shape
    phi2 = chi2 / n
    phi2c = max(0.0, phi2 - (c - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    cc = c - (c - 1) ** 2 / (n - 1)
    denom = min(rc - 1, cc - 1)
    return float(np.sqrt(phi2c / denom)) if denom > 0 else float("nan")


def test_chi2_matches_scipy_and_cramers_v(tmp_path: Path) -> None:
    tab = np.array([[20, 30, 10], [25, 10, 35]])
    rl, cl = ["A", "B"], ["x", "y", "z"]
    df = _df_from_table(tab, rl, cl)
    csv = tmp_path / "t.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"var1": "row", "var2": "col"})
    chi2_ref, p_ref, df_ref, _ = chi2_contingency(tab, correction=False)
    assert abs(res.estimates["chi2"] - chi2_ref) < 1e-2
    assert abs(res.estimates["p_value"] - p_ref) < 1e-4
    assert res.estimates["df"] == float(df_ref) == 2.0
    assert res.estimates["n"] == float(tab.sum())
    assert abs(res.estimates["cramers_v"] - _cramers_v_biascorr(tab)) < 1e-3
    out = Path(res.output_dir)
    assert (out / "chi_square_observed.csv").exists()
    assert (out / "chi_square_expected.csv").exists()


def test_strong_association_rejects(tmp_path: Path) -> None:
    tab = np.array([[80, 5, 5], [5, 80, 5], [5, 5, 80]])
    rl, cl = ["r0", "r1", "r2"], ["c0", "c1", "c2"]
    df = _df_from_table(tab, rl, cl)
    csv = tmp_path / "assoc.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"var1": "row", "var2": "col"})
    assert res.estimates["p_value"] < 0.001
    assert res.estimates["cramers_v"] > 0.3
    assert "不独立" in res.summary


def test_no_association_does_not_reject(tmp_path: Path) -> None:
    # product (independent) table: chi2 ~ 0, large p, V ~ 0.
    row_m = np.array([100.0, 200.0, 300.0])
    col_m = np.array([0.2, 0.3, 0.5])
    tab = np.round(np.outer(row_m, col_m)).astype(int)
    rl, cl = ["r0", "r1", "r2"], ["c0", "c1", "c2"]
    df = _df_from_table(tab, rl, cl)
    csv = tmp_path / "indep.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"var1": "row", "var2": "col"})
    assert res.estimates["p_value"] > 0.20
    assert res.estimates["cramers_v"] < 0.1


def test_2x2_yates_surfaced(tmp_path: Path) -> None:
    tab = np.array([[40, 10], [10, 40]])
    rl, cl = ["A", "B"], ["x", "y"]
    df = _df_from_table(tab, rl, cl)
    csv = tmp_path / "twobytwo.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"var1": "row", "var2": "col"})
    cy_ref, py_ref, _, _ = chi2_contingency(tab, correction=True)
    assert "chi2_yates" in res.estimates
    assert abs(res.estimates["chi2_yates"] - cy_ref) < 1e-2
    assert abs(res.estimates["p_value_yates"] - py_ref) < 1e-4


def test_goodness_of_fit_uniform(tmp_path: Path) -> None:
    # one categorical column, far from uniform -> reject; recompute chi2.
    counts = {"a": 50, "b": 10, "c": 10, "d": 10}
    rows = []
    for k, v in counts.items():
        rows += [{"cat": k}] * v
    df = pd.DataFrame(rows)
    csv = tmp_path / "gof.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"var1": "cat"})
    obs = np.array([50.0, 10.0, 10.0, 10.0])
    n = obs.sum()
    exp = np.full(4, n / 4)
    chi2_ref = float(np.sum((obs - exp) ** 2 / exp))
    assert abs(res.estimates["chi2"] - chi2_ref) < 1e-2
    assert res.estimates["df"] == 3.0
    assert res.estimates["p_value"] < 0.001
    assert "拟合优度" in res.summary
    assert (Path(res.output_dir) / "chi_square_gof_table.csv").exists()


def test_goodness_of_fit_config_expected(tmp_path: Path) -> None:
    # observed roughly matches a 2:1:1 expected -> should NOT reject.
    rows = [{"cat": "a"}] * 50 + [{"cat": "b"}] * 25 + [{"cat": "c"}] * 25
    df = pd.DataFrame(rows)
    csv = tmp_path / "gofcfg.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"var1": "cat", "expected": {"a": 2, "b": 1, "c": 1}})
    assert res.estimates["p_value"] > 0.20
    assert "config 指定的期望分布" in res.summary


def test_sparse_cells_flagged(tmp_path: Path) -> None:
    tab = np.array([[3, 1, 0], [1, 0, 2], [0, 2, 1]])
    rl, cl = ["r0", "r1", "r2"], ["c0", "c1", "c2"]
    df = _df_from_table(tab, rl, cl)
    csv = tmp_path / "sparse.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"var1": "row", "var2": "col"})
    assert res.estimates["n_sparse_cells"] > 0
    assert "期望频数<5" in res.summary


def test_no_categorical_column_skips(tmp_path: Path) -> None:
    df = pd.DataFrame({"cont": np.arange(30.0), "cont2": np.arange(30.0) * 2})
    csv = tmp_path / "none.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "chi2" not in res.estimates
    assert "跳过" in res.summary
    assert (Path(res.output_dir) / "report.md").exists()
