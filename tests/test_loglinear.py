"""Tests for loglinear — log-linear independence model + G-squared LR test.

Cross-checks:
  * G-squared and Pearson chi-square match an independent recompute on a known table;
  * G-squared df = (r-1)(c-1);
  * a strongly-associated table rejects independence (small p); an independent
    (product) table does not (large p);
  * standardized residuals are recomputed independently;
  * config factors override; too-few-columns honest skip; sparse-cell flag.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="loglinear", method="Log-linear model (G-squared test of independence)",
        domain="statistics", family="categorical", goal="explain",
        preconditions=Precondition(min_categorical_cols=2, min_rows=4),
    )


def _g2_ref(tab: np.ndarray) -> tuple[float, float, int]:
    """Independent reference: (G2, pearson_chi2, df) for an r x c count table."""
    O = tab.astype(float)
    n = O.sum()
    E = O.sum(1, keepdims=True) @ O.sum(0, keepdims=True) / n
    g2 = 2.0 * np.sum(np.where(O > 0, O * np.log(np.where(O > 0, O / E, 1.0)), 0.0))
    pear = np.sum((O - E) ** 2 / E)
    df = (O.shape[0] - 1) * (O.shape[1] - 1)
    return float(g2), float(pear), int(df)


def _df_from_table(tab: np.ndarray, rlabels, clabels) -> pd.DataFrame:
    """Expand a count table into long-form rows so crosstab rebuilds it exactly."""
    rows = []
    for i, rl in enumerate(rlabels):
        for j, cl in enumerate(clabels):
            rows += [{"row": rl, "col": cl}] * int(tab[i, j])
    return pd.DataFrame(rows)


def test_g2_matches_reference_and_df(tmp_path: Path) -> None:
    tab = np.array([[20, 30, 10], [25, 10, 35]])
    rl, cl = ["A", "B"], ["x", "y", "z"]
    df = _df_from_table(tab, rl, cl)
    csv = tmp_path / "t.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"factors": ["row", "col"]})
    g2_ref, pear_ref, df_ref = _g2_ref(tab)
    assert abs(res.estimates["g2"] - g2_ref) < 1e-2
    assert abs(res.estimates["pearson_chi2"] - pear_ref) < 1e-2
    assert res.estimates["df"] == float(df_ref) == 2.0
    assert res.estimates["n"] == float(tab.sum())
    out = Path(res.output_dir)
    assert (out / "loglinear_cells.csv").exists()
    assert (out / "loglinear_observed.csv").exists()


def test_strong_association_rejects_independence(tmp_path: Path) -> None:
    # near-diagonal table: strong dependence -> tiny p.
    tab = np.array([[80, 5, 5], [5, 80, 5], [5, 5, 80]])
    rl, cl = ["r0", "r1", "r2"], ["c0", "c1", "c2"]
    df = _df_from_table(tab, rl, cl)
    csv = tmp_path / "assoc.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"factors": ["row", "col"]})
    assert res.estimates["p_value"] < 0.001
    assert "不独立" in res.summary


def test_independent_table_does_not_reject(tmp_path: Path) -> None:
    # product (independent) table: G2 ~ 0, large p.
    row_m = np.array([100.0, 200.0, 300.0])
    col_m = np.array([0.2, 0.3, 0.5])
    tab = np.round(np.outer(row_m, col_m)).astype(int)
    rl, cl = ["r0", "r1", "r2"], ["c0", "c1", "c2"]
    df = _df_from_table(tab, rl, cl)
    csv = tmp_path / "indep.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"factors": ["row", "col"]})
    assert res.estimates["p_value"] > 0.20
    assert res.estimates["g2"] < 1.0


def test_standardized_residuals_recompute(tmp_path: Path) -> None:
    tab = np.array([[40, 10], [10, 40]])
    rl, cl = ["A", "B"], ["x", "y"]
    df = _df_from_table(tab, rl, cl)
    csv = tmp_path / "res.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"factors": ["row", "col"]})
    cells = pd.read_csv(Path(res.output_dir) / "loglinear_cells.csv")
    O = tab.astype(float)
    n = O.sum()
    E = O.sum(1, keepdims=True) @ O.sum(0, keepdims=True) / n
    # ADJUSTED (standardized) Pearson residual ~ N(0,1) under independence
    row_p = O.sum(1, keepdims=True) / n
    col_p = O.sum(0, keepdims=True) / n
    sr = (O - E) / np.sqrt(E * (1 - row_p) * (1 - col_p))
    # match each cell residual to the reference (order-insensitive via the labels)
    got = {(str(r["row"]), str(r["col"])): r["std_resid"] for _, r in cells.iterrows()}
    for i, a in enumerate(rl):
        for j, b in enumerate(cl):
            assert abs(got[(a, b)] - sr[i, j]) < 1e-3


def test_sparse_cells_flagged(tmp_path: Path) -> None:
    # small counts -> some expected < 5 -> sparse flag in summary + estimate.
    tab = np.array([[3, 1, 0], [1, 0, 2], [0, 2, 1]])
    rl, cl = ["r0", "r1", "r2"], ["c0", "c1", "c2"]
    df = _df_from_table(tab, rl, cl)
    csv = tmp_path / "sparse.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"factors": ["row", "col"]})
    assert res.estimates["n_sparse_cells"] > 0
    assert "期望频数<5" in res.summary


def test_too_few_columns_skips(tmp_path: Path) -> None:
    df = pd.DataFrame({"only": ["a", "b", "c"] * 10, "cont": np.arange(30.0)})
    csv = tmp_path / "one.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "g2" not in res.estimates
    assert "跳过" in res.summary
    assert (Path(res.output_dir) / "report.md").exists()
