"""Tests for the `correlation_matrix` executor branch.

Known structure: four numeric columns where exactly ONE pair (a, b) is strongly
correlated and the rest are noise. We assert the strongest pair is (a, b), the
matrix is square/symmetric, BH-FDR p-values are reported and >= raw p, the FDR
significant-pair count is sensible, config method/columns work, and degrade paths.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from researchforge.catalog import AnalysisEntry
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_ENTRY = AnalysisEntry(
    id="correlation_matrix",
    method="Correlation matrix (pairwise p + BH-FDR)",
    domain="statistics",
    family="statistics",
    goal="explore",
    preconditions={"min_numeric_cols": 2, "min_rows": 3},
)


def _matrix_csv(tmp_path: Path, n: int = 300):
    """a, b strongly correlated (~0.85); c, d are independent noise."""
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, n)
    b = 0.85 * a + np.sqrt(1 - 0.85**2) * rng.normal(0, 1, n)
    c = rng.normal(0, 1, n)
    dcol = rng.normal(0, 1, n)
    df = pd.DataFrame({"a": a.round(4), "b": b.round(4), "c": c.round(4), "d": dcol.round(4)})
    csv = tmp_path / "mat.csv"
    df.to_csv(csv, index=False)
    return csv, df


def test_strongest_pair_and_fdr(tmp_path):
    csv, df = _matrix_csv(tmp_path, n=400)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "out"))
    out = Path(res.output_dir)

    assert "完成" in res.summary
    assert res.estimates["n_vars"] == 4.0
    assert res.estimates["n"] == 400.0
    # strongest pair magnitude near the designed 0.85
    assert 0.75 < res.estimates["max_abs_corr"] < 0.92, res.estimates

    # square symmetric matrix
    corr = pd.read_csv(out / "correlation_matrix.csv", index_col=0)
    assert corr.shape == (4, 4)
    assert np.allclose(corr.to_numpy(float), corr.to_numpy(float).T, atol=1e-9)
    assert np.allclose(np.diag(corr.to_numpy(float)), 1.0)

    # long table: strongest pair is (a, b); p_fdr present and >= raw p
    long = pd.read_csv(out / "pairwise_correlations.csv")
    assert {"var1", "var2", "r", "abs_r", "p", "p_fdr", "sig_fdr"} <= set(long.columns)
    top = long.sort_values("abs_r", ascending=False).iloc[0]
    assert {top["var1"], top["var2"]} == {"a", "b"}, long
    assert (long["p_fdr"] >= long["p"] - 1e-9).all(), "BH-FDR p must be >= raw p"

    # independent recompute of the (a,b) r
    r_ref, _ = stats.pearsonr(df["a"].to_numpy(float), df["b"].to_numpy(float))
    assert abs(top["r"] - r_ref) < 1e-4, (top["r"], r_ref)

    # the strong a-b pair survives FDR; n_sig_pairs_fdr counts the sig flags
    ab = long[(long["var1"].isin(["a", "b"])) & (long["var2"].isin(["a", "b"]))].iloc[0]
    assert bool(ab["sig_fdr"]) is True
    assert res.estimates["n_sig_pairs_fdr"] == float(long["sig_fdr"].sum())


def test_config_spearman_and_columns(tmp_path):
    csv, df = _matrix_csv(tmp_path, n=300)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "out"),
                       config={"method": "spearman", "columns": ["a", "b", "c"]})
    assert "spearman" in res.summary.lower()
    assert res.estimates["n_vars"] == 3.0
    corr = pd.read_csv(Path(res.output_dir) / "correlation_matrix.csv", index_col=0)
    assert list(corr.columns) == ["a", "b", "c"]


def test_constant_column_dropped(tmp_path):
    csv, df = _matrix_csv(tmp_path, n=200)
    df["const"] = 3.0
    csv2 = tmp_path / "withconst.csv"
    df.to_csv(csv2, index=False)
    fp = profile_dataset(csv2)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "out"))
    # const dropped -> matrix is still the 4 real columns
    corr = pd.read_csv(Path(res.output_dir) / "correlation_matrix.csv", index_col=0)
    assert "const" not in corr.columns
    assert res.estimates["n_vars"] == 4.0


def test_degrade_one_numeric_col(tmp_path):
    rng = np.random.default_rng(9)
    df = pd.DataFrame({"only": rng.normal(0, 1, 30).round(4), "g": [f"x{i % 4}" for i in range(30)]})
    csv = tmp_path / "one.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "out"))
    assert "跳过" in res.summary
    assert "max_abs_corr" not in res.estimates
