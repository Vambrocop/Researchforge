"""Tests for ordinal_association — Goodman-Kruskal gamma, Kendall tau-b, Somers D.

Cross-checks:
  * gamma matches an independent concordant/discordant recompute;
  * tau-b matches scipy.stats.kendalltau on the rank codes;
  * a perfectly monotone table -> gamma = 1; a reversed one -> gamma = -1;
  * gamma magnitude >= |tau-b| (gamma ignores ties); Somers D between them;
  * config var1/var2 override; too-few-columns honest skip.
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
        id="ordinal_association", method="Ordinal association (gamma, Kendall tau-b, Somers D)",
        domain="statistics", family="categorical", goal="describe",
        preconditions=Precondition(min_categorical_cols=2, min_rows=4),
    )


def _cd_ref(tab: np.ndarray) -> tuple[float, float]:
    """Independent reference for concordant/discordant pair counts."""
    A = tab.astype(float)
    r, c = A.shape
    C = D = 0.0
    for i in range(r):
        for j in range(c):
            C += A[i, j] * (A[i + 1:, j + 1:].sum() + A[:i, :j].sum())
            D += A[i, j] * (A[i + 1:, :j].sum() + A[:i, j + 1:].sum())
    return C / 2.0, D / 2.0


def _df_from_table(tab: np.ndarray, xcats, ycats) -> pd.DataFrame:
    rows = []
    for i, x in enumerate(xcats):
        for j, y in enumerate(ycats):
            rows += [{"x": x, "y": y}] * int(tab[i, j])
    return pd.DataFrame(rows)


def test_gamma_and_tau_match_reference(tmp_path: Path) -> None:
    tab = np.array([[20, 10, 5], [8, 25, 12], [3, 9, 30]])
    xc, yc = [1, 2, 3], [1, 2, 3]
    df = _df_from_table(tab, xc, yc)
    csv = tmp_path / "t.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"var1": "x", "var2": "y"})
    C, D = _cd_ref(tab)
    gamma_ref = (C - D) / (C + D)
    assert abs(res.estimates["gamma"] - gamma_ref) < 1e-3
    assert abs(res.estimates["concordant"] - C) < 0.5
    assert abs(res.estimates["discordant"] - D) < 0.5
    # tau-b matches scipy on the rank codes
    tau_ref, p_ref = stats.kendalltau(df["x"].to_numpy(), df["y"].to_numpy())
    assert abs(res.estimates["tau_b"] - tau_ref) < 1e-3
    assert abs(res.estimates["tau_b_p"] - p_ref) < 1e-3
    out = Path(res.output_dir)
    assert (out / "ordinal_table.csv").exists()
    assert (out / "ordinal_measures.csv").exists()


def test_perfect_monotone_gamma_one(tmp_path: Path) -> None:
    # purely diagonal table: no discordant pairs -> gamma = 1.
    tab = np.diag([30, 30, 30]).astype(int)
    df = _df_from_table(tab, [1, 2, 3], [1, 2, 3])
    csv = tmp_path / "mono.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"var1": "x", "var2": "y"})
    assert abs(res.estimates["gamma"] - 1.0) < 1e-9
    assert res.estimates["tau_b"] > 0.5
    assert res.estimates["somers_d_yx"] > 0.5


def test_reversed_monotone_gamma_minus_one(tmp_path: Path) -> None:
    # anti-diagonal: no concordant pairs -> gamma = -1.
    tab = np.array([[0, 0, 30], [0, 30, 0], [30, 0, 0]])
    df = _df_from_table(tab, [1, 2, 3], [1, 2, 3])
    csv = tmp_path / "rev.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"var1": "x", "var2": "y"})
    assert abs(res.estimates["gamma"] - (-1.0)) < 1e-9
    assert res.estimates["tau_b"] < -0.5


def test_gamma_magnitude_ge_tau_and_somers(tmp_path: Path) -> None:
    # gamma ignores ties -> |gamma| >= |tau-b| and |Somers D|.
    tab = np.array([[20, 10, 5], [8, 25, 12], [3, 9, 30]])
    df = _df_from_table(tab, [1, 2, 3], [1, 2, 3])
    csv = tmp_path / "mag.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"var1": "x", "var2": "y"})
    g = abs(res.estimates["gamma"])
    assert g >= abs(res.estimates["tau_b"]) - 1e-9
    assert g >= abs(res.estimates["somers_d_yx"]) - 1e-9
    assert g >= abs(res.estimates["somers_d_xy"]) - 1e-9


def test_too_few_columns_skips(tmp_path: Path) -> None:
    df = pd.DataFrame({"only": [1, 2, 3] * 10, "cont": np.arange(30.0) + 0.5})
    csv = tmp_path / "one.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "gamma" not in res.estimates
    assert "跳过" in res.summary
    assert (Path(res.output_dir) / "report.md").exists()
