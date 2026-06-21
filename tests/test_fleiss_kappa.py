"""Tests for fleiss_kappa — Fleiss (1971) multi-rater kappa.

Known-value cross-checks:
  * a small, fully hand-computable count matrix whose kappa = 1/3 (worked out
    in test_fleiss_hand_computed below);
  * an independent recompute of P-bar, P-bar-e and kappa from a generated matrix
    (two independent implementations must agree);
  * perfect agreement -> kappa = 1;
  * config raters override (wide layout) and the <3-rater honest skip.

Hand computation for the kappa=1/3 case (4 subjects, 3 raters, 2 categories):
  counts = [[3,0],[0,3],[2,1],[1,2]]  (each row sums to n=3, N=4)
  p_j = colsum/(N n) = [6/12, 6/12] = [0.5, 0.5]
  P_i = (sum n_ij^2 - n)/(n(n-1)):  [1, 1, 1/3, 1/3]
  P_bar = 2/3 ;  P_e = 0.5^2 + 0.5^2 = 0.5
  kappa = (2/3 - 0.5)/(1 - 0.5) = (1/6)/(1/2) = 1/3.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog import AnalysisEntry
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_ENTRY = AnalysisEntry(
    id="fleiss_kappa",
    method="Fleiss' kappa (multi-rater agreement)",
    domain="statistics",
    family="agreement",
    goal="describe",
    preconditions={"requires_group": True, "min_rows": 5},
)


def _fleiss_ref(counts: np.ndarray) -> dict[str, float]:
    """Independent reference implementation of Fleiss (1971) kappa."""
    N, q = counts.shape
    n = int(counts.sum(1)[0])
    p_j = counts.sum(0) / (N * n)
    P_i = (np.square(counts).sum(1) - n) / (n * (n - 1))
    P_bar = P_i.mean()
    P_e = np.square(p_j).sum()
    return {
        "P_bar": float(P_bar),
        "P_e": float(P_e),
        "kappa": float((P_bar - P_e) / (1 - P_e)),
    }


def test_fleiss_hand_computed(tmp_path: Path) -> None:
    # 4 subjects x 3 raters x 2 categories -> kappa = 1/3 (see module docstring).
    counts = np.array([[3, 0], [0, 3], [2, 1], [1, 2]], dtype=float)
    df = pd.DataFrame(counts, columns=["cat0", "cat1"])
    csv = tmp_path / "hand.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _ENTRY, output_root=str(tmp_path / "o"),
        config={"count_matrix": True, "raters": ["cat0", "cat1"]},
    )
    assert abs(res.estimates["fleiss_kappa"] - 1.0 / 3.0) < 1e-6
    assert abs(res.estimates["P_bar_observed"] - 2.0 / 3.0) < 1e-6
    assert abs(res.estimates["P_e_expected"] - 0.5) < 1e-6
    assert res.estimates["n_subjects"] == 4.0
    assert res.estimates["n_raters"] == 3.0
    assert res.estimates["n_categories"] == 2.0


def test_fleiss_matches_independent_recompute(tmp_path: Path) -> None:
    # A larger self-consistent count matrix (every row sums to n=5); the engine
    # must reproduce the independent reference exactly.
    rng = np.random.default_rng(42)
    N, q, n = 40, 4, 5
    counts = np.zeros((N, q), dtype=float)
    for i in range(N):
        # bias most ratings toward a per-subject "true" category, some spread
        truth = rng.integers(0, q)
        for _ in range(n):
            cat = truth if rng.random() < 0.7 else rng.integers(0, q)
            counts[i, cat] += 1
    df = pd.DataFrame(counts, columns=[f"cat{j}" for j in range(q)])
    csv = tmp_path / "gen.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _ENTRY, output_root=str(tmp_path / "o"),
        config={"count_matrix": True, "raters": [f"cat{j}" for j in range(q)]},
    )
    ref = _fleiss_ref(counts)
    assert abs(res.estimates["P_bar_observed"] - ref["P_bar"]) < 1e-4
    assert abs(res.estimates["P_e_expected"] - ref["P_e"]) < 1e-4
    assert abs(res.estimates["fleiss_kappa"] - ref["kappa"]) < 1e-4
    # SE / z-test present and significant for this strongly-agreeing matrix
    assert res.estimates["p_value"] < 0.001


def test_fleiss_perfect_agreement(tmp_path: Path) -> None:
    # Wide layout: 8 raters all give the same code to each of 10 subjects.
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 3, 10)
    data = {f"rater{r}": codes for r in range(8)}
    df = pd.DataFrame(data)
    csv = tmp_path / "perfect.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))
    assert abs(res.estimates["fleiss_kappa"] - 1.0) < 1e-9
    assert abs(res.estimates["P_bar_observed"] - 1.0) < 1e-9


def test_fleiss_wide_layout_config(tmp_path: Path) -> None:
    # 5 raters, categorical labels, with a distractor column excluded via config.
    rng = np.random.default_rng(11)
    n = 25
    labels = np.array(["low", "mid", "high"])
    cols = {}
    truth = rng.integers(0, 3, n)
    for r in range(5):
        flip = rng.random(n) < 0.2
        codes = np.where(flip, rng.integers(0, 3, n), truth)
        cols[f"j{r}"] = labels[codes]
    cols["distractor"] = ["p", "q", "r", "s", "t"] * 5
    df = pd.DataFrame(cols)
    csv = tmp_path / "wide.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _ENTRY, output_root=str(tmp_path / "o"),
        config={"raters": ["j0", "j1", "j2", "j3", "j4"]},
    )
    assert res.estimates["n_raters"] == 5.0
    assert res.estimates["n_subjects"] == 25.0
    assert 0.0 < res.estimates["fleiss_kappa"] <= 1.0
    out = Path(res.output_dir)
    assert (out / "fleiss_category_kappa.csv").exists()
    assert (out / "fleiss_estimates.csv").exists()


def test_fleiss_too_few_raters_skips(tmp_path: Path) -> None:
    df = pd.DataFrame({"r1": ["a", "b"] * 10, "r2": ["a", "b"] * 10})
    csv = tmp_path / "two.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))
    assert "fleiss_kappa" not in res.estimates
    assert "跳过" in res.summary
    assert (Path(res.output_dir) / "report.md").exists()
