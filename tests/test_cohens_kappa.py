"""Tests for cohens_kappa — Cohen (1960) kappa + Cohen (1968) weighted kappa.

Known-value cross-checks:
  * perfect agreement -> kappa = 1;
  * a hand-computable 2x2 table whose kappa is recomputed independently here;
  * pure-chance ratings -> kappa ~ 0;
  * quadratic weighted kappa on an ordinal table vs an independent recompute;
  * the prevalence (kappa-paradox) case is flagged;
  * config rater1/rater2 override; too-few-columns honest skip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog import AnalysisEntry
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_ENTRY = AnalysisEntry(
    id="cohens_kappa",
    method="Cohen's kappa (two-rater agreement)",
    domain="statistics",
    family="agreement",
    goal="describe",
    preconditions={"requires_group": True, "min_rows": 10},
)


def _ck_unweighted(r1: list, r2: list) -> float:
    """Independent reference for unweighted Cohen's kappa."""
    cats = sorted(set(r1) | set(r2), key=str)
    idx = {c: i for i, c in enumerate(cats)}
    q = len(cats)
    n = len(r1)
    cm = np.zeros((q, q))
    for a, b in zip(r1, r2):
        cm[idx[a], idx[b]] += 1
    P = cm / n
    po = np.trace(P)
    pe = (P.sum(1) @ P.sum(0))
    return (po - pe) / (1 - pe)


def _ck_weighted_quad(r1: list, r2: list) -> float:
    """Independent reference for quadratic weighted kappa (numeric ordinal codes)."""
    cats = sorted(set(r1) | set(r2), key=float)
    idx = {c: i for i, c in enumerate(cats)}
    q = len(cats)
    n = len(r1)
    cm = np.zeros((q, q))
    for a, b in zip(r1, r2):
        cm[idx[a], idx[b]] += 1
    P = cm / n
    Pe = np.outer(P.sum(1), P.sum(0))
    ii, jj = np.meshgrid(np.arange(q), np.arange(q), indexing="ij")
    w = (ii - jj) ** 2 / (q - 1) ** 2
    return 1 - (w * P).sum() / (w * Pe).sum()


def test_perfect_agreement(tmp_path: Path) -> None:
    vals = (["a", "b", "c", "a", "b", "c"] * 4)
    df = pd.DataFrame({"r1": vals, "r2": list(vals)})
    csv = tmp_path / "perfect.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))
    assert abs(res.estimates["cohens_kappa"] - 1.0) < 1e-9
    assert abs(res.estimates["observed_agreement"] - 1.0) < 1e-9


def test_known_2x2_table(tmp_path: Path) -> None:
    # Hand-computable 2x2: build raters with a known confusion structure.
    #  a=both yes=20, b=r1 yes r2 no=5, c=r1 no r2 yes=10, d=both no=15  (N=50)
    r1 = ["yes"] * 25 + ["no"] * 25
    r2 = ["yes"] * 20 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 15
    df = pd.DataFrame({"r1": r1, "r2": r2})
    csv = tmp_path / "t.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))
    ref = _ck_unweighted(r1, r2)
    assert abs(res.estimates["cohens_kappa"] - ref) < 1e-6
    # SE + CI present
    assert "se" in res.estimates
    assert res.estimates["ci_low"] < res.estimates["cohens_kappa"] < res.estimates["ci_high"]


def test_chance_agreement_near_zero(tmp_path: Path) -> None:
    # Independent random ratings -> kappa ~ 0.
    rng = np.random.default_rng(7)
    n = 600
    df = pd.DataFrame(
        {
            "r1": rng.integers(0, 3, n),
            "r2": rng.integers(0, 3, n),
        }
    )
    csv = tmp_path / "chance.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))
    assert abs(res.estimates["cohens_kappa"]) < 0.12


def test_quadratic_weighted_kappa_ordinal(tmp_path: Path) -> None:
    # Ordinal 1..4 codes; off-by-one disagreements should give weighted > unweighted.
    rng = np.random.default_rng(3)
    base = rng.integers(1, 5, 120)
    noise = rng.integers(-1, 2, 120)  # +/-1 ordinal drift
    r2 = np.clip(base + noise, 1, 4)
    df = pd.DataFrame({"r1": base, "r2": r2})
    csv = tmp_path / "ord.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _ENTRY, output_root=str(tmp_path / "o"), config={"weights": "quadratic"}
    )
    ref = _ck_weighted_quad(list(base), list(r2))
    assert abs(res.estimates["kappa_quadratic_weighted"] - ref) < 1e-3  # estimate rounded to 4dp
    # quadratic weighting rewards near-misses -> >= unweighted kappa here
    assert res.estimates["kappa_quadratic_weighted"] >= res.estimates["cohens_kappa"] - 1e-9
    assert "kappa_linear_weighted" in res.estimates


def test_prevalence_paradox_flagged(tmp_path: Path) -> None:
    # High observed agreement but skewed prevalence -> low kappa, paradox flagged.
    #  both "yes" = 85, both "no" = 2, r1 yes/r2 no = 7, r1 no/r2 yes = 6 (N=100)
    #  -> p_o = 0.87, kappa ~ 0.165 (the classic kappa paradox).
    r1 = ["yes"] * 92 + ["no"] * 8
    r2 = ["yes"] * 85 + ["no"] * 7 + ["yes"] * 6 + ["no"] * 2
    df = pd.DataFrame({"r1": r1, "r2": r2})
    csv = tmp_path / "para.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))
    assert res.estimates["observed_agreement"] >= 0.85
    assert res.estimates["cohens_kappa"] < 0.40
    assert "κ 悖论" in res.summary  # paradox disclosure present
    assert "prevalence_index" in res.estimates


def test_config_override_and_products(tmp_path: Path) -> None:
    vals = ["a", "b", "c"] * 20
    df = pd.DataFrame(
        {
            "noise": ["x", "y", "z"] * 20,
            "judge_A": vals,
            "judge_B": vals,
        }
    )
    csv = tmp_path / "cfg.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp,
        _ENTRY,
        output_root=str(tmp_path / "o"),
        config={"rater1": "judge_A", "rater2": "judge_B"},
    )
    assert abs(res.estimates["cohens_kappa"] - 1.0) < 1e-9
    out = Path(res.output_dir)
    assert (out / "confusion_matrix.csv").exists()
    assert (out / "kappa_estimates.csv").exists()


def test_too_few_columns_skips(tmp_path: Path) -> None:
    df = pd.DataFrame({"only": ["a", "b", "c"] * 10, "cont": np.arange(30.0)})
    csv = tmp_path / "one.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))
    assert "cohens_kappa" not in res.estimates
    assert "跳过" in res.summary
    assert (Path(res.output_dir) / "report.md").exists()
