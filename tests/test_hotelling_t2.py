"""Tests for the `hotelling_t2` executor branch (two-sample Hotelling's T-squared).

Synthetic data with KNOWN structure:
- shifted mean vector  -> reject (p < 0.05).
- identical distributions -> do not reject (p > 0.05).
Numerical cross-checks of the T-squared -> F conversion against:
  (a) an independent hand-computed reference (the load-bearing formula), and
  (b) pingouin.multivariate_ttest if importable (gracefully skipped otherwise).
Plus precondition skips (1 outcome, >2 groups).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog import AnalysisEntry
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_ENTRY = AnalysisEntry(
    id="hotelling_t2",
    method="Hotelling's T-squared (two-sample mean-vector test)",
    domain="statistics",
    family="statistics",
    goal="explain",
    preconditions={"requires_group": True, "min_continuous": 2, "min_rows": 10},
)


# --- independent reference implementation of the load-bearing math ----------
def _reference_t2(g1: np.ndarray, g2: np.ndarray):
    n1, n2 = len(g1), len(g2)
    p = g1.shape[1]
    diff = g1.mean(0) - g2.mean(0)
    S1 = np.cov(g1, rowvar=False, ddof=1)
    S2 = np.cov(g2, rowvar=False, ddof=1)
    Sp = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)
    t2 = (n1 * n2 / (n1 + n2)) * diff @ np.linalg.inv(Sp) @ diff
    df2 = n1 + n2 - p - 1
    f = t2 * df2 / ((n1 + n2 - 2) * p)
    from scipy import stats
    pval = float(stats.f.sf(f, p, df2))
    return float(t2), float(f), p, df2, pval


def _shift_csv(tmp_path: Path, n_per: int = 50, shift: float = 1.5):
    rng = np.random.default_rng(0)
    rows = []
    for g, mu in [("A", 0.0), ("B", shift)]:
        for _ in range(n_per):
            rows.append({"o1": rng.normal(mu, 1.0),
                         "o2": rng.normal(mu, 1.0),
                         "o3": rng.normal(0.0, 1.0),
                         "grp": g})
    csv = tmp_path / "shift.csv"
    df = pd.DataFrame(rows)
    df.to_csv(csv, index=False)
    return csv, df


def _null_csv(tmp_path: Path, n_per: int = 50):
    rng = np.random.default_rng(99)
    rows = []
    for g in ["A", "B"]:
        for _ in range(n_per):
            rows.append({"o1": rng.normal(0, 1), "o2": rng.normal(0, 1),
                         "o3": rng.normal(0, 1), "grp": g})
    csv = tmp_path / "null.csv"
    df = pd.DataFrame(rows)
    df.to_csv(csv, index=False)
    return csv, df


def test_hotelling_shifted_rejects_and_matches_reference(tmp_path):
    csv, df = _shift_csv(tmp_path)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "out"),
                       config={"outcomes": ["o1", "o2", "o3"], "group": "grp"})
    out = Path(res.output_dir)

    assert (out / "hotelling_test.csv").exists()
    assert (out / "hotelling_mean_diffs.csv").exists()
    # Shifted mean vector -> reject equality of mean vectors.
    assert res.estimates["p_value"] < 0.05, res.estimates

    # --- cross-check against an independent reference ---
    levels = df["grp"].unique().tolist()
    g1 = df.loc[df["grp"] == levels[0], ["o1", "o2", "o3"]].values
    g2 = df.loc[df["grp"] == levels[1], ["o1", "o2", "o3"]].values
    t2_ref, f_ref, df1_ref, df2_ref, p_ref = _reference_t2(g1, g2)

    assert res.estimates["T2"] == pytest.approx(t2_ref, rel=1e-5), (res.estimates["T2"], t2_ref)
    assert res.estimates["F"] == pytest.approx(f_ref, rel=1e-5), (res.estimates["F"], f_ref)
    assert res.estimates["df1"] == df1_ref
    assert res.estimates["df2"] == df2_ref
    assert res.estimates["p_value"] == pytest.approx(p_ref, rel=1e-4, abs=1e-6)


def test_hotelling_null_does_not_reject(tmp_path):
    csv, df = _null_csv(tmp_path)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "out"),
                       config={"outcomes": ["o1", "o2", "o3"], "group": "grp"})
    assert res.estimates["p_value"] > 0.05, res.estimates


@pytest.mark.skipif(
    importlib.util.find_spec("pingouin") is None,
    reason="pingouin not installed — reference cross-check skipped",
)
def test_hotelling_matches_pingouin(tmp_path):
    csv, df = _shift_csv(tmp_path)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "out"),
                       config={"outcomes": ["o1", "o2", "o3"], "group": "grp"})
    import pingouin as pg
    levels = df["grp"].unique().tolist()
    X = df.loc[df["grp"] == levels[0], ["o1", "o2", "o3"]]
    Y = df.loc[df["grp"] == levels[1], ["o1", "o2", "o3"]]
    pg_res = pg.multivariate_ttest(X, Y)
    # pingouin reports T2, F, df1, df2, pval
    assert res.estimates["T2"] == pytest.approx(float(pg_res["T2"].iloc[0]), rel=1e-4)
    assert res.estimates["F"] == pytest.approx(float(pg_res["F"].iloc[0]), rel=1e-4)
    assert res.estimates["p_value"] == pytest.approx(float(pg_res["pval"].iloc[0]), rel=1e-3, abs=1e-6)


def test_hotelling_one_outcome_skips(tmp_path):
    rng = np.random.default_rng(1)
    n = 40
    df = pd.DataFrame({"o1": rng.normal(0, 1, n), "grp": ["A", "B"] * (n // 2)})
    csv = tmp_path / "one_out.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "out"))
    assert "T2" not in res.estimates
    assert "跳过" in res.summary


def test_hotelling_three_groups_skips(tmp_path):
    rng = np.random.default_rng(2)
    rows = []
    for g in ["A", "B", "C"]:
        for _ in range(20):
            rows.append({"o1": rng.normal(0, 1), "o2": rng.normal(0, 1), "grp": g})
    df = pd.DataFrame(rows)
    csv = tmp_path / "three.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "out"),
                       config={"outcomes": ["o1", "o2"], "group": "grp"})
    assert "T2" not in res.estimates
    assert "跳过" in res.summary
    assert "MANOVA" in res.summary
