"""Tests for the `partial_correlation` executor branch.

Known structure: a confounder z drives BOTH x and y, so the zero-order r(x,y) is
large but the PARTIAL r(x,y | z) collapses toward zero. We assert that drop, that
the residualisation partial r matches an independent statsmodels-style recompute,
and exercise config + honest-degrade paths.
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
    id="partial_correlation",
    method="Partial correlation (controlling for covariates)",
    domain="statistics",
    family="statistics",
    goal="explore",
    preconditions={"min_numeric_cols": 3, "min_rows": 5},
)


def _confounded_csv(tmp_path: Path, n: int = 300):
    """z is a confounder: x = z + noise, y = z + noise. So x–y is spuriously
    correlated through z; controlling for z should drive the partial r ~0."""
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, n)
    x = z + rng.normal(0, 0.5, n)
    y = z + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"x": x.round(4), "y": y.round(4), "z": z.round(4)})
    csv = tmp_path / "confound.csv"
    df.to_csv(csv, index=False)
    return csv, df


def _partial_r_ref(df, x, y, controls):
    """Independent recompute of partial r via residualisation (the contract)."""
    X = df[x].to_numpy(float)
    Y = df[y].to_numpy(float)
    Z = df[controls].to_numpy(float)
    Zc = np.column_stack([np.ones(len(X)), Z])
    rx = X - Zc @ np.linalg.lstsq(Zc, X, rcond=None)[0]
    ry = Y - Zc @ np.linalg.lstsq(Zc, Y, rcond=None)[0]
    return float(stats.pearsonr(rx, ry)[0])


def test_confounder_collapses_partial(tmp_path):
    csv, df = _confounded_csv(tmp_path, n=400)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "out"),
                       config={"x": "x", "y": "y", "covariates": ["z"]})
    out = Path(res.output_dir)

    assert "完成" in res.summary
    zero = res.estimates["zero_order_r"]
    partial = res.estimates["partial_r"]
    # zero-order is large (shared confounder), partial collapses toward 0
    assert zero > 0.5, res.estimates
    assert abs(partial) < 0.2, res.estimates
    # THE key assertion: controlling for z drops the magnitude substantially
    assert abs(zero) - abs(partial) > 0.3, (zero, partial)
    # confounding is called out in the summary
    assert "混淆" in res.summary

    # independent recompute pins the residualisation result
    ref = _partial_r_ref(df, "x", "y", ["z"])
    assert abs(partial - ref) < 1e-4, (partial, ref)

    for k in ("partial_r", "partial_p", "zero_order_r", "ci_low", "ci_high", "n_controls", "n"):
        assert k in res.estimates, k
    assert res.estimates["n_controls"] == 1.0

    tab = pd.read_csv(out / "partial_vs_zero_order.csv")
    assert set(tab["type"]) == {"zero_order", "partial"}
    assert "partial_vs_zero_order.csv" in res.files


def test_multiple_covariates(tmp_path):
    rng = np.random.default_rng(5)
    n = 300
    z1 = rng.normal(0, 1, n)
    z2 = rng.normal(0, 1, n)
    x = z1 + 0.5 * z2 + rng.normal(0, 0.4, n)
    y = z1 + 0.5 * z2 + rng.normal(0, 0.4, n)
    df = pd.DataFrame({"x": x.round(4), "y": y.round(4), "z1": z1.round(4), "z2": z2.round(4)})
    csv = tmp_path / "multi.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "out"),
                       config={"x": "x", "y": "y", "covariates": ["z1", "z2"]})
    assert res.estimates["n_controls"] == 2.0
    ref = _partial_r_ref(df, "x", "y", ["z1", "z2"])
    assert abs(res.estimates["partial_r"] - ref) < 1e-4


def test_degrade_no_covariates(tmp_path):
    rng = np.random.default_rng(6)
    n = 50
    x = rng.normal(0, 1, n)
    df = pd.DataFrame({"x": x.round(4), "y": (x + rng.normal(0, 0.5, n)).round(4)})
    csv = tmp_path / "twocol.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    # only x,y available; with config covariates pointing at nonexistent col -> skip
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "out"),
                       config={"x": "x", "y": "y", "covariates": ["nope"]})
    assert "跳过" in res.summary
    assert "partial_r" not in res.estimates


def test_degrade_too_few_numeric(tmp_path):
    rng = np.random.default_rng(7)
    df = pd.DataFrame({"only": rng.normal(0, 1, 20).round(4), "g": [f"c{i % 2}" for i in range(20)]})
    csv = tmp_path / "one.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "out"))
    assert "跳过" in res.summary
    assert "partial_r" not in res.estimates
