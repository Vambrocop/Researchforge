"""Tests for the `pearson_correlation` executor branch.

Known structure: x, y built so the population correlation is ~0.8; we assert the
reported Pearson r lands near that, independently recompute it with scipy to pin
correctness, check Spearman/Kendall are also reported, and exercise config + the
honest-degrade paths (too few cols, constant column).
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
    id="pearson_correlation",
    method="Pearson correlation (with Spearman & Kendall)",
    domain="statistics",
    family="statistics",
    goal="explore",
    preconditions={"min_numeric_cols": 2, "min_rows": 3},
)


def _corr_csv(tmp_path: Path, n: int = 200, rho: float = 0.8):
    """x, y with population correlation ~rho via a shared component."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, n)
    y = rho * x + np.sqrt(1 - rho * rho) * noise
    df = pd.DataFrame({"x": x.round(4), "y": y.round(4)})
    csv = tmp_path / "corr.csv"
    df.to_csv(csv, index=False)
    return csv, df


def test_pearson_recovers_known_r_and_matches_scipy(tmp_path):
    csv, df = _corr_csv(tmp_path, n=300, rho=0.8)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "out"))
    out = Path(res.output_dir)

    assert "完成" in res.summary
    # reported r near the designed 0.8
    assert 0.7 < res.estimates["pearson_r"] < 0.9, res.estimates
    # independent recompute with scipy on the same data
    r_ref, p_ref = stats.pearsonr(df["x"].to_numpy(float), df["y"].to_numpy(float))
    assert abs(res.estimates["pearson_r"] - r_ref) < 1e-4, (res.estimates, r_ref)
    assert abs(res.estimates["pearson_p"] - p_ref) < 1e-4
    # all three coefficients + n present
    for k in ("pearson_r", "pearson_p", "ci_low", "ci_high", "spearman_rho", "kendall_tau", "n"):
        assert k in res.estimates, k
    assert res.estimates["n"] == 300.0
    # CI brackets the point estimate
    assert res.estimates["ci_low"] < res.estimates["pearson_r"] < res.estimates["ci_high"]
    # CSV with the three coefficients
    tab = pd.read_csv(out / "correlation_coefficients.csv")
    assert set(tab["coefficient"]) == {"pearson_r", "spearman_rho", "kendall_tau"}
    assert "correlation_coefficients.csv" in res.files


def test_config_x_y(tmp_path):
    rng = np.random.default_rng(1)
    n = 150
    a = rng.normal(0, 1, n)
    df = pd.DataFrame({
        "a": a.round(4),
        "b": rng.normal(0, 1, n).round(4),       # noise
        "c": (-0.9 * a + rng.normal(0, 0.3, n)).round(4),  # strongly anti-correlated with a
    })
    csv = tmp_path / "abc.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "out"), config={"x": "a", "y": "c"})
    assert res.estimates["pearson_r"] < -0.7, res.estimates
    r_ref, _ = stats.pearsonr(df["a"].to_numpy(float), df["c"].to_numpy(float))
    assert abs(res.estimates["pearson_r"] - r_ref) < 1e-4


def test_degrade_one_numeric_col(tmp_path):
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"only": rng.normal(0, 1, 30).round(4), "label": [f"g{i % 3}" for i in range(30)]})
    csv = tmp_path / "one.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "out"))
    assert "跳过" in res.summary
    assert "pearson_r" not in res.estimates


def test_degrade_constant_column(tmp_path):
    n = 40
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"x": rng.normal(0, 1, n).round(4), "y": np.full(n, 5.0)})
    csv = tmp_path / "const.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "out"))
    assert "跳过" in res.summary
    assert "pearson_r" not in res.estimates
