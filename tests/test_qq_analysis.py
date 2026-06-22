"""Tests for qq_analysis — Q-Q + P-P diagnostics, Filliben PPCC, reference line.

Cross-checks:
  * normal data vs norm -> PPCC very close to 1, slope ~ std, intercept ~ mean;
  * non-normal (heavy-skew) data vs norm -> PPCC noticeably below 1;
  * PPCC recomputed independently from the CSV columns matches the engine;
  * positive-support dist on data with <=0 honest-skips;
  * config column/dist override; too-few-rows honest skip; products present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog import AnalysisEntry
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_ENTRY = AnalysisEntry(
    id="qq_analysis",
    method="Q-Q and P-P probability-plot diagnostics (Filliben PPCC)",
    domain="statistics",
    family="distribution",
    goal="describe",
    preconditions={"min_numeric_cols": 1, "min_rows": 8},
)


def _run(df: pd.DataFrame, tmp_path: Path, config=None):
    csv = tmp_path / "q.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    return run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"), config=config)


def test_normal_high_ppcc(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(20, 5, 600)})
    res = _run(df, tmp_path)
    for k in ("ppcc", "qq_slope", "qq_intercept", "n"):
        assert k in res.estimates
    assert res.estimates["n"] == 600.0
    assert res.estimates["ppcc"] > 0.99
    # Q-Q is plotted against the FITTED-distribution quantiles (dist.ppf with the
    # estimated mean/sd), so a good fit lies on the 45° line: slope ~ 1, intercept ~ 0.
    assert abs(res.estimates["qq_slope"] - 1.0) < 0.15
    assert abs(res.estimates["qq_intercept"] - 0.0) < 1.0
    out = Path(res.output_dir)
    assert (out / "qq_analysis.csv").exists()


def test_skewed_lower_ppcc(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    # heavy right skew -> Q-Q against normal departs from the line
    df = pd.DataFrame({"x": rng.exponential(3.0, 600)})
    res = _run(df, tmp_path)  # default dist = norm
    assert res.estimates["ppcc"] < 0.99


def test_ppcc_independent_recompute(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"x": rng.normal(0, 1, 300)})
    res = _run(df, tmp_path)
    out = Path(res.output_dir)
    tbl = pd.read_csv(out / "qq_analysis.csv")
    a = tbl["data_sorted"].to_numpy(float)
    b = tbl["theoretical_quantile"].to_numpy(float)
    finite = np.isfinite(a) & np.isfinite(b)
    ref = float(np.corrcoef(a[finite], b[finite])[0, 1])
    assert abs(res.estimates["ppcc"] - ref) < 1e-3


def test_positive_dist_on_negative_skips(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"x": rng.normal(0, 2, 200)})  # has negatives
    res = _run(df, tmp_path, config={"dist": "gamma"})
    assert "跳过" in res.summary
    assert "ppcc" not in res.estimates


def test_config_column_and_dist(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    df = pd.DataFrame({
        "junk": rng.uniform(0, 1, 200),
        "target": rng.exponential(2.0, 200),
    })
    res = _run(df, tmp_path, config={"column": "target", "dist": "expon"})
    assert "target" in res.summary
    assert "expon" in res.summary
    out = Path(res.output_dir)
    assert (out / "qq_analysis.csv").exists()
    # expon fits its own data well -> high PPCC
    assert res.estimates["ppcc"] > 0.97


def test_too_few_rows_skips(tmp_path: Path) -> None:
    df = pd.DataFrame({"x": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]})  # n=7 < 8
    res = _run(df, tmp_path)
    assert "ppcc" not in res.estimates
    assert "跳过" in res.summary
    assert (Path(res.output_dir) / "report.md").exists()
