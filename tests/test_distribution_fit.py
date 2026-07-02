"""Tests for distribution_fit — MLE-fit candidate distributions, rank by AIC/BIC/KS.

Cross-checks:
  * normal data -> norm wins (lowest AIC) and the estimates keys are present;
  * exponential data -> expon wins;
  * data with non-positive values -> positive-support dists skipped (only norm fits),
    and the skip is disclosed in the summary;
  * config column override picks the named column;
  * AIC/logL recomputed independently for the best dist matches the engine;
  * too-few-rows honest skip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog import AnalysisEntry
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_ENTRY = AnalysisEntry(
    id="distribution_fit",
    method="Distribution fitting by maximum likelihood (AIC/BIC/KS ranking)",
    domain="statistics",
    family="distribution",
    goal="describe",
    preconditions={"min_numeric_cols": 1, "min_rows": 8},
)


def _run(df: pd.DataFrame, tmp_path: Path, config=None):
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    return run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"), config=config)


def test_normal_data_norm_wins(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(50, 8, 500)})
    res = _run(df, tmp_path)
    # estimates contract
    for k in ("best_aic", "best_bic", "best_ks", "delta_aic_second", "n"):
        assert k in res.estimates
    assert res.estimates["n"] == 500.0
    # best distribution name lives in the summary (estimates are floats only)
    assert "norm" in res.summary
    out = Path(res.output_dir)
    assert (out / "distribution_fit.csv").exists()
    # best AIC should equal the smallest AIC in the CSV table
    tbl = pd.read_csv(out / "distribution_fit.csv")
    assert abs(res.estimates["best_aic"] - tbl["AIC"].min()) < 1e-3
    assert tbl.sort_values("AIC").iloc[0]["dist"] == "norm"


def test_exponential_data_expon_wins(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.exponential(3.0, 600)})
    res = _run(df, tmp_path)
    out = Path(res.output_dir)
    tbl = pd.read_csv(out / "distribution_fit.csv")
    # expon (or its nesting parents) should be near the top; check expon present + best is positive-support
    assert "expon" in set(tbl["dist"])
    best = tbl.sort_values("AIC").iloc[0]["dist"]
    assert best in {"expon", "gamma", "weibull_min", "lognorm"}


def test_nonpositive_skips_positive_support(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"x": rng.normal(0, 5, 300)})  # has negatives
    res = _run(df, tmp_path)
    out = Path(res.output_dir)
    tbl = pd.read_csv(out / "distribution_fit.csv")
    fitted = set(tbl["dist"])
    # positive-only families must be absent
    assert fitted.isdisjoint({"lognorm", "gamma", "weibull_min", "expon"})
    assert "norm" in fitted
    assert "跳过" in res.summary  # skip disclosed


def test_independent_aic_recompute(tmp_path: Path) -> None:
    from scipy import stats

    rng = np.random.default_rng(5)
    x = rng.normal(10, 2, 400)
    df = pd.DataFrame({"x": x})
    res = _run(df, tmp_path)
    # independent best-dist AIC = norm here
    params = stats.norm.fit(x)
    logL = float(np.sum(stats.norm.logpdf(x, *params)))
    aic_ref = 2 * len(params) - 2 * logL
    assert abs(res.estimates["best_aic"] - aic_ref) < 1e-2


def test_config_column_override(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "junk": rng.normal(0, 1, 200),
        "target": rng.normal(100, 3, 200),
    })
    res = _run(df, tmp_path, config={"column": "target"})
    assert "target" in res.summary
    out = Path(res.output_dir)
    assert (out / "distribution_fit.csv").exists()


def test_too_few_rows_skips(tmp_path: Path) -> None:
    df = pd.DataFrame({"x": [1.5, 2.5, 3.5, 4.5, 5.5]})  # n=5 < 8
    res = _run(df, tmp_path)
    assert "best_aic" not in res.estimates
    assert "跳过" in res.summary
    assert (Path(res.output_dir) / "report.md").exists()


def test_free_loc_disclosure_present_for_positive_support_data(tmp_path: Path) -> None:
    """Disclosure regression: when lognorm/gamma/weibull_min are actually fit (positive
    data), the summary must disclose they use scipy's free-loc (3-parameter shifted)
    form, not the canonical 2-parameter families."""
    rng = np.random.default_rng(6)
    df = pd.DataFrame({"x": rng.exponential(3.0, 400)})  # strictly positive
    res = _run(df, tmp_path)
    assert "自由 loc" in res.summary or "free" in res.summary.lower()
    out = Path(res.output_dir)
    txt = (out / "distribution_fit_summary.txt").read_text(encoding="utf-8")
    assert "自由 loc" in txt


def test_free_loc_disclosure_absent_when_only_norm_fits(tmp_path: Path) -> None:
    # data with negatives -> positive-support families are skipped -> no loc disclosure
    rng = np.random.default_rng(9)
    df = pd.DataFrame({"x": rng.normal(0, 5, 300)})
    res = _run(df, tmp_path)
    assert "自由 loc" not in res.summary
