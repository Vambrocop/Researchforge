"""Tests for influence_diagnostics: leverage / Cook's D / DFFITS.

Known-structure check: a clean linear dataset gets ONE planted high-leverage,
off-the-line point (extreme x, off-trend y). That point must dominate Cook's D
(it is the max, and is flagged), while a clean dataset flags very few points.
Plus config override and the no-predictor degrade.
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
        id="influence_diagnostics",
        method="Influence diagnostics (leverage / Cook's D / DFFITS)",
        domain="statistics",
        family="regression",
        goal="explain",
        preconditions=Precondition(min_continuous=1, min_numeric_cols=2, min_rows=12),
    )


def test_influence_detects_planted_point(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 100
    x = rng.normal(0, 1, n)
    y = 2.0 * x + rng.normal(0, 0.3, n)
    # plant an extreme high-leverage, off-the-line observation at the LAST index
    x[-1] = 12.0          # far out in x -> high leverage
    y[-1] = -25.0         # far off the y = 2x trend -> large residual -> big Cook's D
    df = pd.DataFrame({"y": y, "x": x})  # y first -> outcome
    csv = tmp_path / "infl.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    tab = pd.read_csv(out / "influence.csv")
    assert len(tab) == n
    planted = tab.iloc[-1]
    # the planted point dominates Cook's D and is flagged
    assert float(planted["cooks_d"]) == float(tab["cooks_d"].max())
    assert bool(planted["flag"]) is True
    assert res.estimates["n_influential_cooks"] >= 1
    assert res.estimates["n_high_leverage"] >= 1
    assert res.estimates["max_cooks_d"] > 0
    assert res.estimates["n"] == float(n)
    assert (out / "influence_plot.png").exists()


def test_influence_clean_few_flags(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n = 200
    x = rng.normal(0, 1, n)
    y = 1.5 * x + rng.normal(0, 0.5, n)  # no outliers
    df = pd.DataFrame({"y": y, "x": x})
    csv = tmp_path / "clean.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)
    tab = pd.read_csv(out / "influence.csv")
    # clean data: only a small fraction flagged (the 4/n rule alone flags a few by chance)
    assert int(tab["flag"].sum()) < 0.15 * n


def test_influence_config_override(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    n = 80
    x = rng.normal(0, 1, n)
    target = 3.0 * x + rng.normal(0, 0.3, n)
    target[-1] = 30.0
    x[-1] = 8.0
    df = pd.DataFrame({"other": rng.normal(0, 1, n), "target": target, "x": x})
    csv = tmp_path / "ovr.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"outcome": "target", "predictors": ["x"]},
    )
    out = Path(res.output_dir)
    tab = pd.read_csv(out / "influence.csv")
    assert len(tab) == n
    assert res.estimates["n_influential_cooks"] >= 1


def test_influence_degrade_no_predictor(tmp_path: Path) -> None:
    df = pd.DataFrame({"y": np.arange(30, dtype=float)})
    csv = tmp_path / "one.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
