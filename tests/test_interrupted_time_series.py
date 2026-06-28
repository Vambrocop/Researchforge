"""Tests for interrupted_time_series: segmented regression recovers the level + slope
change at a configured intervention; honest skip / placeholder handling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="interrupted_time_series", method="Interrupted time series", domain="epidemiology",
        family="causal", goal="explain", preconditions=Precondition(min_continuous=1, min_rows=8),
    )


def _its_data(seed: int = 0, n: int = 120, T0: int = 60,
              b0=10.0, b1=0.1, b2=5.0, b3=0.15):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    D = (t >= T0).astype(float)
    ts = (t - T0) * D
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = 0.5 * e[i - 1] + rng.normal(0, 1)   # AR(1) errors → exercise HAC
    y = b0 + b1 * t + b2 * D + b3 * ts + e
    return pd.DataFrame({"year": t, "value": y})


def test_its_recovers_level_and_slope_change(tmp_path: Path) -> None:
    csv = tmp_path / "its.csv"
    _its_data(b2=5.0, b3=0.15, T0=60).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "value", "time": "year", "intervention": 60})
    out = Path(res.output_dir)
    assert (out / "its_segmented.png").exists()
    e = res.estimates
    # level change ~5, slope change ~0.15 (generous tolerance; AR(1) noise)
    assert abs(e["level_change"] - 5.0) < 2.0
    assert abs(e["slope_change"] - 0.15) < 0.15
    assert e["level_change_p"] < 0.05 and e["slope_change_p"] < 0.05
    assert e["n_pre"] == 60.0 and e["n_post"] == 60.0
    assert "durbin_watson" in e and "post_slope" in e


def test_its_placeholder_intervention_flagged(tmp_path: Path) -> None:
    csv = tmp_path / "its.csv"
    _its_data().to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "value", "time": "year"})  # no intervention
    # runs (midpoint placeholder) but flags it prominently
    assert "level_change" in res.estimates
    assert "占位" in res.summary and "intervention" in res.summary


def test_its_skips_too_few_points(tmp_path: Path) -> None:
    df = pd.DataFrame({"value": [1.0, 2, 3, 4, 5]})  # n=5 < 8
    csv = tmp_path / "x.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "value"})
    assert "level_change" not in res.estimates
    assert "中断时间序列跳过" in res.summary
