"""Tests for kernel_density — Gaussian KDE, Silverman/Scott bandwidths, modes.

Known structure:
  * a well-separated bimodal mixture -> 2 modes detected at ~the planted means;
  * a clean unimodal normal (negative control) -> 1 mode near the mean;
  * estimates contract is satisfied, both rule-of-thumb bandwidths reported;
  * a numeric bandwidth override is honoured;
  * config column override picks the named column;
  * too-few-rows honest skip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_ENTRY = AnalysisEntry(
    id="kernel_density",
    method="Kernel density estimation with bandwidth rules and mode detection",
    domain="statistics",
    family="distribution_extra",
    goal="describe",
    preconditions={"min_numeric_cols": 1, "min_rows": 8},
)


def _run(df: pd.DataFrame, tmp_path: Path, config=None):
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    return run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"), config=config)


def test_bimodal_two_modes(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 1.0, 500)
    b = rng.normal(12.0, 1.0, 500)
    df = pd.DataFrame({"x": np.concatenate([a, b])})
    res = _run(df, tmp_path)

    for k in ("n_modes", "silverman_bw", "scott_bw", "primary_mode", "n"):
        assert k in res.estimates
    assert res.estimates["n"] == 1000.0
    assert res.estimates["n_modes"] == 2.0  # bimodal recovered
    assert res.estimates["silverman_bw"] > 0
    assert res.estimates["scott_bw"] > 0

    out = Path(res.output_dir)
    modes = pd.read_csv(out / "kernel_density_modes.csv")
    assert len(modes) == 2
    locs = np.sort(modes["location"].to_numpy())
    # the two modes recover the planted means within a tolerance
    assert abs(locs[0] - 0.0) < 1.5
    assert abs(locs[1] - 12.0) < 1.5
    grid = pd.read_csv(out / "kernel_density_grid.csv")
    assert set(grid.columns) == {"x", "density"}
    assert (grid["density"] >= 0).all()
    assert "多峰" in res.summary


def test_unimodal_normal_one_mode(tmp_path: Path) -> None:
    # negative control: clean single normal -> exactly one mode near the mean
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(50.0, 5.0, 800)})
    res = _run(df, tmp_path)
    assert res.estimates["n_modes"] == 1.0
    assert abs(res.estimates["primary_mode"] - 50.0) < 2.0
    assert "单峰" in res.summary


def test_bandwidth_override_oversmooths(tmp_path: Path) -> None:
    # a large bandwidth oversmooths a bimodal sample down to a single mode
    rng = np.random.default_rng(2)
    a = rng.normal(0.0, 1.0, 400)
    b = rng.normal(8.0, 1.0, 400)
    df = pd.DataFrame({"x": np.concatenate([a, b])})
    res = _run(df, tmp_path, config={"bandwidth": 8.0})
    assert res.estimates["n_modes"] == 1.0  # oversmoothing hides the second mode


def test_config_column_override(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "junk": rng.normal(0, 1, 400),
        "target": np.concatenate([rng.normal(0, 1, 200), rng.normal(20, 1, 200)]),
    })
    res = _run(df, tmp_path, config={"column": "target"})
    assert "target" in res.summary
    assert res.estimates["n_modes"] == 2.0


def test_too_few_rows_skips(tmp_path: Path) -> None:
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})  # n=5 < 8
    res = _run(df, tmp_path)
    assert "n_modes" not in res.estimates
    assert "跳过" in res.summary
    assert (Path(res.output_dir) / "report.md").exists()
