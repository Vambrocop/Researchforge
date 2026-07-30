"""Tests for `monotonic_constraints` (HistGradientBoosting + per-feature monotone, Wave S).

Planted structure: y increases in x0, decreases in x1. Config directions must be honored
and the fitted partial dependence must actually be monotone; auto-inference (no config)
must disclose it is data-inferred; multiclass is declined.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog import Catalog
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_CAT = Catalog.load()


def _run(csv, tmp_path, config=None):
    return run_analysis(profile_dataset(csv), _CAT.by_id("monotonic_constraints"),
                        output_root=str(tmp_path / "o"), config=config)


def _signal(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0, 10, n)
    x1 = rng.uniform(0, 10, n)
    return pd.DataFrame({"y": 2 * x0 - 1.5 * x1 + rng.normal(0, 1, n), "x0": x0, "x1": x1})


def test_catalog_loads():
    e = _CAT.by_id("monotonic_constraints")
    assert e is not None and e.executor_ref == "py::monotonic_constraints"
    assert isinstance(e.biases, list) and len(e.biases) >= 3


def test_config_directions_are_monotone(tmp_path):
    csv = tmp_path / "r.csv"
    _signal().to_csv(csv, index=False)
    res = _run(csv, tmp_path, config={"outcome": "y", "increasing": ["x0"], "decreasing": ["x1"]})
    assert res.estimates["cv_score"] > 0.5
    assert res.estimates["n_increasing"] == 1.0 and res.estimates["n_decreasing"] == 1.0
    tab = pd.read_csv(Path(res.output_dir) / "monotonic_constraints.csv").set_index("feature")
    assert tab.loc["x0", "constraint"] == "increasing"
    assert tab.loc["x1", "constraint"] == "decreasing"
    # the fitted partial dependence is actually monotone in the asked direction
    assert bool(tab.loc["x0", "pdp_monotone_ok"]) and bool(tab.loc["x1", "pdp_monotone_ok"])


def test_auto_infer_discloses_data_inferred(tmp_path):
    rng = np.random.default_rng(1)
    n = 200
    x0 = rng.uniform(0, 10, n)
    x1 = rng.uniform(0, 10, n)
    df = pd.DataFrame({"cls": (x0 - x1 > 0).astype(int), "x0": x0, "x1": x1})
    csv = tmp_path / "c.csv"
    df.to_csv(csv, index=False)
    res = _run(csv, tmp_path, config={"outcome": "cls"})
    assert res.estimates["cv_score"] > 0.6
    assert "自动推断" in res.summary                      # disclosed NOT a domain prior
    tab = pd.read_csv(Path(res.output_dir) / "monotonic_constraints.csv")
    assert tab["source"].astype(str).str.contains("数据推断").any()


def test_multiclass_declined(tmp_path):
    rng = np.random.default_rng(2)
    n = 150
    df = pd.DataFrame({"g": rng.integers(0, 3, n),
                       "x0": rng.uniform(0, 10, n), "x1": rng.uniform(0, 10, n)})
    csv = tmp_path / "m.csv"
    df.to_csv(csv, index=False)
    res = _run(csv, tmp_path, config={"outcome": "g"})
    assert "跳过" in res.summary and "二值" in res.summary
    assert "cv_score" not in res.estimates
