"""Tests for `bayesian_model_averaging` (BIC-approximation BMA over a linear model space).

Planted structure: y depends on x1 (+2) and x2 (-1.5) only; x3/x4 are noise. BMA must give
PIP ~1 to the true predictors, PIP ~0 to the noise, recover the true model as the top model,
and its model-averaged coefficients must match. Degrades honestly on a non-continuous outcome
or too few predictors.
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
    return run_analysis(profile_dataset(csv), _CAT.by_id("bayesian_model_averaging"),
                        output_root=str(tmp_path / "o"), config=config)


def test_catalog_loads():
    e = _CAT.by_id("bayesian_model_averaging")
    assert e is not None and e.executor_ref == "py::bayesian_model_averaging"
    assert isinstance(e.biases, list) and len(e.biases) >= 3


def test_recovers_true_predictors(tmp_path):
    rng = np.random.default_rng(3)
    n = 120
    x1, x2 = rng.normal(0, 1, n), rng.normal(0, 1, n)
    x3, x4 = rng.normal(0, 1, n), rng.normal(0, 1, n)
    y = 2.0 * x1 - 1.5 * x2 + rng.normal(0, 1, n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3, "x4": x4})
    csv = tmp_path / "b.csv"
    df.to_csv(csv, index=False)
    res = _run(csv, tmp_path, config={"outcome": "y"})

    assert res.estimates["n_models"] == 16.0            # 2^4 predictor subsets
    assert res.estimates["pip_x1"] > 0.9 and res.estimates["pip_x2"] > 0.9
    assert res.estimates["pip_x3"] < 0.35 and res.estimates["pip_x4"] < 0.35

    pip = pd.read_csv(Path(res.output_dir) / "bma_pip.csv").set_index("predictor")
    assert abs(pip.loc["x1", "ma_coef"] - 2.0) < 0.4     # model-averaged coef ~ true
    assert abs(pip.loc["x2", "ma_coef"] + 1.5) < 0.4
    assert (pip.loc["x1", "ma_sd"] > 0) and (pip.loc["x3", "ma_coef"] == pip.loc["x3", "ma_coef"])

    top = pd.read_csv(Path(res.output_dir) / "bma_top_models.csv")
    # the true 2-variable model should be the highest-weight model
    assert set(top.iloc[0]["predictors"].split(" + ")) == {"x1", "x2"}
    assert top.iloc[0]["posterior_weight"] > 0.4


def test_degrades_on_binary_outcome(tmp_path):
    rng = np.random.default_rng(1)
    n = 80
    df = pd.DataFrame({"flag": rng.integers(0, 2, n), "x1": rng.normal(0, 1, n),
                       "x2": rng.normal(0, 1, n)})
    csv = tmp_path / "bin.csv"
    df.to_csv(csv, index=False)
    res = _run(csv, tmp_path, config={"outcome": "flag"})
    # flag is binary -> not continuous -> resolve picks a continuous outcome or skips; either
    # way BMA must not treat the binary column as the continuous response.
    assert "flag" not in res.summary or "跳过" in res.summary


def test_degrades_too_few_predictors(tmp_path):
    rng = np.random.default_rng(2)
    n = 60
    df = pd.DataFrame({"y": rng.normal(0, 1, n), "x1": rng.normal(0, 1, n)})
    csv = tmp_path / "one.csv"
    df.to_csv(csv, index=False)
    res = _run(csv, tmp_path, config={"outcome": "y"})
    assert "跳过" in res.summary and "n_models" not in res.estimates
