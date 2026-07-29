"""Tests for the `naive_bayes` branch (Gaussian Naive Bayes, Wave S).

Small-data classification baseline: strong on separable classes, degrades honestly on a
continuous outcome, and (integration) ranks above data-hungry ensembles on small data.
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
    return run_analysis(profile_dataset(csv), _CAT.by_id("naive_bayes"),
                        output_root=str(tmp_path / "o"), config=config)


def test_catalog_loads():
    e = _CAT.by_id("naive_bayes")
    assert e is not None and e.executor_ref == "py::naive_bayes"
    assert isinstance(e.biases, list) and len(e.biases) >= 3


def test_classifies_separable_classes(tmp_path):
    rng = np.random.default_rng(0)
    n = 80
    x1 = rng.normal(0, 1, n)
    y = (x1 + rng.normal(0, 0.4, n) > 0).astype(int)     # x1 separates the classes
    df = pd.DataFrame({"x1": x1, "x2": rng.normal(0, 1, n), "cls": y})
    csv = tmp_path / "sep.csv"
    df.to_csv(csv, index=False)
    res = _run(csv, tmp_path, config={"outcome": "cls"})
    assert res.estimates["cv_accuracy"] > 0.7             # well above 0.5 chance
    assert res.estimates["n_classes"] == 2.0
    assert (Path(res.output_dir) / "nb_confusion_matrix.csv").exists()
    assert (Path(res.output_dir) / "nb_class_priors.csv").exists()
    assert "完成" in res.summary and "条件独立" in res.summary


def test_multiclass(tmp_path):
    rng = np.random.default_rng(1)
    n = 150
    grp = rng.integers(0, 3, n)
    df = pd.DataFrame({"x1": grp + rng.normal(0, 0.5, n),
                       "x2": rng.normal(0, 1, n), "target": grp})
    csv = tmp_path / "mc.csv"
    df.to_csv(csv, index=False)
    res = _run(csv, tmp_path, config={"outcome": "target"})
    assert res.estimates["n_classes"] == 3.0
    assert res.estimates["cv_accuracy"] > 0.7


def test_degrades_on_continuous_outcome(tmp_path):
    rng = np.random.default_rng(2)
    n = 60
    df = pd.DataFrame({"y": rng.normal(0, 1, n), "x1": rng.normal(0, 1, n),
                       "x2": rng.normal(0, 1, n)})
    csv = tmp_path / "cont.csv"
    df.to_csv(csv, index=False)
    res = _run(csv, tmp_path, config={"outcome": "y"})
    assert "跳过" in res.summary and "分类器" in res.summary
    assert "cv_accuracy" not in res.estimates
