"""Tests for multinomial_logit: multi-class gate + MNLogit fit (statsmodels)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset
from researchforge.recommender.match import check_preconditions


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="multinomial_logit",
        method="Multinomial logistic regression",
        domain="statistics",
        family="statistics",
        goal="explain",
        preconditions=Precondition(requires_ordinal=True, min_rows=40),
    )


def test_multinomial_logit_executor(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 300
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    lin = np.column_stack([np.zeros(n), 1.0 * x1, 1.0 * x2])
    p = np.exp(lin)
    p /= p.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(3, p=p[i]) for i in range(n)])
    # outcome as a 3-level integer column (count kind) + 2 continuous predictors
    df = pd.DataFrame({"choice": y, "x1": np.round(x1, 3), "x2": np.round(x2, 3)})
    csv = tmp_path / "mnl.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)
    tab = pd.read_csv(out / "coefficients.csv")

    assert {"class_vs_baseline", "term", "coef", "RRR", "p_value"}.issubset(tab.columns)
    assert res.estimates["n_classes"] == 3
    assert 0.0 <= res.estimates["accuracy"] <= 1.0
    # better than the ~33% random-guess baseline for 3 balanced classes
    assert res.estimates["accuracy"] > 0.4


def test_multinomial_logit_precondition_unmet(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"y": rng.normal(0, 1, 50), "x": rng.normal(0, 1, 50)})  # no multi-class col
    csv = tmp_path / "cont.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)

    assert not ok
