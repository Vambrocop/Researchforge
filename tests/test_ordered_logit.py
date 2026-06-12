"""Tests for ordered_logit: ordinal-outcome gate + proportional-odds fit."""

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
        id="ordered_logit",
        method="Ordered logistic regression (proportional odds)",
        domain="statistics",
        family="statistics",
        goal="explain",
        preconditions=Precondition(requires_ordinal=True, min_rows=30),
    )


def test_ordered_logit_executor(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 300
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    latent = 1.2 * x1 - 0.7 * x2 + rng.normal(0, 1, n)
    sat = np.digitize(latent, [-1, 0, 1]) + 1  # 4 ordered levels (1..4)
    df = pd.DataFrame({"sat": sat.astype(int), "x1": x1, "x2": x2})
    csv = tmp_path / "likert.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    assert (out / "coefficients.csv").exists()
    assert (out / "summary.txt").exists()
    # recovered slope signs match the data-generating process
    assert res.estimates["x1"] > 0
    assert res.estimates["x2"] < 0


def test_ordered_logit_precondition_unmet(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    # all-continuous data: no 3–10 level ordinal column to serve as outcome
    df = pd.DataFrame({"y": rng.normal(0, 1, 50), "x": rng.normal(0, 1, 50)})
    csv = tmp_path / "cont.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)

    assert not ok
    assert any("有序" in u for u in unmet)
