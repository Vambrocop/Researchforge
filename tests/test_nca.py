"""Tests for nca: continuous gate + CE-FDH ceiling effect size (Dul 2016)."""

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
        id="nca",
        method="Necessary Condition Analysis (NCA)",
        domain="social science",
        family="configurational",
        goal="explain",
        preconditions=Precondition(min_continuous=2, min_rows=20),
    )


def test_nca_distinguishes_necessity(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 200
    cond = rng.uniform(0, 1, n)
    outcome = rng.uniform(0, 1, n) * cond  # outcome high REQUIRES cond high -> necessity
    noise = rng.uniform(0, 1, n)  # independent of outcome -> not necessary
    # outcome must be the first continuous column (engine convention)
    df = pd.DataFrame({"outcome": outcome, "condition": cond, "noise": noise})
    csv = tmp_path / "nca.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    assert (out / "nca_effect_sizes.csv").exists()
    # the genuine necessary condition has a large d; the noise does not
    assert res.estimates["condition"] > 0.3
    assert res.estimates["noise"] < 0.15
    assert res.estimates["condition"] > res.estimates["noise"]


def test_nca_precondition_unmet(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"y": rng.normal(0, 1, 30), "g": ["a", "b"] * 15})  # only 1 continuous
    csv = tmp_path / "one.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)

    assert not ok
    assert any("连续" in u for u in unmet)
