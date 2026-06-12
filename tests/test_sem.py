"""Tests for sem: continuous-indicator gate + single-factor CFA fit (semopy)."""

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
        id="sem",
        method="Structural equation modeling (single-factor CFA template)",
        domain="statistics",
        family="sem",
        goal="explain",
        preconditions=Precondition(min_continuous=3, min_rows=50),
    )


def test_sem_single_factor_fit(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 300
    latent = rng.normal(0, 1, n)
    # four indicators driven by one latent factor -> single-factor CFA fits well
    df = pd.DataFrame({f"v{i}": 0.8 * latent + rng.normal(0, 0.5, n) for i in range(1, 5)})
    csv = tmp_path / "sem.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    assert (out / "loadings.csv").exists()
    assert (out / "fit_indices.csv").exists()
    load = pd.read_csv(out / "loadings.csv")
    assert len(load) == 4  # one loading per indicator
    # data is genuinely single-factor -> good fit
    assert res.estimates["cfi"] > 0.9
    assert res.estimates["rmsea"] < 0.1


def test_sem_precondition_unmet(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(0, 1, 80), "y": rng.normal(0, 1, 80)})  # only 2 continuous
    csv = tmp_path / "two.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)

    assert not ok
    assert any("连续" in u for u in unmet)
