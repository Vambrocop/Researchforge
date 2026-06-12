"""Tests for diversity_indices: precondition gate + Shannon/Simpson/richness."""

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
        id="diversity_indices",
        method="Ecological diversity indices",
        domain="ecology",
        family="ecology",
        goal="describe",
        preconditions=Precondition(min_count_cols=2, min_rows=2),
    )


def test_diversity_executor(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({f"sp{i}": rng.integers(0, 8, 12) for i in range(4)})
    csv = tmp_path / "abund.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    assert sum(1 for c in fp.columns if c.kind == "count") >= 2

    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    div = pd.read_csv(Path(res.output_dir) / "diversity.csv")

    assert {"shannon", "simpson", "richness"} <= set(div.columns)
    assert (div["shannon"] >= 0).all()
    assert "mean_shannon" in res.estimates


def test_diversity_known_values(tmp_path: Path) -> None:
    # row 0 = monoculture (shannon 0, richness 1); row 1 = 4 equal species (shannon ln4)
    df = pd.DataFrame(
        {
            "sp0": [10, 5, 3, 3, 1, 1, 2, 2, 4, 4],
            "sp1": [0, 5, 3, 1, 1, 1, 2, 0, 4, 0],
            "sp2": [0, 5, 0, 1, 0, 1, 0, 2, 0, 4],
            "sp3": [0, 5, 0, 0, 1, 0, 2, 0, 4, 0],
        }
    )
    csv = tmp_path / "known.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    div = pd.read_csv(Path(res.output_dir) / "diversity.csv")

    assert abs(div["shannon"].iloc[0] - 0.0) < 1e-6
    assert div["richness"].iloc[0] == 1
    assert abs(div["shannon"].iloc[1] - np.log(4)) < 0.01
    assert div["richness"].iloc[1] == 4


def test_diversity_precondition_unmet(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(0, 1, 30), "y": rng.normal(0, 1, 30)})
    csv = tmp_path / "cont.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)

    assert not ok
    assert any("计数列" in u for u in unmet)
