"""Tests for nmds (non-metric MDS ordination): gate + 2D coords + stress."""

from __future__ import annotations

from pathlib import Path

import math

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset
from researchforge.recommender.match import check_preconditions


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="nmds",
        method="Non-metric multidimensional scaling (NMDS)",
        domain="ecology",
        family="ecology",
        goal="explore",
        preconditions=Precondition(min_count_cols=2, min_rows=4),
    )


def test_nmds_executor(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    # +1 guarantees every site has organisms (no empty rows -> Bray-Curtis defined)
    df = pd.DataFrame({f"sp{i}": rng.integers(0, 5, 12) + 1 for i in range(4)})
    csv = tmp_path / "abund.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    assert sum(1 for c in fp.columns if c.kind == "count") >= 2

    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    coords = pd.read_csv(Path(res.output_dir) / "nmds_coords.csv", index_col=0)

    assert list(coords.columns) == ["NMDS1", "NMDS2"]
    assert "stress" in res.estimates
    assert math.isfinite(res.estimates["stress"])
    assert res.estimates["stress"] >= 0
    # disclosure: which stress flavor (normalized Kruskal Stress-1 vs raw) is reported
    assert "stress" in res.summary.lower()
    assert (
        ("normalized" in res.summary)
        or ("raw" in res.summary)
        or ("un-normalized" in res.summary)
    ), f"summary={res.summary}"


def test_nmds_precondition_unmet(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(0, 1, 20), "y": rng.normal(0, 1, 20)})
    csv = tmp_path / "cont.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)

    assert not ok
    assert any("计数列" in u for u in unmet)
