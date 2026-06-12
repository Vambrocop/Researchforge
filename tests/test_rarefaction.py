"""Tests for rarefaction: count-matrix gate + Hurlbert expected-richness curves."""

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
        id="rarefaction",
        method="Rarefaction curves (Hurlbert)",
        domain="microbiology",
        family="ecology",
        goal="describe",
        preconditions=Precondition(min_count_cols=2, min_rows=1),
    )


def test_rarefaction_executor(tmp_path: Path) -> None:
    # 8 sites x 4 OTUs; repeated values so columns read as `count`, not `id`.
    df = pd.DataFrame(
        {
            "otu0": [50, 40, 0, 10, 30, 50, 0, 10],
            "otu1": [30, 20, 25, 5, 10, 30, 25, 5],
            "otu2": [15, 0, 25, 20, 40, 15, 25, 20],
            "otu3": [5, 10, 0, 15, 20, 5, 0, 15],
        }
    )
    csv = tmp_path / "abund.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    assert sum(1 for c in fp.columns if c.kind == "count") >= 2

    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)
    tab = pd.read_csv(out / "rarefaction.csv")

    assert list(tab.columns) == ["site", "depth", "expected_richness"]
    # curve is monotonic non-decreasing within each site
    for s in tab["site"].unique():
        vals = tab[tab["site"] == s].sort_values("depth")["expected_richness"].to_numpy()
        assert np.all(np.diff(vals) >= -1e-6)
    # at full depth, E[S] -> observed richness; site 0 has all 4 OTUs present
    site0 = tab[tab["site"] == 0].sort_values("depth")
    assert abs(site0["expected_richness"].iloc[-1] - 4.0) < 1e-6
    assert res.estimates["n_sites"] == 8


def test_rarefaction_precondition_unmet(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(0, 1, 20), "y": rng.normal(0, 1, 20)})  # no count cols
    csv = tmp_path / "cont.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)

    assert not ok
    assert any("计数列" in u for u in unmet)
