"""Tests for differential_abundance: count+group gate + CLR/Wilcoxon/FDR."""

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
        id="differential_abundance",
        method="Differential abundance (CLR + Wilcoxon)",
        domain="microbiology",
        family="ecology",
        goal="explain",
        preconditions=Precondition(min_count_cols=2, requires_group=True, min_rows=10),
    )


def test_differential_abundance_finds_enriched_taxon(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    rows = []
    for grp in ("ctrl", "case"):
        for _ in range(18):
            base = rng.integers(8, 20, 6).astype(int)
            if grp == "case":
                base[0] += 40  # otu0 strongly enriched in cases -> should be significant
            rows.append({"group": grp, **{f"otu{i}": int(base[i]) for i in range(6)}})
    df = pd.DataFrame(rows)
    csv = tmp_path / "abund.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    assert sum(1 for c in fp.columns if c.kind == "count") >= 2
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)
    tab = pd.read_csv(out / "differential_abundance.csv")

    assert {"taxon", "p_value", "q_value", "significant"}.issubset(tab.columns)
    assert res.estimates["n_significant"] >= 1
    otu0 = tab[tab["taxon"] == "otu0"].iloc[0]
    assert bool(otu0["significant"])  # the enriched taxon is flagged


def test_differential_abundance_precondition_unmet(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(0, 1, 20), "y": rng.normal(0, 1, 20)})  # no taxa/group
    csv = tmp_path / "cont.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)

    assert not ok
