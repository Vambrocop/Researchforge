"""Tests for the network/graph analysis (networkx) executor branch."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_HAS_NX = importlib.util.find_spec("networkx") is not None


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="network_analysis", method="Network analysis", domain="network science",
        family="ml", goal="explore", preconditions=Precondition(requires_edgelist=True, min_rows=3),
    )


def _edges(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    comms = [[f"n{c}_{i}" for i in range(8)] for c in range(3)]
    edges = []
    for c in comms:
        for i in range(len(c)):
            for j in range(i + 1, len(c)):
                if rng.uniform() < 0.6:
                    edges.append((c[i], c[j]))
    alln = [x for c in comms for x in c]
    for _ in range(5):
        a, b = rng.choice(alln, 2, replace=False)
        edges.append((a, b))
    df = pd.DataFrame(edges, columns=["source", "target"])
    csv = tmp_path / "net.csv"
    df.to_csv(csv, index=False)
    return csv


@pytest.mark.skipif(not _HAS_NX, reason="networkx not available")
def test_network_recovers_communities(tmp_path: Path) -> None:
    fp = profile_dataset(_edges(tmp_path))
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "networkx" in res.summary
    assert res.estimates["n_nodes"] == 24
    # 3 planted communities -> strong modularity, ~3 communities recovered
    assert res.estimates["modularity"] > 0.4
    assert res.estimates["n_communities"] >= 2
    # node centrality file produced
    cent = pd.read_csv(Path(res.output_dir) / "node_centrality.csv")
    assert len(cent) == 24 and "betweenness" in cent.columns


def test_network_needs_two_id_columns(tmp_path: Path) -> None:
    # only one identifier column (val is clearly continuous, not an id) -> honest failure
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"node": [f"n{i}" for i in range(10)], "val": rng.normal(0, 1, 10).round(3)})
    csv = tmp_path / "x.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "网络分析失败" in res.summary
