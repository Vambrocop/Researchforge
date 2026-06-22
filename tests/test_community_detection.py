"""Tests for the community_detection (Louvain) executor branch.

Known structure: a graph with planted communities (3 dense cliques, sparse
bridges) -> Louvain should recover ~the right number of communities with high
modularity. Plus honest-degrade and config tests.
"""

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
_HAS_LOUVAIN = importlib.util.find_spec("community") is not None


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="community_detection", method="Community detection", domain="network science",
        family="ml", goal="explore", preconditions=Precondition(requires_edgelist=True, min_rows=3),
    )


def _planted_communities(tmp_path: Path, n_comm: int = 3, size: int = 8, p_in: float = 0.8) -> Path:
    """Edge list with n_comm dense communities + a few sparse cross-community bridges."""
    rng = np.random.default_rng(0)
    comms = [[f"c{c}_{i}" for i in range(size)] for c in range(n_comm)]
    edges = []
    for c in comms:
        for i in range(len(c)):
            for j in range(i + 1, len(c)):
                if rng.uniform() < p_in:
                    edges.append((c[i], c[j]))
    # a few bridges between communities (kept sparse so structure survives)
    alln = [x for c in comms for x in c]
    for _ in range(n_comm):
        a, b = rng.choice(alln, 2, replace=False)
        edges.append((a, b))
    df = pd.DataFrame(edges, columns=["source", "target"])
    csv = tmp_path / "planted.csv"
    df.to_csv(csv, index=False)
    return csv


@pytest.mark.skipif(not (_HAS_NX and _HAS_LOUVAIN), reason="networkx/python-louvain not available")
def test_recovers_planted_communities(tmp_path: Path) -> None:
    fp = profile_dataset(_planted_communities(tmp_path, n_comm=3, size=8))
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "完成" in res.summary
    assert res.estimates["n_nodes"] == 24
    # 3 planted dense communities -> strong modularity, ~3 communities recovered
    assert res.estimates["modularity"] > 0.4
    assert 2 <= res.estimates["n_communities"] <= 5
    # node->community CSV covers every node
    nc = pd.read_csv(Path(res.output_dir) / "node_communities.csv")
    assert len(nc) == 24 and set(nc.columns) == {"node", "community"}
    # community sizes CSV sums to the node count
    cs = pd.read_csv(Path(res.output_dir) / "community_sizes.csv")
    assert cs["size"].sum() == 24
    assert "community_summary.txt" in res.files


@pytest.mark.skipif(not (_HAS_NX and _HAS_LOUVAIN), reason="networkx/python-louvain not available")
def test_reproducible_seeded(tmp_path: Path) -> None:
    fp = profile_dataset(_planted_communities(tmp_path, n_comm=3, size=8))
    a = run_analysis(fp, _entry(), output_root=str(tmp_path / "a"))
    b = run_analysis(fp, _entry(), output_root=str(tmp_path / "b"))
    assert a.estimates["modularity"] == b.estimates["modularity"]
    assert a.estimates["n_communities"] == b.estimates["n_communities"]


@pytest.mark.skipif(not (_HAS_NX and _HAS_LOUVAIN), reason="networkx/python-louvain not available")
def test_config_source_target(tmp_path: Path) -> None:
    # rename columns -> handler must honour config source/target
    csv = _planted_communities(tmp_path, n_comm=2, size=7)
    df = pd.read_csv(csv).rename(columns={"source": "u", "target": "v"})
    csv2 = tmp_path / "renamed.csv"
    df.to_csv(csv2, index=False)
    fp = profile_dataset(csv2)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"source": "u", "target": "v"})
    assert "完成" in res.summary
    assert res.estimates["n_nodes"] == 14


def test_degrade_no_edge_list(tmp_path: Path) -> None:
    # one identifier + one continuous column -> cannot form an edge list -> honest skip
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"node": [f"n{i}" for i in range(10)], "val": rng.normal(0, 1, 10).round(3)})
    csv = tmp_path / "x.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "网络分析失败" in res.summary
    assert "modularity" not in res.estimates


@pytest.mark.skipif(not (_HAS_NX and _HAS_LOUVAIN), reason="networkx/python-louvain not available")
def test_degrade_too_few_nodes(tmp_path: Path) -> None:
    # only 2 nodes -> graph too small -> honest failure (no crash)
    df = pd.DataFrame({"source": ["a"], "target": ["b"]})
    csv = tmp_path / "tiny.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "社团发现失败" in res.summary or "网络分析失败" in res.summary
    assert "modularity" not in res.estimates
