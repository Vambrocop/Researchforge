"""Tests for the centrality_suite executor branch.

Known structure: a star graph -> the centre node must have the maximum degree,
betweenness, and closeness centrality. All five centralities are computed for
every node. Plus the Spearman agreement matrix, config, and honest-degrade tests.
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


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="centrality_suite", method="Centrality suite", domain="network science",
        family="ml", goal="explore", preconditions=Precondition(requires_edgelist=True, min_rows=3),
    )


def _star(tmp_path: Path, leaves: int = 9) -> Path:
    """Star graph: centre 'hub' connected to `leaves` leaf nodes."""
    edges = [("hub", f"leaf{i}") for i in range(leaves)]
    df = pd.DataFrame(edges, columns=["source", "target"])
    csv = tmp_path / "star.csv"
    df.to_csv(csv, index=False)
    return csv


def _ring(tmp_path: Path, n: int = 10) -> Path:
    """Cycle graph: every node has degree 2 (centralities all symmetric)."""
    edges = [(f"n{i}", f"n{(i + 1) % n}") for i in range(n)]
    df = pd.DataFrame(edges, columns=["source", "target"])
    csv = tmp_path / "ring.csv"
    df.to_csv(csv, index=False)
    return csv


@pytest.mark.skipif(not _HAS_NX, reason="networkx not available")
def test_star_centre_is_most_central(tmp_path: Path) -> None:
    fp = profile_dataset(_star(tmp_path, leaves=9))
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "完成" in res.summary
    assert res.estimates["n_nodes"] == 10
    cent = pd.read_csv(Path(res.output_dir) / "node_centrality.csv")
    assert len(cent) == 10
    # all five centralities computed for every node
    for col in ("degree", "betweenness", "closeness", "eigenvector", "pagerank"):
        assert col in cent.columns
    # the hub is the unique maximiser of degree / betweenness / closeness
    for col in ("degree", "betweenness", "closeness", "pagerank"):
        top = cent.sort_values(col, ascending=False).iloc[0]["node"]
        assert top == "hub", f"{col} top should be hub, got {top}"
    # hub degree centrality == 1.0 (connected to all other nodes)
    hub_deg = cent.loc[cent["node"] == "hub", "degree"].iloc[0]
    assert abs(hub_deg - 1.0) < 1e-6


@pytest.mark.skipif(not _HAS_NX, reason="networkx not available")
def test_spearman_matrix_and_agreement(tmp_path: Path) -> None:
    fp = profile_dataset(_star(tmp_path, leaves=8))
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    spear = pd.read_csv(Path(res.output_dir) / "centrality_spearman.csv", index_col=0)
    assert spear.shape == (5, 5)
    # diagonal is 1 (a measure perfectly agrees with itself)
    for m in spear.index:
        assert abs(spear.loc[m, m] - 1.0) < 1e-6
    assert "mean_spearman_agreement" in res.estimates
    # top_nodes table produced with one row per measure-rank
    top = pd.read_csv(Path(res.output_dir) / "top_nodes.csv")
    assert set(top["measure"].unique()) == {"degree", "betweenness", "closeness", "eigenvector", "pagerank"}


@pytest.mark.skipif(not _HAS_NX, reason="networkx not available")
def test_eigenvector_converges_on_ring(tmp_path: Path) -> None:
    # a symmetric ring is a well-behaved graph -> eigenvector should converge
    fp = profile_dataset(_ring(tmp_path, n=10))
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert res.estimates["eigenvector_converged"] == 1.0
    cent = pd.read_csv(Path(res.output_dir) / "node_centrality.csv")
    # on a ring every node is structurally equivalent -> equal degree centrality
    assert cent["degree"].nunique() == 1


@pytest.mark.skipif(not _HAS_NX, reason="networkx not available")
def test_config_size_by(tmp_path: Path) -> None:
    fp = profile_dataset(_star(tmp_path, leaves=8))
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"size_by": "pagerank"})
    assert "完成" in res.summary
    # plot best-effort; node_centrality.csv must always exist
    assert "node_centrality.csv" in res.files


def test_degrade_no_edge_list(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"node": [f"n{i}" for i in range(10)], "val": rng.normal(0, 1, 10).round(3)})
    csv = tmp_path / "x.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "网络分析失败" in res.summary
    assert "n_nodes" not in res.estimates
