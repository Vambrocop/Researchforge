"""Tests for the stochastic_block_model executor branch.

Planted 2-block assortative SBM (within-prob >> between-prob) -> ICL should
select ~2 blocks, recover the planted partition, and report within > between
(assortative). Plus forced-K config and honest-degrade tests.
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
_HAS_SK = importlib.util.find_spec("sklearn") is not None


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="stochastic_block_model", method="Stochastic block model", domain="network science",
        family="ml", goal="explore", preconditions=Precondition(requires_edgelist=True, min_rows=10),
    )


def _planted_blocks(tmp_path: Path, size: int = 30, p_in: float = 0.4, p_out: float = 0.02) -> Path:
    rng = np.random.default_rng(0)
    z = [0] * size + [1] * size
    nodes = [f"n{i}" for i in range(2 * size)]
    edges = []
    for i in range(2 * size):
        for j in range(i + 1, 2 * size):
            p = p_in if z[i] == z[j] else p_out
            if rng.uniform() < p:
                edges.append((nodes[i], nodes[j]))
    csv = tmp_path / "blocks.csv"
    pd.DataFrame(edges, columns=["source", "target"]).to_csv(csv, index=False)
    return csv


@pytest.mark.skipif(not (_HAS_NX and _HAS_SK), reason="networkx/sklearn not available")
def test_recovers_planted_blocks(tmp_path: Path) -> None:
    fp = profile_dataset(_planted_blocks(tmp_path))
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "完成" in res.summary
    assert res.estimates["n_nodes"] == 60
    # strong planted 2-block structure -> spectrum has exactly 2 eigenvalues beyond bulk
    assert res.estimates["n_blocks"] == 2
    # assortative planted graph -> within-block prob clearly above between
    assert res.estimates["mean_within_block_prob"] > res.estimates["mean_between_block_prob"]
    assert res.estimates["assortativity"] > 0
    # node-to-block CSV covers every node; block matrix is K x K
    nb = pd.read_csv(Path(res.output_dir) / "sbm_node_blocks.csv")
    assert len(nb) == 60 and set(nb.columns) == {"node", "block"}
    K = int(res.estimates["n_blocks"])
    bm = pd.read_csv(Path(res.output_dir) / "sbm_block_matrix.csv", index_col=0)
    assert bm.shape == (K, K)
    # selection evidence = adjacency-spectrum table (beyond-bulk eigenvalues)
    sel = pd.read_csv(Path(res.output_dir) / "sbm_block_selection.csv")
    assert {"rank", "abs_eigenvalue", "bulk_threshold", "beyond_bulk"} <= set(sel.columns)
    assert int(sel["beyond_bulk"].sum()) == 2  # 2 informative eigenvalues


@pytest.mark.skipif(not (_HAS_NX and _HAS_SK), reason="networkx/sklearn not available")
def test_forced_n_blocks(tmp_path: Path) -> None:
    fp = profile_dataset(_planted_blocks(tmp_path))
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"n_blocks": 4})
    assert res.estimates["n_blocks"] == 4  # config forces K, overriding the spectral estimate
    bm = pd.read_csv(Path(res.output_dir) / "sbm_block_matrix.csv", index_col=0)
    assert bm.shape == (4, 4)


@pytest.mark.skipif(not (_HAS_NX and _HAS_SK), reason="networkx/sklearn not available")
def test_reproducible_seeded(tmp_path: Path) -> None:
    fp = profile_dataset(_planted_blocks(tmp_path))
    a = run_analysis(fp, _entry(), output_root=str(tmp_path / "a"))
    b = run_analysis(fp, _entry(), output_root=str(tmp_path / "b"))
    assert a.estimates["n_blocks"] == b.estimates["n_blocks"]
    assert a.estimates["icl"] == b.estimates["icl"]


@pytest.mark.skipif(not (_HAS_NX and _HAS_SK), reason="networkx/sklearn not available")
def test_degrade_too_few_nodes(tmp_path: Path) -> None:
    # 5 nodes (< 10) -> honest skip, no crash
    df = pd.DataFrame({"source": ["a", "b", "c", "d", "a"], "target": ["b", "c", "d", "e", "c"]})
    csv = tmp_path / "tiny.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "随机块模型失败" in res.summary
    assert "n_blocks" not in res.estimates


def test_degrade_no_edge_list(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"node": [f"n{i}" for i in range(12)], "val": rng.normal(0, 1, 12).round(3)})
    csv = tmp_path / "x.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "网络分析失败" in res.summary
    assert "n_blocks" not in res.estimates
