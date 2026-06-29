"""Tests for the link_prediction executor branch.

Planted-community structure: held-out within-community edges keep many common
neighbours, while cross-community non-edges have few -> neighbourhood predictors
should recover the held-out edges well above chance (AUC > 0.6). Plus seeded
reproducibility, predicted-links output, and honest-degrade tests.
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
        id="link_prediction", method="Link prediction", domain="network science",
        family="ml", goal="predict", preconditions=Precondition(requires_edgelist=True, min_rows=10),
    )


def _planted(tmp_path: Path, n_comm: int = 3, size: int = 9, p_in: float = 0.85) -> Path:
    rng = np.random.default_rng(0)
    comms = [[f"c{c}_{i}" for i in range(size)] for c in range(n_comm)]
    edges = []
    for c in comms:
        for i in range(len(c)):
            for j in range(i + 1, len(c)):
                if rng.uniform() < p_in:
                    edges.append((c[i], c[j]))
    alln = [x for c in comms for x in c]
    for _ in range(n_comm):  # a few sparse bridges
        a, b = rng.choice(alln, 2, replace=False)
        edges.append((a, b))
    csv = tmp_path / "planted.csv"
    pd.DataFrame(edges, columns=["source", "target"]).to_csv(csv, index=False)
    return csv


@pytest.mark.skipif(not (_HAS_NX and _HAS_SK), reason="networkx/sklearn not available")
def test_recovers_held_out_edges(tmp_path: Path) -> None:
    fp = profile_dataset(_planted(tmp_path))
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "完成" in res.summary
    assert res.estimates["n_nodes"] == 27
    # neighbourhood predictors should beat chance on planted structure
    assert res.estimates["best_predictor_auc"] > 0.6
    for name in ("common_neighbors", "jaccard", "adamic_adar",
                 "resource_allocation", "preferential_attachment"):
        assert f"auc_{name}" in res.estimates
    auc = pd.read_csv(Path(res.output_dir) / "link_prediction_auc.csv")
    assert set(auc.columns) == {"predictor", "auc"} and len(auc) == 5
    # predicted new links present, well-formed
    pl = pd.read_csv(Path(res.output_dir) / "predicted_links.csv")
    assert set(pl.columns) == {"source", "target", "score"}
    assert (pl["score"] >= 0).all()
    assert "link_prediction_summary.txt" in res.files


@pytest.mark.skipif(not (_HAS_NX and _HAS_SK), reason="networkx/sklearn not available")
def test_reproducible_seeded(tmp_path: Path) -> None:
    fp = profile_dataset(_planted(tmp_path))
    a = run_analysis(fp, _entry(), output_root=str(tmp_path / "a"))
    b = run_analysis(fp, _entry(), output_root=str(tmp_path / "b"))
    assert a.estimates["best_predictor_auc"] == b.estimates["best_predictor_auc"]
    assert a.estimates["auc_adamic_adar"] == b.estimates["auc_adamic_adar"]


@pytest.mark.skipif(not (_HAS_NX and _HAS_SK), reason="networkx/sklearn not available")
def test_degrade_too_few_edges(tmp_path: Path) -> None:
    # a 4-edge graph -> below the 10-edge floor for a stable held-out split
    df = pd.DataFrame({"source": ["a", "b", "c", "d"], "target": ["b", "c", "d", "a"]})
    csv = tmp_path / "tiny.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "链路预测失败" in res.summary
    assert "best_predictor_auc" not in res.estimates


def test_degrade_no_edge_list(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"node": [f"n{i}" for i in range(12)], "val": rng.normal(0, 1, 12).round(3)})
    csv = tmp_path / "x.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "网络分析失败" in res.summary
    assert "best_predictor_auc" not in res.estimates
