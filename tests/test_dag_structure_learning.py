"""Tests for dag_structure_learning: PC algorithm causal discovery (CPDAG).

Simulate a KNOWN linear DAG  X1 → X2 → X3  and  X1 → X3  (so X2 mediates), plus an
INDEPENDENT X4, with Gaussian noise. Assert the PC SKELETON recovers the true
adjacencies (X1-X2, X2-X3, X1-X3 present; the X4 non-edges absent) and that orientation
runs (n_directed_edges >= 0). Generous assertions — PC may leave edges undirected
(a Markov-equivalence class). Plus a skip test (<3 vars / <50 rows). Seeded RNG.
Pure numpy/scipy — no optional-package skip needed."""

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
        id="dag_structure_learning",
        method="Causal discovery (PC algorithm, CPDAG)",
        domain="statistics",
        family="causal",
        goal="explore",
        preconditions=Precondition(min_continuous=3, min_rows=50),
    )


def _known_dag(seed: int = 0, n: int = 600) -> pd.DataFrame:
    """Linear-Gaussian SEM for the DAG  X1 → X2 → X3  and  X1 → X3 ;  X4 independent.

    True skeleton edges: X1-X2, X2-X3, X1-X3. Non-edges: X4 to everything, and
    (notably) X1-X3 is a DIRECT edge here so the triple X1-X2-X3 is SHIELDED (no
    v-structure) — a fair, non-trivial recovery target.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = 1.2 * x1 + rng.normal(0, 1, n)
    x3 = 0.9 * x2 + 0.7 * x1 + rng.normal(0, 1, n)
    x4 = rng.normal(0, 1, n)  # independent of the others
    return pd.DataFrame(
        {"X1": x1.round(5), "X2": x2.round(5), "X3": x3.round(5), "X4": x4.round(5)}
    )


def _collider_dag(seed: int = 0, n: int = 600) -> pd.DataFrame:
    """A genuine v-structure (collider):  X1 → X3 ← X2  with X1 ⟂ X2.

    X1, X2 are marginally independent but become dependent given the collider X3, so
    PC must NOT condition on X3 to delete X1-X2 — and the unshielded triple X1-X3-X2
    (X1,X2 non-adjacent, X3 not in their sepset) must orient X1→X3←X2."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = 1.1 * x1 + 1.1 * x2 + rng.normal(0, 0.6, n)
    return pd.DataFrame(
        {"X1": x1.round(5), "X2": x2.round(5), "X3": x3.round(5)}
    )


def test_pc_recovers_skeleton_and_orients(tmp_path: Path) -> None:
    csv = tmp_path / "dag.csv"
    _known_dag().to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"variables": ["X1", "X2", "X3", "X4"], "alpha": 0.05},
    )
    out = Path(res.output_dir)

    assert (out / "dag_edges.csv").exists()
    assert (out / "dag_adjacency.csv").exists()

    adj = pd.read_csv(out / "dag_adjacency.csv", index_col=0)
    # symmetric "edge present" view: an edge exists if either direction is marked
    def edge(a: str, b: str) -> bool:
        return bool(adj.loc[a, b]) or bool(adj.loc[b, a])

    # true skeleton edges present
    assert edge("X1", "X2")
    assert edge("X2", "X3")
    assert edge("X1", "X3")
    # X4 is independent → no edge to anything
    assert not edge("X4", "X1")
    assert not edge("X4", "X2")
    assert not edge("X4", "X3")

    # estimates sane
    assert res.estimates["n_variables"] == 4.0
    assert res.estimates["n"] == 600.0
    assert res.estimates["alpha"] == 0.05
    # the three true edges (X1-X2, X2-X3, X1-X3) recovered, no spurious X4 edges →
    # exactly 3 edges expected (generous range allows an occasional extra/undirected).
    assert 3 <= res.estimates["n_edges"] <= 4
    assert res.estimates["n_directed_edges"] >= 0
    assert (
        res.estimates["n_directed_edges"] + res.estimates["n_undirected_edges"]
        == res.estimates["n_edges"]
    )
    # honest CPDAG disclosure present
    assert "CPDAG" in res.summary or "马尔可夫等价" in res.summary
    assert "⚠" in res.summary


def test_pc_orients_v_structure(tmp_path: Path) -> None:
    """A real collider X1 → X3 ← X2 (X1 ⟂ X2) must give exactly the 2 edges X1-X3,
    X2-X3 (NOT X1-X2) and orient at least one direction (the v-structure)."""
    csv = tmp_path / "collider.csv"
    _collider_dag().to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"variables": ["X1", "X2", "X3"], "alpha": 0.05},
    )
    out = Path(res.output_dir)
    adj = pd.read_csv(out / "dag_adjacency.csv", index_col=0)

    def edge(a: str, b: str) -> bool:
        return bool(adj.loc[a, b]) or bool(adj.loc[b, a])

    assert edge("X1", "X3")
    assert edge("X2", "X3")
    assert not edge("X1", "X2")  # marginally independent → no edge
    # the collider should be oriented X1→X3←X2 (both arrowheads into X3)
    assert res.estimates["n_directed_edges"] >= 1
    assert bool(adj.loc["X1", "X3"])  # X1 -> X3 oriented
    assert bool(adj.loc["X2", "X3"])  # X2 -> X3 oriented


def test_pc_skips_too_few_variables(tmp_path: Path) -> None:
    """<3 continuous variables → honest skip (no crash, Chinese message)."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(0, 1, 100), "b": rng.normal(0, 1, 100)})
    csv = tmp_path / "two.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"variables": ["a", "b"]},
    )
    assert "n_edges" not in res.estimates
    assert "PC 算法" in res.summary and "跳过" in res.summary


def test_pc_skips_too_few_rows(tmp_path: Path) -> None:
    """<50 rows → honest skip."""
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {"X1": rng.normal(0, 1, 30), "X2": rng.normal(0, 1, 30), "X3": rng.normal(0, 1, 30)}
    )
    csv = tmp_path / "short.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"variables": ["X1", "X2", "X3"]},
    )
    assert "n_edges" not in res.estimates
    assert "PC 算法" in res.summary and "跳过" in res.summary


def test_dag_precondition_gate(tmp_path: Path) -> None:
    """The catalog preconditions (min_continuous=3, min_rows=50) gate correctly:
    a 2-column dataset fails the continuous-count precondition."""
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"y": rng.normal(0, 1, 80), "x": rng.normal(0, 1, 80)})
    csv = tmp_path / "flat.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)
    assert not ok
