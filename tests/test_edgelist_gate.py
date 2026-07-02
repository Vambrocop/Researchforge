"""Tests for the requires_edgelist hard-gate (P0-3): match.py must use the SAME
node-column cardinality definition as affinity.data_signals, so plain low-cardinality
categorical data (gender/region) doesn't pass as a feasible "edge list" and float
network-science methods to the top for ordinary data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import Precondition
from researchforge.profiler import profile_dataset
from researchforge.recommender.match import check_preconditions

_EDGELIST_PRE = Precondition(requires_edgelist=True)


def test_low_cardinality_categoricals_fail_edgelist_gate(tmp_path: Path) -> None:
    """gender/region/income data has 2-category and 5-category columns — far below the
    12-distinct-value node threshold — so it must NOT be read as an edge list."""
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({
        "gender": rng.choice(["m", "f"], n),
        "region": rng.choice(["N", "S", "E", "W", "C"], n),
        "income": rng.normal(50000, 10000, n).round(2),
    })
    csv = tmp_path / "plain.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)

    feasible, unmet = check_preconditions(fp, _EDGELIST_PRE)
    assert feasible is False
    assert any("节点标识列" in m for m in unmet)


def test_high_cardinality_edgelist_passes_gate(tmp_path: Path) -> None:
    """A real edge list: source/target node columns each with >=12 distinct labels."""
    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame({
        "source": [f"u{i}" for i in rng.integers(0, 15, n)],
        "target": [f"u{i}" for i in rng.integers(0, 15, n)],
    })
    csv = tmp_path / "edges.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)

    feasible, unmet = check_preconditions(fp, _EDGELIST_PRE)
    assert feasible is True
    assert unmet == []
