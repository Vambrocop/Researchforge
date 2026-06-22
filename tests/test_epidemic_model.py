"""Tests for the epidemic_model (network SIR/SIS diffusion) executor branch.

Known dynamics:
  * high beta / low gamma -> large attack rate; low beta -> small attack rate
  * SIR R count is monotone non-decreasing over time (recovered never leave R)
Plus config (model/beta/gamma) and honest-degrade tests.
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
        id="epidemic_model", method="Network epidemic model", domain="network science",
        family="ml", goal="explore", preconditions=Precondition(requires_edgelist=True, min_rows=3),
    )


def _dense(tmp_path: Path, n: int = 20, p: float = 0.5) -> Path:
    """A reasonably dense Erdos-Renyi-style edge list (good mixing for spread)."""
    rng = np.random.default_rng(0)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.uniform() < p:
                edges.append((f"n{i}", f"n{j}"))
    df = pd.DataFrame(edges, columns=["source", "target"])
    csv = tmp_path / f"dense_{n}.csv"
    df.to_csv(csv, index=False)
    return csv


@pytest.mark.skipif(not _HAS_NX, reason="networkx not available")
def test_high_beta_large_attack_rate(tmp_path: Path) -> None:
    fp = profile_dataset(_dense(tmp_path, n=25, p=0.5))
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"model": "sir", "beta": 0.9, "gamma": 0.05, "steps": 60, "n_runs": 8},
    )
    assert "完成" in res.summary
    # high transmission + slow recovery on a dense graph -> most nodes get infected
    assert res.estimates["attack_rate"] > 0.7
    assert res.estimates["peak_infected"] > 1.0


@pytest.mark.skipif(not _HAS_NX, reason="networkx not available")
def test_low_beta_small_attack_rate(tmp_path: Path) -> None:
    fp = profile_dataset(_dense(tmp_path, n=25, p=0.5))
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"model": "sir", "beta": 0.001, "gamma": 0.6, "steps": 60, "n_runs": 8},
    )
    # negligible transmission + fast recovery -> epidemic barely spreads beyond seed(s)
    assert res.estimates["attack_rate"] < 0.3
    # ordering sanity: high-beta attack rate >> low-beta attack rate
    hi = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "hi"),
        config={"model": "sir", "beta": 0.9, "gamma": 0.05, "steps": 60, "n_runs": 8},
    )
    assert hi.estimates["attack_rate"] > res.estimates["attack_rate"]


@pytest.mark.skipif(not _HAS_NX, reason="networkx not available")
def test_sir_R_monotone_non_decreasing(tmp_path: Path) -> None:
    fp = profile_dataset(_dense(tmp_path, n=20, p=0.4))
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"model": "sir", "beta": 0.5, "gamma": 0.2, "steps": 50, "n_runs": 6},
    )
    curve = pd.read_csv(Path(res.output_dir) / "epidemic_curve.csv")
    r = curve["R"].to_numpy()
    # recovered never leave R in SIR -> mean R(t) is monotone non-decreasing
    assert np.all(np.diff(r) >= -1e-9)
    # S is monotone non-increasing (a node only ever leaves S)
    s = curve["S"].to_numpy()
    assert np.all(np.diff(s) <= 1e-9)
    # conservation: S + I + R == N at every step (exact in the sim; the CSV rounds the
    # run-averaged S/I/R independently to 3 dp, so allow that ≤1.5e-3 display rounding)
    n = res.estimates["n_nodes"]
    assert np.allclose((curve["S"] + curve["I"] + curve["R"]).to_numpy(), n, atol=2e-3)


@pytest.mark.skipif(not _HAS_NX, reason="networkx not available")
def test_sis_no_R_compartment(tmp_path: Path) -> None:
    fp = profile_dataset(_dense(tmp_path, n=20, p=0.5))
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"model": "sis", "beta": 0.6, "gamma": 0.3, "steps": 40, "n_runs": 6},
    )
    curve = pd.read_csv(Path(res.output_dir) / "epidemic_curve.csv")
    # SIS has no recovered compartment -> R stays 0 throughout
    assert np.allclose(curve["R"].to_numpy(), 0.0, atol=1e-9)
    assert "r0_proxy" in res.estimates


@pytest.mark.skipif(not _HAS_NX, reason="networkx not available")
def test_reproducible_seeded(tmp_path: Path) -> None:
    fp = profile_dataset(_dense(tmp_path, n=20, p=0.5))
    cfg = {"model": "sir", "beta": 0.4, "gamma": 0.2, "steps": 40, "n_runs": 6}
    a = run_analysis(fp, _entry(), output_root=str(tmp_path / "a"), config=cfg)
    b = run_analysis(fp, _entry(), output_root=str(tmp_path / "b"), config=cfg)
    assert a.estimates["attack_rate"] == b.estimates["attack_rate"]
    assert a.estimates["peak_infected"] == b.estimates["peak_infected"]


def test_degrade_no_edge_list(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"node": [f"n{i}" for i in range(10)], "val": rng.normal(0, 1, 10).round(3)})
    csv = tmp_path / "x.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "网络分析失败" in res.summary
    assert "attack_rate" not in res.estimates
