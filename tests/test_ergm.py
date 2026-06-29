"""Tests for the ergm executor branch.

ergm delegates to R's statnet/ergm (MCMC-MLE) when available, else degrades to a
pure-Python CUG (conditional uniform graph) test of transitivity. A union of
cliques has transitivity ~1.0, far above random graphs of the same size/density,
so the CUG test should flag significant clustering. Also covers term-allowlist
injection safety and honest degrade.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.executor import rbridge
from researchforge.executor.branches.network_science import _ergm_terms_ok
from researchforge.profiler import profile_dataset

_HAS_NX = importlib.util.find_spec("networkx") is not None
_HAS_ERGM = rbridge.r_available() and rbridge.r_package_available("ergm")


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="ergm", method="Exponential random graph model", domain="network science",
        family="ml", goal="explain", preconditions=Precondition(requires_edgelist=True, min_rows=5),
    )


def _cliques(tmp_path: Path, n_cliques: int = 3, size: int = 6) -> Path:
    """Union of disjoint cliques -> global transitivity = 1.0 (every triple closed)."""
    edges = []
    for c in range(n_cliques):
        nodes = [f"k{c}_{i}" for i in range(size)]
        for i in range(size):
            for j in range(i + 1, size):
                edges.append((nodes[i], nodes[j]))
    csv = tmp_path / "cliques.csv"
    pd.DataFrame(edges, columns=["source", "target"]).to_csv(csv, index=False)
    return csv


def test_term_allowlist_injection_safe() -> None:
    # legitimate ergm formulas pass
    assert _ergm_terms_ok("edges + gwesp(0.25, fixed=TRUE)")
    assert _ergm_terms_ok("edges")
    assert _ergm_terms_ok("edges + triangle + mutual")
    assert _ergm_terms_ok("edges + gwdegree(0.5, fixed=TRUE)")
    # injection / unknown-term attempts are rejected
    assert not _ergm_terms_ok("edges + I(system('rm -rf'))")   # I() not in allowlist
    assert not _ergm_terms_ok("edges; system('x')")            # semicolon
    assert not _ergm_terms_ok("edges + `system`('x')")         # backtick
    assert not _ergm_terms_ok("nodematch('grp')")              # quote char + not allowlisted
    assert not _ergm_terms_ok("")                               # empty


@pytest.mark.skipif(not _HAS_NX, reason="networkx not available")
@pytest.mark.skipif(_HAS_ERGM, reason="R ergm present -> exercises R path, not the CUG degrade")
def test_cug_degrade_flags_clustering(tmp_path: Path) -> None:
    fp = profile_dataset(_cliques(tmp_path))
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"n_sim": 200})
    assert "CUG" in res.summary
    assert res.estimates["n_nodes"] == 18
    # disjoint cliques: transitivity 1.0 >> random density-matched null
    assert res.estimates["transitivity_observed"] > 0.99
    assert res.estimates["transitivity_observed"] > res.estimates["transitivity_null_mean"]
    assert res.estimates["transitivity_p"] < 0.05
    cug = pd.read_csv(Path(res.output_dir) / "ergm_cug_test.csv")
    assert {"statistic", "observed", "null_mean", "null_sd", "z", "p_one_sided"} <= set(cug.columns)
    assert "ergm_summary.txt" in res.files


@pytest.mark.skipif(not _HAS_ERGM, reason="R ergm not installed")
def test_r_ergm_path(tmp_path: Path) -> None:
    fp = profile_dataset(_cliques(tmp_path))
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "ergm" in res.summary.lower()
    assert "coef_edges" in res.estimates
    coef = pd.read_csv(Path(res.output_dir) / "ergm_coefficients.csv")
    assert {"term", "estimate", "std_err", "p_value"} <= set(coef.columns)
    assert (coef["term"] == "edges").any()


@pytest.mark.skipif(not _HAS_NX, reason="networkx not available")
def test_reproducible_seeded(tmp_path: Path) -> None:
    if _HAS_ERGM:
        pytest.skip("seeded determinism asserted on the CUG degrade path")
    fp = profile_dataset(_cliques(tmp_path))
    a = run_analysis(fp, _entry(), output_root=str(tmp_path / "a"), config={"n_sim": 150})
    b = run_analysis(fp, _entry(), output_root=str(tmp_path / "b"), config={"n_sim": 150})
    assert a.estimates["transitivity_p"] == b.estimates["transitivity_p"]
    assert a.estimates["transitivity_z"] == b.estimates["transitivity_z"]


def test_degrade_no_edge_list(tmp_path: Path) -> None:
    import numpy as np

    rng = np.random.default_rng(1)
    df = pd.DataFrame({"node": [f"n{i}" for i in range(12)], "val": rng.normal(0, 1, 12).round(3)})
    csv = tmp_path / "x.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "网络分析失败" in res.summary
    assert "transitivity_observed" not in res.estimates
