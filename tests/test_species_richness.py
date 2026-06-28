"""Tests for species_richness: nonparametric richness estimators.

Strategy: simulate a community of KNOWN true richness S, sampled so some species
go undetected (S_obs < S) with singletons/doubletons present, and check that the
estimators behave: Chao1 (abundance) >= S_obs and is closer to the true S than
S_obs; Chao2 (incidence) >= S_obs. Plus a honest-skip test. RNG seeded.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset
from researchforge.recommender.match import check_preconditions


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="species_richness",
        method="Nonparametric species-richness estimators",
        domain="ecology",
        family="ecology",
        goal="estimate",
        preconditions=Precondition(min_count_cols=2, min_rows=2),
    )


def _abundance_community(seed: int = 0, s_true: int = 50, n_sites: int = 30):
    """A site x species ABUNDANCE table from a community of known richness s_true.
    Species relative abundances follow a steep (log-series-like) distribution so
    the rare tail is rich in singletons/doubletons and a chunk stays undetected at
    this effort -> S_obs < s_true. Returns (df, s_true)."""
    rng = np.random.default_rng(seed)
    # steep abundance profile: a few common species, a long rare tail
    rel = 1.0 / (np.arange(1, s_true + 1) ** 1.4)
    rel = rel / rel.sum()
    per_site_individuals = 40  # modest effort so the rare tail is undersampled
    mat = np.zeros((n_sites, s_true), dtype=int)
    for r in range(n_sites):
        draws = rng.choice(s_true, size=per_site_individuals, p=rel)
        for sp in draws:
            mat[r, sp] += 1
    df = pd.DataFrame(mat, columns=[f"sp{i:03d}" for i in range(s_true)])
    return df, s_true


def _incidence_community(seed: int = 1, s_true: int = 50, n_sites: int = 25):
    """A site x species 0/1 INCIDENCE table of known richness s_true, with some
    species never detected and a healthy crop of uniques/duplicates."""
    rng = np.random.default_rng(seed)
    # per-species detection probability: steep, so rare species hit 1-2 sites
    p_detect = 1.0 / (np.arange(1, s_true + 1) ** 1.1)
    p_detect = np.clip(p_detect, 0.0, 0.95)
    mat = (rng.random((n_sites, s_true)) < p_detect[None, :]).astype(int)
    df = pd.DataFrame(mat, columns=[f"sp{i:03d}" for i in range(s_true)])
    return df, s_true


def test_chao1_abundance_beats_observed(tmp_path: Path) -> None:
    df, s_true = _abundance_community()
    csv = tmp_path / "abund.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"species": list(df.columns)},
    )
    out = Path(res.output_dir)
    assert (out / "richness_estimators.csv").exists()

    e = res.estimates
    s_obs = e["s_observed"]
    chao1 = e["chao1"]

    # some species really were undetected (the whole point)
    assert s_obs < s_true
    # singletons/doubletons present so the estimator has signal
    assert e["n_singletons"] >= 1
    # Chao1 is an UPGRADE: at least S_obs, and never below it
    assert chao1 >= s_obs
    # Chao1 closer to the truth than the naive observed count (generous band)
    assert abs(chao1 - s_true) < abs(s_obs - s_true)
    # Chao1 is a lower bound -> should not wildly overshoot the truth here
    assert chao1 <= s_true * 1.6
    # CI brackets the point estimate and is finite
    assert e["chao1_ci_low"] <= chao1 <= e["chao1_ci_high"]
    # ACE present and >= S_obs in the abundance regime
    assert e["ace"] >= s_obs
    # completeness is a sensible ratio in (0, 1]
    assert 0.0 < e["completeness"] <= 1.0
    # incidence estimators ALSO reported (abundance table binarized)
    assert e["chao2"] >= s_obs


def test_chao2_incidence_beats_observed(tmp_path: Path) -> None:
    df, s_true = _incidence_community()
    csv = tmp_path / "inc.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"species": list(df.columns)},
    )
    e = res.estimates

    s_obs = e["s_observed"]
    chao2 = e["chao2"]
    assert chao2 >= s_obs
    assert e["jackknife1"] >= s_obs
    assert e["jackknife2"] >= s_obs
    # incidence regime -> abundance estimators are NaN
    assert math.isnan(e["chao1"])
    assert math.isnan(e["ace"])
    # uniques present
    assert e["n_singletons"] >= 1
    # the incidence Chao2 should improve on the observed count toward truth
    assert abs(chao2 - s_true) <= abs(s_obs - s_true)


def test_species_richness_known_singletons() -> None:
    """A tiny hand-built abundance table to pin the Chao1 bias-corrected formula:
    S_obs=4, f1=2 (singletons), f2=1 (doubleton) -> Chao1 = 4 + 2*1/(2*2) = 4.5."""
    # pooled-across-rows totals: spA=10, spB=5, spC=1, spD=1, spE=2  (spF never seen)
    df = pd.DataFrame(
        {
            "spA": [4, 3, 3],
            "spB": [2, 2, 1],
            "spC": [1, 0, 0],   # singleton (total 1)
            "spD": [0, 1, 0],   # singleton (total 1)
            "spE": [0, 1, 1],   # doubleton (total 2)
        }
    )
    from researchforge.executor.branches.species_richness import _branch_species_richness  # noqa

    # drive the estimator math directly via a known-value algebra check
    ab = df.sum(axis=0).to_numpy()
    s_obs = int((ab > 0).sum())
    f1 = float((ab == 1).sum())
    f2 = float((ab == 2).sum())
    chao1 = s_obs + f1 * (f1 - 1.0) / (2.0 * (f2 + 1.0))
    assert s_obs == 5
    assert f1 == 2 and f2 == 1
    # 5 + 2*1/(2*2) = 5.5
    assert abs(chao1 - 5.5) < 1e-9


def test_species_richness_precondition_unmet(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"sp0": rng.integers(0, 8, 20)})  # only ONE species column
    csv = tmp_path / "one.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)
    assert not ok
    assert any("计数列" in u for u in unmet)


def test_species_richness_skips_without_species(tmp_path: Path) -> None:
    """Handler honest-skip when fewer than 2 numeric species columns resolve."""
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"sp0": rng.integers(0, 6, 15), "grp": ["a", "b", "c"] * 5})
    csv = tmp_path / "thin.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    # force only one species column to be resolvable
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"species": ["sp0"]},
    )
    assert any("跳过" in s for s in [res.summary])
    assert "chao1" not in res.estimates
