"""Tests for the family data-affinity profiles (Stage 2 of smarter auto-selection).

Verifies every catalog family has a profile, that `data_signals` reads the right
structural flags off a fingerprint, and that `match_score` ranks the appropriate
family above the generic `statistics` family for each structural data shape — the
property Stage 3 will turn into a real `fit`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog import Catalog
from researchforge.profiler import profile_dataset
from researchforge.recommender.affinity import (
    FAMILY_AFFINITY,
    data_signals,
    get_affinity,
    match_score,
)


def _signals(df: pd.DataFrame, tmp_path: Path) -> dict:
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    return data_signals(profile_dataset(csv))


def test_every_catalog_family_has_a_profile() -> None:
    fams = {e.family for e in Catalog.load().entries}
    missing = fams - set(FAMILY_AFFINITY)
    assert not missing, f"families without an affinity profile: {sorted(missing)}"


def test_profiles_are_well_formed() -> None:
    valid_struct = {"panel", "timeseries", "geo", "edgelist", "cross_section", "any"}
    valid_oc = {"continuous", "count", "binary", "categorical", "survival", "multi_numeric", "none"}
    for fam, a in FAMILY_AFFINITY.items():
        assert a.structure in valid_struct, f"{fam}: bad structure {a.structure}"
        assert a.outcomes and a.outcomes <= valid_oc, f"{fam}: bad outcomes {a.outcomes}"
        assert a.min_rows >= 0


def test_data_signals_panel(tmp_path: Path) -> None:
    n = 120
    df = pd.DataFrame({"firm": np.repeat(np.arange(20), 6), "year": np.tile(np.arange(6), 20),
                       "x": np.random.default_rng(0).normal(0, 1, n).round(3),
                       "y": np.random.default_rng(1).normal(0, 1, n).round(3)})
    s = _signals(df, tmp_path)
    assert s["is_panel"] is True
    assert s["n_rows"] == 120


def test_data_signals_survival(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"duration": rng.exponential(10, 150).round(2),
                       "event": rng.binomial(1, 0.6, 150),
                       "age": rng.normal(60, 10, 150).round(1)})
    s = _signals(df, tmp_path)
    assert s["has_survival"] is True


def test_data_signals_edgelist(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"source": [f"u{i}" for i in rng.integers(0, 30, 200)],
                       "target": [f"u{i}" for i in rng.integers(0, 30, 200)]})
    s = _signals(df, tmp_path)
    assert s["has_edgelist"] is True


def test_has_treatment_requires_treatment_named_column(tmp_path: Path) -> None:
    # Wave M1: has_treatment 是排序层语义信号——须存在处理名列(treated/arm/exposed…),
    # 而非"有 binary 列就算"。无处理名的 cohort(disease/smoking/sex 全 binary)不发
    # requires_treatment bonus → 因果族不霸榜;有 treated 列则照发(真因果数据不受影响)。
    rng = np.random.default_rng(21)
    no_trt = pd.DataFrame({"disease": rng.binomial(1, 0.3, 120), "smoking": rng.binomial(1, 0.4, 120),
                           "age": rng.uniform(20, 80, 120).round(0)})
    assert _signals(no_trt, tmp_path)["has_treatment"] is False
    with_trt = pd.DataFrame({"treated": rng.binomial(1, 0.5, 120), "y": rng.normal(0, 1, 120),
                             "x": rng.normal(0, 1, 120)})
    assert _signals(with_trt, tmp_path)["has_treatment"] is True


def test_count_outcome_excludes_ordinal_likert(tmp_path: Path) -> None:
    # Wave K-A1: 1-5 Likert items profile as `count` but are ordinal_like ratings,
    # not count outcomes — they must NOT make Poisson/NB/PERMANOVA look feasible.
    rng = np.random.default_rng(11)
    df = pd.DataFrame({f"item{i}": rng.integers(1, 6, 150) for i in range(1, 6)})
    s = _signals(df, tmp_path)
    assert s["has_count"] is True          # they ARE count-kind…
    assert s["n_count_real"] == 0          # …but none is a real (non-ordinal) count
    assert s["has_count_outcome"] is False


def test_count_outcome_keeps_genuine_unbounded_count(tmp_path: Path) -> None:
    # Wave K-A1: an unbounded event count (has zeros / many distinct values) is a real
    # count outcome — the ordinal_like exclusion must not swallow true Poisson data.
    rng = np.random.default_rng(12)
    df = pd.DataFrame({"x": rng.normal(0, 1, 200).round(3),
                       "events": rng.poisson(6, 200)})
    s = _signals(df, tmp_path)
    assert s["n_count_real"] >= 1
    assert s["has_count_outcome"] is True


def test_match_score_prefers_right_family_per_structure(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    stat = get_affinity("statistics")

    # survival data -> survival family beats statistics
    surv = _signals(pd.DataFrame({"duration": rng.exponential(10, 150).round(2),
                                  "event": rng.binomial(1, 0.6, 150),
                                  "age": rng.normal(60, 10, 150).round(1)}), tmp_path)
    assert match_score(surv, get_affinity("survival")) > match_score(surv, stat)

    # geo data -> spatial family beats statistics
    geo = _signals(pd.DataFrame({"lat": rng.uniform(30, 40, 150).round(4),
                                 "lon": rng.uniform(-120, -110, 150).round(4),
                                 "temp": rng.normal(15, 5, 150).round(3)}), tmp_path)
    assert match_score(geo, get_affinity("spatial")) > match_score(geo, stat)

    # time-series data -> time-series family beats statistics
    ts = _signals(pd.DataFrame({"month": np.arange(140),
                                "sales": np.cumsum(rng.normal(0, 1, 140)).round(3) + 50}), tmp_path)
    assert match_score(ts, get_affinity("time-series")) > match_score(ts, stat)

    # panel data -> econometrics family beats statistics
    n = 120
    panel = _signals(pd.DataFrame({"firm": np.repeat(np.arange(20), 6),
                                   "year": np.tile(np.arange(6), 20),
                                   "cap": rng.normal(0, 1, n).round(3),
                                   "invest": rng.normal(0, 1, n).round(3)}), tmp_path)
    assert match_score(panel, get_affinity("econometrics")) > match_score(panel, stat)


def test_match_score_bounded(tmp_path: Path) -> None:
    s = _signals(pd.DataFrame({"x": [1.0, 2.0, 3.0] * 10, "y": [0, 1, 0] * 10}), tmp_path)
    for fam in FAMILY_AFFINITY.values():
        v = match_score(s, fam)
        assert 0.0 <= v <= 100.0
