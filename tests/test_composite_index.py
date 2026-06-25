"""Tests for the RESOURCE / SUSTAINABILITY family method ``composite_index`` — an
OECD/JRC composite-indicator builder (the core of an EWF-nexus security index).

Known-value cases are hand-computed in the docstrings; honest-degrade paths assert
the Chinese "跳过" message and no crash. Mirrors tests/test_techno_economic.py.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="composite_index",
        method="Composite indicator (OECD-JRC)",
        domain="sustainability",
        family="resource",
        goal="describe",
        preconditions=Precondition(min_rows=3, min_numeric_cols=2),
    )


def _csv(tmp_path: Path, name: str, df: pd.DataFrame) -> Path:
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


def _run(tmp_path: Path, df: pd.DataFrame, cfg: dict | None = None, sub: str = "o"):
    csv = _csv(tmp_path, f"{sub}.csv", df)
    fp = profile_dataset(csv)
    return run_analysis(fp, _entry(), output_root=str(tmp_path / sub), config=cfg)


def _scores(res) -> dict[str, float]:
    """unit -> composite_score from the ranked CSV."""
    out = pd.read_csv(Path(res.output_dir) / "composite_scores.csv")
    return dict(zip(out["unit"].astype(str), out["composite_score"]))


def _rank_of(res, unit: str) -> int:
    out = pd.read_csv(Path(res.output_dir) / "composite_scores.csv")
    return int(out.loc[out["unit"].astype(str) == unit, "rank"].iloc[0])


# --------------------------------------------------------------------------- #
# (a) a unit dominating ALL indicators ranks #1 (linear, equal weights, minmax)
# --------------------------------------------------------------------------- #
def test_dominant_unit_ranks_first(tmp_path: Path) -> None:
    """A=(10,10) dominates B=(5,5) and C=(1,1) on both benefit indicators.
    minmax: i1/i2 -> A=1, B=(5-1)/9=0.444, C=0. linear equal -> A=1 (rank 1)."""
    df = pd.DataFrame({
        "unit": ["A", "B", "C"],
        "i1": [10.0, 5.0, 1.0],
        "i2": [10.0, 5.0, 1.0],
    })
    res = _run(tmp_path, df, {"unit": "unit"})
    assert _rank_of(res, "A") == 1
    assert _rank_of(res, "C") == 3
    sc = _scores(res)
    assert math.isclose(sc["A"], 1.0, abs_tol=1e-6)
    assert math.isclose(sc["B"], (5.0 - 1.0) / 9.0, abs_tol=1e-6)
    assert math.isclose(sc["C"], 0.0, abs_tol=1e-6)
    # estimates: equal weights sum to 1, two indicators
    assert res.estimates["n_units"] == 3.0
    assert res.estimates["n_indicators"] == 2.0
    assert math.isclose(res.estimates["weight__i1"], 0.5, abs_tol=1e-9)
    assert math.isclose(res.estimates["weight__i2"], 0.5, abs_tol=1e-9)
    out = Path(res.output_dir)
    assert (out / "composite_scores.csv").exists()
    assert (out / "indicator_weights.csv").exists()


# --------------------------------------------------------------------------- #
# (b) a cost indicator is inverted (worst-on-cost unit is NOT rewarded)
# --------------------------------------------------------------------------- #
def test_cost_indicator_inverted(tmp_path: Path) -> None:
    """benefit b tied (both 10 -> zero-range -> flat 0.5); cost c in {100, 0}.
    With c marked cost: norm_c = (max-x)/range -> A(c=100)=0, B(c=0)=1.
    linear equal: A=0.5*0.5+0.5*0=0.25, B=0.5*0.5+0.5*1=0.75 -> B above A.
    The unit WORST on the cost indicator (A, highest cost) must NOT be rewarded."""
    df = pd.DataFrame({
        "unit": ["A", "B"],
        "b": [10.0, 10.0],
        "c": [100.0, 0.0],
    })
    res = _run(tmp_path, df, {"unit": "unit", "cost_indicators": ["c"]})
    sc = _scores(res)
    assert sc["B"] > sc["A"], "low-cost unit B must outrank high-cost unit A"
    assert math.isclose(sc["A"], 0.25, abs_tol=1e-6)
    assert math.isclose(sc["B"], 0.75, abs_tol=1e-6)
    # control: WITHOUT marking c as cost, the high-c unit A is (wrongly) rewarded
    res2 = _run(tmp_path, df, {"unit": "unit"}, sub="o2")
    sc2 = _scores(res2)
    assert sc2["A"] > sc2["B"], "without inversion the high-c unit ranks higher"


# --------------------------------------------------------------------------- #
# (c) geometric aggregation penalizes an imbalanced unit vs linear
# --------------------------------------------------------------------------- #
def test_geometric_penalizes_imbalance(tmp_path: Path) -> None:
    """Anchors Lo=(0,0) Hi=(10,10) set the [0,1] scale. Balanced=(5,5) and
    Imbalanced=(10,0) get the SAME linear score (0.5 each) but geometric punishes
    the imbalanced one: Balanced geo=sqrt(.5*.5)=0.5, Imbalanced geo=sqrt(1*~0)~0."""
    df = pd.DataFrame({
        "unit": ["Lo", "Balanced", "Imbalanced", "Hi"],
        "i1": [0.0, 5.0, 10.0, 10.0],
        "i2": [0.0, 5.0, 0.0, 10.0],
    })
    lin = _run(tmp_path, df, {"unit": "unit", "aggregation": "linear"}, sub="lin")
    geo = _run(tmp_path, df, {"unit": "unit", "aggregation": "geometric"}, sub="geo")
    sl = _scores(lin)
    # linear: Balanced and Imbalanced tie
    assert math.isclose(sl["Balanced"], sl["Imbalanced"], abs_tol=1e-6)
    assert math.isclose(sl["Balanced"], 0.5, abs_tol=1e-6)
    # geometric: Balanced is rewarded over Imbalanced
    sg = _scores(geo)
    assert sg["Balanced"] > sg["Imbalanced"]
    assert _rank_of(geo, "Imbalanced") > _rank_of(geo, "Balanced")


# --------------------------------------------------------------------------- #
# (d) entropy weighting returns weights summing to ~1
# --------------------------------------------------------------------------- #
def test_entropy_weights_sum_to_one(tmp_path: Path) -> None:
    """Entropy objective weights over all indicators must form a valid weight
    vector (each >= 0, summing to 1). The dispersed indicator earns more weight."""
    df = pd.DataFrame({
        "unit": ["A", "B", "C", "D"],
        # i_spread is highly dispersed; i_flat barely varies -> i_spread gets more weight
        "i_spread": [1.0, 30.0, 60.0, 100.0],
        "i_flat": [50.0, 51.0, 49.0, 50.0],
    })
    res = _run(tmp_path, df, {"unit": "unit", "weighting": "entropy"})
    w_spread = res.estimates["weight__i_spread"]
    w_flat = res.estimates["weight__i_flat"]
    assert w_spread >= 0.0 and w_flat >= 0.0
    assert math.isclose(w_spread + w_flat, 1.0, abs_tol=1e-6)
    # more-dispersed indicator -> lower entropy -> higher weight
    assert w_spread > w_flat
    # cross-check against the weights CSV
    wt = pd.read_csv(Path(res.output_dir) / "indicator_weights.csv")
    assert math.isclose(float(wt["weight"].sum()), 1.0, abs_tol=1e-6)


# --------------------------------------------------------------------------- #
# zscore normalization still produces a valid ranking
# --------------------------------------------------------------------------- #
def test_zscore_normalization_ranks(tmp_path: Path) -> None:
    """A dominant unit still ranks #1 under z-score normalization."""
    df = pd.DataFrame({
        "unit": ["A", "B", "C"],
        "i1": [10.0, 5.0, 1.0],
        "i2": [9.0, 6.0, 2.0],
    })
    res = _run(tmp_path, df, {"unit": "unit", "normalization": "zscore"})
    assert _rank_of(res, "A") == 1
    assert _rank_of(res, "C") == 3


# --------------------------------------------------------------------------- #
# indicators subset via config
# --------------------------------------------------------------------------- #
def test_indicators_subset(tmp_path: Path) -> None:
    """config indicators selects a subset; ignored columns don't affect the score."""
    df = pd.DataFrame({
        "unit": ["A", "B", "C"],
        "keep1": [10.0, 5.0, 1.0],
        "keep2": [10.0, 5.0, 1.0],
        "ignore": [1.0, 100.0, 50.0],  # would flip ranking if it were used
    })
    res = _run(tmp_path, df, {"unit": "unit", "indicators": ["keep1", "keep2"]})
    assert res.estimates["n_indicators"] == 2.0
    assert "weight__ignore" not in res.estimates
    assert _rank_of(res, "A") == 1


# --------------------------------------------------------------------------- #
# honest degrade: fewer than two numeric indicators
# --------------------------------------------------------------------------- #
def test_degrade_too_few_indicators(tmp_path: Path) -> None:
    df = pd.DataFrame({"unit": ["A", "B", "C"], "only": [1.0, 2.0, 3.0]})
    res = _run(tmp_path, df, {"unit": "unit"})
    assert "跳过" in res.summary
    assert "composite_score" not in res.estimates
    assert "n_units" not in res.estimates


def test_degrade_no_numeric(tmp_path: Path) -> None:
    df = pd.DataFrame({"unit": ["A", "B", "C"], "label": ["x", "y", "z"]})
    res = _run(tmp_path, df, {"unit": "unit"})
    assert "跳过" in res.summary


# --------------------------------------------------------------------------- #
# zero-range indicator is disclosed and contributes nothing
# --------------------------------------------------------------------------- #
def test_zero_range_indicator_disclosed(tmp_path: Path) -> None:
    """A constant indicator carries no information -> mapped to flat 0.5 (minmax),
    disclosed in the summary, and does not change the relative ranking."""
    df = pd.DataFrame({
        "unit": ["A", "B", "C"],
        "i1": [10.0, 5.0, 1.0],
        "const": [7.0, 7.0, 7.0],
    })
    res = _run(tmp_path, df, {"unit": "unit"})
    assert "常数" in res.summary or "零方差" in res.summary
    # i1 alone decides the order
    assert _rank_of(res, "A") == 1
    assert _rank_of(res, "C") == 3
