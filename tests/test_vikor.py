"""Tests for vikor: VIKOR compromise ranking (S / R / Q + acceptance conditions).

Hand-checks: a dominant alternative gets S=R=Q=0 and ranks #1; Q rank order matches a
hand-computed Q; config v and weights override; the < 2 criteria gate produces an
honest skip.

NOTE: criteria columns must read as "continuous" (the profiler flags all-distinct
all-whole-number columns as ids), so each matrix carries a non-integer value. VIKOR
normalises by (f*-x)/(f*-f-), which is scale-invariant, so the hand math is unaffected.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset
from researchforge.recommender.match import check_preconditions


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="vikor",
        method="VIKOR compromise ranking",
        domain="evaluation",
        family="mcda",
        goal="describe",
        preconditions=Precondition(min_continuous=2, min_rows=3),
    )


def test_vikor_dominant_alternative_has_zero_Q_and_ranks_first(tmp_path: Path) -> None:
    # A is the best (largest) on every benefit criterion -> A is the ideal f* on all,
    # so S_A=0, R_A=0 -> Q_A=0, the global best.
    df = pd.DataFrame(
        {
            "alt": ["A", "B", "C", "D"],
            "c1": [10.5, 6.0, 4.0, 2.0],
            "c2": [9.5, 5.0, 7.0, 3.0],
            "c3": [8.5, 4.0, 6.0, 2.0],
        }
    )
    csv = tmp_path / "v.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    scores = pd.read_csv(out / "vikor_scores.csv")
    assert set(["alternative", "S", "R", "Q", "rank"]).issubset(scores.columns)
    top = scores.sort_values("rank").iloc[0]
    assert top["alternative"] == "A"
    assert abs(top["Q"] - 0.0) < 1e-9  # dominant -> Q=0
    assert abs(top["S"] - 0.0) < 1e-9
    assert abs(top["R"] - 0.0) < 1e-9


def test_vikor_Q_matches_hand_computation(tmp_path: Path) -> None:
    # 3 alternatives, 2 equal-weight benefit criteria, v=0.5. Hand-compute S, R, Q.
    # Raw values lie in [0,1] (with a .5 so they read continuous); f*=1, f-=0 on each.
    df = pd.DataFrame(
        {
            "alt": ["A", "B", "C"],
            "c1": [1.0, 0.5, 0.0],   # f*=1, f-=0
            "c2": [0.0, 0.5, 1.0],   # f*=1, f-=0
        }
    )
    csv = tmp_path / "vh.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)
    sc = pd.read_csv(out / "vikor_scores.csv").set_index("alternative")

    # weights = 0.5 each. term_ij = w * (f*_j - x_ij)/(f*_j - f-_j).
    # A: c1=0.5*(1-1)=0; c2=0.5*(1-0)=0.5 -> S=0.5, R=0.5
    # B: c1=0.5*0.5=0.25; c2=0.25 -> S=0.5, R=0.25
    # C: c1=0.5*1=0.5; c2=0 -> S=0.5, R=0.5
    for a in ("A", "B", "C"):
        assert abs(sc.loc[a, "S"] - 0.5) < 1e-9
    assert abs(sc.loc["A", "R"] - 0.5) < 1e-9
    assert abs(sc.loc["B", "R"] - 0.25) < 1e-9
    assert abs(sc.loc["C", "R"] - 0.5) < 1e-9
    # S*=S-=0.5 -> S term denom 0 -> S contributes 0 to Q. R*=0.25, R-=0.5.
    # Q = 0.5*0 + 0.5*(R-0.25)/0.25.  A:0.5; B:0; C:0.5.
    assert abs(sc.loc["B", "Q"] - 0.0) < 1e-9   # B is the compromise (min Q)
    assert abs(sc.loc["A", "Q"] - 0.5) < 1e-9
    assert abs(sc.loc["C", "Q"] - 0.5) < 1e-9
    assert sc.sort_values("rank").index[0] == "B"


def test_vikor_v_and_weights_config_override(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "alt": ["A", "B", "C"],
            "c1": [1.0, 0.5, 0.0],
            "c2": [0.0, 0.5, 1.0],
        }
    )
    csv = tmp_path / "vc.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    # Heavily weight c1 -> A (best on c1) should now win.
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"weights": [0.9, 0.1], "v": 0.5},
    )
    out = Path(res.output_dir)
    sc = pd.read_csv(out / "vikor_scores.csv")
    assert sc.sort_values("rank").iloc[0]["alternative"] == "A"


def test_vikor_precondition_unmet(tmp_path: Path) -> None:
    df = pd.DataFrame({"only_one": [1.1, 2.2, 3.3, 4.4]})  # < 2 numeric criteria
    csv = tmp_path / "one.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)
    assert not ok
    assert any("连续" in u for u in unmet)
