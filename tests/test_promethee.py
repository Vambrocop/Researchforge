"""Tests for promethee: PROMETHEE II net-flow complete ranking.

Hand-checks: a dominant alternative has the highest net flow; net flows sum to ~0
(antisymmetry of the preference index); a hand-computed net flow on a tiny matrix;
config weights override; the < 2 criteria gate produces an honest skip.

NOTE: criteria columns must read as "continuous" (the profiler flags all-distinct
all-whole-number columns as ids), so each matrix carries a non-integer value. The
default preference function uses p = the per-criterion range, so absolute scale does
not change preferences once the ordering is fixed.
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
        id="promethee",
        method="PROMETHEE II complete ranking",
        domain="evaluation",
        family="mcda",
        goal="describe",
        preconditions=Precondition(min_continuous=2, min_rows=3),
    )


def test_promethee_dominant_alternative_highest_net_flow(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "alt": ["A", "B", "C", "D"],
            "c1": [10.5, 7.0, 4.0, 1.0],
            "c2": [9.5, 6.0, 5.0, 2.0],
            "c3": [8.5, 7.0, 3.0, 1.0],
        }
    )
    csv = tmp_path / "p.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    flows = pd.read_csv(out / "promethee_flows.csv")
    assert set(["alternative", "phi_plus", "phi_minus", "phi_net", "rank"]).issubset(
        flows.columns
    )
    top = flows.sort_values("rank").iloc[0]
    assert top["alternative"] == "A"  # A dominates on every criterion
    assert top["phi_net"] == flows["phi_net"].max()
    # Net flows of a complete preorder sum to ~0 (π aggregated symmetrically).
    assert abs(flows["phi_net"].sum()) < 1e-6


def test_promethee_net_flow_matches_hand_computation(tmp_path: Path) -> None:
    # 3 alternatives, 1 effective benefit criterion (range 1.0, A>B>C) + a constant
    # tie criterion, equal weights. Default linear function: q=0, p=range.
    df = pd.DataFrame(
        {
            "alt": ["A", "B", "C"],
            "c1": [1.5, 1.0, 0.5],     # range 1.0; A>B>C (the .5 keeps it continuous)
            "c2": [1.0, 1.0, 1.0],     # constant -> contributes 0 preference
        }
    )
    csv = tmp_path / "ph.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)
    fl = pd.read_csv(out / "promethee_flows.csv").set_index("alternative")

    # weights = 0.5 each; c2 contributes 0 (all diffs 0).
    # c1: p=1.0, q=0. P_c1(a,b)=clip((x_a-x_b)/1.0,0,1).
    #   A vs B: 0.5; A vs C: 1.0; B vs C: 0.5
    # π(a,b)=0.5*P_c1(a,b). m=3 -> flows /2.
    # φ+(A)=(0.25+0.5)/2=0.375; φ+(B)=(0+0.25)/2=0.125; φ+(C)=0
    # φ-(A)=0; φ-(B)=0.125; φ-(C)=0.375
    assert abs(fl.loc["A", "phi_plus"] - 0.375) < 1e-6
    assert abs(fl.loc["B", "phi_plus"] - 0.125) < 1e-6
    assert abs(fl.loc["C", "phi_plus"] - 0.0) < 1e-6
    assert abs(fl.loc["A", "phi_net"] - 0.375) < 1e-6
    assert abs(fl.loc["B", "phi_net"] - 0.0) < 1e-6
    assert abs(fl.loc["C", "phi_net"] - (-0.375)) < 1e-6


def test_promethee_weights_config_override(tmp_path: Path) -> None:
    # c2 favours C; weighting c2 heavily should make C the net-flow winner.
    df = pd.DataFrame(
        {
            "alt": ["A", "B", "C"],
            "c1": [10.5, 5.0, 1.0],   # favours A
            "c2": [1.0, 5.0, 10.5],   # favours C
        }
    )
    csv = tmp_path / "pc.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"weights": [0.1, 0.9]},
    )
    out = Path(res.output_dir)
    fl = pd.read_csv(out / "promethee_flows.csv")
    assert fl.sort_values("rank").iloc[0]["alternative"] == "C"  # c2-heavy -> C wins


def test_promethee_precondition_unmet(tmp_path: Path) -> None:
    df = pd.DataFrame({"only_one": [1.1, 2.2, 3.3, 4.4]})  # < 2 numeric criteria
    csv = tmp_path / "one.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)
    assert not ok
    assert any("连续" in u for u in unmet)
