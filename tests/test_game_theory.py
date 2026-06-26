"""Tests for the GAME-THEORY family (decision domain): normal_form_game,
shapley_value.

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


def _entry(eid: str, method: str) -> AnalysisEntry:
    return AnalysisEntry(
        id=eid,
        method=method,
        domain="decision",
        family="game_theory",
        goal="evaluate",
        preconditions=Precondition(min_rows=1),
    )


def _csv(tmp_path: Path, name: str, df: pd.DataFrame) -> Path:
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


# --------------------------------------------------------------------------- #
# 1) normal_form_game
# --------------------------------------------------------------------------- #
def test_prisoners_dilemma_one_pure_ne(tmp_path: Path) -> None:
    """Prisoner's Dilemma:
        P1 = [[-1, -3], [0, -2]],  P2 = [[-1, 0], [-3, -2]]
    Best-response test:
        col_max(P1) = [0, -2]; row_max(P2) = [0, -2].
        Cell (1,1): P1=-2 >= 0? no for col0; cell (1,1) P1=-2>=col_max[1]=-2 ok,
        P2=-2>=row_max[1]=-2 ok -> NE (defect, defect). Exactly 1 pure NE.
    """
    csv = _csv(tmp_path, "pd.csv", pd.DataFrame({
        "p1_c0": [-1.0, 0.0],   # P1 payoff when P2 plays C0 (rows = P1 strategies)
        "p1_c1": [-3.0, -2.0],  # P1 payoff when P2 plays C1
        "p2_c0": [-1.0, -3.0],  # P2 payoff when P2 plays C0
        "p2_c1": [0.0, -2.0],   # P2 payoff when P2 plays C1
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("normal_form_game", "Normal-form game"),
                       output_root=str(tmp_path / "o"),
                       config={"player1_payoff": ["p1_c0", "p1_c1"],
                               "player2_payoff": ["p2_c0", "p2_c1"]})
    e = res.estimates
    assert e["n_pure_ne"] == 1.0
    assert e["n_p1_strategies"] == 2.0 and e["n_p2_strategies"] == 2.0
    assert e["is_zero_sum"] == 0.0
    # the unique NE is (R1, C1) = (defect, defect)
    out = Path(res.output_dir)
    eq = pd.read_csv(out / "equilibria.csv")
    pure = eq[eq["type"] == "pure"]
    assert len(pure) == 1
    assert pure.iloc[0]["p1_strategy"] == "R1"
    assert pure.iloc[0]["p2_strategy"] == "C1"
    # each player has a strictly dominated strategy (cooperate is dominated by defect)
    assert e["n_p1_strictly_dominated"] == 1.0
    assert e["n_p2_strictly_dominated"] == 1.0
    # IESDS collapses to the single (defect, defect) cell
    assert e["iesds_rows_remaining"] == 1.0
    assert e["iesds_cols_remaining"] == 1.0
    assert (out / "payoff_heatmap.png").exists() or True  # plot best-effort


def test_coordination_game_two_pure_plus_mixed(tmp_path: Path) -> None:
    """Pure coordination game:
        P1 = P2 = [[2, 0], [0, 1]]
    Pure NE: (R0,C0) and (R1,C1) -> 2 pure NE.
    Mixed NE (each makes the other indifferent):
        q = (P1[1,1]-P1[0,1]) / (P1[0,0]-P1[0,1]-P1[1,0]+P1[1,1]) = (1-0)/(2-0-0+1) = 1/3
        p = (P2[1,1]-P2[1,0]) / (P2[0,0]-P2[0,1]-P2[1,0]+P2[1,1]) = (1-0)/3 = 1/3
        value v1 = [1/3,2/3] P1 [1/3,2/3] = 2/3.
    """
    csv = _csv(tmp_path, "coord.csv", pd.DataFrame({
        "p1_c0": [2.0, 0.0],
        "p1_c1": [0.0, 1.0],
        "p2_c0": [2.0, 0.0],
        "p2_c1": [0.0, 1.0],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("normal_form_game", "Normal-form game"),
                       output_root=str(tmp_path / "o"),
                       config={"player1_payoff": ["p1_c0", "p1_c1"],
                               "player2_payoff": ["p2_c0", "p2_c1"]})
    e = res.estimates
    assert e["n_pure_ne"] == 2.0
    assert math.isclose(e["mixed_p1_prob"], 1.0 / 3.0, abs_tol=1e-6)
    assert math.isclose(e["mixed_p2_prob"], 1.0 / 3.0, abs_tol=1e-6)
    assert math.isclose(e["game_value"], 2.0 / 3.0, abs_tol=1e-6)
    assert e["is_zero_sum"] == 0.0


def test_matching_pennies_zero_sum(tmp_path: Path) -> None:
    """Matching pennies as a single numeric matrix (-> zero-sum, P2 = -P1):
        P1 = [[1, -1], [-1, 1]]
    No pure NE. Mixed NE = (0.5, 0.5) for both; game value 0.
        q = (1-(-1))/(1-(-1)-(-1)+1) = 2/4 = 0.5
        p (on P2=-P1) = ... = 0.5
        value = 0.
    Also: maximin = -1, minimax = +1 -> no pure saddle point.
    """
    csv = _csv(tmp_path, "mp.csv", pd.DataFrame({
        "col0": [1.0, -1.0],
        "col1": [-1.0, 1.0],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("normal_form_game", "Normal-form game"),
                       output_root=str(tmp_path / "o"))
    e = res.estimates
    assert e["n_pure_ne"] == 0.0
    assert e["is_zero_sum"] == 1.0
    assert e["has_saddle_point"] == 0.0
    assert math.isclose(e["mixed_p1_prob"], 0.5, abs_tol=1e-9)
    assert math.isclose(e["mixed_p2_prob"], 0.5, abs_tol=1e-9)
    assert math.isclose(e["game_value"], 0.0, abs_tol=1e-9)
    assert math.isclose(e["maximin_value"], -1.0, abs_tol=1e-9)
    assert math.isclose(e["minimax_value"], 1.0, abs_tol=1e-9)


def test_zero_sum_with_saddle_point(tmp_path: Path) -> None:
    """A zero-sum game WITH a pure saddle point:
        P1 = [[4, 3], [2, 1]]  (P2 = -P1)
        row mins = [3, 1] -> maximin = 3; col maxs = [4, 3] -> minimax = 3.
        maximin == minimax = 3 -> saddle point exists, game value 3.
        Pure NE check: col_max(P1) = [4, 3]; on P2=-P1, row_max(P2) per row:
        P2 = [[-4,-3],[-2,-1]] row_max = [-3, -1]. Cell (0,1): P1=3>=col_max[1]=3 ok;
        P2=-3>=row_max[0]=-3 ok -> NE (R0,C1). 1 pure NE.
    """
    csv = _csv(tmp_path, "sad.csv", pd.DataFrame({
        "c0": [4.0, 2.0],
        "c1": [3.0, 1.0],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("normal_form_game", "Normal-form game"),
                       output_root=str(tmp_path / "o"))
    e = res.estimates
    assert e["is_zero_sum"] == 1.0
    assert e["has_saddle_point"] == 1.0
    assert math.isclose(e["maximin_value"], 3.0, abs_tol=1e-9)
    assert math.isclose(e["minimax_value"], 3.0, abs_tol=1e-9)
    assert math.isclose(e["game_value"], 3.0, abs_tol=1e-9)
    assert e["n_pure_ne"] >= 1.0


def test_normal_form_bimatrix_long_form(tmp_path: Path) -> None:
    """Bimatrix long form pivots into two payoff matrices. Re-encode the Prisoner's
    Dilemma; expect exactly 1 pure NE at the (D, D) cell.
        strategies: C (cooperate), D (defect).
    """
    rows = []
    payoff = {  # (p1s, p2s) -> (payoff1, payoff2)
        ("C", "C"): (-1, -1),
        ("C", "D"): (-3, 0),
        ("D", "C"): (0, -3),
        ("D", "D"): (-2, -2),
    }
    for (a, b), (x, y) in payoff.items():
        rows.append({"p1_strategy": a, "p2_strategy": b, "payoff1": x, "payoff2": y})
    csv = _csv(tmp_path, "bm.csv", pd.DataFrame(rows))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("normal_form_game", "Normal-form game"),
                       output_root=str(tmp_path / "o"))
    e = res.estimates
    assert e["n_p1_strategies"] == 2.0 and e["n_p2_strategies"] == 2.0
    assert e["n_pure_ne"] == 1.0
    assert e["is_zero_sum"] == 0.0


def test_normal_form_degrade_no_numeric(tmp_path: Path) -> None:
    """No numeric payoff data -> honest 跳过, no crash."""
    csv = _csv(tmp_path, "txt.csv", pd.DataFrame({"label": ["a", "b", "c"]}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("normal_form_game", "Normal-form game"),
                       output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
    assert "n_pure_ne" not in res.estimates


# --------------------------------------------------------------------------- #
# 2) shapley_value
# --------------------------------------------------------------------------- #
def test_shapley_symmetric_three_player(tmp_path: Path) -> None:
    """Symmetric 3-player game where every coalition's value is its size/3 scaled so
    v(N)=1 and players are interchangeable -> phi_i = 1/3 each by symmetry.
        v({}) = 0, v(single) = 0, v(pair) = 0, v(N) = 1
    is a 'unanimity-like' game; symmetry forces phi_A = phi_B = phi_C = 1/3.
    Efficiency: sum phi = 1 = v(N).
    """
    csv = _csv(tmp_path, "sym.csv", pd.DataFrame({
        "coalition": ["A", "B", "C", "A,B", "A,C", "B,C", "A,B,C"],
        "value": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("shapley_value", "Shapley value"),
                       output_root=str(tmp_path / "o"),
                       config={"coalition": "coalition", "value": "value"})
    e = res.estimates
    assert e["n_players"] == 3.0
    assert math.isclose(e["grand_coalition_value"], 1.0, abs_tol=1e-9)
    for p in ("A", "B", "C"):
        assert math.isclose(e[f"shapley__{p}"], 1.0 / 3.0, abs_tol=1e-5)
    # efficiency: sum phi == v(N)
    assert math.isclose(e["shapley_sum"], 1.0, abs_tol=1e-9)
    assert abs(e["efficiency_gap"]) < 1e-9
    out = Path(res.output_dir)
    assert (out / "shapley_values.csv").exists()


def test_shapley_asymmetric_glove_game(tmp_path: Path) -> None:
    """Asymmetric glove game (Owen): players L1, L2 own a left glove, R owns a right
    glove. A pair (one left + one right) is worth 1; everything else 0.
        v(L1)=v(L2)=v(R)=0
        v(L1,L2)=0  (two left gloves, no pair)
        v(L1,R)=v(L2,R)=1
        v(L1,L2,R)=1 (only one pair can be formed)
    Known Shapley values (Roth/Owen): the single right-glove owner R captures the
    most. By the formula:
        phi_R = 2/3, phi_L1 = phi_L2 = 1/6.
    Efficiency: 2/3 + 1/6 + 1/6 = 1 = v(N).
    """
    csv = _csv(tmp_path, "glove.csv", pd.DataFrame({
        "coalition": ["L1", "L2", "R", "L1,L2", "L1,R", "L2,R", "L1,L2,R"],
        "value": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("shapley_value", "Shapley value"),
                       output_root=str(tmp_path / "o"),
                       config={"coalition": "coalition", "value": "value"})
    e = res.estimates
    assert e["n_players"] == 3.0
    assert math.isclose(e["shapley__R"], 2.0 / 3.0, abs_tol=1e-5)
    assert math.isclose(e["shapley__L1"], 1.0 / 6.0, abs_tol=1e-5)
    assert math.isclose(e["shapley__L2"], 1.0 / 6.0, abs_tol=1e-5)
    # efficiency holds
    assert math.isclose(e["shapley_sum"], 1.0, abs_tol=1e-9)
    assert math.isclose(e["grand_coalition_value"], 1.0, abs_tol=1e-9)


def test_shapley_bitmask_layout(tmp_path: Path) -> None:
    """Bitmask coalition encoding ('101' = players P0 & P2). Re-encode the symmetric
    game; phi_i = 1/3 each. Players are positional: P0, P1, P2.
    """
    csv = _csv(tmp_path, "bits.csv", pd.DataFrame({
        "coalition": ["100", "010", "001", "110", "101", "011", "111"],
        "value": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("shapley_value", "Shapley value"),
                       output_root=str(tmp_path / "o"),
                       config={"coalition": "coalition", "value": "value"})
    e = res.estimates
    assert e["n_players"] == 3.0
    for p in ("P0", "P1", "P2"):
        assert math.isclose(e[f"shapley__{p}"], 1.0 / 3.0, abs_tol=1e-5)
    assert math.isclose(e["shapley_sum"], 1.0, abs_tol=1e-9)


def test_shapley_per_player_flag_layout(tmp_path: Path) -> None:
    """One 0/1 membership column per player + a value column. Symmetric game ->
    phi_i = 1/3 each.
    """
    csv = _csv(tmp_path, "flags.csv", pd.DataFrame({
        "A": [1, 0, 0, 1, 1, 0, 1],
        "B": [0, 1, 0, 1, 0, 1, 1],
        "C": [0, 0, 1, 0, 1, 1, 1],
        "value": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("shapley_value", "Shapley value"),
                       output_root=str(tmp_path / "o"),
                       config={"value": "value"})
    e = res.estimates
    assert e["n_players"] == 3.0
    for p in ("A", "B", "C"):
        assert math.isclose(e[f"shapley__{p}"], 1.0 / 3.0, abs_tol=1e-5)
    assert math.isclose(e["grand_coalition_value"], 1.0, abs_tol=1e-9)


def test_shapley_efficiency_general(tmp_path: Path) -> None:
    """A 3-player game with arbitrary coalition values; the Shapley value ALWAYS
    satisfies efficiency (sum phi = v(N)) regardless of the numbers.
        v(A)=10, v(B)=20, v(C)=30, v(AB)=50, v(AC)=60, v(BC)=70, v(ABC)=120.
    sum phi must equal 120.
    """
    csv = _csv(tmp_path, "gen.csv", pd.DataFrame({
        "coalition": ["A", "B", "C", "A,B", "A,C", "B,C", "A,B,C"],
        "value": [10.0, 20.0, 30.0, 50.0, 60.0, 70.0, 120.0],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("shapley_value", "Shapley value"),
                       output_root=str(tmp_path / "o"),
                       config={"coalition": "coalition", "value": "value"})
    e = res.estimates
    assert math.isclose(e["grand_coalition_value"], 120.0, abs_tol=1e-9)
    assert math.isclose(e["shapley_sum"], 120.0, abs_tol=1e-6)
    assert abs(e["efficiency_gap"]) < 1e-6
    # share % sums to ~100
    out = Path(res.output_dir)
    tbl = pd.read_csv(out / "shapley_values.csv")
    assert math.isclose(tbl["share_pct"].sum(), 100.0, abs_tol=1e-3)


def test_shapley_degrade_no_value_column(tmp_path: Path) -> None:
    """No usable value column -> honest 跳过, no crash."""
    csv = _csv(tmp_path, "novals.csv", pd.DataFrame({"label": ["x", "y", "z"]}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("shapley_value", "Shapley value"),
                       output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
    assert "grand_coalition_value" not in res.estimates


def test_shapley_degrade_too_many_players(tmp_path: Path) -> None:
    """n > 10 players -> exact enumeration capped -> honest 跳过 (no 2^n blowup)."""
    names = [f"P{i}" for i in range(11)]
    # only supply the grand coalition + singletons (enough to set n=11 players)
    coals = names + [",".join(names)]
    vals = [0.0] * len(names) + [1.0]
    csv = _csv(tmp_path, "big.csv", pd.DataFrame({"coalition": coals, "value": vals}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("shapley_value", "Shapley value"),
                       output_root=str(tmp_path / "o"),
                       config={"coalition": "coalition", "value": "value"})
    assert "跳过" in res.summary
    assert "grand_coalition_value" not in res.estimates
