"""Tests for the DoE design generator (randomized layouts) + `cli design`."""

from __future__ import annotations

from collections import Counter

import pytest

from researchforge.design import generate_design, recommend_design
from researchforge.design.layout import random_latin_square, randomize_factorial, randomize_rcbd


def test_rcbd_layout_each_treatment_once_per_block() -> None:
    plan = randomize_rcbd(["A", "B", "C", "D"], n_blocks=5, seed=1)
    assert len(plan) == 20
    for b in range(1, 6):
        assert sorted(r["treatment"] for r in plan if r["block"] == b) == ["A", "B", "C", "D"]
    assert randomize_rcbd(["A", "B", "C", "D"], 5, seed=1) == plan  # deterministic


def test_latin_square_is_valid() -> None:
    plan = random_latin_square(["A", "B", "C", "D"], seed=2)
    assert len(plan) == 16
    for i in range(1, 5):
        assert sorted(r["treatment"] for r in plan if r["row"] == i) == ["A", "B", "C", "D"]
        assert sorted(r["treatment"] for r in plan if r["col"] == i) == ["A", "B", "C", "D"]


def test_factorial_layout_all_combos() -> None:
    plan = randomize_factorial(["lo", "hi"], ["x", "y", "z"], n_reps=3, seed=0)
    assert len(plan) == 2 * 3 * 3
    combos = Counter((r["factor_a"], r["factor_b"]) for r in plan)
    assert len(combos) == 6 and all(v == 3 for v in combos.values())


def test_generate_design_validates() -> None:
    with pytest.raises(ValueError):
        generate_design("rcbd", treatments=["A"], n_blocks=3)
    with pytest.raises(ValueError):
        generate_design("latin_square", treatments=["A", "B"])
    with pytest.raises(ValueError):
        generate_design("factorial", factor_a=["a"], factor_b=["x", "y"], n_reps=3)
    with pytest.raises(ValueError):
        generate_design("bogus")
    assert recommend_design(2) == "factorial"
    assert recommend_design(1, blocking_directions=2) == "latin_square"
    assert recommend_design(1) == "rcbd"


def test_cli_design_runs(capsys) -> None:
    from researchforge.cli import main

    rc = main(["design", "rcbd", "--treatments", "A,B,C", "--blocks", "4", "--seed", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "rcbd" in out and "12" in out  # 3 treatments × 4 blocks = 12 plots
