"""Generate randomized experimental-design layouts (field plans) — the DoE-advisory
"provide a design" capability. Deterministic given a seed. Pairs with the matching
analysis branch (rcbd / factorial_anova / latin_square) the user runs after collecting
data.

Each generator returns a list of plot dicts. `recommend_design` suggests a design from
the factor structure; `generate_design` is the single entry the CLI/web call.
"""

from __future__ import annotations

import random

# design id -> the analysis branch to run on the collected data
ANALYSIS_FOR = {"rcbd": "rcbd", "factorial": "factorial_anova", "latin_square": "latin_square"}


def randomize_rcbd(treatments: list, n_blocks: int, seed: int = 0) -> list[dict]:
    """Randomized complete block design: each treatment appears once per block, in a
    randomized plot order within the block."""
    rng = random.Random(seed)
    plan = []
    for b in range(1, n_blocks + 1):
        order = list(treatments)
        rng.shuffle(order)
        for plot, t in enumerate(order, 1):
            plan.append({"block": b, "plot": plot, "treatment": t})
    return plan


def random_latin_square(treatments: list, seed: int = 0) -> list[dict]:
    """A randomized t×t Latin square (each treatment once per row and per column),
    built from a cyclic square then random row / column / symbol permutations."""
    rng = random.Random(seed)
    t = len(treatments)
    base = [[(i + j) % t for j in range(t)] for i in range(t)]  # cyclic Latin square
    rng.shuffle(base)                                            # permute rows
    cols = list(range(t))
    rng.shuffle(cols)
    base = [[r[c] for c in cols] for r in base]                  # permute columns
    sym = list(range(t))
    rng.shuffle(sym)                                             # relabel symbols
    plan = []
    for i in range(t):
        for j in range(t):
            plan.append({"row": i + 1, "col": j + 1, "treatment": treatments[sym[base[i][j]]]})
    return plan


def randomize_factorial(factor_a: list, factor_b: list, n_reps: int, seed: int = 0) -> list[dict]:
    """Full factorial A×B with `n_reps` complete replicates, randomized within each rep."""
    rng = random.Random(seed)
    combos = [(a, b) for a in factor_a for b in factor_b]
    plan = []
    for rep in range(1, n_reps + 1):
        order = list(combos)
        rng.shuffle(order)
        for plot, (a, b) in enumerate(order, 1):
            plan.append({"rep": rep, "plot": plot, "factor_a": a, "factor_b": b})
    return plan


def recommend_design(n_treatment_factors: int, blocking_directions: int = 1) -> str:
    """Rule-of-thumb design recommendation from the factor structure."""
    if n_treatment_factors >= 2:
        return "factorial"
    if blocking_directions >= 2:
        return "latin_square"
    return "rcbd"


def generate_design(design: str, *, treatments=None, n_blocks=3, factor_a=None,
                    factor_b=None, n_reps=3, seed: int = 0) -> dict:
    """Single entry: dispatch to the right generator and return a plan + metadata.
    Raises ValueError on bad inputs (honest fail, no silent garbage)."""
    if design == "rcbd":
        if not treatments or len(treatments) < 2:
            raise ValueError("rcbd 需要 ≥2 个处理（--treatments）")
        if n_blocks < 2:
            raise ValueError("rcbd 需要 ≥2 个区组（--blocks）")
        plan = randomize_rcbd(treatments, n_blocks, seed)
    elif design == "latin_square":
        if not treatments or len(treatments) < 3:
            raise ValueError("latin_square 需要 ≥3 个处理（方阵, --treatments）")
        plan = random_latin_square(treatments, seed)
    elif design == "factorial":
        if not factor_a or not factor_b or len(factor_a) < 2 or len(factor_b) < 2:
            raise ValueError("factorial 需要两个因子各 ≥2 水平（--factor-a / --factor-b）")
        if n_reps < 2:
            raise ValueError("factorial 需要 ≥2 个重复（--reps）以估交互误差")
        plan = randomize_factorial(factor_a, factor_b, n_reps, seed)
    else:
        raise ValueError(f"未知设计类型：{design}（支持 rcbd / factorial / latin_square）")
    return {"design": design, "plan": plan, "n_plots": len(plan),
            "analysis": ANALYSIS_FOR[design], "seed": seed}
