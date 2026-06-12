"""Benchmark cases — synthetic datasets with known ground truth.

Each case knows what the engine *should* conclude (panel or not, which analyses
are feasible) and, where applicable, a true effect the executor should recover.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from researchforge.synth import make_panel


@dataclass
class BenchmarkCase:
    name: str
    build: Callable[[], pd.DataFrame]
    expect_panel: bool
    expect_feasible: set[str] = field(default_factory=set)
    expect_infeasible: set[str] = field(default_factory=set)
    # (analysis_id, variable, true_value, tolerance)
    recover: Optional[tuple[str, str, float, float]] = None


def _cross_section(seed: int = 2, n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    z = rng.normal(0, 1, n)
    y = 1.0 + 2.0 * x - 0.5 * z + rng.normal(0, 0.5, n)
    return pd.DataFrame({"y": y, "x": x, "z": z})  # y first -> outcome


def default_cases() -> list[BenchmarkCase]:
    return [
        BenchmarkCase(
            name="treated_panel",
            build=lambda: make_panel(n_units=8, n_periods=6, treated=True, seed=1),
            expect_panel=True,
            expect_feasible={"descriptive_stats", "did", "panel_fixed_effects", "ols_regression"},
            expect_infeasible={"correlation"},
            recover=("did", "treated", 2.0, 0.6),
        ),
        BenchmarkCase(
            name="cross_section",
            build=lambda: _cross_section(seed=2, n=80),
            expect_panel=False,
            expect_feasible={"descriptive_stats", "correlation", "ols_regression"},
            expect_infeasible={"did", "panel_fixed_effects"},
            recover=("ols_regression", "x", 2.0, 0.4),
        ),
        BenchmarkCase(
            name="untreated_panel",
            build=lambda: make_panel(n_units=8, n_periods=6, treated=False, seed=3),
            expect_panel=True,
            expect_feasible={"descriptive_stats", "panel_fixed_effects"},
            expect_infeasible={"did"},
        ),
    ]
