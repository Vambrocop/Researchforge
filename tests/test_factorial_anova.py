"""Tests for the two-way factorial ANOVA executor branch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="factorial_anova", method="Two-way factorial ANOVA", domain="experimental design",
        family="experimental_design", goal="explain",
        preconditions=Precondition(requires_group=True, min_continuous=1, min_rows=8),
    )


def _run(tmp_path: Path, rows: list[dict]):
    csv = tmp_path / "f.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    return run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                        config={"outcome": "y", "factor_a": "A", "factor_b": "B"})


def test_factorial_detects_interaction(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    rows = []
    for a in ["lo", "hi"]:
        for b in ["ctrl", "trt"]:
            mu = 10 + (5 if a == "hi" else 0) + (1 if b == "trt" else 0) + (6 if (a == "hi" and b == "trt") else 0)
            rows += [{"y": mu + rng.normal(0, 1), "A": a, "B": b} for _ in range(8)]
    res = _run(tmp_path, rows)
    assert "完成" in res.summary
    assert res.estimates["interaction_p"] < 0.05
    assert "交互显著" in res.summary


def test_factorial_no_interaction(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    rows = []
    for a in ["lo", "hi"]:
        for b in ["ctrl", "trt"]:
            mu = 10 + (4 if a == "hi" else 0) + (3 if b == "trt" else 0)  # purely additive
            rows += [{"y": mu + rng.normal(0, 1), "A": a, "B": b} for _ in range(8)]
    res = _run(tmp_path, rows)
    assert res.estimates["A_p"] < 0.05 and res.estimates["B_p"] < 0.05
    assert res.estimates["interaction_p"] > 0.05
    assert "交互不显著" in res.summary


def test_factorial_needs_replication(tmp_path: Path) -> None:
    # 1 obs per A×B cell -> no residual to estimate the interaction -> honest fail
    res = _run(tmp_path, [{"y": v, "A": a, "B": b} for v, a, b in
                          [(1.0, "a", "x"), (2.0, "a", "y"), (3.0, "b", "x"), (4.0, "b", "y")]])
    assert "失败" in res.summary
