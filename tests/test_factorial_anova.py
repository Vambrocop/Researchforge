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


def test_factorial_rejects_complete_nesting(tmp_path: Path) -> None:
    # 城市(A) ⊂ 区域(B): every city belongs to exactly one region -> NOT a crossed
    # factorial design. The A*B formula is exactly rank-deficient (dogfood P6:
    # statsmodels used to return F=nan / leak "not full rank" warnings). Must
    # downgrade to a one-way ANOVA on the finer factor instead of crashing/NaN-ing.
    rng = np.random.default_rng(2)
    city_region = {
        "上海": "华东", "杭州": "华东",
        "广州": "华南", "深圳": "华南",
    }
    city_eff = {"上海": 0.0, "杭州": 3.0, "广州": 6.0, "深圳": 9.0}
    rows = []
    for city, region in city_region.items():
        for _ in range(10):
            rows.append({"y": 10 + city_eff[city] + rng.normal(0, 1), "A": city, "B": region})
    res = _run(tmp_path, rows)
    assert "失败" not in res.summary
    assert "嵌套" in res.summary
    assert np.isfinite(res.estimates["A_F"])
    assert np.isfinite(res.estimates["A_p"])
    # interaction/B are not estimable once nested -> explicitly NaN, not silently
    # fabricated, and never leaked as a spurious "highly significant" number.
    assert not np.isfinite(res.estimates["interaction_p"])


def test_factorial_true_crossed_design_not_flagged_as_nested(tmp_path: Path) -> None:
    # Sanity guard for the nesting heuristic itself: a genuine (if unbalanced)
    # crossed design — every A level paired with BOTH B levels — must still run
    # as a normal two-way ANOVA, not get misdetected as nested.
    rng = np.random.default_rng(3)
    rows = []
    for a in ["lo", "hi"]:
        for b in ["ctrl", "trt"]:
            n = 5 if (a, b) != ("hi", "trt") else 9  # unbalanced but fully crossed
            mu = 10 + (4 if a == "hi" else 0) + (3 if b == "trt" else 0)
            rows += [{"y": mu + rng.normal(0, 1), "A": a, "B": b} for _ in range(n)]
    res = _run(tmp_path, rows)
    assert "嵌套" not in res.summary
    assert "完成" in res.summary
