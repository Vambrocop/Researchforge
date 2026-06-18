"""Tests for the Latin square design ANOVA executor branch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="latin_square", method="Latin square ANOVA", domain="experimental design",
        family="experimental_design", goal="explain",
        preconditions=Precondition(requires_group=True, min_continuous=1, min_rows=9),
    )


def test_latin_square_recovers_treatment(tmp_path: Path) -> None:
    # valid 4×4 Latin square: trt = (row + col) % 4, on top of row + column gradients
    rng = np.random.default_rng(0)
    t = 4
    trt_eff = {k: 2.0 * k for k in range(t)}      # 0,2,4,6 — strong
    row_eff = rng.normal(0, 1.5, t)
    col_eff = rng.normal(0, 1.5, t)
    rows = []
    for i in range(t):
        for j in range(t):
            k = (i + j) % t
            rows.append({"y": 10 + trt_eff[k] + row_eff[i] + col_eff[j] + rng.normal(0, 0.4),
                         "trt": f"T{k}", "row": f"R{i}", "col": f"C{j}"})
    csv = tmp_path / "ls.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "y", "treatment": "trt", "row": "row", "col": "col"})
    assert "完成" in res.summary
    assert res.estimates["n_treatments"] == 4
    assert res.estimates["treatment_p"] < 0.05
    assert res.estimates["treatment_F"] > 3


def test_latin_square_rejects_invalid(tmp_path: Path) -> None:
    # treatment determined by column only -> each column has the same treatment 3× -> not a Latin square
    rows = []
    for i, r in enumerate(["R0", "R1", "R2"]):
        for c, trt in zip(["C0", "C1", "C2"], ["A", "B", "C"]):
            rows.append({"y": float(i) + len(c), "trt": trt, "row": r, "col": c})
    csv = tmp_path / "ls.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "y", "treatment": "trt", "row": "row", "col": "col"})
    assert "拉丁方设计失败" in res.summary
