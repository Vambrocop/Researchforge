"""Tests for the RCBD (randomized complete block design) executor branch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="rcbd", method="RCBD ANOVA", domain="experimental design",
        family="experimental_design", goal="explain",
        preconditions=Precondition(requires_group=True, min_continuous=1, min_rows=6),
    )


def test_rcbd_recovers_treatment_effect(tmp_path: Path) -> None:
    # 4 varieties × 6 blocks, strong treatment effect on top of a block (field) gradient.
    rng = np.random.default_rng(0)
    trt_eff = {"A": 0.0, "B": 2.0, "C": 4.0, "D": 6.0}
    rows = []
    for b in range(1, 7):
        b_eff = rng.normal(0, 1.5)  # block gradient RCBD should soak up
        for t in trt_eff:
            rows.append({"yield": 10 + trt_eff[t] + b_eff + rng.normal(0, 0.5),
                         "variety": t, "block": b})
    csv = tmp_path / "trial.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "yield", "treatment": "variety", "block": "block"})
    assert "完成" in res.summary
    assert res.estimates["n_treatments"] == 4
    assert res.estimates["n_blocks"] == 6
    assert res.estimates["treatment_p"] < 0.01      # strong effect -> highly significant
    assert res.estimates["treatment_F"] > 5
    assert "完全平衡" in res.summary                  # balanced 4x6 design


def test_rcbd_rejects_degenerate_design(tmp_path: Path) -> None:
    # near-saturated incomplete design (perfectly additive, residual MS ~ 0): must honest-fail,
    # NOT report a spurious "highly significant" F (inference-reviewer must-fix regression guard).
    df = pd.DataFrame({
        "y": [10.0, 12.0, 14.0, 11.0, 13.0, 15.0, 12.5],
        "trt": ["A", "B", "C", "A", "B", "C", "A"],
        "block": [1, 1, 1, 2, 2, 2, 3],
    })
    csv = tmp_path / "deg.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "y", "treatment": "trt", "block": "block"})
    assert "RCBD 失败" in res.summary
    assert "treatment_p" not in res.estimates   # no spurious F/p emitted


def test_rcbd_accepts_response_config_key_alias(tmp_path: Path) -> None:
    # D5: outcome column may be specified via config["response"] as well as
    # config["outcome"] (back-compat with the field_trials rcbd_anova key name).
    rng = np.random.default_rng(0)
    trt_eff = {"A": 0.0, "B": 2.0, "C": 4.0, "D": 6.0}
    rows = []
    for b in range(1, 7):
        b_eff = rng.normal(0, 1.5)
        for t in trt_eff:
            rows.append({"yield": 10 + trt_eff[t] + b_eff + rng.normal(0, 0.5),
                         "variety": t, "block": b})
    csv = tmp_path / "trial.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"response": "yield", "treatment": "variety", "block": "block"})
    assert "完成" in res.summary
    assert res.estimates["treatment_p"] < 0.01
    assert res.estimates["treatment_F"] > 5


def test_rcbd_needs_treatment_and_block(tmp_path: Path) -> None:
    # only one factor available -> cannot form a block design
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"y": rng.normal(0, 1, 10), "g": ["A", "B"] * 5})
    csv = tmp_path / "x.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "y", "treatment": "g"})
    assert "RCBD 失败" in res.summary
