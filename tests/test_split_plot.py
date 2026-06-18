"""Tests for the split-plot ANOVA executor branch (two error strata)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="split_plot", method="Split-plot ANOVA", domain="experimental design",
        family="experimental_design", goal="explain",
        preconditions=Precondition(requires_group=True, min_continuous=1, min_rows=8),
    )


def _balanced_rows(seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    a_eff = {"dry": 0.0, "med": 2.0, "wet": 4.0}      # whole-plot (irrigation)
    b_eff = {"v1": 0.0, "v2": 3.0, "v3": 6.0}         # sub-plot (variety), strong
    rows = []
    for blk in range(1, 5):  # 4 blocks
        for a in a_eff:
            wp_err = rng.normal(0, 1.0)  # whole-plot error shared by all sub-plots in this whole-plot
            for b in b_eff:
                rows.append({"yield": 10 + a_eff[a] + b_eff[b] + wp_err + rng.normal(0, 0.4),
                             "block": blk, "irrigation": a, "variety": b})
    return rows


def test_split_plot_recovers_subplot_effect(tmp_path: Path) -> None:
    csv = tmp_path / "sp.csv"
    pd.DataFrame(_balanced_rows()).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "yield", "block": "block",
                               "whole_plot": "irrigation", "sub_plot": "variety"})
    assert "完成" in res.summary
    assert res.estimates["n_blocks"] == 4
    assert res.estimates["n_wholeplots"] == 3
    assert res.estimates["n_subplots"] == 3
    # strong sub-plot effect, tested against the residual stratum -> highly significant
    assert res.estimates["subplot_p"] < 0.01
    assert "两个误差层" in res.summary


def test_split_plot_requires_balance(tmp_path: Path) -> None:
    rows = _balanced_rows()[:-1]  # drop one obs -> unbalanced
    csv = tmp_path / "sp.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "yield", "block": "block",
                               "whole_plot": "irrigation", "sub_plot": "variety"})
    assert "裂区设计失败" in res.summary
