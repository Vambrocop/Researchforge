"""Tests for vif_multicollinearity: VIF + condition number.

Known-structure check: x2 is built as x1 + small noise, so x1/x2 are near-collinear
-> their VIFs must be large (>10) and the condition number high. An (almost)
independent x3 stays low-VIF. Plus config override and the <2-predictor degrade.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="vif_multicollinearity",
        method="Multicollinearity diagnostics (VIF + condition number)",
        domain="statistics",
        family="regression",
        goal="explain",
        preconditions=Precondition(min_continuous=1, min_numeric_cols=3, min_rows=10),
    )


def test_vif_detects_collinearity(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.normal(0, 1, n)
    x2 = x1 + rng.normal(0, 0.02, n)  # near-collinear with x1 -> high VIF
    x3 = rng.normal(0, 1, n)  # independent -> low VIF
    y = 1.0 * x1 + 0.5 * x3 + rng.normal(0, 0.5, n)  # y first -> outcome
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})
    csv = tmp_path / "vif.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    tab = pd.read_csv(out / "vif.csv")
    vif_x1 = float(tab[tab["predictor"] == "x1"].iloc[0]["VIF"])
    vif_x2 = float(tab[tab["predictor"] == "x2"].iloc[0]["VIF"])
    vif_x3 = float(tab[tab["predictor"] == "x3"].iloc[0]["VIF"])

    # collinear pair has hugely inflated VIF; independent predictor stays low
    assert vif_x1 > 10 and vif_x2 > 10
    assert vif_x3 < 5
    assert res.estimates["max_vif"] > 10
    assert res.estimates["n_high_vif"] >= 2
    assert res.estimates["condition_number"] > 30
    assert res.estimates["n_predictors"] == 3.0
    assert (out / "vif.png").exists()
    assert (out / "condition_indices.csv").exists()


def test_vif_clean_predictors_low(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n = 300
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    y = x1 - x2 + 0.5 * x3 + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})
    csv = tmp_path / "clean.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    # independent predictors -> all VIFs near 1, none flagged severe
    assert res.estimates["max_vif"] < 5
    assert res.estimates["n_high_vif"] == 0.0


def test_vif_config_override(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    n = 150
    x1 = rng.normal(0, 1, n)
    x2 = x1 + rng.normal(0, 0.02, n)
    target = x1 + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"other": rng.normal(0, 1, n), "target": target, "x1": x1, "x2": x2})
    csv = tmp_path / "ovr.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"outcome": "target", "predictors": ["x1", "x2"]},
    )
    out = Path(res.output_dir)
    tab = pd.read_csv(out / "vif.csv")
    assert set(tab["predictor"]) == {"x1", "x2"}
    assert res.estimates["max_vif"] > 10  # x1/x2 collinear


def test_vif_degrade_single_predictor(tmp_path: Path) -> None:
    # only one continuous predictor (plus outcome) -> VIF needs >=2 -> skip
    rng = np.random.default_rng(3)
    n = 40
    x = rng.normal(0, 1, n)
    y = 2 * x + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"y": y, "x": x})
    csv = tmp_path / "one.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
