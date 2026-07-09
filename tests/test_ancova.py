"""Tests for the ANCOVA executor branch (experimental_stats family)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="ancova", method="ANCOVA", domain="experimental design",
        family="experimental_stats", goal="explain",
        preconditions=Precondition(requires_group=True, min_continuous=2, min_rows=8),
    )


def _run(tmp_path: Path, df: pd.DataFrame, config: dict):
    csv = tmp_path / "c.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    return run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config=config)


def test_ancova_covariate_matters_and_adjusts(tmp_path: Path) -> None:
    # covariate x drives y strongly; groups DIFFER in x, so adjusting for x changes the
    # group means. parallel slopes (same x->y coefficient in every group).
    rng = np.random.default_rng(0)
    rows = []
    for g, x_center, trt in [("A", 0.0, 0.0), ("B", 5.0, 2.0)]:
        for _ in range(40):
            x = x_center + rng.normal(0, 1.0)
            y = 1.0 + 2.0 * x + trt + rng.normal(0, 1.0)  # shared slope 2.0
            rows.append({"y": y, "grp": g, "x": x})
    df = pd.DataFrame(rows)
    res = _run(tmp_path, df, {"outcome": "y", "group": "grp", "covariates": ["x"]})
    assert "完成" in res.summary
    e = res.estimates
    # covariate is strongly significant
    assert e["covariate_p"] < 1e-6
    # parallel slopes hold -> homogeneity-of-slopes interaction NOT significant
    assert e["slopes_interaction_p"] > 0.05
    assert "partial_eta_sq" in e and 0.0 <= e["partial_eta_sq"] <= 1.0
    assert e["n_groups"] == 2.0
    assert "adjusted_means.csv" in res.files and "ancova_table.csv" in res.files

    # adjusted means differ from unadjusted (because groups differ on the covariate)
    adj = pd.read_csv(Path(res.output_dir) / "adjusted_means.csv")
    diffs = (adj["adjusted_mean"] - adj["unadjusted_mean"]).abs()
    assert diffs.max() > 1.0  # adjustment moved the means substantially


def test_ancova_detects_heterogeneous_slopes(tmp_path: Path) -> None:
    # opposite slopes per group -> homogeneity-of-regression-slopes is VIOLATED
    rng = np.random.default_rng(3)
    rows = []
    for g, slope in [("A", 3.0), ("B", -3.0)]:
        for _ in range(50):
            x = rng.normal(0, 1.5)
            y = 5.0 + slope * x + rng.normal(0, 1.0)
            rows.append({"y": y, "grp": g, "x": x})
    df = pd.DataFrame(rows)
    res = _run(tmp_path, df, {"outcome": "y", "group": "grp", "covariates": ["x"]})
    assert res.estimates["slopes_interaction_p"] < 0.05
    assert "斜率假定被违反" in res.summary


def test_ancova_resolver_picks_named_outcome_not_first(tmp_path: Path) -> None:
    """A decoy continuous column ('noise_metric', unrelated to x/group) is placed
    BEFORE 'y' — the shared resolver must still pick 'y', not cont[0]='noise_metric'."""
    rng = np.random.default_rng(22)
    rows = []
    for g, x_center, trt in [("A", 0.0, 0.0), ("B", 5.0, 2.0)]:
        for _ in range(40):
            x = x_center + rng.normal(0, 1.0)
            rows.append({
                "noise_metric": rng.normal(0, 1.0),
                "grp": g, "x": x,
                "y": 1.0 + 2.0 * x + trt + rng.normal(0, 1.0),
            })
    df = pd.DataFrame(rows)
    res = _run(tmp_path, df, {"group": "grp", "covariates": ["x"]})  # no "outcome" in config
    assert "完成" in res.summary
    # covariate x truly drives y -> tiny covariate_p; a wrong (positional) pick of the
    # unrelated noise_metric column would NOT show this strong covariate signal.
    assert res.estimates["covariate_p"] < 1e-6


def test_ancova_needs_covariate(tmp_path: Path) -> None:
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                       "grp": ["A", "A", "A", "A", "B", "B", "B", "B"]})
    res = _run(tmp_path, df, {"outcome": "y", "group": "grp"})
    assert "失败" in res.summary
