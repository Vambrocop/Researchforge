"""Tests for raking — iterative proportional fitting (IPF) on 2+ margins.

Cross-checks:
  * supplied 2-variable margins -> IPF converges and the achieved weighted margins
    match the targets within tolerance;
  * the raked design effect (>= 1) and converged flag are reported;
  * honest skip when config margins is absent (never fabricate population targets).
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
        id="raking",
        method="Raking (iterative proportional fitting)",
        domain="statistics",
        family="survey_methods",
        goal="describe",
        preconditions=Precondition(min_categorical_cols=2, min_rows=3),
    )


def _data(tmp_path: Path) -> Path:
    # Two categorical variables, sample marginals skewed away from the population.
    rng = np.random.default_rng(0)
    n = 300
    sex = rng.choice(["M", "F"], n, p=[0.65, 0.35])  # sample over-samples M
    region = rng.choice(["north", "south", "east"], n, p=[0.5, 0.3, 0.2])
    df = pd.DataFrame({"sex": sex, "region": region})
    csv = tmp_path / "rk.csv"
    df.to_csv(csv, index=False)
    return csv


def test_raking_converges_and_hits_margins(tmp_path: Path) -> None:
    csv = _data(tmp_path)
    fp = profile_dataset(csv)
    margins = {
        "sex": {"M": 0.50, "F": 0.50},
        "region": {"north": 0.34, "south": 0.33, "east": 0.33},
    }
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"rake_vars": ["sex", "region"], "margins": margins})

    assert res.estimates["n_rake_vars"] == 2.0
    assert res.estimates["converged"] == 1.0
    assert res.estimates["max_margin_error"] < 1e-6
    # calibration raises variance -> raked deff >= 1
    assert res.estimates["raked_design_effect"] >= 1.0 - 1e-9
    assert res.estimates["iterations"] >= 1.0

    out = Path(res.output_dir)
    assert (out / "raking_margins.csv").exists()
    assert (out / "raking_weight_summary.csv").exists()
    md = pd.read_csv(out / "raking_margins.csv")
    # every achieved margin matches its target within tolerance
    for _, r in md.iterrows():
        assert abs(r["achieved_prop"] - r["target_prop"]) < 1e-6


def test_raking_skips_without_margins(tmp_path: Path) -> None:
    csv = _data(tmp_path)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"rake_vars": ["sex", "region"]})
    assert "converged" not in res.estimates
    assert "跳过" in res.summary
    assert (Path(res.output_dir) / "report.md").exists()


def test_raking_skips_with_one_variable(tmp_path: Path) -> None:
    csv = _data(tmp_path)
    fp = profile_dataset(csv)
    # only one rake var -> needs >= 2; honest skip.
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"rake_vars": ["sex"], "margins": {"sex": {"M": 0.5, "F": 0.5}}})
    assert "converged" not in res.estimates
    assert "跳过" in res.summary
