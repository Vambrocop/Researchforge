"""Tests for poststratification — calibrate weights to KNOWN population proportions.

Cross-checks:
  * supplied pop_props -> the post-stratified weighted proportions hit the targets;
  * the per-cell adjustment factor = target_prop / sample_prop (hand check);
  * an adjusted value mean is reported when config value is given;
  * honest skip when config pop_props is absent (never fabricate population targets).
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
        id="poststratification",
        method="Post-stratification weight calibration",
        domain="statistics",
        family="survey_methods",
        goal="describe",
        preconditions=Precondition(min_categorical_cols=1, min_rows=3),
    )


def _data(tmp_path: Path) -> Path:
    # Sample over-represents "young": 60 young / 30 mid / 10 old (sample props .6/.3/.1).
    age = (["young"] * 60) + (["mid"] * 30) + (["old"] * 10)
    rng = np.random.default_rng(0)
    val = rng.normal(50, 5, len(age)).round(3)
    df = pd.DataFrame({"age_group": age, "spend": val})
    csv = tmp_path / "ps.csv"
    df.to_csv(csv, index=False)
    return csv


def test_postrat_hits_target_proportions(tmp_path: Path) -> None:
    csv = _data(tmp_path)
    fp = profile_dataset(csv)
    # known population: 30% young / 40% mid / 30% old.
    pop = {"young": 0.30, "mid": 0.40, "old": 0.30}
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"strata": "age_group", "pop_props": pop, "value": "spend"})

    out = Path(res.output_dir)
    cells = pd.read_csv(out / "poststrat_cells.csv")
    assert (out / "poststrat_cells.csv").exists()
    assert res.estimates["n_cells"] == 3.0
    assert res.estimates["n"] == 100.0

    # the post-stratified weighted proportions must equal the targets.
    # recompute weighted prop from the per-cell adjusted weights.
    cells = cells.set_index("cell")
    total_adj = (cells["n"] * cells["mean_adjusted_weight"]).sum()
    for lv, tp in pop.items():
        adj_w = cells.loc[lv, "n"] * cells.loc[lv, "mean_adjusted_weight"]
        assert abs(adj_w / total_adj - tp) < 1e-9
        # adjustment factor = target / sample_prop (sample props are .6/.3/.1)
        samp_p = cells.loc[lv, "sample_prop"]
        assert abs(cells.loc[lv, "adjustment_factor"] - tp / samp_p) < 1e-9

    # adjusted value mean reported
    assert "adjusted_value_mean" in res.estimates
    assert "unadjusted_value_mean" in res.estimates
    # weighting efficiency in (0, 1]
    assert 0 < res.estimates["weighting_efficiency"] <= 1.0 + 1e-9


def test_postrat_skips_without_pop_props(tmp_path: Path) -> None:
    csv = _data(tmp_path)
    fp = profile_dataset(csv)
    # no pop_props -> honest skip; targets must never be fabricated.
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"strata": "age_group"})
    assert "n_cells" not in res.estimates
    assert "跳过" in res.summary
    assert (Path(res.output_dir) / "report.md").exists()


def test_postrat_skips_when_level_lacks_target(tmp_path: Path) -> None:
    csv = _data(tmp_path)
    fp = profile_dataset(csv)
    # "old" missing from targets -> skip rather than invent its target.
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"strata": "age_group", "pop_props": {"young": 0.5, "mid": 0.5}})
    assert "n_cells" not in res.estimates
    assert "跳过" in res.summary
