"""Guard tests for the P1 outcome/group-selection hardening (Opus history sweep)."""

from pathlib import Path

import pandas as pd

from researchforge.catalog import Catalog
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset
from researchforge.synth import make_panel


def test_group_comparison_prefers_binary_over_unit_id(tmp_path):
    # Panel has unit (8 categories), year (time), y (continuous), treated (binary).
    # The grouping variable must be the binary 'treated', NOT the high-cardinality 'unit'.
    csv = tmp_path / "panel.csv"
    make_panel(n_units=8, n_periods=6, treated=True, seed=1).to_csv(csv, index=False)
    fp = profile_dataset(csv)

    res = run_analysis(fp, Catalog.load().by_id("group_comparison"), output_root=str(tmp_path / "o"))
    gm = pd.read_csv(Path(res.output_dir) / "group_means.csv")

    assert len(gm) == 2  # 2 groups (treated 0/1), not 8 (unit ids)


def test_did_picks_within_unit_varying_treatment(tmp_path):
    # 'region' is a fixed-within-unit group flag (binary, ordered first); 'treated' is the
    # real staggered DID treatment. DID must use 'treated', not the first binary 'region'.
    df = make_panel(n_units=8, n_periods=6, treated=True, seed=1)
    units = list(df["unit"].unique())
    region_map = {u: i % 2 for i, u in enumerate(units)}  # constant within unit, binary
    df["region"] = df["unit"].map(region_map)
    df = df[["unit", "year", "y", "region", "treated"]]  # region before treated
    csv = tmp_path / "p.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)

    res = run_analysis(fp, Catalog.load().by_id("did"), output_root=str(tmp_path / "o"))

    assert "treated" in res.estimates  # within-unit-varying treatment chosen
    assert "region" not in res.estimates  # fixed group flag NOT used as the treatment


def test_iv_regression_gives_honest_message(tmp_path):
    csv = tmp_path / "panel.csv"
    make_panel(seed=2).to_csv(csv, index=False)
    fp = profile_dataset(csv)

    res = run_analysis(fp, Catalog.load().by_id("iv_regression"), output_root=str(tmp_path / "o"))

    assert "工具变量" in res.summary  # explains it needs a user-specified instrument
    assert "暂未接入" not in res.summary  # no longer the generic placeholder
