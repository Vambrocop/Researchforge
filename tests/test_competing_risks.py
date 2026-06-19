"""Tests for competing_risks: Aalen-Johansen CIF < naive 1-KM, multi-state gate.

Synthetic structure: two competing exponential events (cause 1 = event of interest,
cause 2 = competing). With a real competing risk present, the proper Aalen-Johansen
CIF for cause 1 must be BELOW the naive 1-KM (which censors competing events and
over-estimates). We also check the honest skip when duration/event is missing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="competing_risks",
        method="Competing risks (Cumulative Incidence Functions, Aalen-Johansen)",
        domain="statistics",
        family="survival",
        goal="explain",
        preconditions=Precondition(min_continuous=1, min_rows=20),
    )


def _competing_data(n=600, seed=0) -> pd.DataFrame:
    """Two competing exponential causes + administrative censoring."""
    rng = np.random.default_rng(seed)
    # latent times for cause 1 and cause 2 (comparable hazards -> strong competition)
    t1 = rng.exponential(scale=10.0, size=n)
    t2 = rng.exponential(scale=10.0, size=n)
    cens = rng.exponential(scale=40.0, size=n)  # light independent censoring
    time = np.minimum.reduce([t1, t2, cens])
    status = np.where(
        (t1 <= t2) & (t1 <= cens), 1,
        np.where((t2 < t1) & (t2 <= cens), 2, 0),
    )
    return pd.DataFrame({
        "duration": np.round(time, 4),
        "status": status.astype(int),
        "group": rng.integers(0, 2, n),
    })


def test_cif_below_naive_1mkm(tmp_path: Path) -> None:
    pytest.importorskip("lifelines")
    df = _competing_data()
    csv = tmp_path / "cr.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    assert (out / "cif_curves.png").exists()
    assert (out / "cif_table.csv").exists()

    cif = res.estimates["cif_eoi_at_max_time"]
    naive = res.estimates["naive_1_minus_km_at_max_time"]
    # the headline guarantee: the proper CIF is strictly below the naive 1-KM,
    # which over-estimates by treating the competing event as censoring.
    assert 0.0 < cif < naive <= 1.0
    # competing events are genuinely present in the data
    assert res.estimates["n_event_2"] > 0
    assert res.estimates["n_event_1"] > 0

    tab = pd.read_csv(out / "cif_table.csv")
    # both event types reported, CIFs are valid probabilities and non-decreasing
    assert set(tab["event_type"].unique()) >= {1, 2}
    assert (tab["CIF"].between(0, 1)).all()


def test_cif_table_monotone_per_event(tmp_path: Path) -> None:
    pytest.importorskip("lifelines")
    df = _competing_data(seed=3)
    csv = tmp_path / "cr.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    tab = pd.read_csv(Path(res.output_dir) / "cif_table.csv")
    # CIF is a cumulative function -> non-decreasing in time within each group/event
    for (_g, _e), block in tab.groupby(["group", "event_type"]):
        vals = block.sort_values("time")["CIF"].values
        assert np.all(np.diff(vals) >= -1e-9), "CIF must be non-decreasing in time"


def test_competing_risks_missing_cols_skips(tmp_path: Path) -> None:
    # no duration / event columns at all -> honest skip, no crash
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(0, 1, 40), "b": rng.normal(0, 1, 40)})
    csv = tmp_path / "plain.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "竞争风险失败" in res.summary
    assert "cif_eoi_at_max_time" not in res.estimates


def test_competing_risks_binary_only_is_not_competing(tmp_path: Path) -> None:
    # a binary (0/1) event has no competing event -> branch refuses and points to KM/Cox
    rng = np.random.default_rng(2)
    n = 80
    df = pd.DataFrame({
        "duration": np.round(rng.exponential(10, n), 3),
        "status": rng.integers(0, 2, n),  # only {0,1}
    })
    csv = tmp_path / "bin.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "竞争风险失败" in res.summary
    assert "没有竞争事件" in res.summary or "退化为 1-KM" in res.summary
