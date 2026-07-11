"""Tests for the mixed_effects (linear mixed-effects model) executor branch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset
from researchforge.synth import make_panel


def _make_entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="mixed_effects",
        method="Linear mixed-effects model",
        domain="statistics",
        family="statistics",
        goal="explain",
        preconditions=Precondition(
            requires_group=True,
            min_continuous=1,
            min_rows=20,
        ),
    )


# ---------------------------------------------------------------------------
# 1. Happy path: panel data with unit, year, y, treated
# ---------------------------------------------------------------------------

def test_mixed_effects_panel(tmp_path: Path) -> None:
    df = make_panel(n_units=8, n_periods=6, treated=True, seed=1)
    # columns: unit, year, y, treated
    csv = tmp_path / "panel.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    entry = _make_entry()

    res = run_analysis(fp, entry, output_root=str(tmp_path / "out"))
    out = Path(res.output_dir)

    # required output files
    assert (out / "summary.txt").exists(), "summary.txt missing"
    assert (out / "report.md").exists(), "report.md missing"

    # treated should be recovered as a fixed-effect estimate
    assert "treated" in res.estimates, (
        f"'treated' not in estimates; got {res.estimates}. summary={res.summary}"
    )
    # with time controlled (C(year)), the staggered-treatment estimate ~2.15 ≈ true 2.0;
    # the pre-fix, time-uncontrolled estimate (~2.88) would fail this guard.
    assert abs(res.estimates["treated"] - 2.0) < 0.5, (
        f"treated estimate {res.estimates['treated']} not near true 2.0 — time may be uncontrolled"
    )

    # summary line mentions the group column (unit)
    assert "unit" in res.summary, (
        f"'unit' not found in summary: {res.summary!r}"
    )


# ---------------------------------------------------------------------------
# 2. Graceful skip: no categorical/binary group and no unit_col
# ---------------------------------------------------------------------------

def test_mixed_effects_no_group_skips_gracefully(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    n = 30
    df = pd.DataFrame({"x": rng.normal(0, 1, n), "y": rng.normal(0, 1, n)})
    csv = tmp_path / "two_continuous.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    # Confirm no unit_col and no categorical/binary column
    assert fp.unit_col is None
    assert not any(c.kind in {"categorical", "binary"} for c in fp.columns)

    entry = _make_entry()
    res = run_analysis(fp, entry, output_root=str(tmp_path / "out"))

    # Must not crash; report.md must exist
    assert Path(res.report_path).exists(), "report.md missing after graceful skip"

    # Summary must contain the skip note
    assert "未找到分组变量" in res.summary, (
        f"Expected skip note in summary; got: {res.summary!r}"
    )


# ---------------------------------------------------------------------------
# 3. Wave K-B3: categorical fixed effects must NOT be silently dropped, and a
#    genuine zero-predictor degeneration must report failure, not "完成".
# ---------------------------------------------------------------------------

def test_mixed_effects_keeps_categorical_fixed_effect(tmp_path: Path) -> None:
    """'site' (10 levels) is the random-effect group; 'region' (3 levels,
    categorical) is a real fixed-effect covariate that must be dummy-coded
    into the model, not dropped for not being continuous/count/binary."""
    rng = np.random.default_rng(5)
    n_sites = 10
    rows = []
    for s in range(n_sites):
        site_effect = rng.normal(0, 1.5)  # genuine random intercept per site
        for _ in range(30):
            region = rng.choice(["A", "B", "C"])
            offset = {"A": 0.0, "B": 5.0, "C": -5.0}[region]
            x = rng.normal(0, 1)
            y = 10.0 + site_effect + offset + 0.5 * x + rng.normal(0, 0.5)
            # y 是真结果(offset 直接进 y)；列序须让 y 在 x 前——outcome=第一个连续列，
            # 否则 outcome 误选 x、region 只经 y 后门相关，断言语义就错了(B3 冷审 SHOULD)。
            rows.append({"site": f"S{s}", "region": region, "y": y, "x": x})
    df = pd.DataFrame(rows)
    csv = tmp_path / "region_panel.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    assert any(c.name == "region" and c.kind == "categorical" for c in fp.columns), (
        f"expected 'region' profiled as categorical; got {[(c.name, c.kind) for c in fp.columns]}"
    )

    entry = _make_entry()
    res = run_analysis(fp, entry, output_root=str(tmp_path / "out"))

    assert "失败" not in res.summary, f"unexpected failure: {res.summary}"
    region_keys = [k for k in res.estimates if k.startswith("region")]
    assert region_keys, f"'region' dummy levels missing from estimates: {res.estimates}"
    # B has a strong positive offset relative to A/C -> at least one region
    # dummy coefficient should be clearly positive and large.
    assert any(v > 2.0 for v in (res.estimates[k] for k in region_keys)), (
        f"region dummy coefficients look wrong: {res.estimates}"
    )
    assert "含" in res.summary and "分类固定效应" in res.summary


def test_mixed_effects_zero_predictors_reports_failure(tmp_path: Path) -> None:
    """Only a group column + outcome, nothing else — the model would degenerate
    to an intercept-only fit. That must be reported as a failure, not '完成'."""
    rng = np.random.default_rng(1)
    n_groups = 8
    rows = []
    for g in range(n_groups):
        for _ in range(10):
            rows.append({"grp": f"G{g}", "y": rng.normal(0, 1)})
    df = pd.DataFrame(rows)
    csv = tmp_path / "grp_only.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    assert fp.unit_col is None  # no time-like column -> no panel structure detected

    entry = _make_entry()
    res = run_analysis(fp, entry, output_root=str(tmp_path / "out"))

    assert "失败" in res.summary, f"expected an honest failure, got: {res.summary}"
    assert "无可用固定效应预测变量" in res.summary
    assert "完成" not in res.summary
    assert not res.estimates
