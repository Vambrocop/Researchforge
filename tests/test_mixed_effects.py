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
