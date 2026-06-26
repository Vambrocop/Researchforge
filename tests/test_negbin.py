"""Tests for negative_binomial_regression: precondition guard + executor recovery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _make_entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="negative_binomial_regression",
        method="Negative binomial regression (overdispersed counts)",
        domain="statistics",
        family="statistics",
        goal="explain",
        preconditions=Precondition(
            requires_count_outcome=True,
            min_rows=30,
        ),
    )


# ---------------------------------------------------------------------------
# Executor recovery test — overdispersed counts
# ---------------------------------------------------------------------------


def test_negbin_executor_recovers_coef(tmp_path: Path) -> None:
    """NB regression should recover log-rate coefficient for x1 ~= 0.5 on overdispersed data."""
    rng = np.random.default_rng(42)
    n = 400
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    mu = np.exp(0.3 + 0.5 * x1 - 0.4 * x2)
    # Inject overdispersion via a gamma-Poisson mixture (NB data generation)
    shape = 2.0
    counts = rng.poisson(mu * rng.gamma(shape, 1.0 / shape, n)).astype(int)

    df = pd.DataFrame({"count": counts, "x1": x1, "x2": x2})
    csv = tmp_path / "negbin_data.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)

    # Verify count column was profiled correctly
    count_cols = [c for c in fp.columns if c.kind == "count"]
    assert count_cols, (
        f"'count' column not detected; actual kinds: {[(c.name, c.kind) for c in fp.columns]}"
    )

    entry = _make_entry()
    res = run_analysis(fp, entry, output_root=str(tmp_path / "out"))
    out = Path(res.output_dir)

    # Required output files
    assert (out / "summary.txt").exists(), "summary.txt missing"
    assert (out / "coefficients.csv").exists(), "coefficients.csv missing"
    assert (out / "report.md").exists(), "report.md missing"

    # x1 must be in estimates
    assert "x1" in res.estimates, (
        f"'x1' not in estimates; got {res.estimates}. summary={res.summary}"
    )

    # Coefficient recovery: should be close to true 0.5
    x1_coef = res.estimates["x1"]
    assert abs(x1_coef - 0.5) < 0.25, (
        f"x1 coefficient {x1_coef:.4f} not close enough to 0.5"
    )


# ---------------------------------------------------------------------------
# Graceful skip — no count column
# ---------------------------------------------------------------------------


def test_negbin_graceful_skip_no_count(tmp_path: Path) -> None:
    """Without a count column the executor should not crash and report the skip reason."""
    rng = np.random.default_rng(7)
    n = 60
    df = pd.DataFrame(
        {
            "y": rng.normal(5, 1, n),   # continuous, not count
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
        }
    )
    csv = tmp_path / "no_count.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)

    # Confirm no count column
    assert not any(c.kind == "count" for c in fp.columns), (
        f"Expected no count columns; got: {[c.kind for c in fp.columns]}"
    )

    entry = _make_entry()
    res = run_analysis(fp, entry, output_root=str(tmp_path / "out"))
    out = Path(res.output_dir)

    # Should not crash; report.md must exist
    assert (out / "report.md").exists(), "report.md missing"
    assert "未找到计数型结果变量" in res.summary, (
        f"Expected skip message; got: {res.summary}"
    )


# ---------------------------------------------------------------------------
# Multi-count ambiguity note
# ---------------------------------------------------------------------------


def test_negbin_multi_count_ambiguity_note(tmp_path: Path) -> None:
    """With >1 count column the chosen outcome should be flagged in the summary."""
    rng = np.random.default_rng(13)
    n = 120
    df = pd.DataFrame(
        {
            "c_outcome": rng.poisson(3, n).astype(int),
            "c_other": rng.poisson(2, n).astype(int),
            "x1": rng.normal(0, 1, n),
        }
    )
    csv = tmp_path / "multi_count.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)

    assert sum(1 for c in fp.columns if c.kind == "count") >= 2

    res = run_analysis(fp, _make_entry(), output_root=str(tmp_path / "out"))
    assert "个计数列" in res.summary  # ambiguity note surfaced
