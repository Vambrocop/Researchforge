"""Tests for poisson_regression: precondition guard + executor recovery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset
from researchforge.recommender.match import check_preconditions


def _make_entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="poisson_regression",
        method="Poisson regression (count outcome)",
        domain="statistics",
        family="statistics",
        goal="explain",
        preconditions=Precondition(
            requires_count_outcome=True,
            min_rows=20,
        ),
    )


# ---------------------------------------------------------------------------
# Recommender / precondition tests
# ---------------------------------------------------------------------------


def test_precondition_met_with_count_column(tmp_path: Path) -> None:
    """When a count column exists, requires_count_outcome should be satisfied."""
    rng = np.random.default_rng(0)
    n = 50
    # Small non-negative integers with repeats -> profiler detects as "count"
    counts = rng.poisson(lam=3, size=n).astype(int)
    df = pd.DataFrame(
        {
            "count": counts,
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
        }
    )
    csv = tmp_path / "with_count.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)

    # Verify profiler tagged it as count
    count_cols = [c for c in fp.columns if c.kind == "count"]
    assert count_cols, (
        f"Expected at least one 'count' column; got kinds: {[c.kind for c in fp.columns]}"
    )

    entry = _make_entry()
    ok, unmet = check_preconditions(fp, entry.preconditions)
    assert ok, f"Expected preconditions met; unmet={unmet}"
    assert "需要计数型结果变量" not in unmet


def test_precondition_unmet_without_count_column(tmp_path: Path) -> None:
    """When no count column exists, 'needs count outcome' should appear in unmet."""
    rng = np.random.default_rng(1)
    n = 50
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

    # Confirm no count column was detected
    assert not any(c.kind == "count" for c in fp.columns), (
        f"Expected no count columns; got: {[c.kind for c in fp.columns]}"
    )

    entry = _make_entry()
    ok, unmet = check_preconditions(fp, entry.preconditions)
    assert not ok, "Expected preconditions NOT met"
    assert "需要计数型结果变量" in unmet, f"Expected unmet reason; got: {unmet}"


# ---------------------------------------------------------------------------
# Executor recovery test
# ---------------------------------------------------------------------------


def test_poisson_executor_recovers_coef(tmp_path: Path) -> None:
    """Poisson GLM should recover log-rate coefficient for x1 ~= 0.5."""
    rng = np.random.default_rng(42)
    n = 300
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    lam = np.exp(0.3 + 0.5 * x1 - 0.4 * x2)
    counts = rng.poisson(lam).astype(int)

    # count column must have repeats (it will since Poisson generates few unique values)
    df = pd.DataFrame({"count": counts, "x1": x1, "x2": x2})
    csv = tmp_path / "poisson_data.csv"
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
    assert abs(x1_coef - 0.5) < 0.2, (
        f"x1 coefficient {x1_coef:.4f} not close enough to 0.5"
    )


def test_poisson_multi_count_ambiguity_note(tmp_path: Path) -> None:
    """With >1 count column the chosen outcome is flagged (it may be an ID/code)."""
    rng = np.random.default_rng(7)
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


# ---------------------------------------------------------------------------
# Wave K-A2 — genuine count OUTCOME vs demographic integer covariate
# ---------------------------------------------------------------------------


def test_a2_valve_keeps_lone_unnamed_count_outcome(tmp_path: Path) -> None:
    """不误伤 lock: a LONE unbounded count response with a NON-count-ish name, no continuous
    column and no Likert array is a true Poisson outcome — the K-A2 safety valve must keep
    requires_count_outcome feasible even though the name gives no hint and role detection is
    silent."""
    rng = np.random.default_rng(21)
    n = 80
    # a single count-kind column, neutrally named (no count/n_/events hint), wide range so it
    # is neither ordinal_like nor id-like; no continuous / no Likert siblings.
    df = pd.DataFrame({"harvest": rng.poisson(9, n).astype(int)})
    csv = tmp_path / "lone_count.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)

    kinds = {c.name: c.kind for c in fp.columns}
    assert kinds.get("harvest") == "count", f"expected 'harvest' count-kind; got {kinds}"
    assert not getattr(fp.column("harvest"), "ordinal_like", False), "must not be ordinal_like"
    assert not any(c.kind == "continuous" for c in fp.columns)

    ok, unmet = check_preconditions(fp, _make_entry().preconditions)
    assert ok, f"safety valve should keep a lone true-count outcome feasible; unmet={unmet}"


def test_a2_demographic_age_amid_likert_is_not_a_count_outcome(tmp_path: Path) -> None:
    """误路由 lock: a questionnaire's demographic `age` (lone count amid a Likert block, no
    continuous column) must NOT satisfy requires_count_outcome — the ordinal guard blocks the
    safety valve, so Poisson/NB/ZIP stay infeasible on bounded-rating survey data (发现1)."""
    rng = np.random.default_rng(23)
    n = 200
    # age placed BEFORE the Likert block (as in the real P1 questionnaire) so the last-numeric
    # positional heuristic does not mistake it for the outcome — the point is that a demographic
    # covariate must not open count models purely by being count-kind.
    data = {"age": rng.integers(18, 70, n)}                          # demographic count
    for i in range(1, 7):
        data[f"item{i}"] = rng.integers(1, 6, n)                     # 6 Likert items (1-5)
    df = pd.DataFrame(data)
    csv = tmp_path / "likert_plus_age.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)

    age_col = fp.column("age")
    assert age_col is not None and age_col.kind == "count", "age should profile as count"
    assert not getattr(age_col, "ordinal_like", False), "age is not a bounded rating"
    assert sum(1 for c in fp.columns if getattr(c, "ordinal_like", False)) >= 1, (
        "Likert items must profile as ordinal_like for this test to exercise the guard"
    )

    ok, unmet = check_preconditions(fp, _make_entry().preconditions)
    assert not ok, "a demographic age amid a Likert block must NOT satisfy requires_count_outcome"
    assert "需要计数型结果变量" in unmet, f"unmet reason missing; got {unmet}"
