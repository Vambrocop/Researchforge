"""Semantic role-hint detection (smarter auto-selection, v1.x).

The hints are NON-BINDING: they don't change run-time defaults, only suggest a
config. Tests cover name-based + position-based outcome detection, treatment/time
hints, fingerprint wiring, and the (gated, additive) run-time nudge.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, ParamSpec, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset
from researchforge.profiler.fingerprint import ColumnInfo
from researchforge.profiler.roles import detect_roles


def _col(name, kind):
    return ColumnInfo(name=name, kind=kind, dtype="float64", n_missing=0, n_unique=10)


def test_outcome_by_name():
    cols = [_col("x1", "continuous"), _col("target", "continuous"), _col("x2", "continuous")]
    roles = detect_roles(cols)
    assert roles["likely_outcome"] == "target"
    assert "name" in roles["reason"]


def test_outcome_by_position_when_no_name():
    # last numeric column, >=3 numeric, no name signal -> position heuristic
    cols = [_col("a", "continuous"), _col("b", "continuous"), _col("c", "count")]
    roles = detect_roles(cols)
    assert roles["likely_outcome"] == "c"
    assert "last numeric" in roles["reason"]


def test_no_outcome_when_too_few_numeric():
    cols = [_col("a", "continuous"), _col("g", "categorical")]
    roles = detect_roles(cols)
    assert roles["likely_outcome"] is None


def test_binary_outcome_beats_name_matched_predictor():
    # 'approved' (binary target) must win over 'score' (a continuous predictor that
    # merely name-matches the outcome pattern) — binary-outcome detection runs first.
    cols = [_col("approved", "binary"), _col("income", "continuous"),
            _col("age", "continuous"), _col("score", "continuous")]
    roles = detect_roles(cols)
    assert roles["likely_outcome"] == "approved"
    assert "binary" in roles["reason"]
    # the binary target is not also flagged as the treatment
    assert roles["likely_treatment"] != "approved"


def test_demographic_binary_not_mistaken_for_outcome():
    # a plain demographic binary (gender) is NOT an outcome name -> no false positive
    cols = [_col("gender", "binary"), _col("income", "continuous")]
    roles = detect_roles(cols)
    assert roles["likely_outcome"] is None


def test_group_binary_not_outcome_continuous_is():
    # group(binary) + outcome(continuous): the continuous outcome wins, group stays a group
    cols = [_col("group", "binary"), _col("outcome", "continuous")]
    roles = detect_roles(cols)
    assert roles["likely_outcome"] == "outcome"


def test_survival_event_not_mistaken_for_classification_outcome():
    # in time-to-event data 'event' is a censoring indicator, NOT a classification target,
    # so it must not be auto-picked as the likely outcome
    cols = [_col("duration", "continuous"), _col("event", "binary"), _col("age", "continuous")]
    roles = detect_roles(cols)
    assert roles["likely_outcome"] != "event"


def test_outcome_confidence_high_for_unambiguous_name():
    # an unambiguous DV name ('target') → HIGH confidence (safe to bind execution to it)
    cols = [_col("x1", "continuous"), _col("target", "continuous"), _col("x2", "continuous")]
    roles = detect_roles(cols)
    assert roles["likely_outcome"] == "target"
    assert roles["likely_outcome_confidence"] == "high"


def test_outcome_confidence_medium_for_domain_word():
    # a domain word ('price') that could just as well be a predictor → MEDIUM (hint only,
    # must NOT bind — here 'price' is a feature and 'sales' is the real outcome, but by name
    # alone that is unknowable, so detection stays non-binding rather than guess wrong).
    cols = [_col("adspend", "continuous"), _col("price", "continuous"), _col("sales", "continuous")]
    roles = detect_roles(cols)
    assert roles["likely_outcome_confidence"] == "medium"


def test_outcome_confidence_low_for_position():
    cols = [_col("a", "continuous"), _col("b", "continuous"), _col("c", "count")]
    roles = detect_roles(cols)
    assert roles["likely_outcome"] == "c"
    assert roles["likely_outcome_confidence"] == "low"


def test_binary_outcome_is_high_confidence():
    cols = [_col("approved", "binary"), _col("income", "continuous"), _col("score", "continuous")]
    roles = detect_roles(cols)
    assert roles["likely_outcome"] == "approved"
    assert roles["likely_outcome_confidence"] == "high"


def test_treatment_and_time_hints():
    cols = [_col("treated", "binary"), _col("year", "count"), _col("y", "continuous")]
    roles = detect_roles(cols)
    assert roles["likely_treatment"] == "treated"
    assert roles["likely_time"] == "year"


def test_fingerprint_carries_hint_diabetes_like(tmp_path: Path):
    # diabetes-shaped: many continuous features + an integer target last (the real
    # e2e finding — progression profiles as count and was missed as the outcome).
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({f"x{i}": rng.normal(0, 1, n) for i in range(6)})
    df["progression"] = rng.integers(25, 320, n)  # integer target -> count kind
    csv = tmp_path / "diab.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    assert fp.likely_outcome == "progression"     # caught by position heuristic
    assert fp.column("progression").kind == "count"


def test_run_nudge_appears_for_outcome_method_without_config(tmp_path: Path):
    df = pd.DataFrame({"a": range(40), "b": [i * 1.3 for i in range(40)],
                       "outcome_score": [i * 0.5 + 3 for i in range(40)]})
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    assert fp.likely_outcome == "outcome_score"   # name match
    entry = AnalysisEntry(
        id="x", method="m", domain="d", family="regression", goal="explain",
        params=[ParamSpec(name="outcome", type="column")],
        preconditions=Precondition(),
    )
    # no handler registered for "x" -> placeholder, but the nudge is added in setup
    res = run_analysis(fp, entry, output_root=str(tmp_path / "o"))
    assert "💡" in res.summary and "outcome_score" in res.summary


def test_run_no_nudge_when_outcome_configured(tmp_path: Path):
    df = pd.DataFrame({"a": range(40), "b": [i * 1.3 for i in range(40)],
                       "target": [i * 0.5 for i in range(40)]})
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    entry = AnalysisEntry(
        id="x", method="m", domain="d", family="regression", goal="explain",
        params=[ParamSpec(name="outcome", type="column")],
    )
    res = run_analysis(fp, entry, output_root=str(tmp_path / "o"), config={"outcome": "target"})
    assert "💡" not in res.summary


def test_regression_binds_high_confidence_outcome(tmp_path: Path):
    # execution coherence: a high-confidence named outcome ('target') that is NOT the first
    # continuous column must be the modeled dependent variable — not a right-hand predictor.
    from researchforge.catalog.registry import Catalog
    from researchforge.executor.run import _regression

    rng = np.random.default_rng(0)
    n = 120
    x1 = rng.normal(0, 1, n); x2 = rng.normal(0, 1, n)
    df = pd.DataFrame({"x1": x1.round(3), "x2": x2.round(3),
                       "target": (2 + 1.5 * x1 - 0.8 * x2 + rng.normal(0, 0.5, n)).round(3)})
    csv = tmp_path / "m.csv"; df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    assert fp.likely_outcome_confidence == "high"
    entry = {e.id: e for e in Catalog.load().all()}["ols_regression"]
    y, rhs, formula, _ = _regression(df, fp, entry, {})
    assert y == "target"          # bound, not cont[0]=x1
    assert "target" not in rhs    # and not also a predictor


def test_regression_medium_confidence_does_not_bind(tmp_path: Path):
    # a MEDIUM domain word ('sales') must NOT override the first-continuous default (it could
    # be a predictor) — safety: never model the wrong column on an ambiguous name.
    from researchforge.catalog.registry import Catalog
    from researchforge.executor.run import _regression

    rng = np.random.default_rng(1)
    n = 120
    a = rng.normal(0, 1, n)
    df = pd.DataFrame({"adspend": a.round(3), "price": rng.normal(0, 1, n).round(3),
                       "sales": (3 * a + rng.normal(0, 1, n)).round(3)})
    csv = tmp_path / "m.csv"; df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    assert fp.likely_outcome_confidence == "medium"
    entry = {e.id: e for e in Catalog.load().all()}["ols_regression"]
    y, _, _, _ = _regression(df, fp, entry, {})
    assert y == "adspend"         # first continuous, NOT the medium 'sales' hint


def test_config_outcome_overrides_detection(tmp_path: Path):
    from researchforge.catalog.registry import Catalog
    from researchforge.executor.run import _regression

    rng = np.random.default_rng(2)
    n = 80
    df = pd.DataFrame({"x1": rng.normal(0, 1, n).round(3),
                       "target": rng.normal(0, 1, n).round(3)})
    csv = tmp_path / "m.csv"; df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    entry = {e.id: e for e in Catalog.load().all()}["ols_regression"]
    y, _, _, _ = _regression(df, fp, entry, {"outcome": "x1"})
    assert y == "x1"              # explicit config beats the high-confidence 'target'


def test_run_no_nudge_when_method_has_no_outcome_param(tmp_path: Path):
    df = pd.DataFrame({"a": range(40), "b": [i * 1.3 for i in range(40)], "target": range(40)})
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    entry = AnalysisEntry(id="x", method="m", domain="d", family="statistics", goal="describe")
    res = run_analysis(fp, entry, output_root=str(tmp_path / "o"))
    assert "💡" not in res.summary
