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


def test_regression_medium_with_structure_evidence_binds(tmp_path: Path):
    # Wave H3: TWO medium domain words (price, sales) — name alone is ambiguous, but data
    # structure corroborates 'sales' (last numeric + top explained-R² tier; note R² alone is
    # SYMMETRIC between sales and its generator adspend, so the tier + asymmetric name/position
    # signals decide) → promoted to HIGH and bound as the dependent variable.
    from researchforge.catalog.registry import Catalog
    from researchforge.executor.run import _regression

    rng = np.random.default_rng(1)
    n = 120
    a = rng.normal(0, 1, n)
    df = pd.DataFrame({"adspend": a.round(3), "price": rng.normal(0, 1, n).round(3),
                       "sales": (3 * a + rng.normal(0, 1, n)).round(3)})
    csv = tmp_path / "m.csv"; df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    assert fp.likely_outcome == "sales"
    assert fp.likely_outcome_confidence == "high"  # promoted by structural evidence
    entry = {e.id: e for e in Catalog.load().all()}["ols_regression"]
    y, rhs, _, _ = _regression(df, fp, entry, {})
    assert y == "sales" and "sales" not in rhs


def test_regression_medium_without_evidence_does_not_bind(tmp_path: Path):
    # a MEDIUM domain word with NO structural corroboration (mid-position, independent of the
    # other columns) must NOT bind — the old safety stands where evidence is absent.
    from researchforge.catalog.registry import Catalog
    from researchforge.executor.run import _regression

    rng = np.random.default_rng(2)
    n = 120
    df = pd.DataFrame({"x1": rng.normal(0, 1, n).round(3),
                       "price": rng.normal(0, 1, n).round(3),
                       "x2": rng.normal(0, 1, n).round(3)})
    csv = tmp_path / "m.csv"; df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    assert fp.likely_outcome_confidence == "medium"
    entry = {e.id: e for e in Catalog.load().all()}["ols_regression"]
    y, _, _, _ = _regression(df, fp, entry, {})
    assert y == "x1"              # first continuous, the unevidenced hint stays a hint


def test_medium_conflict_with_position_only_stays_medium(tmp_path: Path):
    # conservative: TWO medium names, evidence is position-only (no R² tier — all independent)
    # → no promotion; a name tie needs the structure signal, position alone can't break it.
    rng = np.random.default_rng(3)
    n = 120
    df = pd.DataFrame({"revenue": rng.normal(0, 1, n).round(3),
                       "x": rng.normal(0, 1, n).round(3),
                       "profit": rng.normal(0, 1, n).round(3)})
    csv = tmp_path / "m.csv"; df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    assert fp.likely_outcome_confidence == "medium"


def test_two_column_medium_kept_by_treatment_veto(tmp_path: Path):
    # with only 2 numeric columns there is no structural evidence (R² is exactly symmetric),
    # so 'condition_score' stays MEDIUM — and the G1 outcome-signal veto (not promotion) is
    # what keeps the treatment-worded DV from being skipped by the resolver.
    from researchforge.executor.run import resolve_outcome

    rng = np.random.default_rng(4)
    n = 120
    x = rng.normal(0, 1, n)
    df = pd.DataFrame({"condition_score": (2 + 1.5 * x + rng.normal(0, .5, n)).round(3),
                       "x": x.round(3)})
    csv = tmp_path / "m.csv"; df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    assert fp.likely_outcome == "condition_score"
    assert fp.likely_outcome_confidence == "medium"
    cont = [c.name for c in fp.columns if c.kind == "continuous"]
    assert resolve_outcome(fp, {}, cont) == "condition_score"


def test_medium_not_promoted_when_r2_contradicts_position(tmp_path: Path):
    # cold-review M1: a med-named NOISE column last in order must NOT bind while the true
    # (unnamed) DV carries the explained variance. Promotion requires R² structure; position
    # alone never promotes → 'price' stays medium (unbound) and the regression models the DV.
    from researchforge.executor.run import _regression
    from researchforge.catalog.registry import Catalog

    rng = np.random.default_rng(11)
    n = 150
    tenure = rng.normal(0, 1, n); promo = rng.normal(0, 1, n)
    df = pd.DataFrame({"satisfaction": (3 + 0.8 * tenure + 0.5 * promo + rng.normal(0, 0.5, n)).round(3),
                       "tenure": tenure.round(3), "promo": promo.round(3),
                       "price": rng.normal(20, 5, n).round(2)})  # noise, med-named, LAST
    csv = tmp_path / "m.csv"; df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    assert fp.likely_outcome_confidence != "high"   # price NOT promoted (R² contradicts)
    entry = {e.id: e for e in Catalog.load().all()}["ols_regression"]
    y, _, _, _ = _regression(df, fp, entry, {})
    assert y == "satisfaction"                        # the truly-explained DV, not the noise


def test_count_medium_not_promoted(tmp_path: Path):
    # cold-review M2 (honesty): a med-named COUNT column must not promote to HIGH — it would
    # make the run nudge claim a column the continuous regression can't bind and drops.
    rng = np.random.default_rng(12)
    n = 150
    a = rng.normal(0, 1, n)
    df = pd.DataFrame({"adspend": a.round(3), "foot": rng.normal(0, 1, n).round(3),
                       "sales": rng.poisson(np.exp(0.3 + 0.4 * a)).astype(int)})  # genuine count
    csv = tmp_path / "m.csv"; df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    assert fp.column("sales").kind == "count"
    assert fp.likely_outcome_confidence == "medium"   # not promoted


def test_medium_promotion_stable_on_weak_borderline_n():
    # cold-review S1: pure-noise med column near the n-floor must NOT bind on sampling luck —
    # the F-test controls false promotion (was a ~30% coin-flip with a fixed R²≥0.10 floor).
    import numpy as np
    import pandas as pd
    import tempfile
    from pathlib import Path
    from researchforge.profiler import profile_dataset

    d = Path(tempfile.mkdtemp())
    promoted = 0
    for s in range(40):
        r = np.random.default_rng(500 + s)
        m = 25
        df = pd.DataFrame({"x1": r.normal(0, 1, m).round(3), "x2": r.normal(0, 1, m).round(3),
                           "score": r.normal(0, 1, m).round(3)})  # score = pure noise
        csv = d / f"s{s}.csv"; df.to_csv(csv, index=False)
        promoted += profile_dataset(csv).likely_outcome_confidence == "high"
    assert promoted <= 6  # ≈5% type-I on 40 nulls; far below the old coin-flip


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


# ── Wave G: selection→execution coherence across outcome TYPES (binary/count/ordinal) ──

def test_resolve_outcome_skips_treatment_named():
    # fallback rule: a treatment-named candidate (treat/arm/exposed/dose…) is skipped when
    # a non-treatment candidate exists; if ALL are treatment-named, first-candidate stands.
    from researchforge.executor.run import resolve_outcome
    from researchforge.profiler.fingerprint import DataFingerprint

    fp = DataFingerprint(path="x", n_rows=100, n_cols=2, columns=[])  # no role hints
    assert resolve_outcome(fp, {}, ["treated", "died"]) == "died"
    assert resolve_outcome(fp, {}, ["exposed", "flag"]) == "flag"
    assert resolve_outcome(fp, {}, ["dose_a", "arm_b"]) == "dose_a"   # all treatment-named
    assert resolve_outcome(fp, {}, ["flag", "treated"]) == "flag"      # order kept otherwise
    assert resolve_outcome(fp, {"outcome": "treated"}, ["treated", "died"]) == "treated"  # config wins


def _rp_fp(**overrides):
    cols = [
        ColumnInfo(name="unit", kind="id", dtype="object", n_missing=0, n_unique=20),
        ColumnInfo(name="time", kind="datetime", dtype="object", n_missing=0, n_unique=5),
        ColumnInfo(name="y", kind="continuous", dtype="float64", n_missing=0, n_unique=50),
        ColumnInfo(name="x1", kind="continuous", dtype="float64", n_missing=0, n_unique=50),
        ColumnInfo(name="x2", kind="count", dtype="int64", n_missing=0, n_unique=10),
        ColumnInfo(name="x3", kind="binary", dtype="int64", n_missing=0, n_unique=2),
        ColumnInfo(name="x4", kind="categorical", dtype="object", n_missing=0, n_unique=4),
    ]
    kwargs = dict(path="x", n_rows=100, n_cols=len(cols), columns=cols,
                  unit_col="unit", time_col="time")
    kwargs.update(overrides)
    from researchforge.profiler.fingerprint import DataFingerprint

    return DataFingerprint(**kwargs)


def test_resolve_predictors_forced_wins_and_filters():
    # explicit config["predictors"] wins over auto-selection; filtered to columns that
    # actually exist and excludes the outcome even if the user listed it by mistake.
    from researchforge.executor.run import resolve_predictors

    fp = _rp_fp()
    df_cols = ["unit", "time", "y", "x1", "x2", "x3", "x4"]
    preds = resolve_predictors(
        fp, {"predictors": ["x3", "y", "bogus", "x1"]}, "y", df=pd.DataFrame(columns=df_cols)
    )
    assert preds == ["x3", "x1"]   # 'y' (outcome) and 'bogus' (not a real column) dropped


def test_resolve_predictors_auto_excludes_outcome_and_unit_time():
    # with no config override, predictors = numeric/binary columns in fp order, excluding
    # the outcome and unit/time role columns (never modeled as predictors).
    from researchforge.executor.run import resolve_predictors

    fp = _rp_fp()
    preds = resolve_predictors(fp, {}, "y")
    assert preds == ["x1", "x2", "x3"]   # unit/time/y excluded; categorical x4 not in kinds


def test_resolve_predictors_cap_honored():
    # cap truncates both the forced and the auto path (regression's own convention: forced
    # up to 8, auto up to 5 — this test exercises the generic `cap` knob directly).
    from researchforge.executor.run import resolve_predictors

    fp = _rp_fp()
    assert resolve_predictors(fp, {}, "y", cap=2) == ["x1", "x2"]
    df_cols = ["unit", "time", "y", "x1", "x2", "x3", "x4"]
    forced_preds = resolve_predictors(
        fp, {"predictors": ["x1", "x2", "x3"]}, "y", cap=5, forced_cap=2,
        df=pd.DataFrame(columns=df_cols),
    )
    assert forced_preds == ["x1", "x2"]


def test_logistic_binds_event_outcome_over_leading_treatment(tmp_path: Path):
    # {treated, age, dose, died}: 'died' is the high-confidence binary outcome; the leading
    # 'treated' flag must NOT be modeled as the dependent variable (the old first-binary grab).
    from researchforge.catalog.registry import Catalog

    rng = np.random.default_rng(0)
    n = 200
    treated = rng.binomial(1, 0.5, n)
    age = rng.normal(60, 10, n).round(1)
    p = 1 / (1 + np.exp(-(-1 + 1.2 * treated + 0.03 * (age - 60))))
    df = pd.DataFrame({"treated": treated, "age": age,
                       "dose": rng.normal(5, 1, n).round(2), "died": rng.binomial(1, p)})
    csv = tmp_path / "t.csv"; df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    assert fp.likely_outcome == "died" and fp.likely_outcome_confidence == "high"
    entry = {e.id: e for e in Catalog.load().all()}["logistic_regression"]
    res = run_analysis(fp, entry, output_root=str(tmp_path / "o"))
    assert "结果变量 died" in res.summary
    # config override still wins over the role binding
    res2 = run_analysis(fp, entry, output_root=str(tmp_path / "o2"), config={"outcome": "treated"})
    assert "结果变量 treated" in res2.summary


def test_logistic_skips_treatment_named_without_high_confidence(tmp_path: Path):
    # {exposed, x, flag}: no high-confidence outcome name fires, but 'exposed' is
    # treatment-named → the resolver falls back to 'flag', not the leading treatment column.
    from researchforge.catalog.registry import Catalog

    rng = np.random.default_rng(3)
    n = 200
    exposed = rng.binomial(1, 0.5, n)
    flag = rng.binomial(1, 1 / (1 + np.exp(-0.8 * (exposed - 0.5))))
    df = pd.DataFrame({"exposed": exposed, "x": rng.normal(0, 1, n).round(3), "flag": flag})
    csv = tmp_path / "t.csv"; df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    entry = {e.id: e for e in Catalog.load().all()}["logistic_regression"]
    res = run_analysis(fp, entry, output_root=str(tmp_path / "o"))
    assert "结果变量 flag" in res.summary


def test_poisson_binds_high_confidence_count_outcome(tmp_path: Path):
    # {events, x1, y}: both 'events' and 'y' are count columns; 'y' is the high-confidence
    # outcome name and NOT first — the count family must model y, not grab events.
    from researchforge.catalog.registry import Catalog

    rng = np.random.default_rng(4)
    n = 220
    x1 = rng.normal(0, 1, n)
    events = rng.poisson(3, n)                       # a count PREDICTOR (first count col)
    y = rng.poisson(np.exp(0.4 + 0.3 * x1))          # the named count outcome
    df = pd.DataFrame({"events": events, "x1": x1.round(3), "y": y})
    csv = tmp_path / "p.csv"; df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    assert fp.likely_outcome == "y" and fp.likely_outcome_confidence == "high"
    entry = {e.id: e for e in Catalog.load().all()}["poisson_regression"]
    res = run_analysis(fp, entry, output_root=str(tmp_path / "o"))
    assert "计数结果 y" in res.summary  # exact summary line, not an incidental substring


def test_is_treatment_named_word_boundaries():
    # word-boundary behavior: segments match (group_size), embeddings don't (grouping).
    from researchforge.profiler.roles import is_treatment_named

    assert is_treatment_named("treated")
    assert is_treatment_named("group_size")           # 'group' as a segment
    assert is_treatment_named("condition_score")      # 'condition' as a segment
    assert not is_treatment_named("grouping")          # embedded, no boundary
    assert not is_treatment_named("programming_score") # 'program' embedded in 'programming'
    assert not is_treatment_named("blood_pressure")


def test_resolve_outcome_rescues_outcome_signal(tmp_path: Path):
    # M1 regression (cold review): a compound DV name carrying a treatment word as a
    # segment ('body_condition_score' — standard animal-science outcome, first column per
    # convention) must NOT be skipped by the treatment-name rule: the role detector flags
    # it as the likely outcome (medium via 'score'), and that outcome signal vetoes the
    # skip. Old first-column behavior stands.
    from researchforge.executor.run import _regression, resolve_outcome
    from researchforge.catalog.registry import Catalog

    rng = np.random.default_rng(6)
    n = 150
    age = rng.uniform(2, 12, n)
    weight = rng.normal(500, 60, n)
    bcs = (3 + 0.1 * age - 0.002 * (weight - 500) + rng.normal(0, 0.3, n))
    df = pd.DataFrame({"body_condition_score": bcs.round(2), "age": age.round(1),
                       "weight": weight.round(1)})
    csv = tmp_path / "bcs.csv"; df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    assert fp.likely_outcome == "body_condition_score"  # medium, via 'score'
    cont = [c.name for c in fp.columns if c.kind == "continuous"]
    assert resolve_outcome(fp, {}, cont) == "body_condition_score"  # veto: not skipped
    entry = {e.id: e for e in Catalog.load().all()}["ols_regression"]
    y, _, _, _ = _regression(df, fp, entry, {})
    assert y == "body_condition_score"


def test_poisson_config_can_force_any_column(tmp_path: Path):
    # count-family config contract (cold review S1): config['outcome'] may force a column
    # the profiler did NOT tag as count (the id-trap: all-unique ints profile as 'id') —
    # aligned with count_models._resolve_count_outcome's wider check.
    from researchforge.catalog.registry import Catalog

    rng = np.random.default_rng(7)
    n = 220
    x1 = rng.normal(0, 1, n)
    base = rng.poisson(5, n)
    df = pd.DataFrame({"visits": base, "x1": x1.round(3),
                       "case_no": np.arange(1000, 1000 + n)})  # all-unique ints -> 'id'
    csv = tmp_path / "f.csv"; df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    assert fp.column("case_no").kind == "id"
    entry = {e.id: e for e in Catalog.load().all()}["poisson_regression"]
    res = run_analysis(fp, entry, output_root=str(tmp_path / "o"),
                       config={"outcome": "case_no"})
    assert "计数结果 case_no" in res.summary
