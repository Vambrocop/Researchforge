"""Tests for the repeated-measures ANOVA executor branch (experimental_stats family)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="repeated_measures_anova", method="Repeated-measures ANOVA",
        domain="experimental design", family="experimental_stats", goal="explain",
        preconditions=Precondition(requires_group=True, min_numeric_cols=2, min_rows=6),
    )


def _run(tmp_path: Path, df: pd.DataFrame, config: dict):
    csv = tmp_path / "r.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    return run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config=config)


def test_rm_long_detects_within_effect(tmp_path: Path) -> None:
    # 24 subjects, 3 conditions with a clear monotone within-subject effect.
    rng = np.random.default_rng(0)
    rows = []
    for s in range(24):
        base = rng.normal(0, 2.0)  # subject random intercept
        for ci, cond in enumerate(["pre", "mid", "post"]):
            y = base + ci * 4.0 + rng.normal(0, 1.0)
            rows.append({"subj": s, "cond": cond, "score": y})
    df = pd.DataFrame(rows)
    res = _run(tmp_path, df, {"subject": "subj", "within": "cond", "outcome": "score"})
    assert "完成" in res.summary
    e = res.estimates
    assert e["p_value"] < 1e-6
    assert e["f_stat"] > 0
    assert 0.0 <= e["partial_eta_sq"] <= 1.0
    assert e["n_subjects"] == 24.0 and e["n_conditions"] == 3.0
    # sphericity diagnostics present and sane
    assert "gg_epsilon" in e and 0.0 < e["gg_epsilon"] <= 1.0 + 1e-9
    assert "mauchly_p" in e
    # GG-corrected p must exist and (because effect is huge) still be significant
    assert e["gg_corrected_p"] < 1e-3
    assert "rm_anova_table.csv" in res.files and "condition_means.csv" in res.files


def test_rm_null_negative_control(tmp_path: Path) -> None:
    # no within-subject condition effect -> should not reject
    rng = np.random.default_rng(11)
    rows = []
    for s in range(30):
        base = rng.normal(0, 2.0)
        for cond in ["c1", "c2", "c3"]:
            rows.append({"subj": s, "cond": cond, "score": base + rng.normal(0, 1.0)})
    df = pd.DataFrame(rows)
    res = _run(tmp_path, df, {"subject": "subj", "within": "cond", "outcome": "score"})
    assert res.estimates["p_value"] > 0.05


def test_rm_wide_format_melts(tmp_path: Path) -> None:
    # wide: one row per subject, three repeated-measure columns
    rng = np.random.default_rng(5)
    rows = []
    for s in range(20):
        base = rng.normal(0, 2.0)
        rows.append({
            "subject": s,
            "t1": base + 0.0 + rng.normal(0, 1.0),
            "t2": base + 3.0 + rng.normal(0, 1.0),
            "t3": base + 6.0 + rng.normal(0, 1.0),
        })
    df = pd.DataFrame(rows)
    res = _run(tmp_path, df, {"subject": "subject", "measures": ["t1", "t2", "t3"]})
    assert "完成" in res.summary
    assert "宽表" in res.summary
    assert res.estimates["p_value"] < 1e-3
    assert res.estimates["n_conditions"] == 3.0


def test_rm_resolver_picks_named_outcome_not_first(tmp_path: Path) -> None:
    """A decoy continuous column ('other_metric', no within-subject effect) is
    placed BEFORE 'y' — the shared resolver must still pick 'y', not cont[0]."""
    rng = np.random.default_rng(23)
    rows = []
    for s in range(24):
        base = rng.normal(0, 2.0)
        for ci, cond in enumerate(["pre", "mid", "post"]):
            rows.append({
                "other_metric": rng.normal(0, 1.0),
                "subj": s, "cond": cond,
                "y": base + ci * 4.0 + rng.normal(0, 1.0),
            })
    df = pd.DataFrame(rows)
    res = _run(tmp_path, df, {"subject": "subj", "within": "cond"})  # no "outcome" in config
    assert "完成" in res.summary
    # real y has a huge monotone within-effect -> tiny p; other_metric has none, so a
    # wrong (positional) pick would fail to reject.
    assert res.estimates["p_value"] < 1e-6


def test_rm_single_condition_degrades(tmp_path: Path) -> None:
    df = pd.DataFrame({"subj": [1, 2, 3, 4, 5, 6],
                       "cond": ["a"] * 6,
                       "score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    res = _run(tmp_path, df, {"subject": "subj", "within": "cond", "outcome": "score"})
    assert "失败" in res.summary
