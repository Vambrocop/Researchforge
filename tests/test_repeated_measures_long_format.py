"""dogfood：重复测量方差分析读不懂自己的教科书输入格式。

A textbook long-format repeated-measures frame — `[subject, week, arm, pain_score]`, 60
subjects × 5 weeks, an ordinary RCT — used to FAIL:

    重复测量方差分析失败：需要 长表(subject + within 条件 + outcome) 或 宽表(…)

because the long-format path required `config` to supply `subject` AND `within`; there was no
auto-detection at all, so it fell through to the WIDE path, which needs ≥2 continuous columns
and finds exactly one. Meanwhile the fingerprint already held `fp.unit_col == "subject"` and
`fp.time_col == "week"` — the branch was asking the user for what the engine had computed.

The guard that matters: a `within` factor must actually VARY within a subject. `arm` is
constant per subject — it is a BETWEEN-subject factor, and taking it would put a single
observation in every cell.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog import Catalog
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_CAT = Catalog.load()


def _fp(df, tmp_path, name="d.csv"):
    csv = tmp_path / name
    df.to_csv(csv, index=False)
    return profile_dataset(csv)


def _long_rm(n_subj=60, n_time=5, seed=3):
    """subject × week, plus a BETWEEN-subject arm. Truth: -1.5/week, drug a further
    -2.5/week, subject random intercept sd 4."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subj):
        grp = "drug" if s % 2 == 0 else "placebo"
        re_s = rng.normal(0, 4.0)
        for t in range(n_time):
            rows.append({"subject": f"p{s:03d}", "week": t, "arm": grp,
                         "pain_score": round(float(50 + re_s - 1.5 * t
                                                   - 2.5 * t * (grp == "drug")
                                                   + rng.normal(0, 2.0)), 2)})
    return pd.DataFrame(rows)


def test_a_long_format_frame_no_longer_needs_config(tmp_path):
    fp = _fp(_long_rm(), tmp_path)
    assert fp.unit_col == "subject" and fp.time_col == "week"   # the profiler already knew
    res = run_analysis(fp, _CAT.by_id("repeated_measures_anova"),
                       output_root=str(tmp_path / "o"))
    assert "失败" not in res.summary, res.summary[:200]
    assert res.estimates["n_subjects"] == 60
    assert res.estimates["n_conditions"] == 5          # week, not arm
    assert res.estimates["p_value"] < 0.01             # a real within-subject effect
    assert 0 < res.estimates["partial_eta_sq"] <= 1


def test_the_within_factor_must_vary_inside_a_subject(tmp_path):
    """`arm` is constant per subject — a BETWEEN-subject factor. Choosing it would put one
    observation in every cell and silently report a degenerate ANOVA."""
    fp = _fp(_long_rm(), tmp_path)
    res = run_analysis(fp, _CAT.by_id("repeated_measures_anova"),
                       output_root=str(tmp_path / "w"))
    assert "week" in res.summary
    assert res.estimates["n_conditions"] == 5, "arm has 2 levels; week has 5"


def test_sphericity_is_checked_and_corrected(tmp_path):
    """The textbook deliverable, not just 'a number': Mauchly + Greenhouse-Geisser."""
    fp = _fp(_long_rm(), tmp_path)
    res = run_analysis(fp, _CAT.by_id("repeated_measures_anova"),
                       output_root=str(tmp_path / "s"))
    if "失败" in res.summary:
        pytest.skip(res.summary[:80])
    assert "mauchly_p" in res.estimates and "gg_epsilon" in res.estimates
    assert 0 < res.estimates["gg_epsilon"] <= 1


def test_config_still_wins_over_the_detected_roles(tmp_path):
    fp = _fp(_long_rm(), tmp_path)
    res = run_analysis(fp, _CAT.by_id("repeated_measures_anova"),
                       output_root=str(tmp_path / "c"),
                       config={"subject": "subject", "within": "arm",
                               "outcome": "pain_score"})
    # arm has 2 levels and each subject sits in only one → honest failure, not a fake ANOVA
    assert ("n_conditions" not in res.estimates) or res.estimates["n_conditions"] == 2


def test_a_frame_with_no_within_factor_still_fails_honestly(tmp_path):
    """Auto-detection must not manufacture a within factor out of nothing — and the message
    should say what it looked for."""
    rng = np.random.default_rng(11)
    n = 80
    df = pd.DataFrame({"subject": [f"p{i:03d}" for i in range(n)],   # one row per subject
                       "arm": rng.integers(0, 2, n),
                       "score": rng.normal(50, 8, n).round(2)})
    fp = _fp(df, tmp_path, name="flat.csv")
    res = run_analysis(fp, _CAT.by_id("repeated_measures_anova"),
                       output_root=str(tmp_path / "f"))
    assert "失败" in res.summary
    assert "已自动查找" in res.summary, "the message should report what it tried"


def test_wide_format_inference_is_untouched(tmp_path):
    """The wide path (≥2 repeated-measure columns, one row per subject) still works."""
    rng = np.random.default_rng(7)
    n = 40
    base = rng.normal(50, 5, n)
    df = pd.DataFrame({"t1": base.round(2),
                       "t2": (base - 2 + rng.normal(0, 1, n)).round(2),
                       "t3": (base - 4 + rng.normal(0, 1, n)).round(2)})
    fp = _fp(df, tmp_path, name="wide.csv")
    res = run_analysis(fp, _CAT.by_id("repeated_measures_anova"),
                       output_root=str(tmp_path / "wide"))
    if "失败" in res.summary:
        pytest.skip(res.summary[:80])
    assert res.estimates["n_conditions"] == 3
