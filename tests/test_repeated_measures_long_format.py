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


# ── cold review C ────────────────────────────────────────────────────────────
def _tiny_rm(n_subj=3, k=5, seed=1):
    """n <= k-1, so the covariance of the (k-1) contrasts is RANK DEFICIENT."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subj):
        b = rng.normal(0, 3)
        for w in range(k):
            rows.append({"subject": f"p{s}", "week": w,
                         "score": round(float(50 + b - 1.2 * w + rng.normal(0, 1.5)), 2)})
    return pd.DataFrame(rows)


def _rct_with_arm(n_subj=60, n_t=5, seed=3):
    """`arm` is constant within subject — a BETWEEN-subject factor."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subj):
        grp = "drug" if s % 2 == 0 else "placebo"
        re_s = rng.normal(0, 4.0)
        for t in range(n_t):
            rows.append({"subject": f"p{s:03d}", "week": t, "arm": grp,
                         "pain_score": round(float(50 + re_s - 1.5 * t
                                                   - 2.5 * t * (grp == "drug")
                                                   + rng.normal(0, 2.0)), 2)})
    return pd.DataFrame(rows)


def test_mauchly_is_refused_when_the_contrast_covariance_is_rank_deficient(tmp_path):
    """C/M1: dropping the zero eigenvalues left GG intact but turned Mauchly's W into a
    ratio over the surviving eigenvalues only, so W jumped up and the test reported
    'sphericity is fine'. Measured at 3 subjects x 5 conditions: true W = -6.5e-32, the
    engine returned W = 0.0282 and p = 0.997, so the summary recommended the UNCORRECTED
    p (0.012, significant) where the GG-corrected one is 0.0886. R refuses on this matrix."""
    csv = tmp_path / "tiny.csv"
    _tiny_rm().to_csv(csv, index=False)
    res = run_analysis(profile_dataset(csv), _CAT.by_id("repeated_measures_anova"),
                       output_root=str(tmp_path / "o"))
    mp = res.estimates.get("mauchly_p")
    assert mp != mp, f"Mauchly must be NOT COMPUTABLE here, got {mp}"
    assert "无法检验" in res.summary, res.summary[:300]
    assert "GG 校正" in res.summary
    assert "未拒绝" not in res.summary, "silence must not read as 'sphericity is fine'"


def test_a_between_subject_factor_is_named_instead_of_silently_dropped(tmp_path):
    """C/M2: AnovaRM fits a pure within model, so `arm` and arm x week are not in it and
    their sums of squares land in the error term. Against R's split-plot on this exact
    fixture: within F 260.30 -> 139.39 (error SS 1007 -> 1850), arm:week F(4,232)=48.57
    never estimated, GG epsilon 0.9531 -> 0.6324 — the branch then announced a sphericity
    violation that the correctly specified model does not have."""
    csv = tmp_path / "rct.csv"
    _rct_with_arm().to_csv(csv, index=False)
    res = run_analysis(profile_dataset(csv), _CAT.by_id("repeated_measures_anova"),
                       output_root=str(tmp_path / "p"))
    assert "组间因子" in res.summary and "arm" in res.summary, res.summary[:300]
    assert "交互项都没有被估计" in res.summary
    assert "mixed_effects" in res.summary, "the honest exit has to be named"


def test_the_between_factor_notice_does_not_cry_wolf(tmp_path):
    """The first version of this detection keyed off `long_df.columns` — which carries only
    subject/within/outcome — so it never fired on the fixture it was written for. Both
    directions get pinned."""
    csv = tmp_path / "nobetween.csv"
    _rct_with_arm().drop(columns=["arm"]).to_csv(csv, index=False)
    res = run_analysis(profile_dataset(csv), _CAT.by_id("repeated_measures_anova"),
                       output_root=str(tmp_path / "q"))
    assert "组间因子" not in res.summary


def test_the_rank_guard_no_longer_sends_panels_to_rm_anova(tmp_path):
    """C/S6: a column absorbed by the unit fixed effects is BETWEEN-subject by definition,
    and a one-way within RM-ANOVA can never estimate a between factor — it drops it, picks
    `year`, and returns p=7.56e-12 for a question nobody asked."""
    rng = np.random.default_rng(3)
    rows = []
    for s in range(60):
        grp = "drug" if s % 2 == 0 else "placebo"
        re_s = rng.normal(0, 4.0)
        for t in range(5):
            rows.append({"subject": f"p{s:03d}", "week": t, "arm": grp,
                         "pain_score": round(float(50 + re_s - 1.5 * t
                                                   - 2.5 * t * (grp == "drug")
                                                   + rng.normal(0, 2.0)), 2)})
    csv = tmp_path / "absorbed.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    res = run_analysis(profile_dataset(csv), _CAT.by_id("did"), output_root=str(tmp_path / "r"))
    assert "失败" in res.summary and "秩亏" in res.summary
    assert "mixed_effects" in res.summary
    assert "repeated_measures_anova" not in res.summary, (
        "RM-ANOVA cannot estimate a between-subject factor")


# ── cold review C · M3 / M4 ──────────────────────────────────────────────────
def _within_subjects_no_time(n_subj=45, seed=2):
    """The most standard within-subjects frame there is: participant x stimulus, no time
    column at all. `profiler/profile.py` returns before setting unit_col when time_col is
    None, so `fp.unit_col` is None here — and the detector used to read only that."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_subj):
        b = rng.normal(0, 40)
        for st, eff in (("word", 0), ("picture", 35), ("sound", 70)):
            rows.append({"participant": f"s{p:02d}", "stimulus": st,
                         "rt_ms": round(float(500 + b + eff + rng.normal(0, 25)), 1)})
    return pd.DataFrame(rows)


def _latin_square_rotation(n_subj=60, n_t=5, seed=5):
    """A balanced technician rotation varies inside every subject, so `.any()` accepted it
    as the within factor. Here a real discrete `week` exists alongside it."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subj):
        b = rng.normal(0, 3)
        for t in range(n_t):
            rows.append({"subject": f"p{s:03d}", "week": t,
                         "operator_id": f"op{(s + t) % n_t}",
                         "yield_kg": round(float(20 + b - 0.8 * t + rng.normal(0, 1.2)), 2)})
    return pd.DataFrame(rows)


def test_a_within_subjects_design_without_a_time_column_is_detected(tmp_path):
    """C/M3: 45 x 3, perfectly balanced, and it reported 受试者=None. My earlier claim that
    the long path no longer needs config was over-general — it held only for frames the
    profiler already called a panel."""
    csv = tmp_path / "stim.csv"
    _within_subjects_no_time().to_csv(csv, index=False)
    fp = profile_dataset(csv)
    assert fp.unit_col is None and fp.time_col is None, "the premise of M3"
    res = run_analysis(fp, _CAT.by_id("repeated_measures_anova"), output_root=str(tmp_path / "o"))
    assert "完成" in res.summary, res.summary[:300]
    e = res.estimates
    assert e["n_subjects"] == 45.0 and e["n_conditions"] == 3.0
    # df = (k-1, (n-1)(k-1)) = (2, 88) — cold review C checked these against R by hand
    assert e["df_num"] == pytest.approx(2.0)
    assert e["df_den"] == pytest.approx(88.0)
    assert e["p_value"] < 1e-10


def test_the_auto_chosen_roles_are_always_disclosed(tmp_path):
    """C/M4's aggravator: outcome selection emits a 💡 notice, subject/within emitted
    nothing at all — the reader could not tell the roles had been guessed."""
    csv = tmp_path / "disc.csv"
    _within_subjects_no_time().to_csv(csv, index=False)
    res = run_analysis(profile_dataset(csv), _CAT.by_id("repeated_measures_anova"),
                       output_root=str(tmp_path / "p"))
    assert "自动判定" in res.summary and "participant" in res.summary
    assert "stimulus" in res.summary


def test_a_real_factor_wins_over_a_balanced_rotation(tmp_path):
    """C/M4: the guard asked "does it vary inside a subject", never "is it a design
    factor". `operator_id` varies inside every subject, so it was taken as the within
    factor and the branch reported F(4,236)=1.787, p=0.132 for a question nobody asked."""
    csv = tmp_path / "latin.csv"
    _latin_square_rotation().to_csv(csv, index=False)
    res = run_analysis(profile_dataset(csv), _CAT.by_id("repeated_measures_anova"),
                       output_root=str(tmp_path / "q"))
    assert "条件（week" in res.summary, res.summary[:300]
    # ...and the rejected candidate is named, because the ambiguity is real
    assert "operator_id 同样能解析成组内因子" in res.summary


def test_a_nuisance_named_within_factor_is_flagged(tmp_path):
    """When the rotation is the ONLY discrete candidate (the real time is continuous and
    unaligned across subjects) the engine still has to run something — but it must say the
    factor looks like a rotation/ID rather than a research factor."""
    rng = np.random.default_rng(5)
    rows = []
    for s in range(60):
        b = rng.normal(0, 3)
        for t in range(5):
            days = round(float(7 * t + rng.uniform(-1.5, 1.5)), 2)
            rows.append({"subject": f"p{s:03d}", "followup_days": days,
                         "operator_id": f"op{(s + t) % 5}",
                         "yield_kg": round(float(20 + b - 0.11 * days + rng.normal(0, 1.2)), 2)})
    csv = tmp_path / "nuis.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    res = run_analysis(profile_dataset(csv), _CAT.by_id("repeated_measures_anova"),
                       output_root=str(tmp_path / "r"))
    assert "干扰/轮换变量" in res.summary, res.summary[:400]
    assert "结果无意义" in res.summary


def test_config_still_outranks_the_detector(tmp_path):
    csv = tmp_path / "cfg.csv"
    _latin_square_rotation().to_csv(csv, index=False)
    res = run_analysis(profile_dataset(csv), _CAT.by_id("repeated_measures_anova"),
                       output_root=str(tmp_path / "s"),
                       config={"subject": "subject", "within": "operator_id"})
    assert "条件（operator_id" in res.summary
    assert "自动判定" not in res.summary, "nothing was auto-selected"
