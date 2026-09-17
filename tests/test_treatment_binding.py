"""Wave H4d — the TREATMENT side of role binding (the twin of H4/H4b/H4c's outcome side).

``fp.binary_columns`` is literally *every binary column* (profiler/profile.py), so the
family-wide idiom

    treatment = cfg["treatment"] else fp.binary_columns[0]

meant "the first binary column in file order" with no role signal whatsoever — while
``roles.py`` had already computed a NAME-aware ``fp.likely_treatment`` that nobody read.

On a perfectly ordinary survival frame ``[duration, event, treatment, age, biomarker]`` that
made PSM match on the **survival event indicator** as if it were the intervention, with a
column literally named ``treatment`` sitting right next to it. The damage was not cosmetic:
the truth is ``duration ~ Exp(20·exp(-0.4·treatment + …))``, i.e. treatment SHORTENS duration
by about 6.7, and the engine reported ATT = **+4.64, p = 0.16 (not significant)** — wrong
sign, effect missed entirely. With the name signal it reports ATT ≈ −8.7, p ≈ 0.0003.

``resolve_treatment`` (executor/_helpers/core.py) is the shared fix, and it records what it
bound on ``RunResult.treatment`` so this is auditable the way outcomes became auditable in H4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog import Catalog
from researchforge.executor import run_analysis
from researchforge.executor.run import _pick_did_treatment, resolve_treatment
from researchforge.profiler import profile_dataset

_CAT = Catalog.load()
# 判据必须是**降级标记**，不是散文里的词。原先用裸子串 "失败"/"跳过" 判降级，而同一波新增的披露文案里就有「事件率饱和时甚至直接**失败**」——旗舰 survival frame 上 psm/ipw/aipw/rosenbaum 的 _ran() 因此恒为 False，10 个断言集体休眠、棘轮短路（冷审 A#12）。降级消息一律是「<方法>失败：」/「<方法>跳过：」的冒号形式。
_DEGRADED = ("失败：", "跳过：", "暂未接入", "未检测到")

# the causal branches that resolve a treatment through the shared helper
_WIRED = ["psm", "ipw", "aipw", "rosenbaum_bounds", "evalue"]


def _fp(df, tmp_path, name="d.csv"):
    csv = tmp_path / name
    df.to_csv(csv, index=False)
    return profile_dataset(csv)


def _ran(res) -> bool:
    return bool(res.estimates) and not any(k in res.summary for k in _DEGRADED)


def _survival_frame(n=300, seed=2):
    """`event` comes BEFORE `treatment` in file order, so the old positional pick lands on
    the survival event indicator. Treatment shortens duration (true ATT ≈ -6.7)."""
    rng = np.random.default_rng(seed)
    age = rng.integers(30, 80, n)
    trt = rng.integers(0, 2, n)
    dur = np.round(rng.exponential(20 * np.exp(-0.4 * trt + 0.01 * (age - 50))), 2)
    return pd.DataFrame({"duration": dur,
                         "event": (rng.random(n) < 0.7).astype(int),
                         "treatment": trt,
                         "age": age,
                         "biomarker": rng.normal(5, 1.5, n).round(3)})


# ── the resolver itself ──────────────────────────────────────────────────────
def test_resolver_prefers_a_treatment_named_candidate(tmp_path):
    fp = _fp(_survival_frame(), tmp_path)
    assert fp.binary_columns == ["event", "treatment"]      # every binary, file order
    assert resolve_treatment(fp, {}, fp.binary_columns) == "treatment"


def test_resolver_honors_explicit_config(tmp_path):
    fp = _fp(_survival_frame(), tmp_path)
    assert resolve_treatment(fp, {"treatment": "event"}, fp.binary_columns) == "event"


def test_resolver_falls_back_to_column_order_without_a_signal(tmp_path):
    """No treatment-named candidate → the long-standing positional default is preserved."""
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"flag_a": rng.integers(0, 2, 120),
                       "flag_b": rng.integers(0, 2, 120),
                       "value": rng.normal(0, 1, 120).round(3)})
    fp = _fp(df, tmp_path)
    assert resolve_treatment(fp, {}, ["flag_a", "flag_b"]) in {"flag_a", "flag_b"}


def test_resolver_returns_none_on_no_candidates(tmp_path):
    fp = _fp(_survival_frame(), tmp_path)
    assert resolve_treatment(fp, {}, []) is None


# ── the branches ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("cid", _WIRED)
def test_causal_family_binds_the_named_treatment(cid, tmp_path):
    fp = _fp(_survival_frame(), tmp_path)
    entry = _CAT.by_id(cid)
    if entry is None:
        pytest.skip(f"{cid} not in catalog")
    res = run_analysis(fp, entry, output_root=str(tmp_path / f"o_{cid}"))
    if not _ran(res):
        pytest.skip(f"{cid} degraded: {res.summary[:60]}")
    assert res.treatment == "treatment", (
        f"{cid} treated {res.treatment!r} as the intervention "
        "(first-binary fallback would give 'event')"
    )


def test_psm_recovers_the_real_effect_not_the_event_contrast(tmp_path):
    """The regression this wave exists for. Matching on `event` gave ATT=+4.64 (p=0.16):
    wrong sign, effect missed. The truth is a large NEGATIVE ATT (~-6.7)."""
    fp = _fp(_survival_frame(), tmp_path)
    res = run_analysis(fp, _CAT.by_id("psm"), output_root=str(tmp_path / "psm"))
    if not _ran(res):
        pytest.skip(f"psm degraded: {res.summary[:60]}")
    att, se = res.estimates["att"], res.estimates["se"]
    assert att < 0, f"ATT should be negative (treatment shortens duration), got {att:.3f}"
    assert res.estimates["pvalue"] < 0.05, "the real effect is large; it must be detected"
    assert abs(att - (-6.7)) < 3 * se, f"ATT {att:.2f} ± {se:.2f} should cover the truth ≈ -6.7"
    # and the matched sample must be the ASSIGNMENT split (~50%), not the event rate (~70%)
    assert 0.35 < res.estimates["n_treated"] / 300 < 0.60


def test_config_treatment_still_wins_end_to_end(tmp_path):
    fp = _fp(_survival_frame(), tmp_path)
    res = run_analysis(fp, _CAT.by_id("psm"), output_root=str(tmp_path / "cfg"),
                       config={"treatment": "event"})
    if not _ran(res):
        pytest.skip("psm degraded")
    assert res.treatment == "event"


def test_wired_branches_never_run_without_reporting_their_treatment(tmp_path):
    """Ratchet: RunResult.treatment is the audit instrument for the treatment side."""
    fp = _fp(_survival_frame(seed=9), tmp_path)
    silent = []
    for cid in _WIRED:
        entry = _CAT.by_id(cid)
        if entry is None:
            continue
        res = run_analysis(fp, entry, output_root=str(tmp_path / f"r_{cid}"))
        if _ran(res) and res.treatment is None:
            silent.append(cid)
    assert not silent, f"ran but reported no treatment: {silent}"


# ═════════════════════════════════════════════════════════════════════════════
# Cold-review regressions (2026-09-08). Each of these reproduces a scenario where
# the FIRST version of this wave was worse than the code it replaced, plus the
# tier-discriminating cases the original 12 tests could not distinguish.
# ═════════════════════════════════════════════════════════════════════════════
def _staggered_panel(n_unit=30, n_t=14, seed=11):
    """Ordinary staggered adoption, true ATT = +3.0, with a time-invariant NAME-matched
    decoy (`region_group`). The DiD estimand is built from onset = min(time|D==1) per unit,
    so a flag that never switches has no onset — the reviewer measured staggered_did going
    +3.497 -> -0.332 (p=0.033, wrong sign) and five sibling methods degrading to a skip."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_unit):
        fe = rng.normal(0, 1)
        g = 6 if u % 3 == 0 else (10 if u % 3 == 1 else 10 ** 9)
        for t in range(n_t):
            d = 1 if t >= g else 0
            rows.append({"unit": f"u{u:02d}", "year": 2005 + t,
                         "y": round(float(10 + fe + 3.0 * d + 0.1 * t + rng.normal(0, 0.6)), 3),
                         "adopted": d, "region_group": int(u % 2)})
    return pd.DataFrame(rows)


def _named_covariate_frame(n=400, seed=5):
    """`age_group` matched the WEAK treatment regex (the `(?:^|_|\b)` alternation contains a
    literal `_`, so `age_group` matched `group`). Harmless while it only made resolve_outcome
    SKIP a column; once it could BIND, PSM reported ATT=+3.11 (p=0.008) for a covariate on
    data whose true ATT is -8."""
    rng = np.random.default_rng(seed)
    age_group = rng.integers(0, 2, n)
    bmi = rng.normal(26, 4, n)
    vacc = (rng.random(n) < 1 / (1 + np.exp(-(0.4 * age_group)))).astype(int)
    sev = 50 - 8 * vacc + 3 * age_group + 0.5 * bmi + rng.normal(0, 5, n)
    return pd.DataFrame({"severity_index": sev.round(2), "vaccinated": vacc,
                         "age_group": age_group, "bmi": bmi.round(2)})


@pytest.mark.parametrize("cid", ["staggered_did", "event_study", "callaway_santanna",
                                 "goodman_bacon", "honest_did", "chaisemartin_did"])
def test_panel_did_binds_the_switching_indicator_not_a_named_flag(cid, tmp_path):
    fp = _fp(_staggered_panel(), tmp_path)
    entry = _CAT.by_id(cid)
    if entry is None:
        pytest.skip(f"{cid} not in catalog")
    res = run_analysis(fp, entry, output_root=str(tmp_path / f"did_{cid}"))
    if not _ran(res):
        pytest.skip(f"{cid} degraded: {res.summary[:70]}")
    assert res.treatment == "adopted", (
        f"{cid} bound {res.treatment!r}; only the within-unit SWITCHING indicator has an onset"
    )


def test_staggered_did_recovers_the_positive_att(tmp_path):
    """The number, not just the column: binding `region_group` produced -0.332 (p=0.033)."""
    fp = _fp(_staggered_panel(), tmp_path)
    res = run_analysis(fp, _CAT.by_id("staggered_did"), output_root=str(tmp_path / "sd"))
    if not _ran(res):
        pytest.skip(f"staggered_did degraded: {res.summary[:70]}")
    att = res.estimates.get("att_overall")
    assert att is not None and 2.0 < att < 4.0, f"true ATT is +3.0, got {att}"


@pytest.mark.parametrize("cid", ["psm", "ipw", "aipw"])
def test_a_named_covariate_never_beats_the_real_arm(cid, tmp_path):
    fp = _fp(_named_covariate_frame(), tmp_path)
    entry = _CAT.by_id(cid)
    if entry is None:
        pytest.skip(f"{cid} not in catalog")
    res = run_analysis(fp, entry, output_root=str(tmp_path / f"cov_{cid}"))
    if not _ran(res):
        pytest.skip(f"{cid} degraded: {res.summary[:70]}")
    assert res.treatment == "vaccinated", f"{cid} bound {res.treatment!r}"
    key = "att" if "att" in res.estimates else "ate"
    assert res.estimates[key] < 0, "true ATT is -8; a positive estimate means the wrong column"


def test_likely_treatment_hint_is_not_a_covariate(tmp_path):
    """`likely_treatment` is shown to the user (cli / study_report / web), so its vocabulary
    has to be the precise one too — it was announcing `age_group` as 可能的处理变量."""
    fp = _fp(_named_covariate_frame(), tmp_path)
    assert fp.likely_treatment == "vaccinated"


# ── tier discrimination: mutants M1/M2/M5 all survived the original suite ────
def _tier_frame(n=300, seed=7):
    """`likely_treatment` is 'arm', but 'treatment_response' is the high-confidence binary
    OUTCOME and also matches the treatment vocabulary. Tier 2 without the outcome exclusion
    binds the outcome; deleting tier 2 or tier 3 changes the answer — so this frame tells
    the tiers apart, which the flagship frame could not."""
    rng = np.random.default_rng(seed)
    arm = rng.integers(0, 2, n)
    return pd.DataFrame({"baseline": rng.normal(50, 8, n).round(2),
                         "treatment_response": (rng.random(n) < 0.3 + 0.3 * arm).astype(int),
                         "arm": arm,
                         "age": rng.integers(20, 70, n)})


def test_a_binary_outcome_is_never_bound_as_the_treatment(tmp_path):
    fp = _fp(_tier_frame(), tmp_path)
    assert fp.likely_outcome == "treatment_response"       # high-confidence binary outcome
    assert resolve_treatment(fp, {}, fp.binary_columns) == "arm"


def test_first_named_candidate_wins_in_dataframe_order(tmp_path):
    """Pins `named[0]`, not `named[-1]` (mutant M2 survived the original suite)."""
    rng = np.random.default_rng(13)
    n = 200
    df = pd.DataFrame({"y": rng.normal(0, 1, n).round(3),
                       "arm": rng.integers(0, 2, n),
                       "dose_high": rng.integers(0, 2, n)})
    fp = _fp(df, tmp_path)
    assert resolve_treatment(fp, {}, ["arm", "dose_high"]) == "arm"
    assert resolve_treatment(fp, {}, ["dose_high", "arm"]) == "dose_high"


def test_config_can_name_a_column_outside_the_candidate_list(tmp_path):
    """Pins the `df=` widening (mutant M3 survived): a user may name a column the profiler
    typed as something other than binary, and it must bind AND be recorded."""
    fp = _fp(_named_covariate_frame(), tmp_path)
    csv0 = tmp_path / "widen.csv"
    _named_covariate_frame().to_csv(csv0, index=False)
    import pandas as _pd
    d = _pd.read_csv(csv0)
    # with df=, an explicit config may name ANY column (bmi is continuous, not a candidate)
    assert resolve_treatment(fp, {"treatment": "bmi"}, ["vaccinated"], df=d) == "bmi"
    # without df= the widening must NOT happen — config has to name a candidate
    assert resolve_treatment(fp, {"treatment": "bmi"}, ["vaccinated"]) == "vaccinated"
    csv = tmp_path / "w.csv"
    _named_covariate_frame().to_csv(csv, index=False)
    res = run_analysis(profile_dataset(csv), _CAT.by_id("psm"),
                       output_root=str(tmp_path / "wcfg"), config={"treatment": "age_group"})
    assert res.treatment == "age_group"


def test_the_first_binding_is_the_recorded_one(tmp_path):
    """Pins first-wins (mutant M4 recorded the LAST binding and survived)."""
    from researchforge.executor.run import capture_bound_treatment

    fp = _fp(_named_covariate_frame(), tmp_path)
    with capture_bound_treatment() as rec:
        resolve_treatment(fp, {}, ["vaccinated"])
        resolve_treatment(fp, {}, ["age_group"])
    assert rec == ["vaccinated"]


def test_the_bound_treatment_is_disclosed_to_the_user(tmp_path):
    """MUST-FIX 3: the report used to name neither the intervention nor the alternative."""
    fp = _fp(_named_covariate_frame(), tmp_path)
    res = run_analysis(fp, _CAT.by_id("psm"), output_range := str(tmp_path / "disc"))
    assert "处理/暴露变量" in res.summary and "vaccinated" in res.summary
    assert output_range  # keep the path referenced


def test_explicit_treatment_is_named_but_not_second_guessed(tmp_path):
    fp = _fp(_named_covariate_frame(), tmp_path)
    res = run_analysis(fp, _CAT.by_id("psm"), output_root=str(tmp_path / "d2"),
                       config={"treatment": "age_group"})
    line = next((ln for ln in res.summary.split("\n") if "处理/暴露变量" in ln), "")
    assert "age_group" in line and "config" in line


def test_the_binary_outcome_is_excluded_even_without_any_name_signal(tmp_path):
    """The case that separates tiers 3/4 from a bare ``candidates[0]``.

    No candidate carries a treatment name here, so tier 2 is silent. The first binary in
    file order IS the high-confidence binary OUTCOME (`died`), so a fallback that just takes
    candidates[0] binds the event being explained as the intervention. Tier 3
    (fp.likely_treatment, which roles.py builds excluding the outcome) and tier 4 (first
    non-outcome candidate) both prevent that — deleting them was the one mutant the first
    version of this suite could not kill.
    """
    rng = np.random.default_rng(21)
    n = 300
    flag_b = rng.integers(0, 2, n)
    died = (rng.random(n) < 0.2 + 0.3 * flag_b).astype(int)
    df = pd.DataFrame({"died": died, "flag_b": flag_b,
                       "score": rng.normal(0, 1, n).round(3)})
    fp = _fp(df, tmp_path)
    assert fp.likely_outcome == "died"                      # high-confidence binary event
    assert fp.binary_columns[0] == "died"             # ...and it is first in file order
    assert resolve_treatment(fp, {}, fp.binary_columns) == "flag_b"


# ═════════════════════════════════════════════════════════════════════════════
# Delta-review regressions (2026-09-08). The first round of cold-review fixes
# traded one error class for another, and the panel test above turned out to be
# confounded — it passed with the switching signal entirely disabled.
# ═════════════════════════════════════════════════════════════════════════════
def _panel_with_strong_named_decoy(n_unit=30, n_t=14, seed=11):
    """True ATT +3.0. `treatment_arm` is STRONG-named but TIME-INVARIANT, so only the
    switching signal can pick `adopted`. The earlier decoy (`region_group`) was already
    rejected by the vocabulary, which is why that test passed without the switch signal."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_unit):
        fe = rng.normal(0, 1)
        g = 6 if u % 3 == 0 else (10 if u % 3 == 1 else 10 ** 9)
        for t in range(n_t):
            d = 1 if t >= g else 0
            rows.append({"unit": f"u{u:02d}", "year": 2005 + t,
                         "y": round(float(10 + fe + 3.0 * d + 0.1 * t + rng.normal(0, 0.6)), 3),
                         "adopted": d, "treatment_arm": int(u % 2)})
    return pd.DataFrame(rows)


def _panel_two_switchers(extra="post", n_unit=30, n_t=14, seed=11):
    """True ATT +3.0 with a SECOND switching binary. `post` is a calendar dummy that switches
    for 100% of units vs `policy_on`'s 50%, so an argmax-only switch pick hands back `post`
    and the name ladder never runs — event_study emitted 0.513 against a truth of 3.0."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_unit):
        fe = rng.normal(0, 1)
        g = 6 if u % 3 == 0 else (10 if u % 3 == 1 else 10 ** 9)
        for t in range(n_t):
            d = 1 if t >= g else 0
            row = {"unit": f"u{u:02d}", "year": 2005 + t,
                   "y": round(float(10 + fe + 3.0 * d + 0.1 * t + rng.normal(0, 0.6)), 3),
                   "policy_on": d}
            row["post" if extra == "post" else "hospitalized"] = (
                (1 if t >= 7 else 0) if extra == "post" else int(rng.random() < 0.3 + 0.2 * d))
            rows.append(row)
    return pd.DataFrame(rows)


def _site_group_frame(n=400, seed=31):
    """`group` is the study SITE (a weak whole-name match), `treated` is the arm. Ranking the
    weak escape by file order bound `group` and reported ATT=+3.87 against a truth of -8."""
    rng = np.random.default_rng(seed)
    grp = rng.integers(0, 2, n)
    treated = (rng.random(n) < 1 / (1 + np.exp(-(0.5 * grp)))).astype(int)
    score = 50 - 8 * treated + 2 * grp + rng.normal(0, 5, n)
    return pd.DataFrame({"score": score.round(2), "group": grp, "treated": treated,
                         "age": rng.integers(20, 70, n)})


def _payments_frame(n=400, seed=41):
    """`tx` as a SUBSTRING fired across the payments domain; `tx_online` was bound as the
    treatment and PSM reported +7.4 against a truth of -15."""
    rng = np.random.default_rng(seed)
    tx_online = rng.integers(0, 2, n)
    promo = (rng.random(n) < 0.5).astype(int)
    spend = 200 - 15 * promo + 6 * tx_online + rng.normal(0, 12, n)
    return pd.DataFrame({"spend": spend.round(2), "promo_shown": promo,
                         "tx_online": tx_online, "tenure_days": rng.integers(30, 900, n)})


@pytest.mark.parametrize("cid", ["staggered_did", "event_study", "callaway_santanna"])
def test_switch_signal_beats_a_strong_named_time_invariant_flag(cid, tmp_path):
    """Kills the mutant that disables _pick_did_treatment entirely."""
    fp = _fp(_panel_with_strong_named_decoy(), tmp_path)
    entry = _CAT.by_id(cid)
    if entry is None:
        pytest.skip(f"{cid} not in catalog")
    res = run_analysis(fp, entry, output_root=str(tmp_path / f"sw_{cid}"))
    if not _ran(res):
        pytest.skip(f"{cid} degraded: {res.summary[:70]}")
    assert res.treatment == "adopted", (
        f"{cid} bound {res.treatment!r}; `treatment_arm` never switches, so it has no onset"
    )
    key = next((k for k in ("att_overall", "att_post_mean") if k in res.estimates), None)
    if key:
        assert 2.0 < res.estimates[key] < 4.0, f"true ATT is +3.0, got {res.estimates[key]}"


@pytest.mark.parametrize("extra", ["post", "tv_outcome"])
def test_name_ladder_still_decides_among_several_switchers(extra, tmp_path):
    """The switch signal must NARROW the candidates, not replace the ladder: a calendar
    `post` dummy switches for more units than the real `policy_on`."""
    fp = _fp(_panel_two_switchers(extra), tmp_path, name=f"{extra}.csv")
    res = run_analysis(fp, _CAT.by_id("staggered_did"), output_root=str(tmp_path / f"tw_{extra}"))
    if not _ran(res):
        pytest.skip(f"degraded: {res.summary[:70]}")
    assert res.treatment == "policy_on", f"bound {res.treatment!r}"
    assert 2.0 < res.estimates["att_overall"] < 4.0


def test_pick_did_treatment_returns_every_switcher_most_switching_first(tmp_path):
    """Pins the arity (a single-column return annihilates the ladder) and the ordering."""
    df = _panel_two_switchers("post")
    fp = _fp(df, tmp_path, name="sw.csv")
    got = _pick_did_treatment(df, fp)
    assert got == ["post", "policy_on"], got          # frac 1.0 before frac 0.5
    assert _pick_did_treatment(df, fp, unit="unit", time="year") == got   # explicit overrides


def test_did_formula_takes_only_one_treatment_term(tmp_path):
    """_pick_did_treatment now returns several columns; the `did` formula must take one."""
    fp = _fp(_panel_two_switchers("post"), tmp_path, name="didf.csv")
    res = run_analysis(fp, _CAT.by_id("did"), output_root=str(tmp_path / "didf"))
    if not _ran(res):
        pytest.skip(f"did degraded: {res.summary[:70]}")
    assert "post" in res.summary or "policy_on" in res.summary


def test_a_weak_whole_name_never_outranks_a_strong_word(tmp_path):
    fp = _fp(_site_group_frame(), tmp_path)
    assert resolve_treatment(fp, {}, fp.binary_columns) == "treated"
    assert fp.likely_treatment == "treated"      # the user-visible hint must agree
    res = run_analysis(fp, _CAT.by_id("psm"), output_root=str(tmp_path / "site"))
    if _ran(res):
        assert res.treatment == "treated" and res.estimates["att"] < 0


def test_a_bare_group_column_still_binds(tmp_path):
    """The exact-name escape must survive the strength ranking: with no strong word present,
    a bare `group` column IS the arm."""
    rng = np.random.default_rng(61)
    n = 300
    g = rng.integers(0, 2, n)
    df = pd.DataFrame({"y": (10 - 4 * g + rng.normal(0, 2, n)).round(3),
                       "flag_a": rng.integers(0, 2, n), "group": g})
    fp = _fp(df, tmp_path, name="bare.csv")
    assert resolve_treatment(fp, {}, ["flag_a", "group"]) == "group"


def test_payments_columns_are_not_treatments(tmp_path):
    fp = _fp(_payments_frame(), tmp_path)
    assert fp.likely_treatment == "promo_shown"
    res = run_analysis(fp, _CAT.by_id("psm"), output_root=str(tmp_path / "tx"))
    if not _ran(res):
        pytest.skip("psm degraded")
    assert res.treatment == "promo_shown", f"bound {res.treatment!r}"
    assert res.estimates["att"] < 0, "true ATT is -15"


def test_evalue_records_an_explicitly_configured_exposure(tmp_path):
    """Kills the mutant that reverts SHOULD-FIX 6 for evalue."""
    fp = _fp(_survival_frame(), tmp_path)
    res = run_analysis(fp, _CAT.by_id("evalue"), output_root=str(tmp_path / "ev"),
                       config={"exposure": "treatment"})
    if not _ran(res):
        pytest.skip("evalue degraded")
    assert res.treatment == "treatment"


def test_disclosure_explains_the_switch_rule_instead_of_misdirecting(tmp_path):
    """On a 2x2 panel the engine binds `post` (the switcher) on purpose; warning that
    `treated` "looks more like the treatment" would send the user to a column that cannot
    define an onset."""
    rng = np.random.default_rng(51)
    rows = []
    for u in range(40):
        fe = rng.normal(0, 1)
        tr = int(u % 2)
        for t in range(8):
            po = 1 if t >= 4 else 0
            rows.append({"unit": f"u{u:02d}", "year": 2010 + t,
                         "y": round(float(5 + fe + 2.5 * tr * po + rng.normal(0, 0.5)), 3),
                         "post": po, "treated": tr})
    fp = _fp(pd.DataFrame(rows), tmp_path, name="disc2.csv")
    res = run_analysis(fp, _CAT.by_id("staggered_did"), output_root=str(tmp_path / "disc2"))
    line = next((ln for ln in res.summary.split("\n") if "处理/暴露变量" in ln), "")
    assert "post" in line
    assert "无法定义处理起始期" in line and "⚠ 角色检测认为" not in line


# ── mutants that survived the first delta round (each needed a frame where no OTHER
#    mechanism can rescue the assertion) ──────────────────────────────────────────
def test_pick_did_treatment_honors_the_callers_unit_override(tmp_path):
    """N1: the branch resolves `unit` from config; the fingerprint may have detected another
    column. Asserting that the TWO groupings give DIFFERENT answers pins the override
    regardless of which one the profiler happened to pick — ignoring it makes them identical.

    `flag_x` switches within every `unit_a`; `flag_y` is constant within `unit_a` but varies
    within the coarser `unit_b`, so the admissible treatment set genuinely depends on the
    grouping the caller asked for."""
    rows = []
    for a in range(12):
        for t in range(6):
            rows.append({"unit_a": f"a{a:02d}", "unit_b": f"b{a % 3}", "year": 2010 + t,
                         "y": float(t + a),
                         "flag_x": 1 if t >= 3 else 0,     # switches within unit_a
                         "flag_y": a % 2})                 # constant in unit_a, varies in unit_b
    df = pd.DataFrame(rows)
    fp = _fp(df, tmp_path, name="units.csv")
    by_a = _pick_did_treatment(df, fp, unit="unit_a", time="year")
    by_b = _pick_did_treatment(df, fp, unit="unit_b", time="year")
    assert by_a == ["flag_x"], by_a
    assert "flag_y" in by_b, by_b
    assert by_a != by_b, "the caller's unit override changed nothing — it was ignored"


def test_pick_did_treatment_skips_candidates_absent_from_the_frame(tmp_path):
    """N2: fp.binary_columns is computed from the fingerprint, so a caller working on a
    narrowed frame can list a column the dataframe no longer has. Dropping the guard raises
    KeyError instead of skipping."""
    df = _panel_two_switchers("post")
    fp = _fp(df, tmp_path, name="guard.csv")
    narrowed = df.drop(columns=["post"])
    assert "post" in fp.binary_columns          # still listed by the fingerprint
    got = _pick_did_treatment(narrowed, fp, unit="unit", time="year")
    assert got == ["policy_on"], got                  # skipped, not crashed


def test_the_exact_escape_survives_the_strength_ranking(tmp_path):
    """N4: `group` must still bind when it is the ONLY signal. The earlier version of this
    test was rescued by tier 3 (fp.likely_treatment was also `group`); here the fingerprint's
    hint is `treated`, which is NOT among the candidates, so only the escape can save it."""
    rng = np.random.default_rng(71)
    n = 300
    df = pd.DataFrame({"y": rng.normal(0, 1, n).round(3),
                       "flag_a": rng.integers(0, 2, n),
                       "group": rng.integers(0, 2, n),
                       "treated": rng.integers(0, 2, n)})
    fp = _fp(df, tmp_path, name="escape.csv")
    assert fp.likely_treatment == "treated"                       # strong word wins the hint
    assert resolve_treatment(fp, {}, ["flag_a", "group"]) == "group"   # ...but it isn't offered


def test_did_uses_exactly_one_of_several_switchers(tmp_path):
    """N11: `_pick_did_treatment` now returns every switcher, and `did`'s formula takes one
    treatment term. Its estimates are keyed by coefficient name, so a naive multi-return shows
    up as two treatment coefficients."""
    fp = _fp(_panel_two_switchers("post"), tmp_path, name="didone.csv")
    res = run_analysis(fp, _CAT.by_id("did"), output_root=str(tmp_path / "didone"))
    if not _ran(res):
        pytest.skip(f"did degraded: {res.summary[:70]}")
    used = [k for k in ("post", "policy_on") if k in res.estimates]
    assert len(used) == 1, f"did put {used} in the formula; it takes ONE treatment term"


def test_tx_is_only_a_treatment_when_it_is_the_whole_name(tmp_path):
    """N13: `tx` as a substring fires across the payments domain."""
    from researchforge.profiler.semantics import treatment_name_strength

    assert treatment_name_strength("tx") == 1
    for n in ("tx_date", "tx_id", "tx_amount", "tx_online", "tx_count", "post_tx"):
        assert treatment_name_strength(n) == 0, n
    # ...while `trt` stays safe as a substring
    assert treatment_name_strength("trt") == 2
    assert treatment_name_strength("trt_group") == 2
    for n in ("part", "sort", "trtn"):
        assert treatment_name_strength(n) == 0, n
    # `pre_trt` was asserted at 2 here until cold review A MUST-FIX 2. That assertion encoded
    # the bug: a `pre_*` column marks the period BEFORE treatment, so binding it as the
    # treatment inverts (or, for a calendar dummy, hijacks) the contrast.
    assert treatment_name_strength("pre_trt") == 0


def test_control_named_columns_are_never_bound(tmp_path):
    """Binding a `control_group` indicator would make 1 = control, so PSM/IPW would report
    -ATT with no warning. The reviewer flagged this as a trap to stay out of."""
    from researchforge.profiler.semantics import treatment_name_strength

    assert treatment_name_strength("control") == 0
    assert treatment_name_strength("control_group") == 0
    assert treatment_name_strength("ctrl") == 0


# ── 地基(二): the rename surfaced two more positional treatment picks ─────────
def _sc_panel(decoy="invariant", seed=11, n_unit=30, n_t=14):
    """ONE truly-treated unit (u00 from t=7), true effect +3.0, with a decoy binary placed
    FIRST in file order. Synthetic control needs a donor pool, so picking the decoy does not
    merely bias the estimate — it destroys the analysis (half the panel looks treated)."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_unit):
        fe = rng.normal(0, 1)
        for t in range(n_t):
            d = 1 if (u == 0 and t >= 7) else 0
            row = {"unit": f"u{u:02d}", "year": 2005 + t}
            row["region_flag" if decoy == "invariant" else "post"] = (
                int(u % 2) if decoy == "invariant" else (1 if t >= 7 else 0))
            row["y"] = round(float(10 + fe + 3.0 * d + 0.1 * t + rng.normal(0, 0.5)), 3)
            row["policy_on"] = d
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.mark.parametrize("decoy", ["invariant", "post"])
def test_synthetic_control_does_not_take_the_first_binary_column(decoy, tmp_path):
    fp = _fp(_sc_panel(decoy), tmp_path, name=f"sc_{decoy}.csv")
    assert fp.binary_columns[0] != "policy_on"        # a positional pick would land on the decoy
    res = run_analysis(fp, _CAT.by_id("synthetic_control"),
                       output_root=str(tmp_path / f"sc_{decoy}"))
    if not _ran(res):
        pytest.skip(f"synthetic_control degraded: {res.summary[:80]}")
    assert res.treatment == "policy_on", f"bound {res.treatment!r}"
    assert 2.0 < res.estimates["att"] < 4.0, res.estimates["att"]
    assert res.estimates["n_donors_used"] > 20, "the decoy pick leaves almost no donor pool"


def test_gsynth_does_not_take_the_first_binary_column(tmp_path):
    fp = _fp(_sc_panel("post"), tmp_path, name="gs.csv")
    res = run_analysis(fp, _CAT.by_id("gsynth"), output_root=str(tmp_path / "gs"))
    if not _ran(res):
        pytest.skip(f"gsynth degraded (needs R): {res.summary[:80]}")
    assert res.treatment == "policy_on", f"bound {res.treatment!r}"
    assert 2.0 < res.estimates["att"] < 4.0, res.estimates["att"]


def test_the_binary_column_list_makes_no_role_claim(tmp_path):
    """The field is a SHAPE fact. It was named `treatment_candidates`, which asserted a
    treatment signal it never carried — three measured bugs came from trusting that name."""
    fp = _fp(_survival_frame(), tmp_path, name="shape.csv")
    assert fp.binary_columns == ["event", "treatment"]        # file order, every binary
    assert not hasattr(fp, "treatment_candidates"), "the misleading alias must be gone"


# ── cold review A MUST-FIX 1/2/4: the veto has to reach every tier ───────────
def _confounded(flag_name, n=600, seed=7, complement=True):
    """True ATT = -8. `flag_name` holds 1 - treated when complement, else treated."""
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(0, 1, n), rng.normal(0, 1, n)
    treated = (1 / (1 + np.exp(-(0.8 * x1 - 0.5 * x2))) > rng.random(n)).astype(int)
    y = 20 + 2 * x1 - 1.5 * x2 - 8.0 * treated + rng.normal(0, 1, n)
    return pd.DataFrame({"severity": x1.round(3), "biomarker": x2.round(3),
                         flag_name: (1 - treated) if complement else treated,
                         "outcome_score": y.round(3)})


@pytest.mark.parametrize(
    "flag", ["control_arm", "placebo", "untreated", "unexposed", "no_treatment",
             "non_exposed", "never_exposed", "treatment_naive", "baseline_arm",
             "dose_0", "pre_treatment"])
def test_an_absence_of_treatment_flag_is_never_bound_even_when_it_is_the_only_one(
        flag, tmp_path):
    """The test that used to guard this had `treated` in the frame too, so it pinned the
    ORDERING and nothing else. As the SOLE binary column each of these was bound and the
    estimate came back with the sign inverted: measured IPW +7.652 / PSM +7.042 against a
    truth of -8 (cold review A MUST-FIX 1 and 2)."""
    fp = _fp(_confounded(flag), tmp_path, name=f"{flag}.csv")
    res = run_analysis(fp, _CAT.by_id("ipw"), output_root=str(tmp_path / "o"))
    assert res.treatment is None, f"bound {res.treatment!r}"
    assert "未处理/对照" in res.summary, res.summary[:200]
    assert not res.estimates


def test_the_user_can_still_force_one_through_config(tmp_path):
    """A veto on the AUTO path, not a prohibition — the estimand is then the user's call."""
    fp = _fp(_confounded("control_arm"), tmp_path, name="forced.csv")
    res = run_analysis(fp, _CAT.by_id("ipw"), output_root=str(tmp_path / "p"),
                       config={"treatment": "control_arm"})
    assert res.treatment == "control_arm"
    assert res.estimates.get("ate", 0) > 0, "1 = control, so the sign is legitimately flipped"


@pytest.mark.parametrize("name", ["处理组", "干预组", "治疗组", "实验组", "是否用药",
                                  "TRT01PN", "TRTA", "armcd", "ACTARM", "grupo_tratado",
                                  "tratamiento", "behandelt"])
def test_the_binding_vocabulary_is_not_english_only(name, tmp_path):
    """The module calls itself bilingual and ROLE_HINTS already carries 处理/组别, but the
    BINDING vocabulary consulted neither. Measured on a Chinese frame: every one of these
    scored 0, the resolver fell through to `sex`, and the report said +1.385 where the
    truth was -7.84 — with no warning, because the fallback tier issues none.
    CDISC is the same hole in Latin script: TRT01P / TRTA / ARMCD are THE standard
    treatment variables in clinical submissions."""
    df = _confounded(name, complement=False)
    df["sex"] = np.random.default_rng(1).integers(0, 2, len(df))
    fp = _fp(df, tmp_path, name="i18n.csv")
    res = run_analysis(fp, _CAT.by_id("ipw"), output_root=str(tmp_path / "q"))
    assert res.treatment == name, f"bound {res.treatment!r} instead"
    assert res.estimates["ate"] < -6.0, res.estimates


def test_the_control_mirror_does_not_enter_the_propensity_model(tmp_path):
    """Found while verifying the veto: with `treated` correctly bound, `control_arm`
    (= 1 - treated) still went into the AUTO covariate set, separating the propensity model
    perfectly. The user got a naked '逆概率加权失败：Singular matrix' and estimates={}."""
    df = _confounded("control_arm")
    df["treated"] = 1 - df["control_arm"]
    fp = _fp(df, tmp_path, name="mirror.csv")
    res = run_analysis(fp, _CAT.by_id("ipw"), output_root=str(tmp_path / "r"))
    assert res.treatment == "treated"
    # NOT `"Singular matrix" not in summary` — the disclosure this fix adds quotes that very
    # phrase to explain what used to happen, so the naive check fails on the fixed build.
    # Same disease as cold review A#12: prose and degrade marker sharing a vocabulary.
    assert "失败：" not in res.summary, res.summary[:200]
    assert -9.0 < res.estimates["ate"] < -7.0, res.estimates
    assert "确定性函数" in res.summary and "control_arm" in res.summary


# ── cold review A MUST-FIX 10: did re-introduced the argmax bug inside _regression ──
def _decoy_switcher_panel(n_unit=40, n_t=14, seed=5):
    """True ATT = +3.0 on `policy_on`. `hospitalized` also switches within unit — more
    often, in fact — so a switch-rate argmax picks the decoy."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_unit):
        fe = rng.normal(0, 1)
        g = 7 if u % 2 == 0 else 10 ** 9
        for t in range(n_t):
            d = 1 if t >= g else 0
            rows.append({"unit": f"u{u:02d}", "year": 2005 + t,
                         "hospitalized": int(rng.random() < 0.45), "policy_on": d,
                         "y": round(10 + fe + 3.0 * d + 0.1 * t + rng.normal(0, 0.6), 3)})
    return pd.DataFrame(rows)


def test_did_uses_the_name_ladder_not_the_highest_switch_rate(tmp_path):
    """`_pick_did_treatment(df, fp)[:1]` inside _regression took the top switcher and stopped
    — the same argmax bug D1 removed from _pick_did_treatment itself, moved one level up.
    Measured: did reported 关键系数 hospitalized = 0.2744 (p=0.006) on this frame."""
    fp = _fp(_decoy_switcher_panel(), tmp_path, name="decoy.csv")
    res = run_analysis(fp, _CAT.by_id("did"), output_root=str(tmp_path / "o"))
    assert res.treatment == "policy_on", f"bound {res.treatment!r}"
    assert 2.5 < res.estimates["policy_on"] < 3.5, res.estimates
    assert "hospitalized" not in res.estimates


@pytest.mark.parametrize("named", ["policy_on", "hospitalized"])
def test_did_config_treatment_is_no_longer_inert(named, tmp_path):
    """config={"treatment": ...} had NO effect on did — the branch never consulted cfg."""
    fp = _fp(_decoy_switcher_panel(), tmp_path, name="cfgdid.csv")
    res = run_analysis(fp, _CAT.by_id("did"), output_root=str(tmp_path / named),
                       config={"treatment": named})
    assert res.treatment == named
    assert named in res.estimates
