"""Wave H4b — finish the outcome-binding sweep started by H4.

H4 made ``resolve_outcome`` record what it actually bound, which turned
``RunResult.outcome is None`` into an AUDIT INSTRUMENT: a branch that ran for real yet
reports no outcome never went through the shared resolver, so the engine cannot say what
was modeled (and the smart-selection nudge must stay silent). Auditing all 111 catalog
entries that take an ``outcome``/``y`` parameter surfaced 19 such branches. H4b closes them
in three ways, each pinned below:

  * SHARED RESOLVERS — ``ml_supervised._resolve_learning_roles`` (5 learners) and
    ``limited_dependent._resolve_outcome_predictors`` (tobit + truncated) resolve roles themselves; one
    recording call each covers every branch in the family.
  * REAL GAPS — ``iv_regression``, the mediation Y (serial/parallel/moderated) and
    ``johnson_neyman``'s Y, and ``evalue``'s outcome took whichever column came FIRST;
    they now bind the DETECTED outcome. evalue's is a behaviour fix, not just bookkeeping:
    with ``[treated, died]`` it used to model *treated* as the outcome and *died* as the
    exposure — exactly backwards.
  * CLASS-LABEL METHODS — ``mnl_choice`` / ``linear_discriminant`` /
    ``discriminant_analysis`` predict a class label with level-count logic
    ``resolve_outcome`` does not have, so they keep their own pick and only RECORD it.

Genuinely outcome-free methods (correlations, spatial statistics, Friedman's conditions)
stay unbound on purpose — see docs/deferred-log.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog import Catalog
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_CAT = Catalog.load()
# 判据必须是**降级标记**，不是散文里的词。原先用裸子串 "失败"/"跳过" 判降级，而同一波新增的披露文案里就有「事件率饱和时甚至直接**失败**」——旗舰 survival frame 上 psm/ipw/aipw/rosenbaum 的 _ran() 因此恒为 False，10 个断言集体休眠、棘轮短路（冷审 A#12）。降级消息一律是「<方法>失败：」/「<方法>跳过：」的冒号形式。
_DEGRADED = ("失败：", "跳过：", "暂未接入", "未检测到")


def _fp(df, tmp_path, name="d.csv"):
    csv = tmp_path / name
    df.to_csv(csv, index=False)
    return profile_dataset(csv)


def _run(fp, cid, tmp_path, cfg=None):
    entry = _CAT.by_id(cid)
    if entry is None:
        pytest.skip(f"{cid} not in catalog")
    return run_analysis(fp, entry, output_root=str(tmp_path / f"o_{cid}"), config=cfg)


def _ran(res) -> bool:
    """The branch actually modeled something (an honest degrade proves nothing here)."""
    return bool(res.estimates) and not any(k in res.summary for k in _DEGRADED)


# ── frames ───────────────────────────────────────────────────────────────────
def _reg_frame(n=200, seed=0):
    """`y` is the high-confidence DV and deliberately NOT the first continuous column, so a
    branch that silently fell back to cont[0] binds `x1` and the assertion fails."""
    rng = np.random.default_rng(seed)
    x1, x2, z = (rng.normal(0, 1, n) for _ in range(3))
    return pd.DataFrame({"x1": x1.round(3), "x2": x2.round(3), "z": z.round(3),
                         "y": (2 * x1 - x2 + rng.normal(0, 1, n)).round(3)})


def _binary_frame(n=300, seed=1):
    rng = np.random.default_rng(seed)
    treated = rng.integers(0, 2, n)
    died = (rng.random(n) < 0.2 + 0.2 * treated).astype(int)
    return pd.DataFrame({"score": rng.normal(50, 10, n).round(2),
                         "treated": treated, "died": died})


def _label_frame(n=240, seed=2):
    rng = np.random.default_rng(seed)
    grp = rng.choice(["a", "b", "c"], n)
    shift = pd.Series(grp).map({"a": 0.0, "b": 2.0, "c": 4.0}).to_numpy()
    return pd.DataFrame({"f1": (rng.normal(0, 1, n) + shift).round(3),
                         "f2": (rng.normal(0, 1, n) - shift / 2).round(3),
                         "f3": rng.normal(0, 1, n).round(3),
                         "grp": grp})


# ── shared resolvers: one recording call, a whole family ─────────────────────
@pytest.mark.parametrize("cid", ["regularized_regression", "svm_model", "gradient_boosting",
                                 "monotonic_constraints", "random_forest"])
def test_ml_family_reports_its_outcome(cid, tmp_path):
    fp = _fp(_reg_frame(), tmp_path)
    res = _run(fp, cid, tmp_path)
    if not _ran(res):
        pytest.skip(f"{cid} degraded: {res.summary[:60]}")
    assert res.outcome == "y", f"{cid} bound {res.outcome!r}"


@pytest.mark.parametrize("cid", ["truncated_regression", "tobit_regression"])
def test_limited_dependent_binds_detected_outcome(cid, tmp_path):
    fp = _fp(_reg_frame(), tmp_path)
    res = _run(fp, cid, tmp_path)
    if not _ran(res):
        pytest.skip(f"{cid} degraded: {res.summary[:60]}")
    assert res.outcome == "y", f"{cid} bound {res.outcome!r} (positional fallback?)"


def test_ml_family_honors_config_outcome(tmp_path):
    """The recorder must report the CONFIGURED column, not the detected one."""
    fp = _fp(_reg_frame(), tmp_path)
    res = _run(fp, "gradient_boosting", tmp_path, cfg={"outcome": "x2"})
    if not _ran(res):
        pytest.skip("gradient_boosting degraded")
    assert res.outcome == "x2"


# ── real gaps: these used to take whichever column came first ────────────────
def test_iv_regression_binds_detected_outcome(tmp_path):
    fp = _fp(_reg_frame(), tmp_path)
    res = _run(fp, "iv_regression", tmp_path,
               cfg={"instruments": ["z"], "endogenous": "x1"})
    if not _ran(res):
        pytest.skip(f"iv_regression degraded: {res.summary[:60]}")
    assert res.outcome == "y", f"iv bound {res.outcome!r}"


@pytest.mark.parametrize("cid", ["parallel_mediation", "serial_mediation",
                                 "moderated_mediation", "johnson_neyman"])
def test_mediation_family_binds_y_as_outcome(cid, tmp_path):
    fp = _fp(_reg_frame(), tmp_path)
    res = _run(fp, cid, tmp_path)
    if not _ran(res):
        pytest.skip(f"{cid} degraded: {res.summary[:60]}")
    assert res.outcome == "y", f"{cid} bound {res.outcome!r} as Y"


def test_mediation_config_y_still_wins(tmp_path):
    fp = _fp(_reg_frame(), tmp_path)
    res = _run(fp, "parallel_mediation", tmp_path, cfg={"y": "x2"})
    if not _ran(res):
        pytest.skip("parallel_mediation degraded")
    assert res.outcome == "x2"


def test_evalue_does_not_model_the_exposure(tmp_path):
    """Behaviour fix, not bookkeeping: `[treated, died]` used to bind *treated* as the
    outcome (first binary) and leave *died* as the exposure — the two roles swapped."""
    fp = _fp(_binary_frame(), tmp_path)
    res = _run(fp, "evalue", tmp_path)
    if not _ran(res):
        pytest.skip(f"evalue degraded: {res.summary[:60]}")
    assert res.outcome == "died", f"evalue bound {res.outcome!r} as the outcome"


# ── class-label methods: own pick, but it must be RECORDED ───────────────────
@pytest.mark.parametrize("cid", ["mnl_choice", "linear_discriminant", "discriminant_analysis"])
def test_class_label_methods_record_their_target(cid, tmp_path):
    fp = _fp(_label_frame(), tmp_path)
    res = _run(fp, cid, tmp_path)
    if not _ran(res):
        pytest.skip(f"{cid} degraded: {res.summary[:60]}")
    assert res.outcome == "grp", f"{cid} recorded {res.outcome!r}"


# ── the ratchet: nothing in the swept set may go back to silence ─────────────
_SWEPT = ["regularized_regression", "svm_model", "gradient_boosting", "monotonic_constraints",
          "truncated_regression", "tobit_regression", "parallel_mediation", "serial_mediation",
          "moderated_mediation", "johnson_neyman"]


def test_no_swept_branch_runs_without_reporting_its_outcome(tmp_path):
    fp = _fp(_reg_frame(seed=5), tmp_path)
    silent = []
    for cid in _SWEPT:
        entry = _CAT.by_id(cid)
        if entry is None:
            continue
        res = run_analysis(fp, entry, output_root=str(tmp_path / f"s_{cid}"))
        if _ran(res) and res.outcome is None:
            silent.append(cid)
    assert not silent, f"ran but reported no outcome: {silent}"


# ═════════════════════════════════════════════════════════════════════════════
# Wave H4c — the shapes H4b's single probe could not reach.
#
# H4b audited with ONE probe (3 continuous + 1 categorical), so it only ever saw the
# branches that shape lets run: serial_mediation / moderated_mediation need 4 continuous
# columns and degraded on it, invisible to the instrument (they were found by hand).
# Sweeping five probe shapes — wide-continuous, panel, survival, wide counts, categorical —
# surfaced 17 more, almost all of them one copy-pasted line:
#
#     outcome = cfg["outcome"] else next(c for c in cont if c != treatment)
#
# i.e. column order with no role signal. The frames below put the high-confidence DV
# name where that old pick does NOT land, so a regression to positional picking fails.
# ═════════════════════════════════════════════════════════════════════════════
def _causal_frame(n=400, seed=7):
    """`y` is the DV; the OLD pick ("first continuous that isn't the treatment") lands on
    `covar1`, so `res.outcome == "y"` is a real assertion, not bookkeeping."""
    rng = np.random.default_rng(seed)
    covar1 = rng.normal(0, 1, n)
    covar2 = rng.normal(0, 1, n)
    treated = (rng.random(n) < 1 / (1 + np.exp(-0.6 * covar1))).astype(int)
    return pd.DataFrame({"covar1": covar1.round(3), "covar2": covar2.round(3),
                         "treated": treated,
                         "y": (1.5 * treated + covar1 - 0.5 * covar2
                               + rng.normal(0, 1, n)).round(3)})


def _panel_frame(n_unit=25, n_t=12, seed=8):
    """Same trap in panel shape: `size` is the first continuous column, `y` is the DV."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_unit):
        fe = rng.normal(0, 1)
        starts = 6 if u % 2 == 0 else 10 ** 9
        for t in range(n_t):
            d = 1 if t >= starts else 0
            rows.append({"firm": f"f{u:02d}", "year": 2010 + t, "treated": d,
                         "size": round(float(rng.normal(10, 2)), 3),
                         "y": round(float(5 + fe + 0.8 * d + 0.05 * t
                                          + rng.normal(0, 0.5)), 3)})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("cid", ["psm", "ipw", "aipw", "double_ml", "causal_forest",
                                 "rosenbaum_bounds"])
def test_treatment_effect_family_binds_detected_outcome(cid, tmp_path):
    fp = _fp(_causal_frame(), tmp_path)
    res = _run(fp, cid, tmp_path)
    if not _ran(res):
        pytest.skip(f"{cid} degraded: {res.summary[:60]}")
    assert res.outcome == "y", f"{cid} bound {res.outcome!r} (first-continuous fallback?)"


@pytest.mark.parametrize("cid", ["event_study", "staggered_did", "first_difference",
                                 "mundlak", "hausman_test", "system_gmm"])
def test_panel_family_binds_detected_outcome(cid, tmp_path):
    fp = _fp(_panel_frame(), tmp_path)
    res = _run(fp, cid, tmp_path)
    if not _ran(res):
        pytest.skip(f"{cid} degraded: {res.summary[:60]}")
    assert res.outcome == "y", f"{cid} bound {res.outcome!r} instead of the DV"


def test_glmm_records_its_non_gaussian_outcome(tmp_path):
    """GLMM models the binary/count response, so it will not pick the continuous `y`; what
    it must not do is model something without saying so."""
    fp = _fp(_panel_frame(), tmp_path)
    res = _run(fp, "glmm", tmp_path)
    if not _ran(res):
        pytest.skip(f"glmm degraded: {res.summary[:60]}")
    assert res.outcome is not None


@pytest.mark.parametrize("cid", ["heckman_selection", "moderated_moderation"])
def test_remaining_h4c_branches_bind_detected_outcome(cid, tmp_path):
    fp = _fp(_reg_frame(), tmp_path)          # x1, x2, z, y — `y` is the high-conf DV
    res = _run(fp, cid, tmp_path)
    if not _ran(res):
        pytest.skip(f"{cid} degraded: {res.summary[:60]}")
    assert res.outcome == "y", f"{cid} bound {res.outcome!r}"


def test_no_h4c_branch_runs_without_reporting_its_outcome(tmp_path):
    """Ratchet, panel + causal shapes: same invariant as _SWEPT above."""
    silent = []
    for frame, ids in ((_causal_frame(seed=11), ["psm", "ipw", "aipw", "rosenbaum_bounds"]),
                       (_panel_frame(seed=12), ["event_study", "staggered_did",
                                                "first_difference", "mundlak",
                                                "hausman_test", "glmm"])):
        fp = _fp(frame, tmp_path, name=f"{ids[0]}.csv")
        for cid in ids:
            entry = _CAT.by_id(cid)
            if entry is None:
                continue
            res = run_analysis(fp, entry, output_root=str(tmp_path / f"h4c_{cid}"))
            if _ran(res) and res.outcome is None:
                silent.append(cid)
    assert not silent, f"ran but reported no outcome: {silent}"
