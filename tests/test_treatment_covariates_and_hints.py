"""H4d 冷审剩下的三条 nice-to-have —— 量完之后有两条其实是真 bug。

① 处理后协变量：`event = 1{T <= C}` is a descendant of the duration and hence of the
   treatment, so adjusting for it is post-treatment adjustment. Wave H4d made it reachable:
   before, `event` was (wrongly) taken as the TREATMENT and never sat in the covariate set.
   Measured (reference arm = age + biomarker only):
       事件率 1.00  含 event: PSM 失败(Singular matrix)   排除: -4.004
       事件率 0.99  含 event: -2.537   排除: -3.790   衰减 33%
       事件率 0.89  含 event: -0.606   排除: -1.825   衰减 67%
       事件率 0.64  含 event: -0.264   排除: -0.978   衰减 73%
   The cold reviewer measured |Δ| ≤ 0.04 under their censoring setup and filed it as a
   nice-to-have. Their caveat ("only bites when the event indicator is strongly determined by
   the duration") is right — that condition is just far more common than it sounds.

② `likely_treatment` 的置信度：the hint is shown to users (cli / study_report / web) AND it
   binds (resolve_treatment tier 3), yet a positional fallback read exactly like a name match.

③ 绑定词表：half the reviewer's "lost recall" list was never lost. The real find was the
   opposite direction — `control_arm` scored 2 (it contains `arm`), so the engine bound the
   CONTROL flag and IPW reported **+6.683** where the truth is **-8**.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog import Catalog
from researchforge.executor import run_analysis
from researchforge.executor.run import resolve_treatment
from researchforge.profiler import profile_dataset
from researchforge.profiler.semantics import survival_event_column, treatment_name_strength

_CAT = Catalog.load()


def _fp(df, tmp_path, name="d.csv"):
    csv = tmp_path / name
    df.to_csv(csv, index=False)
    return profile_dataset(csv)


def _ran(res) -> bool:
    return bool(res.estimates) and not any(
        k in res.summary for k in ("失败", "跳过", "暂未接入", "未检测到"))


def _survival(censor_scale=10.0, n=400, seed=2):
    rng = np.random.default_rng(seed)
    age = rng.integers(30, 80, n)
    trt = rng.integers(0, 2, n)
    T = rng.exponential(20 * np.exp(-0.4 * trt + 0.01 * (age - 50)))
    C = rng.exponential(censor_scale)
    return pd.DataFrame({"duration": np.round(np.minimum(T, C), 2),
                         "event": (T <= C).astype(int), "treatment": trt, "age": age,
                         "biomarker": rng.normal(5, 1.5, n).round(3)})


# ── ① 处理后协变量 ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("cid", ["psm", "ipw", "aipw"])
def test_the_event_indicator_stays_out_of_the_auto_covariates(cid, tmp_path):
    fp = _fp(_survival(), tmp_path)
    assert survival_event_column(fp) == "event"
    entry = _CAT.by_id(cid)
    if entry is None:
        pytest.skip(f"{cid} not in catalog")
    res = run_analysis(fp, entry, output_root=str(tmp_path / f"o_{cid}"))
    if not _ran(res):
        pytest.skip(f"{cid} degraded: {res.summary[:70]}")
    assert "排除在自动协变量之外" in res.summary, "silently dropping it is its own problem"
    assert "'event'" in res.summary


def test_excluding_the_event_moves_the_estimate_materially(tmp_path):
    """Not bookkeeping: including it attenuated the ATT by 67% on this frame."""
    fp = _fp(_survival(), tmp_path)
    auto = run_analysis(fp, _CAT.by_id("psm"), output_root=str(tmp_path / "auto"))
    forced = run_analysis(fp, _CAT.by_id("psm"), output_root=str(tmp_path / "forced"),
                          config={"covariates": ["age", "biomarker", "event"]})
    if not (_ran(auto) and _ran(forced)):
        pytest.skip("psm degraded")
    assert auto.estimates["att"] < forced.estimates["att"], (
        f"auto={auto.estimates['att']:.3f} should be further from 0 than "
        f"forced={forced.estimates['att']:.3f}"
    )


def test_an_explicit_covariate_list_is_still_the_users_call(tmp_path):
    fp = _fp(_survival(), tmp_path)
    res = run_analysis(fp, _CAT.by_id("psm"), output_root=str(tmp_path / "cfg"),
                       config={"covariates": ["age", "biomarker", "event"]})
    if not _ran(res):
        pytest.skip("psm degraded")
    bal = pd.read_csv(f"{res.output_dir}/balance.csv")
    assert "event" in bal.iloc[:, 0].tolist()


def test_a_constant_covariate_is_dropped_not_a_singular_matrix(tmp_path):
    """Surfaced by ①: at a saturated event rate the indicator has ONE value, so the profiler
    types it `count`, the survival exclusion cannot see it, and the propensity model went
    singular — the user got a raw "Singular matrix"."""
    df = _survival(censor_scale=1e6)          # nobody is censored -> event is constant 1
    assert df["event"].nunique() == 1
    fp = _fp(df, tmp_path, name="sat.csv")
    res = run_analysis(fp, _CAT.by_id("psm"), output_root=str(tmp_path / "sat"))
    # NB: assert on the OUTCOME of the run, not on the absence of the string — the disclosure
    # message itself names "Singular matrix" as the thing being avoided.
    assert "倾向得分匹配失败" not in res.summary, res.summary[:200]
    assert "已剔除常数协变量" in res.summary and "'event'" in res.summary
    assert _ran(res) and res.estimates["att"] < 0


def test_a_non_survival_frame_keeps_its_binary_covariates(tmp_path):
    """The exclusion is gated on survival SHAPE — a binary called `status` in ordinary
    cross-sectional data is nobody's censoring indicator."""
    rng = np.random.default_rng(8)
    n = 300
    df = pd.DataFrame({"income": rng.normal(50, 10, n).round(2),
                       "treated": rng.integers(0, 2, n),
                       "status": rng.integers(0, 2, n),
                       "age": rng.integers(20, 70, n)})
    fp = _fp(df, tmp_path, name="cs.csv")
    assert survival_event_column(fp) is None
    res = run_analysis(fp, _CAT.by_id("psm"), output_root=str(tmp_path / "cs"))
    if not _ran(res):
        pytest.skip("psm degraded")
    assert "排除在自动协变量之外" not in res.summary


# ── ② likely_treatment 的置信度 ──────────────────────────────────────────────
@pytest.mark.parametrize("col,expected", [("treated", "high"), ("group", "medium"),
                                          ("study_group", "medium"), ("flag_b", "low")])
def test_the_treatment_hint_carries_its_evidence(col, expected, tmp_path):
    rng = np.random.default_rng(5)
    n = 300
    df = pd.DataFrame({"y": rng.normal(0, 1, n).round(3),
                       "flag_a": rng.integers(0, 2, n),
                       col: rng.integers(0, 2, n)})
    fp = _fp(df, tmp_path, name=f"h_{col}.csv")
    assert fp.likely_treatment_confidence == expected, (
        f"{col}: hint={fp.likely_treatment!r} conf={fp.likely_treatment_confidence!r}")


def test_the_report_qualifies_a_positional_treatment_hint(tmp_path):
    from researchforge.study_report import _fp_quality_lines

    rng = np.random.default_rng(5)
    n = 300
    df = pd.DataFrame({"y": rng.normal(0, 1, n).round(3),
                       "flag_a": rng.integers(0, 2, n), "flag_b": rng.integers(0, 2, n)})
    fp = _fp(df, tmp_path, name="qual.csv")
    text = "\n".join(_fp_quality_lines(fp, None))
    assert "可能的处理变量" in text, text
    assert "无名称信号" in text, text


# ── ③ 绑定词表：control 陷阱 + 真正缺的召回 ─────────────────────────────────
@pytest.mark.parametrize("name", ["control", "ctrl", "control_arm", "control_group",
                                  "placebo", "placebo_arm", "comparator", "sham"])
def test_a_control_flag_is_never_bound(name):
    """Binding it makes 1 = control, i.e. the SAME contrast with the sign inverted.
    `control_arm` used to score 2 because it contains `arm`."""
    assert treatment_name_strength(name) == 0


def test_a_control_flag_does_not_flip_the_sign_end_to_end(tmp_path):
    rng = np.random.default_rng(9)
    n = 400
    treated = rng.integers(0, 2, n)
    x = rng.normal(0, 1, n)
    df = pd.DataFrame({"score": (50 - 8 * treated + 2 * x + rng.normal(0, 5, n)).round(2),
                       "control_arm": 1 - treated, "x": x.round(3), "treated": treated})
    fp = _fp(df, tmp_path, name="ctl.csv")
    assert fp.binary_columns[0] == "control_arm"          # first in file order
    assert fp.likely_treatment == "treated"
    assert resolve_treatment(fp, {}, fp.binary_columns) == "treated"
    res = run_analysis(fp, _CAT.by_id("ipw"), output_root=str(tmp_path / "ipw"))
    if not _ran(res):
        pytest.skip("ipw degraded")
    assert res.treatment == "treated"
    assert res.estimates["ate"] < 0, "the truth is -8; a positive ATE means the control flag"


@pytest.mark.parametrize("name", ["study_group", "case_group", "experimental_group",
                                  "case_cohort", "study_grp"])
def test_the_compound_rule_restores_real_recall(name):
    assert treatment_name_strength(name) == 1      # weak: never outranks a strong word


@pytest.mark.parametrize("name", ["intervention_group", "exposure_group", "treatment_group",
                                  "dose_group", "arm_group", "treat_cohort"])
def test_names_the_strong_vocabulary_already_covered(name):
    """Half the reviewer's 'lost recall' list was never lost — these contain a strong word."""
    assert treatment_name_strength(name) == 2


@pytest.mark.parametrize("name", ["age_group", "blood_group", "risk_group", "income_group",
                                  "group_id", "group_size", "region_group", "skin_condition",
                                  "patient_group", "tx_date"])
def test_the_compound_rule_did_not_reopen_the_false_positives(name):
    assert treatment_name_strength(name) == 0
