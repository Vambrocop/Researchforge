"""Tests for the EPIDEMIOLOGY family (clinical biostatistics triangle):
diagnostic_test_eval, epi_risk_measures, calibration_assessment.

Known-value cases are hand-computed in the docstrings. Honest-degrade paths assert the
Chinese "跳过" / honest message and no crash.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry(eid: str, method: str) -> AnalysisEntry:
    return AnalysisEntry(
        id=eid,
        method=method,
        domain="epidemiology",
        family="epidemiology",
        goal="describe",
        preconditions=Precondition(min_rows=1),
    )


def _csv(tmp_path: Path, name: str, df: pd.DataFrame) -> Path:
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


# --------------------------------------------------------------------------- #
# 1) diagnostic_test_eval
# --------------------------------------------------------------------------- #
def test_diagnostic_continuous_perfect_separation(tmp_path: Path) -> None:
    """A score that perfectly separates the two classes -> AUC == 1.0, and at the
    Youden cutoff sensitivity == specificity == 1.0.
       disease = 5 zeros then 5 ones; score increasing -> all cases score higher."""
    n = 30
    disease = np.array([0] * (n // 2) + [1] * (n // 2))
    score = np.arange(n, dtype=float)  # 0..29, increasing with disease
    csv = _csv(tmp_path, "diag.csv", pd.DataFrame({"disease": disease, "score": score}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("diagnostic_test_eval", "Diagnostic"),
                       output_root=str(tmp_path / "o"),
                       config={"truth": "disease", "test": "score"})
    e = res.estimates
    assert math.isclose(e["auc"], 1.0, abs_tol=1e-9)
    assert math.isclose(e["sens"], 1.0, abs_tol=1e-9)
    assert math.isclose(e["spec"], 1.0, abs_tol=1e-9)
    assert math.isclose(e["accuracy"], 1.0, abs_tol=1e-9)
    # AUC CI for a perfect score should hug 1.0 from below
    assert e["auc_ci_low"] <= 1.0 and e["auc_ci_high"] <= 1.0 + 1e-9
    out = Path(res.output_dir)
    assert (out / "roc_curve.csv").exists()
    assert (out / "diagnostic_metrics.csv").exists()


def test_diagnostic_continuous_auc_known_value(tmp_path: Path) -> None:
    """AUC == Mann-Whitney probability P(score_case > score_control). With
       controls = [1, 2, 3], cases = [2, 4, 6]:
       pairs (case>control): (2>1)=1, (2>2)=0.5, (2>3)=0; (4>all)=3; (6>all)=3
       sum = 1.5 + 3 + 3 = 7.5 over 9 pairs -> AUC = 0.8333..."""
    disease = [0, 0, 0, 1, 1, 1]
    score = [1.0, 2.0, 3.0, 2.0, 4.0, 6.0]
    csv = _csv(tmp_path, "diag.csv", pd.DataFrame({"disease": disease, "score": score}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("diagnostic_test_eval", "Diagnostic"),
                       output_root=str(tmp_path / "o"),
                       config={"truth": "disease", "test": "score"})
    assert math.isclose(res.estimates["auc"], 7.5 / 9.0, abs_tol=1e-6)


def test_diagnostic_binary_test_2x2_known(tmp_path: Path) -> None:
    """Binary test vs binary truth, hand-built 2x2:
       TP=40, FN=10 (cases=50) ; FP=5, TN=45 (controls=50)
       sens = 40/50 = 0.80 ; spec = 45/50 = 0.90
       ppv = 40/45 = 0.8889 ; npv = 45/55 = 0.8182
       accuracy = 85/100 = 0.85 ; LR+ = .8/.1 = 8 ; LR- = .2/.9 = 0.2222"""
    truth = [1] * 50 + [0] * 50
    test = [1] * 40 + [0] * 10 + [1] * 5 + [0] * 45
    csv = _csv(tmp_path, "bin.csv", pd.DataFrame({"truth": truth, "testres": test}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("diagnostic_test_eval", "Diagnostic"),
                       output_root=str(tmp_path / "o"),
                       config={"truth": "truth", "test": "testres"})
    e = res.estimates
    assert math.isclose(e["sens"], 0.80, abs_tol=1e-4)
    assert math.isclose(e["spec"], 0.90, abs_tol=1e-4)
    assert math.isclose(e["ppv"], 40.0 / 45.0, abs_tol=1e-4)
    assert math.isclose(e["npv"], 45.0 / 55.0, abs_tol=1e-4)
    assert math.isclose(e["accuracy"], 0.85, abs_tol=1e-4)
    assert math.isclose(e["lr_plus"], 8.0, abs_tol=1e-3)
    assert math.isclose(e["lr_minus"], 0.2 / 0.9, abs_tol=1e-3)
    # no ROC for a binary test
    assert "auc" not in e


def test_diagnostic_degrade_one_class(tmp_path: Path) -> None:
    """Gold standard all one class -> honest 跳过, no crash."""
    csv = _csv(tmp_path, "d.csv", pd.DataFrame({
        "disease": [1, 1, 1, 1, 1], "score": [0.1, 0.2, 0.3, 0.4, 0.5]}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("diagnostic_test_eval", "Diagnostic"),
                       output_root=str(tmp_path / "o"),
                       config={"truth": "disease", "test": "score"})
    assert "跳过" in res.summary
    assert "auc" not in res.estimates


def test_diagnostic_degrade_no_truth(tmp_path: Path) -> None:
    """No binary column at all -> honest 跳过."""
    csv = _csv(tmp_path, "d.csv", pd.DataFrame({"label": ["a", "b", "c", "d"]}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("diagnostic_test_eval", "Diagnostic"),
                       output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary


# --------------------------------------------------------------------------- #
# 2) epi_risk_measures
# --------------------------------------------------------------------------- #
def test_risk_measures_known_2x2(tmp_path: Path) -> None:
    """Cohort 2x2:
       exposed:   20 cases / 100 total  -> risk_exp   = 0.20
       unexposed: 10 cases / 100 total  -> risk_unexp = 0.10
       RR = 2.00 ; OR = (20*90)/(80*10) = 1800/800 = 2.25
       RD = 0.10 ; NNT = 1/0.10 = 10 ; AR% in exposed = (RR-1)/RR = 50%
       a=20,b=80,c=10,d=90."""
    exposure = [1] * 100 + [0] * 100
    outcome = ([1] * 20 + [0] * 80) + ([1] * 10 + [0] * 90)
    csv = _csv(tmp_path, "rr.csv", pd.DataFrame({"exposure": exposure, "disease": outcome}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("epi_risk_measures", "Risk measures"),
                       output_root=str(tmp_path / "o"),
                       config={"exposure": "exposure", "outcome": "disease"})
    e = res.estimates
    assert math.isclose(e["rr"], 2.0, abs_tol=1e-4)
    assert math.isclose(e["or"], 2.25, abs_tol=1e-4)
    assert math.isclose(e["rd"], 0.10, abs_tol=1e-4)
    assert math.isclose(e["nnt"], 10.0, abs_tol=1e-2)
    assert math.isclose(e["ar_pct"], 50.0, abs_tol=1e-2)
    # RR CI brackets the point estimate. NOTE: this 2x2 is borderline — the Woolf
    # log-RR 95% CI is [0.986, 4.06], i.e. it just barely includes 1 (SE(lnRR)=
    # sqrt(1/20-1/100+1/10-1/100)=0.3606, exp(ln2-1.96*0.3606)=0.986), so we do NOT
    # assert exclusion of 1 here — the implementation is right, the effect is just not sig.
    assert e["rr_ci_low"] < 2.0 < e["rr_ci_high"]
    assert math.isclose(e["rr_ci_low"], 0.986, abs_tol=2e-3)
    out = Path(res.output_dir)
    assert (out / "two_by_two.csv").exists()


def test_risk_measures_or_log_ci_known(tmp_path: Path) -> None:
    """Woolf log-CI sanity for OR=2.25, a=20,b=80,c=10,d=90:
       SE(lnOR) = sqrt(1/20 + 1/80 + 1/10 + 1/90) = sqrt(0.18861...) = 0.43429
       lnOR = ln 2.25 = 0.81093
       95% CI = exp(0.81093 +/- 1.95996*0.43429) = [0.9603, 5.2723]."""
    exposure = [1] * 100 + [0] * 100
    outcome = ([1] * 20 + [0] * 80) + ([1] * 10 + [0] * 90)
    csv = _csv(tmp_path, "or.csv", pd.DataFrame({"exposure": exposure, "disease": outcome}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("epi_risk_measures", "Risk measures"),
                       output_root=str(tmp_path / "o"),
                       config={"exposure": "exposure", "outcome": "disease"})
    e = res.estimates
    se = math.sqrt(1 / 20 + 1 / 80 + 1 / 10 + 1 / 90)
    lo = 2.25 * math.exp(-1.959963984540054 * se)
    hi = 2.25 * math.exp(1.959963984540054 * se)
    assert math.isclose(e["or_ci_low"], lo, abs_tol=1e-3)
    assert math.isclose(e["or_ci_high"], hi, abs_tol=1e-3)


def test_risk_measures_zero_cell_correction(tmp_path: Path) -> None:
    """A zero cell triggers the Haldane-Anscombe 0.5 correction and disclosure;
    OR/RR stay finite. exposed: 10 cases/10 (risk 1.0); unexposed: 0 cases/10."""
    exposure = [1] * 10 + [0] * 10
    outcome = [1] * 10 + [0] * 10  # all exposed are cases, no unexposed cases
    csv = _csv(tmp_path, "z.csv", pd.DataFrame({"exposure": exposure, "disease": outcome}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("epi_risk_measures", "Risk measures"),
                       output_root=str(tmp_path / "o"),
                       config={"exposure": "exposure", "outcome": "disease"})
    assert "Haldane" in res.summary or "校正" in res.summary
    assert math.isfinite(res.estimates["or"]) and res.estimates["or"] > 0


def test_risk_measures_degrade(tmp_path: Path) -> None:
    """Only one binary column -> cannot build a 2x2 -> honest 跳过."""
    csv = _csv(tmp_path, "x.csv", pd.DataFrame({"exposure": [1, 0, 1, 0]}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("epi_risk_measures", "Risk measures"),
                       output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
    assert "rr" not in res.estimates


# --------------------------------------------------------------------------- #
# 3) calibration_assessment
# --------------------------------------------------------------------------- #
def test_calibration_well_calibrated_low_brier(tmp_path: Path) -> None:
    """Construct predictions that are well-calibrated: for each of K probability
    levels, make the observed event rate equal that level. With probs in
    {0.1, 0.3, 0.5, 0.7, 0.9} and 100 subjects each whose outcome rate matches,
    the Brier score is the irreducible binomial variance mean = mean p(1-p):
       mean of {0.09,0.21,0.25,0.21,0.09} = 0.17. Calibration slope ~ 1.0."""
    rng = np.random.default_rng(0)
    levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    per = 200
    probs, outs = [], []
    for lv in levels:
        n1 = int(round(lv * per))
        col = [1] * n1 + [0] * (per - n1)
        rng.shuffle(col)
        probs.extend([lv] * per)
        outs.extend(col)
    csv = _csv(tmp_path, "cal.csv", pd.DataFrame({"pred": probs, "event": outs}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("calibration_assessment", "Calibration"),
                       output_root=str(tmp_path / "o"),
                       config={"prob": "pred", "outcome": "event", "bins": 5})
    e = res.estimates
    # Brier ~ mean p(1-p) = 0.17 for perfectly-calibrated predictions
    assert math.isclose(e["brier"], 0.17, abs_tol=0.02)
    # reliability (mis-calibration) should be tiny for a calibrated model
    assert e["brier_reliability"] < 0.01
    # calibration slope close to 1 for a well-calibrated model
    assert 0.7 < e["cal_slope"] < 1.3
    # HL should NOT reject calibration (p > 0.05 expected for a calibrated model)
    assert e["hl_p"] > 0.01
    out = Path(res.output_dir)
    assert (out / "calibration_bins.csv").exists()
    assert (out / "calibration_plot.png").exists()


def test_calibration_overconfident_slope_below_one(tmp_path: Path) -> None:
    """An over-confident model (predictions pushed toward 0/1 harder than the truth
    warrants) should yield a calibration slope < 1. We make predicted probs more
    extreme than the realised rates."""
    rng = np.random.default_rng(1)
    # true event rate 0.5 in both groups, but the model predicts 0.05 and 0.95
    n = 400
    probs = [0.05] * n + [0.95] * n
    outs = ([1] * (n // 2) + [0] * (n // 2)) + ([1] * (n // 2) + [0] * (n // 2))
    csv = _csv(tmp_path, "oc.csv", pd.DataFrame({"pred": probs, "event": outs}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("calibration_assessment", "Calibration"),
                       output_root=str(tmp_path / "o"),
                       config={"prob": "pred", "outcome": "event", "bins": 5})
    assert res.estimates["cal_slope"] < 1.0
    assert res.estimates["brier_reliability"] > 0.05  # clear mis-calibration


def test_calibration_degrade_no_prob_column(tmp_path: Path) -> None:
    """No column lies in [0,1] -> honest message about needing a 0-1 probability column."""
    csv = _csv(tmp_path, "np.csv", pd.DataFrame({
        "score": [10.0, 20.0, 30.0, 40.0] * 10,
        "event": ([1, 0] * 20)}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("calibration_assessment", "Calibration"),
                       output_root=str(tmp_path / "o"),
                       config={"outcome": "event"})
    assert "预测概率" in res.summary or "跳过" in res.summary
    assert "brier" not in res.estimates
