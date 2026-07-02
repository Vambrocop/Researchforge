"""Tests for the RELIABILITY-ENGINEERING family (family=reliability,
domain=engineering): weibull_life_analysis, reliability_growth,
accelerated_life_test.

Synthetic data with KNOWN answers: recovered parameters are checked against the
ground-truth used to simulate them, within tolerances appropriate to the sample
size. Honest-degrade paths assert the Chinese "跳过" message and no crash. Mirrors
tests/test_spc.py / tests/test_techno_economic.py: write CSV -> profile_dataset ->
AnalysisEntry/Precondition -> run_analysis -> assert on res.estimates / res.summary.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry(eid: str, method: str, goal: str = "describe") -> AnalysisEntry:
    return AnalysisEntry(
        id=eid,
        method=method,
        domain="engineering",
        family="reliability",
        goal=goal,
        preconditions=Precondition(min_rows=1),
    )


def _csv(tmp_path: Path, name: str, df: pd.DataFrame) -> Path:
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


# --------------------------------------------------------------------------- #
# 1) weibull_life_analysis
# --------------------------------------------------------------------------- #
def test_weibull_recovers_known_params(tmp_path: Path) -> None:
    """Simulate Weibull(beta=2, eta=100): t = eta * E^(1/beta), E ~ Exp(1)
    (numpy default_rng.weibull(beta) returns eta=1 samples; scale by eta).
    MLE should recover beta ~= 2 and eta ~= 100 within a sampling tolerance; for a
    Weibull, MTTF = eta*Gamma(1.5) ~= 88.6 and B50 (median) ~= eta*(ln2)^(1/2) ~= 83.3."""
    rng = np.random.default_rng(123)
    beta_true, eta_true, n = 2.0, 100.0, 400
    t = eta_true * rng.weibull(beta_true, size=n)
    csv = _csv(tmp_path, "w.csv", pd.DataFrame({"life_hours": t}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("weibull_life_analysis", "Weibull Life Data"),
                       output_root=str(tmp_path / "o"),
                       config={"life": "life_hours"})
    e = res.estimates
    assert math.isclose(e["weibull_beta"], beta_true, rel_tol=0.15)
    assert math.isclose(e["weibull_eta"], eta_true, rel_tol=0.10)
    # MTTF = eta*Gamma(1+1/beta); for beta=2 -> eta*Gamma(1.5)=eta*0.8862
    assert math.isclose(e["mttf"], eta_true * math.gamma(1.5), rel_tol=0.10)
    # B50 (median) = eta*(ln 2)^(1/beta)
    assert math.isclose(e["b50_life"], eta_true * (math.log(2)) ** 0.5, rel_tol=0.12)
    # B10 < B50 < eta (monotone percentiles)
    assert e["b10_life"] < e["b50_life"] < e["weibull_eta"]
    assert e["failure_rate_at_eta"] > 0
    assert e["n_failures"] == float(n)
    assert "aic_weibull" in e
    out = Path(res.output_dir)
    assert (out / "weibull_life_percentiles.csv").exists()
    # complement disclosure vs parametric_survival
    assert "parametric_survival" in res.summary


def test_weibull_censored_case(tmp_path: Path) -> None:
    """Right-censored case: type-I censoring at C=120 of a Weibull(beta=2, eta=100)
    sample. event=1 for observed failures (t<C), event=0 for suspensions (t>=C, time
    set to C). The censored MLE must still recover beta and eta reasonably and report
    the number censored; ignoring censoring would bias eta downward."""
    rng = np.random.default_rng(7)
    beta_true, eta_true, n, C = 2.0, 100.0, 400, 120.0
    raw = eta_true * rng.weibull(beta_true, size=n)
    event = (raw < C).astype(int)
    t_obs = np.minimum(raw, C)
    df = pd.DataFrame({"life_hours": t_obs, "failed": event})
    csv = _csv(tmp_path, "wc.csv", df)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("weibull_life_analysis", "Weibull Life Data"),
                       output_root=str(tmp_path / "o"),
                       config={"life": "life_hours", "event": "failed"})
    e = res.estimates
    assert e["n_censored"] >= 1.0  # some suspensions exist
    assert e["n_failures"] + e["n_censored"] == float(n)
    # censored MLE still close to the truth
    assert math.isclose(e["weibull_beta"], beta_true, rel_tol=0.20)
    assert math.isclose(e["weibull_eta"], eta_true, rel_tol=0.18)
    assert "右删失" in res.summary or "删失" in res.summary


def test_weibull_degrade_non_numeric(tmp_path: Path) -> None:
    """No numeric life column -> honest 跳过, no crash."""
    csv = _csv(tmp_path, "txt.csv", pd.DataFrame({"label": ["a", "b", "c"]}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("weibull_life_analysis", "Weibull Life Data"),
                       output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
    assert "weibull_beta" not in res.estimates


# --------------------------------------------------------------------------- #
# 2) reliability_growth (Crow-AMSAA)
# --------------------------------------------------------------------------- #
def test_growth_recovers_improving_beta(tmp_path: Path) -> None:
    """Generate failure times from a known power-law NHPP N(t)=lambda*t^beta with
    beta<1 (improving). The i-th expected failure time is t_i = (i/lambda)^(1/beta);
    using these noiseless arrival times the time-terminated MLE
    beta = n / sum(ln(T/t_i)) recovers beta closely, and cumulative MTBF (= t/i)
    INCREASES with failure number (reliability growing)."""
    beta_true, lam = 0.6, 0.5
    n = 30
    i = np.arange(1, n + 1, dtype=float)
    t = (i / lam) ** (1.0 / beta_true)  # exact CAMSAA arrival times
    csv = _csv(tmp_path, "g.csv", pd.DataFrame({"failure_time": t}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("reliability_growth", "Crow-AMSAA"),
                       output_root=str(tmp_path / "o"),
                       config={"life": "failure_time"})
    e = res.estimates
    # noiseless mean arrival times give a slightly biased MLE; failure-truncated form
    # β̂=(n-1)/Σ recovers ≈0.64 here — within 10% of the true 0.6.
    assert math.isclose(e["camsaa_beta"], beta_true, rel_tol=0.10)
    assert e["camsaa_beta"] < 1.0  # improving
    assert e["n_failures"] == float(n)
    # instantaneous MTBF > cumulative MTBF while improving (beta<1)
    assert e["mtbf_instantaneous"] > e["mtbf_cumulative"]
    # observed cumulative MTBF (t/i) increases -> reliability growing
    out = Path(res.output_dir)
    tbl = pd.read_csv(out / "reliability_growth.csv")
    cm = tbl["cumulative_mtbf_obs"].to_numpy()
    assert cm[-1] > cm[0]
    assert (out / "duane_plot.png").exists() or "Duane" in res.summary
    assert "改善" in res.summary


def test_growth_inter_arrival_input(tmp_path: Path) -> None:
    """Inter-arrival input (not monotone increasing) is cumulated automatically.
    Same underlying improving process -> beta<1 recovered, and the summary discloses
    the inter-arrival conversion."""
    beta_true, lam = 0.6, 0.5
    n = 30
    i = np.arange(1, n + 1, dtype=float)
    cum = (i / lam) ** (1.0 / beta_true)
    inter = np.diff(np.concatenate([[0.0], cum]))  # inter-arrival gaps
    csv = _csv(tmp_path, "gi.csv", pd.DataFrame({"gap": inter}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("reliability_growth", "Crow-AMSAA"),
                       output_root=str(tmp_path / "o"),
                       config={"life": "gap", "input_type": "interarrival"})
    e = res.estimates
    assert e["camsaa_beta"] < 1.0
    assert math.isclose(e["camsaa_beta"], beta_true, rel_tol=0.10)
    assert "间隔" in res.summary  # disclosed inter-arrival cumulation


def test_growth_default_failure_truncated_disclosure_and_code(tmp_path: Path) -> None:
    """Default run (no termination_time) is FAILURE-truncated (Rigdon & Basu): the
    ⚠ note must say it used 失效截尾 (not 时间截尾, which was the pre-fix bug — the
    note used to unconditionally claim 时间截尾 regardless of what was computed), and
    the emitted analysis_code.py must use the (n-1)-numerator formula and reproduce
    the reported beta (previously the code echo used the n-numerator time-truncated
    form even for the failure-truncated default, so copying it gave a different beta)."""
    beta_true, lam = 0.6, 0.5
    n = 30
    i = np.arange(1, n + 1, dtype=float)
    t = (i / lam) ** (1.0 / beta_true)
    csv = _csv(tmp_path, "g3.csv", pd.DataFrame({"failure_time": t}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("reliability_growth", "Crow-AMSAA"),
                       output_root=str(tmp_path / "o"),
                       config={"life": "failure_time"})
    e = res.estimates
    assert "本次采用失效截尾 MLE" in res.summary
    assert "本次采用时间截尾 MLE" not in res.summary
    out = Path(res.output_dir)
    code_txt = (out / "analysis_code.py").read_text(encoding="utf-8")
    assert "(n - 1)" in code_txt
    # reproduce the reported beta using the emitted (n-1)-numerator formula against the
    # cumulative failure times actually written to the CSV product
    tbl = pd.read_csv(out / "reliability_growth.csv")
    t_arr = tbl["cumulative_time"].to_numpy()
    T = e["total_time"]
    beta_reproduced = (t_arr.size - 1) / np.sum(np.log(T / t_arr))
    assert math.isclose(beta_reproduced, e["camsaa_beta"], rel_tol=1e-3)


def test_growth_termination_time_is_time_truncated_disclosure_and_code(tmp_path: Path) -> None:
    """Explicit config['termination_time'] switches to TIME-truncated MLE: the ⚠ note
    must say 时间截尾 and the emitted code must use the n-numerator formula (no
    (n-1)), reproducing the reported beta."""
    beta_true, lam = 0.6, 0.5
    n = 30
    i = np.arange(1, n + 1, dtype=float)
    t = (i / lam) ** (1.0 / beta_true)
    term = float(t[-1]) * 1.1  # test continued a bit past the last observed failure
    csv = _csv(tmp_path, "g4.csv", pd.DataFrame({"failure_time": t}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("reliability_growth", "Crow-AMSAA"),
                       output_root=str(tmp_path / "o"),
                       config={"life": "failure_time", "termination_time": term})
    e = res.estimates
    assert "本次采用时间截尾 MLE" in res.summary
    assert "本次采用失效截尾 MLE" not in res.summary
    out = Path(res.output_dir)
    code_txt = (out / "analysis_code.py").read_text(encoding="utf-8")
    assert "(n - 1)" not in code_txt
    assert "beta = n / np.sum(np.log(T / t))" in code_txt
    tbl = pd.read_csv(out / "reliability_growth.csv")
    t_arr = tbl["cumulative_time"].to_numpy()
    T = e["total_time"]
    assert math.isclose(T, term, rel_tol=1e-6)
    beta_reproduced = t_arr.size / np.sum(np.log(T / t_arr))
    assert math.isclose(beta_reproduced, e["camsaa_beta"], rel_tol=1e-3)


def test_growth_degrade_too_few(tmp_path: Path) -> None:
    """Fewer than 3 failure times -> honest 跳过, no crash."""
    csv = _csv(tmp_path, "g2.csv", pd.DataFrame({"failure_time": [10.0, 25.0]}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("reliability_growth", "Crow-AMSAA"),
                       output_root=str(tmp_path / "o"),
                       config={"life": "failure_time"})
    assert "跳过" in res.summary
    assert "camsaa_beta" not in res.estimates


# --------------------------------------------------------------------------- #
# 3) accelerated_life_test
# --------------------------------------------------------------------------- #
def test_alt_arrhenius_recovers_ea(tmp_path: Path) -> None:
    """Simulate an Arrhenius life-stress relation with a KNOWN activation energy.
    ln(L) = a + b*(1/T), b = Ea/k. With Ea=0.7 eV, k=8.617e-5 -> b ~= 8124 (K).
    Temperatures in Celsius (50, 85, 125, ...) get converted to Kelvin internally.
    The fitted slope b>0 (life longer at LOWER temperature), the recovered Ea is the
    right sign/magnitude, and the acceleration factor toward the LOWEST stress
    (mildest = use condition) is > 1 (longer life there)."""
    k = 8.617e-5
    Ea = 0.7
    b_true = Ea / k
    rng = np.random.default_rng(11)
    temps_c = np.array([50.0, 85.0, 125.0, 150.0])
    rows = []
    a_true = 5.0
    for tc in temps_c:
        Tk = tc + 273.15
        mean_lnL = a_true + b_true * (1.0 / Tk)
        for _ in range(15):
            lnL = mean_lnL + rng.normal(0.0, 0.05)  # small log-scatter
            rows.append({"temperature": tc, "life_hours": float(np.exp(lnL))})
    df = pd.DataFrame(rows)
    csv = _csv(tmp_path, "alt.csv", df)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("accelerated_life_test", "Accelerated Life Test", "predict"),
                       output_root=str(tmp_path / "o"),
                       config={"life": "life_hours", "stress": "temperature",
                               "model": "arrhenius"})
    e = res.estimates
    assert e["model_slope"] > 0  # life increases as 1/T increases (lower T)
    # recovered activation energy ~ 0.7 eV
    assert math.isclose(e["activation_energy"], Ea, rel_tol=0.15)
    # acceleration factor toward the mildest (lowest) stress > 1 (longer life there)
    assert e["acceleration_factor"] > 1.0
    assert e["median_life_use"] > 0
    # lognormal mean >= median (mean = median * exp(sigma_log^2/2))
    assert e["mean_life_use"] >= e["median_life_use"]
    assert e["n_stress_levels"] == 4.0
    out = Path(res.output_dir)
    assert (out / "accelerated_life_test.csv").exists()
    assert "外推" in res.summary  # extrapolation risk disclosure
    # median-vs-mean label must be present; no bare "MTTF≈" claiming the mean
    # (life_use = exp(a+b*x) is the lognormal MEDIAN, not the mean)
    assert "中位" in res.summary and "均值" in res.summary
    assert "MTTF≈" not in res.summary


def test_alt_inverse_power_explicit(tmp_path: Path) -> None:
    """Inverse power law on a voltage stress: ln(L) = a + b*ln(V), b<0 (higher
    voltage -> shorter life). Forced via config model. Use stress defaults to the
    minimum voltage; acceleration factor to that mildest condition > 1."""
    rng = np.random.default_rng(3)
    a_true, b_true = 30.0, -6.0
    volts = np.array([100.0, 150.0, 200.0, 260.0])
    rows = []
    for v in volts:
        mean_lnL = a_true + b_true * math.log(v)
        for _ in range(15):
            lnL = mean_lnL + rng.normal(0.0, 0.05)
            rows.append({"voltage": v, "life_hours": float(np.exp(lnL))})
    df = pd.DataFrame(rows)
    csv = _csv(tmp_path, "ip.csv", df)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("accelerated_life_test", "Accelerated Life Test", "predict"),
                       output_root=str(tmp_path / "o"),
                       config={"life": "life_hours", "stress": "voltage",
                               "model": "inverse_power"})
    e = res.estimates
    assert math.isclose(e["model_slope"], b_true, rel_tol=0.10)
    assert e["acceleration_factor"] > 1.0  # mildest (lowest V) -> longer life
    assert math.isnan(e["activation_energy"])  # Ea only for Arrhenius
    assert e["use_stress"] == 100.0  # min observed stress (default)


def test_alt_use_stress_override(tmp_path: Path) -> None:
    """A user-supplied use_stress below the tested range yields a larger acceleration
    factor than the default (min observed) and is reported as use_stress."""
    rng = np.random.default_rng(5)
    k = 8.617e-5
    b_true = 0.7 / k
    temps_c = np.array([60.0, 100.0, 140.0])
    rows = []
    for tc in temps_c:
        Tk = tc + 273.15
        mean_lnL = 5.0 + b_true * (1.0 / Tk)
        for _ in range(20):
            rows.append({"temperature": tc,
                         "life_hours": float(np.exp(mean_lnL + rng.normal(0, 0.05)))})
    csv = _csv(tmp_path, "alt2.csv", pd.DataFrame(rows))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("accelerated_life_test", "Accelerated Life Test", "predict"),
                       output_root=str(tmp_path / "o"),
                       config={"life": "life_hours", "stress": "temperature",
                               "model": "arrhenius", "use_stress": 25.0})
    e = res.estimates
    assert e["use_stress"] == 25.0
    assert e["acceleration_factor"] > 1.0


def test_alt_degrade_single_stress(tmp_path: Path) -> None:
    """Only one distinct stress level -> cannot fit a life-stress slope -> 跳过."""
    df = pd.DataFrame({"temperature": [125.0] * 8,
                       "life_hours": [90.0, 110.0, 95.0, 105.0, 88.0, 120.0, 99.0, 101.0]})
    csv = _csv(tmp_path, "alt1.csv", df)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("accelerated_life_test", "Accelerated Life Test", "predict"),
                       output_root=str(tmp_path / "o"),
                       config={"life": "life_hours", "stress": "temperature"})
    assert "跳过" in res.summary
    assert "acceleration_factor" not in res.estimates


def test_alt_celsius_ambiguity_disclosed_for_unlabeled_high_temp(tmp_path: Path) -> None:
    """A stress column just called 'temperature' (no unit token) with values in
    200-300 is a real-world HTOL/HAST high-temperature Celsius test range, but the
    engine's Kelvin heuristic (`S >= 200`) treats it as already-Kelvin and skips the
    +273.15 conversion since the data alone is genuinely ambiguous. The fix does NOT
    change that default (data can't disambiguate), but must surface an explicit ⚠
    note telling the user to confirm units / override config, since silently getting
    this wrong would badly distort Ea and the extrapolated use-stress life."""
    rng = np.random.default_rng(21)
    a_true, b_true = 5.0, 500.0
    temps = np.array([200.0, 250.0, 300.0])
    rows = []
    for tv in temps:
        mean_lnL = a_true + b_true * (1.0 / tv)
        for _ in range(10):
            rows.append({"temperature": tv,
                         "life_hours": float(np.exp(mean_lnL + rng.normal(0, 0.03)))})
    csv = _csv(tmp_path, "altc.csv", pd.DataFrame(rows))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("accelerated_life_test", "Accelerated Life Test", "predict"),
                       output_root=str(tmp_path / "o"),
                       config={"life": "life_hours", "stress": "temperature", "model": "arrhenius"})
    e = res.estimates
    assert e["n_stress_levels"] == 3.0
    # Kelvin default preserved (no silent conversion) ...
    assert "（应力按开尔文使用）" in res.summary
    # ... but the genuine Celsius-vs-Kelvin ambiguity must be explicitly flagged
    assert "仅泛称『温度』" in res.summary
    assert "未标明单位" in res.summary


def test_alt_celsius_ambiguity_not_flagged_when_name_says_kelvin(tmp_path: Path) -> None:
    """Same ambiguous 200-300 numeric range, but the column name explicitly says
    Kelvin -- data and name agree, so there is nothing ambiguous to disclose."""
    rng = np.random.default_rng(22)
    a_true, b_true = 5.0, 500.0
    temps = np.array([200.0, 250.0, 300.0])
    rows = []
    for tv in temps:
        mean_lnL = a_true + b_true * (1.0 / tv)
        for _ in range(10):
            rows.append({"temperature_kelvin": tv,
                         "life_hours": float(np.exp(mean_lnL + rng.normal(0, 0.03)))})
    csv = _csv(tmp_path, "altk.csv", pd.DataFrame(rows))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("accelerated_life_test", "Accelerated Life Test", "predict"),
                       output_root=str(tmp_path / "o"),
                       config={"life": "life_hours", "stress": "temperature_kelvin",
                               "model": "arrhenius"})
    assert "仅泛称『温度』" not in res.summary
