"""Tests for cmh_test — Cochran-Mantel-Haenszel stratified 2x2 test.

Cross-checks:
  * a textbook stratified table: MH common OR + CMH chi-square match an
    independent recompute;
  * a strong common-OR effect is detected (cmh_p small); a null effect is not;
  * Breslow-Day does NOT reject homogeneity when strata share an OR, and DOES
    reject when strata have opposite ORs;
  * config exposure/outcome/stratum override; too-few-columns honest skip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="cmh_test", method="Cochran-Mantel-Haenszel stratified test",
        domain="statistics", family="categorical", goal="explain",
        preconditions=Precondition(min_categorical_cols=3, min_rows=4),
    )


def _df_from_strata(strata: list[tuple]) -> pd.DataFrame:
    """strata = list of (stratum_label, a, b, c, d) where the 2x2 is
    [[a, b], [c, d]] with rows=exposed/unexposed, cols=positive/negative.
    Encodes exposure 1=exposed/0=unexposed, outcome 1=positive/0=negative so the
    branch's 'larger label = exposed/positive' coding lines up."""
    rows = []
    for sk, a, b, c, d in strata:
        rows += [{"stratum": sk, "exp": 1, "out": 1}] * a
        rows += [{"stratum": sk, "exp": 1, "out": 0}] * b
        rows += [{"stratum": sk, "exp": 0, "out": 1}] * c
        rows += [{"stratum": sk, "exp": 0, "out": 0}] * d
    return pd.DataFrame(rows)


def _cmh_ref(strata: list[tuple]) -> tuple[float, float, float]:
    """Independent reference: (mh_or, cmh_chi2_cc, cmh_p)."""
    num = den = sum_a = sum_Ea = sum_Va = 0.0
    for _, a, b, c, d in strata:
        n = a + b + c + d
        num += a * d / n
        den += b * c / n
        sum_a += a
        sum_Ea += (a + b) * (a + c) / n
        sum_Va += (a + b) * (c + d) * (a + c) * (b + d) / (n * n * (n - 1))
    mh_or = num / den
    chi2 = (abs(sum_a - sum_Ea) - 0.5) ** 2 / sum_Va
    return mh_or, chi2, float(stats.chi2.sf(chi2, 1))


def test_mh_or_and_cmh_match_reference(tmp_path: Path) -> None:
    strata = [("S1", 30, 10, 15, 20), ("S2", 25, 15, 10, 25), ("S3", 40, 20, 20, 30)]
    df = _df_from_strata(strata)
    csv = tmp_path / "t.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"exposure": "exp", "outcome": "out", "stratum": "stratum"})
    mh_or, chi2, p = _cmh_ref(strata)
    assert abs(res.estimates["mh_or"] - mh_or) < 1e-2
    assert abs(res.estimates["cmh_chi2"] - chi2) < 1e-2
    assert abs(res.estimates["cmh_p"] - p) < 1e-4
    assert res.estimates["n_strata"] == 3.0
    assert (Path(res.output_dir) / "cmh_strata.csv").exists()


def test_strong_effect_detected(tmp_path: Path) -> None:
    # large common OR across strata -> small cmh_p, OR >> 1.
    strata = [("S1", 60, 10, 10, 60), ("S2", 55, 12, 12, 55)]
    df = _df_from_strata(strata)
    csv = tmp_path / "eff.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"exposure": "exp", "outcome": "out", "stratum": "stratum"})
    assert res.estimates["mh_or"] > 5.0
    assert res.estimates["cmh_p"] < 0.001
    assert "显著相关" in res.summary


def test_null_effect_not_detected(tmp_path: Path) -> None:
    # OR ~ 1 in every stratum -> large cmh_p.
    strata = [("S1", 25, 25, 25, 25), ("S2", 30, 30, 30, 30)]
    df = _df_from_strata(strata)
    csv = tmp_path / "null.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"exposure": "exp", "outcome": "out", "stratum": "stratum"})
    assert abs(res.estimates["mh_or"] - 1.0) < 0.2
    assert res.estimates["cmh_p"] > 0.2


def test_breslow_day_homogeneous(tmp_path: Path) -> None:
    # both strata share the same OR (~6.25) -> Breslow-Day should NOT reject.
    strata = [("S1", 50, 10, 10, 50), ("S2", 25, 5, 5, 25)]
    df = _df_from_strata(strata)
    csv = tmp_path / "homog.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"exposure": "exp", "outcome": "out", "stratum": "stratum"})
    assert res.estimates["bd_p"] > 0.05
    assert "未拒绝 OR 同质性" in res.summary


def test_breslow_day_heterogeneous(tmp_path: Path) -> None:
    # opposite ORs across strata (one OR>1, one OR<1) -> Breslow-Day rejects.
    strata = [("S1", 60, 10, 10, 60), ("S2", 10, 60, 60, 10)]
    df = _df_from_strata(strata)
    csv = tmp_path / "het.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"exposure": "exp", "outcome": "out", "stratum": "stratum"})
    assert res.estimates["bd_p"] < 0.05
    assert "拒绝 OR 同质性" in res.summary


def test_too_few_columns_skips(tmp_path: Path) -> None:
    # only one binary + one continuous -> cannot form exposure/outcome/stratum.
    df = pd.DataFrame({"bin": [0, 1] * 15, "cont": np.arange(30.0)})
    csv = tmp_path / "few.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "mh_or" not in res.estimates
    assert "跳过" in res.summary
    assert (Path(res.output_dir) / "report.md").exists()
