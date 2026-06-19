"""Tests for the dynamic_factor model executor branch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset
from researchforge.recommender.match import check_preconditions


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="dynamic_factor", method="Dynamic factor model",
        domain="economics", family="time-series", goal="explain",
        preconditions=Precondition(min_continuous=3, min_rows=40),
    )


def test_dfm_recovers_common_factor(tmp_path: Path) -> None:
    # build 4 series from a shared AR(1) latent factor + small idiosyncratic noise ->
    # high loadings, high variance-explained, extracted factor strongly correlates with truth.
    rng = np.random.default_rng(0)
    n = 250
    f = np.zeros(n)
    for t in range(1, n):
        f[t] = 0.7 * f[t - 1] + rng.normal(0, 1)
    f = (f - f.mean()) / f.std()
    loads = [1.2, -1.0, 0.9, 1.1]  # mixed signs to exercise the sign convention
    cols = {}
    for i, L in enumerate(loads):
        cols[f"s{i}"] = np.round(L * f + rng.normal(0, 0.4, n), 4)
    df = pd.DataFrame(cols)
    csv = tmp_path / "dfm.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"k_factors": 1, "factor_order": 1})
    assert "完成" in res.summary
    assert res.estimates["k_factors"] == 1.0

    # high variance explained on average (series are mostly common-factor driven)
    assert res.estimates["mean_var_explained"] > 0.5

    # extracted factor correlates strongly (|r|) with the TRUE latent factor
    fac = pd.read_csv(Path(res.output_dir) / "common_factor.csv")["factor1"].to_numpy()
    r = abs(np.corrcoef(fac, f)[0, 1])
    assert r > 0.8

    # loadings recovered with the right relative magnitude; sign convention applied (sum >= 0)
    ld = pd.read_csv(Path(res.output_dir) / "factor_loadings.csv")
    assert ld["loading_factor1"].sum() >= 0  # sign convention
    # all four series have substantial |loading|
    assert (ld["loading_factor1"].abs() > 0.3).all()


def test_dfm_too_few_series_skips(tmp_path: Path) -> None:
    # only 2 continuous series -> need >=3 -> honest skip; precondition gate fails
    rng = np.random.default_rng(2)
    n = 60
    df = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n),
                       "g": ["x", "y"] * (n // 2)})
    csv = tmp_path / "few.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    ok, _ = check_preconditions(fp, _entry().preconditions)
    assert not ok  # min_continuous=3 not met
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "失败" in res.summary and "连续" in res.summary
