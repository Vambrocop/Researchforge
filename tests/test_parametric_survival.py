"""Tests for parametric_survival (AFT): Weibull selected by AIC + covariate sign.

Synthetic structure: Weibull-distributed event times whose scale depends on a
covariate x (AFT form: log T = intercept + beta*x + sigma*W). We assert the
WeibullAFTFitter is picked by AIC and that the recovered acceleration factor for
x has the correct direction. Plus an honest-skip test when columns are missing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="parametric_survival",
        method="Parametric survival (AFT — Weibull / LogNormal / LogLogistic)",
        domain="statistics",
        family="survival",
        goal="explain",
        preconditions=Precondition(requires_binary_outcome=True, min_continuous=1, min_rows=30),
    )


def _weibull_aft_data(n=500, beta=0.8, shape=1.5, seed=0) -> pd.DataFrame:
    """Weibull AFT: T = exp(intercept + beta*x) * Weibull(shape).

    Larger x -> longer survival -> acceleration factor exp(beta) > 1.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    # base Weibull(shape=k, scale=1): -log(U) ~ Exp(1); T0 = (-log U)^(1/k)
    u = rng.uniform(1e-6, 1, n)
    t0 = (-np.log(u)) ** (1.0 / shape)
    scale = np.exp(1.5 + beta * x)
    t = scale * t0
    # administrative censoring at a high quantile (~15% censored)
    c = np.quantile(t, 0.85)
    time = np.minimum(t, c)
    event = (t <= c).astype(int)
    return pd.DataFrame({"time": np.round(time, 4), "event": event, "x": np.round(x, 4)})


def test_weibull_selected_and_sign(tmp_path: Path) -> None:
    pytest.importorskip("lifelines")
    df = _weibull_aft_data()
    csv = tmp_path / "aft.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    assert (out / "aft_aic_comparison.csv").exists()
    assert (out / "aft_acceleration_factors.csv").exists()
    assert "best AIC" in res.summary or "最优分布" in res.summary

    # Weibull-generated data -> WeibullAFT should win on AIC
    assert res.estimates["aic_Weibull"] <= res.estimates["aic_LogNormal"]
    assert res.estimates["aic_Weibull"] <= res.estimates["aic_LogLogistic"]

    # acceleration factor for x: positive beta -> AF = exp(beta) > 1 (longer survival)
    assert res.estimates["AF_x"] > 1.0

    aic_tab = pd.read_csv(out / "aft_aic_comparison.csv")
    assert aic_tab.iloc[0]["distribution"] == "Weibull"  # sorted by AIC ascending
    af_tab = pd.read_csv(out / "aft_acceleration_factors.csv")
    xrow = af_tab[af_tab["covariate"] == "x"].iloc[0]
    assert xrow["AF_lower95"] < xrow["accel_factor"] < xrow["AF_upper95"]
    assert xrow["p_value"] < 0.05  # strong signal recovered


def test_parametric_negative_beta_sign(tmp_path: Path) -> None:
    pytest.importorskip("lifelines")
    df = _weibull_aft_data(beta=-0.7, seed=5)
    csv = tmp_path / "aft.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    # negative beta -> AF = exp(beta) < 1 (shorter survival as x rises)
    assert res.estimates["AF_x"] < 1.0


def test_parametric_missing_cols_skips(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(0, 1, 40), "b": rng.normal(0, 1, 40)})  # no event
    csv = tmp_path / "plain.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "参数生存(AFT)失败" in res.summary
    assert "best_aic" not in res.estimates
