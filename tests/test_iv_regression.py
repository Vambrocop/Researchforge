"""Tests for iv_regression: real 2SLS via linearmodels when a config instrument is
given (recovers the causal coef where naive OLS is biased; Wu-Hausman fires; Sargan on
overid) + honest guidance when no instrument + degrade without linearmodels."""

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
        id="iv_regression", method="Instrumental variables (2SLS)", domain="economics",
        family="causal", goal="explain", preconditions=Precondition(min_rows=30, min_continuous=1),
    )


def _iv_data(seed: int = 0, n: int = 500, beta: float = 2.0):
    """Endogenous x correlated with the error u; instruments z1,z2 affect y only via x."""
    rng = np.random.default_rng(seed)
    z1 = rng.normal(0, 1, n)
    z2 = rng.normal(0, 1, n)
    u = rng.normal(0, 1, n)                       # structural error
    x = 0.7 * z1 + 0.5 * z2 + 0.6 * u + rng.normal(0, 0.5, n)  # endogenous
    w = rng.normal(0, 1, n)                       # exogenous control
    y = 1.0 + beta * x + 0.5 * w + u             # true beta on x
    return pd.DataFrame({"y": y, "x": x, "w": w, "z1": z1, "z2": z2})


def test_iv_recovers_causal_coef_overidentified(tmp_path: Path) -> None:
    pytest.importorskip("linearmodels")
    csv = tmp_path / "iv.csv"
    _iv_data().to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"outcome": "y", "endogenous": ["x"], "instruments": ["z1", "z2"], "controls": ["w"]},
    )
    out = Path(res.output_dir)
    assert (out / "iv_2sls_coefficients.csv").exists()
    # 2SLS recovers the true causal coef (~2.0); naive OLS is biased upward by the endogeneity
    assert abs(res.estimates["iv_coef_x"] - 2.0) < 0.25
    assert res.estimates["naive_ols_coef_x"] > res.estimates["iv_coef_x"] + 0.1  # OLS biased up
    # first-stage strong, endogeneity detected, overid not rejected
    assert res.estimates["first_stage_F_x"] > 10
    assert res.estimates["wu_hausman_p"] < 0.05
    assert "sargan_p" in res.estimates  # overidentified (2 instruments, 1 endogenous)
    assert res.estimates["n_instruments"] == 2.0


def test_iv_single_instrument_just_identified(tmp_path: Path) -> None:
    pytest.importorskip("linearmodels")
    csv = tmp_path / "iv.csv"
    _iv_data().to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"outcome": "y", "endogenous": "x", "instrument": "z1", "controls": ["w"]},
    )
    assert abs(res.estimates["iv_coef_x"] - 2.0) < 0.4
    assert "sargan_p" not in res.estimates  # just-identified: no overid test


def test_iv_no_instrument_returns_guidance(tmp_path: Path) -> None:
    csv = tmp_path / "iv.csv"
    _iv_data().to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "y", "endogenous": "x"})  # no instrument
    assert "iv_coef_x" not in res.estimates
    assert "工具变量" in res.summary and ("instrument" in res.summary or "指定" in res.summary)


def test_iv_degrades_without_linearmodels(monkeypatch, tmp_path: Path) -> None:
    pytest.importorskip("linearmodels")
    import builtins
    csv = tmp_path / "iv.csv"
    _iv_data().to_csv(csv, index=False)
    fp = profile_dataset(csv)
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.startswith("linearmodels"):
            raise ImportError("simulated missing linearmodels")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"outcome": "y", "endogenous": "x", "instrument": "z1"},
    )
    assert "iv_coef_x" not in res.estimates
    assert "linearmodels" in res.summary and "跳过" in res.summary
