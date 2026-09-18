"""导出的 `analysis_code.py` 必须真的复现分析——而它曾经复现的是被修掉的 bug。

Cold review B/A5.3: the arima branch emitted

    SARIMAX(y, order=…, seasonal_order=…,
            enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)

with no `trend` — i.e. the "here is how to reproduce this" snippet handed the user BOTH
defects the previous cold review had just fixed inside `_fit_sarimax`:

  * `enforce_stationarity/invertibility=False` makes statsmodels fall back to an
    approximate-diffuse initialization whose `loglikelihood_burn` grows with the state
    dimension, so each candidate's log-likelihood is summed over a DIFFERENT effective
    sample and AICc stops being comparable (measured then: on a random walk the true
    (0,1,0) was selected 0/30 times, mean p+q = 3.7);
  * no `trend` means SARIMAX fits an undifferenced series as a MEAN-ZERO process — a
    series hovering around 500 forecasts 0.0.

A docstring or a string-matching test would not have caught this: the branch's own code was
right, only its EXPORT was wrong. So this file does the one thing that cannot be faked —
it executes the exported file and compares the numbers with the run's own estimates.

The invariant is general, not arima-specific: **an exported snippet that does not reproduce
the reported number is a lie about the analysis**. New families can be added here.
"""

from __future__ import annotations

import pathlib
import runpy
import warnings

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


def _run_exported(res):
    """Execute the analysis_code.py the run wrote, and hand back its namespace."""
    path = pathlib.Path(res.output_dir) / "analysis_code.py"
    assert path.exists(), res.files
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return runpy.run_path(str(path))


def _seasonal_monthly(n=60, level=500.0, seed=7):
    """Trending + seasonal, so the search differences it (d=1)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    seas = np.array([-8, -6, -2, 2, 6, 9, 8, 5, 1, 4, 12, 20])[t % 12]
    return pd.DataFrame({"month": pd.date_range("2019-01-01", periods=n, freq="MS")
                        .strftime("%Y-%m"),
                         "revenue": (level + 0.8 * t + seas + rng.normal(0, 2.5, n)).round(1)})


def _stationary_around_500(n=120, seed=3):
    """No unit root, so the ADF test fixes d=0 — the case where the missing `trend='c'`
    turned a series around 500 into a mean-zero fit forecasting 0.0."""
    rng = np.random.default_rng(seed)
    y = np.empty(n)
    y[0] = 500.0
    for i in range(1, n):
        y[i] = 500.0 + 0.5 * (y[i - 1] - 500.0) + rng.normal(0, 3.0)
    return pd.DataFrame({"day": pd.date_range("2024-01-01", periods=n, freq="D")
                        .strftime("%Y-%m-%d"),
                         "load": y.round(2)})


@pytest.mark.parametrize("frame,name", [(_seasonal_monthly, "seasonal"),
                                        (_stationary_around_500, "stationary")])
def test_the_exported_arima_code_reproduces_the_reported_forecast(frame, name, tmp_path):
    fp = _fp(frame(), tmp_path, name=f"{name}.csv")
    res = run_analysis(fp, _CAT.by_id("arima"), output_root=str(tmp_path / name))
    reported = res.estimates.get("forecast_next")
    assert reported is not None, res.summary[:200]

    ns = _run_exported(res)
    mean = ns["mean"]
    got = float(np.asarray(mean)[0])
    assert got == pytest.approx(reported, rel=1e-6), (
        f"exported code forecasts {got}, the run reported {reported}")


def test_the_exported_settings_are_the_settings_that_were_fitted(tmp_path):
    """The two flags are load-bearing (see _fit_sarimax); exporting them as False handed the
    user the exact bugs the previous cold review removed."""
    # the cheap frame on purpose — this assertion is about the exported STRING, and the
    # seasonal search costs 10s where the stationary one costs 2.7s.
    fp = _fp(_stationary_around_500(), tmp_path, name="flags.csv")
    res = run_analysis(fp, _CAT.by_id("arima"), output_root=str(tmp_path / "o"))
    code = (pathlib.Path(res.output_dir) / "analysis_code.py").read_text(encoding="utf-8")
    assert "enforce_stationarity=True" in code
    assert "enforce_invertibility=True" in code
    assert "enforce_stationarity=False" not in code
    assert "trend=" in code, "the d==0 constant has to be pinned explicitly either way"


def test_an_undifferenced_series_is_not_exported_as_a_mean_zero_fit(tmp_path):
    """The concrete failure the `trend='c'` rule exists for: a series around 500 fitted with
    no constant forecasts 0.0. Pin it on the EXPORT, where it had regressed."""
    fp = _fp(_stationary_around_500(), tmp_path, name="d0.csv")
    res = run_analysis(fp, _CAT.by_id("arima"), output_root=str(tmp_path / "p"))
    if res.estimates.get("d") != 0.0:
        pytest.skip(f"ADF chose d={res.estimates.get('d')}; this case needs d=0")
    code = (pathlib.Path(res.output_dir) / "analysis_code.py").read_text(encoding="utf-8")
    assert "trend='c'" in code, code
    got = float(np.asarray(_run_exported(res)["mean"])[0])
    assert got > 100.0, f"exported code forecasts {got} for a series around 500"
