"""Stage 5 — diagnostics beyond the GLM family (time-series / survival / missingness).

Each asserts the value-level diagnostic fires on data with that structure and nudges
toward the appropriate methods, and stays silent on plain cross-section data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog import Catalog
from researchforge.profiler import profile_dataset
from researchforge.recommender.diagnostics import build_plan

_CAT = Catalog.load()


def _codes(df: pd.DataFrame, tmp_path: Path):
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    plan = build_plan(profile_dataset(csv), catalog=_CAT)
    return {d.code: d for d in plan.diagnostics}


def test_nonstationary_timeseries(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"month": np.arange(160), "sales": np.cumsum(rng.normal(0.1, 1, 160)).round(3) + 50})
    codes = _codes(df, tmp_path)
    assert "nonstationary" in codes
    assert "arima" in codes["nonstationary"].prefer


def test_volatility_clustering(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n = 320
    eps = rng.normal(0, 1, n)
    y = np.zeros(n)
    for t in range(1, n):
        h = 0.2 + 0.7 * y[t - 1] ** 2  # ARCH(1): variance depends on past shock
        y[t] = np.sqrt(h) * eps[t]
    df = pd.DataFrame({"time": np.arange(n), "ret": y.round(4)})
    codes = _codes(df, tmp_path)
    assert "volatility_clustering" in codes
    assert "garch" in codes["volatility_clustering"].prefer


def test_survival_data(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    n = 200
    df = pd.DataFrame({"duration": rng.exponential(10, n).round(2),
                       "event": rng.binomial(1, 0.6, n),
                       "age": rng.normal(60, 10, n).round(1)})
    codes = _codes(df, tmp_path)
    assert "survival_data" in codes
    assert "survival_analysis" in codes["survival_data"].prefer
    assert "ols_regression" in codes["survival_data"].over


def test_missingness(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    n = 200
    x = rng.normal(0, 1, n)
    x[rng.choice(n, 30, replace=False)] = np.nan  # ~15% missing
    df = pd.DataFrame({"y": rng.normal(0, 1, n).round(3), "x": np.round(x, 3),
                       "z": rng.normal(0, 1, n).round(3)})
    codes = _codes(df, tmp_path)
    assert "missing_data" in codes
    assert "mice_imputation" in codes["missing_data"].prefer


def test_overdispersion_not_fired_on_demographic_count(tmp_path: Path) -> None:
    # Wave M13 (diagnostic twin of M6): a demographic integer (age) that merely profiles as
    # `count` and is only a LOW-confidence positional outcome must NOT trigger the overdispersion
    # → NB nudge — otherwise a no-count-outcome table (customer features) gets routed to count
    # models instead of the intended analysis.
    rng = np.random.default_rng(3)
    n = 220
    df = pd.DataFrame({"spend": rng.normal(3000, 800, n).round(0),
                       "basket": rng.normal(50, 12, n).round(1),
                       "age": rng.integers(18, 70, n)})   # age = last numeric, low-conf, count-kind
    assert "overdispersion" not in _codes(df, tmp_path)


def test_overdispersion_still_fires_on_named_count(tmp_path: Path) -> None:
    # …but a genuinely count-named outcome still triggers it (guard against over-correction).
    rng = np.random.default_rng(4)
    n = 220
    x = rng.normal(0, 1, n)
    mu = np.exp(0.6 + 0.4 * x)
    df = pd.DataFrame({"visits": rng.poisson(mu) + rng.poisson(mu * 2), "x1": x.round(3)})
    assert "overdispersion" in _codes(df, tmp_path)


def test_silent_on_plain_cross_section(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    n = 150
    x1 = rng.normal(0, 1, n)
    df = pd.DataFrame({"y": (2 + 1.5 * x1 + rng.normal(0, 1, n)).round(3),
                       "x1": x1.round(3), "x2": rng.normal(0, 1, n).round(3)})
    codes = _codes(df, tmp_path)
    for c in ("nonstationary", "volatility_clustering", "survival_data", "missing_data"):
        assert c not in codes
