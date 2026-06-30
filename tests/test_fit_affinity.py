"""Stage 3 — the real `fit` (data↔method affinity) replacing fit = rigor.score.

These tests assert that score_method's `fit` now reflects how well a method suits the
data's structure: the appropriate method scores higher than a generic one on each data
shape. (Ranking is NOT changed yet — that's Stage 4 — so the golden-selection suite
still shows the same 4 pass / 5 xfail; that invariant is checked too.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog import Catalog
from researchforge.profiler import profile_dataset
from researchforge.recommender.rigor import assess_rigor
from researchforge.recommender.scoring import score_method

_CAT = Catalog.load()


def _fit(df: pd.DataFrame, mid: str, tmp_path: Path) -> int:
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    e = _CAT.by_id(mid)
    return score_method(fp, e, assess_rigor(fp, e)).fit


def test_fit_binary_prefers_logistic(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    n = 220
    x1 = rng.normal(0, 1, n)
    p = 1 / (1 + np.exp(-(0.4 + 0.9 * x1)))
    df = pd.DataFrame({"approved": rng.binomial(1, p), "income": x1.round(3),
                       "age": rng.normal(40, 8, n).round(2), "score": rng.normal(0, 1, n).round(3)})
    f_log = _fit(df, "logistic_regression", tmp_path)
    assert f_log > _fit(df, "ols_regression", tmp_path)
    assert f_log > _fit(df, "descriptive_stats", tmp_path)


def test_fit_geo_prefers_spatial(tmp_path: Path) -> None:
    rng = np.random.default_rng(8)
    n = 160
    df = pd.DataFrame({"lat": rng.uniform(30, 40, n).round(4), "lon": rng.uniform(-120, -110, n).round(4),
                       "temp": rng.normal(15, 5, n).round(3)})
    assert _fit(df, "moran_i", tmp_path) > _fit(df, "ols_regression", tmp_path)


def test_fit_survival_prefers_survival(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    n = 220
    df = pd.DataFrame({"duration": rng.exponential(10, n).round(2), "event": rng.binomial(1, 0.6, n),
                       "age": rng.normal(60, 10, n).round(2)})
    assert _fit(df, "survival_analysis", tmp_path) > _fit(df, "ols_regression", tmp_path)


def test_fit_panel_prefers_panel(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n = 120
    df = pd.DataFrame({"firm": np.repeat(np.arange(20), 6), "year": np.tile(np.arange(6), 20),
                       "cap": rng.normal(0, 1, n).round(3), "invest": rng.normal(0, 1, n).round(3)})
    assert _fit(df, "panel_fixed_effects", tmp_path) > _fit(df, "descriptive_stats", tmp_path)


def test_fit_count_prefers_negbin(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    n = 220
    mu = np.exp(0.6 + 0.4 * rng.normal(0, 1, n))
    df = pd.DataFrame({"visits": rng.poisson(mu) + rng.poisson(mu * 2),
                       "x1": rng.normal(0, 1, n).round(3), "x2": rng.normal(0, 1, n).round(3)})
    assert _fit(df, "negative_binomial_regression", tmp_path) > _fit(df, "ols_regression", tmp_path)


def test_fit_in_range(tmp_path: Path) -> None:
    df = pd.DataFrame({"x": np.arange(40.0), "y": np.arange(40.0) + 1})
    for mid in ("ols_regression", "logistic_regression", "arima", "moran_i", "descriptive_stats"):
        assert 0 <= _fit(df, mid, tmp_path) <= 100
