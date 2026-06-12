"""Synthetic data generators — lightweight demo/test data, so the repo never
stores large datasets. Used for fixtures, demos, and quick trials.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_timeseries(
    n_periods: int = 60,
    start: str = "2015-01-01",
    seed: int = 0,
) -> pd.DataFrame:
    """A univariate monthly time series with trend + AR(1) noise.

    Returns a DataFrame with columns ['date', 'value']: monthly dates
    via pd.date_range and a realistic AR(1)+trend series. No unit column."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 1, n_periods)
    v = np.zeros(n_periods)
    for t in range(1, n_periods):
        v[t] = 0.6 * v[t - 1] + e[t]
    value = 50 + 0.3 * np.arange(n_periods) + v
    dates = pd.date_range(start, periods=n_periods, freq="MS")
    return pd.DataFrame({"date": dates, "value": value})


def make_panel(
    n_units: int = 6,
    n_periods: int = 6,
    start_year: int = 2010,
    treated: bool = True,
    seed: int = 0,
) -> pd.DataFrame:
    """A unit x year panel with an outcome `y` and (optionally) a staggered
    binary `treated` indicator — the canonical DID / fixed-effects shape."""
    rng = np.random.default_rng(seed)
    units = [f"unit_{i:02d}" for i in range(n_units)]
    treat_start = start_year + n_periods // 2
    rows = []
    for idx, u in enumerate(units):
        unit_effect = rng.normal(0.0, 1.0)
        for t in range(n_periods):
            year = start_year + t
            treated_flag = int(treated and year >= treat_start and idx % 2 == 0)
            y = 10.0 + unit_effect + 0.3 * t + 2.0 * treated_flag + rng.normal(0.0, 0.5)
            row = {"unit": u, "year": year, "y": round(float(y), 3)}
            if treated:
                row["treated"] = treated_flag
            rows.append(row)
    return pd.DataFrame(rows)
