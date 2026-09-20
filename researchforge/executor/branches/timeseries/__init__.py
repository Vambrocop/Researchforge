"""Timeseries family — promoted from a single module to a package at 1131 lines.

The layout convention (CLAUDE.md「引擎架构」): a family is one ``<family>.py`` until it nears
the size guardrail, then becomes ``<family>/`` with one module per analysis and that method's
helpers moved in beside it. ``branches/__init__.py`` walks packages recursively, so every
module here registers itself — nothing to edit when a new analysis is added.

It grew from 1004 to 1131 lines during the ARIMA cost-model work; the hard guardrail is 1500,
but the convention says promote near ~1200 precisely so nobody has to do it under pressure.
This project has been bitten by the monolith once already (run.py at 7935 lines, with
``run_analysis`` alone around 5500 — reading the file once exhausted the context window).

The re-exports below are not decoration. ``branches/forecasting.py``,
``tests/test_seasonal_detection.py`` and ``tests/test_arima_cost_budget.py`` import these
names from ``researchforge.executor.branches.timeseries``; that import path stays valid, the
same way ``run.py`` re-exports ``_helpers``.
"""

from __future__ import annotations

from researchforge.executor.branches.timeseries._order_search import (
    _GRID_FIT_BUDGET,
    _KALMAN_SEC_PER_UNIT,
    _SEARCH_TIME_BUDGET_S,
    _aicc,
    _auto_order,
    _fit_sarimax,
    _fit_seconds,
    _ndiffs_adf,
    _sarimax_k_states,
)
from researchforge.executor.branches.timeseries._seasonal import _periodogram_period

__all__ = [
    "_GRID_FIT_BUDGET",
    "_KALMAN_SEC_PER_UNIT",
    "_SEARCH_TIME_BUDGET_S",
    "_aicc",
    "_auto_order",
    "_fit_sarimax",
    "_fit_seconds",
    "_ndiffs_adf",
    "_periodogram_period",
    "_sarimax_k_states",
]
