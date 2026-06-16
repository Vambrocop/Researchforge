"""Pytest config for the test suite.

Auto-tags heavy / model-fitting test modules as ``slow`` so the fast dev loop
``pytest -m "not slow"`` skips them (runs the light unit tests in seconds). The full
suite still runs everything.

Speedup notes (see CLAUDE.md 运行):
- Fast loop:      ``pytest -m "not slow"``        (light tests only)
- Full, faster:   ``pytest -n 2``                 (bounded parallel, ~2:49 vs ~4:29 serial)
- Do NOT use ``-n auto`` for the full suite: each worker spawns R / heavy libs and the
  box runs out of memory (MemoryError). ``-n 2`` is the memory-safe sweet spot here.

Keep SLOW_MODULES in sync when adding a heavy-method test. Refresh the list from a
durations report: ``pytest -n 2 --durations=25``.
"""

from __future__ import annotations

import pytest

# Test modules whose tests fit a real model — R bridge, econml/doubleml, xgboost,
# dbarts, lifelines/networkx, bootstrap mediation, sklearn clustering, statsmodels
# negbin. These are the slow tail (each ≳2s; test_mediation ~64s, joint model ~39s).
SLOW_MODULES = {
    "test_bart", "test_benchmark", "test_causal_forest", "test_changes_in_changes",
    "test_clustering", "test_cna", "test_conformal_prediction", "test_config",
    "test_csqca", "test_double_ml", "test_dynamic_panel_gmm", "test_fsqca", "test_gam",
    "test_gamm", "test_getis_ord_gi", "test_glmm", "test_gsynth", "test_idw_interpolation",
    "test_joint_longitudinal_survival", "test_kriging", "test_local_moran",
    "test_mediation", "test_meta_analysis", "test_meta_regression", "test_moran_i",
    "test_negbin", "test_network_analysis", "test_panel_qca", "test_qca_necessity",
    "test_rdd", "test_sem", "test_sfa", "test_spatial_regression",
    "test_survival_analysis", "test_xgboost",
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.path.stem in SLOW_MODULES:
            item.add_marker(pytest.mark.slow)
