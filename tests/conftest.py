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
    "test_clustering", "test_cna", "test_competing_risks", "test_conformal_prediction",
    "test_config", "test_csqca", "test_double_ml", "test_dynamic_panel_gmm",
    "test_fsqca", "test_gam", "test_gamm", "test_getis_ord", "test_glmm",
    "test_gsynth", "test_idw", "test_joint_longitudinal_survival", "test_kriging",
    "test_local_moran", "test_mediation", "test_meta_analysis", "test_meta_regression",
    "test_moran_i", "test_negbin", "test_network_analysis", "test_panel_qca",
    "test_parametric_survival", "test_qca_necessity", "test_rdd", "test_rmst",
    "test_sem", "test_sfa", "test_spatial_regression",
    "test_survival_analysis", "test_xgboost",
    # R-backed / EM-heavy additions (R subprocess startup + multi-fit selection)
    "test_callaway_santanna", "test_spatial_panel", "test_rda",
    "test_latent_class_analysis", "test_latent_profile_analysis",
    # lifelines Cox fits (time-varying / stratified / Schoenfeld PH test)
    "test_time_varying_cox", "test_stratified_cox", "test_cox_ph_diagnostics",
    # unit-bootstrap DiD (de Chaisemartin DID_M runs replicate resamples)
    "test_chaisemartin_did",
    # ordinal regression (statsmodels OrderedModel bfgs + Brant per-cut logits)
    "test_proportional_odds_logit", "test_ordered_probit", "test_brant_test",
    # supervised ML (sklearn CV fits: regularized / SVM / gradient boosting)
    "test_regularized_regression", "test_svm_model", "test_gradient_boosting",
    # forecasting (statsmodels ETS/theta) + dimensionality (t-SNE / FA / LDA-CV)
    "test_exponential_smoothing", "test_theta_method", "test_croston",
    "test_tsne", "test_factor_analysis", "test_linear_discriminant",
    # bootstrap-heavy mediation extensions (serial/parallel refit B=2000)
    "test_mediation_extra",
    # ML explainers fit GBM/RF + SHAP TreeExplainer per test
    "test_interpretability",
    # spatial association (permutation inference) + SKATER MST pruning
    "test_spatial_dependence",
    # meta-guard: parses every branch module's AST + loads catalog
    "test_config_params_complete",
    # text mining: LDA fit + TF-IDF vectorization per test
    "test_text_mining",
    # end-to-end: runs many analyses on real datasets through the full pipeline
    "test_end_to_end",
    # AIPW fits propensity + per-arm outcome (GBM) models; UMAP embedding
    "test_aipw", "test_umap",
    # techno-economic: Monte-Carlo cash-flow simulation
    "test_techno_economic",
    # auto-diagnose engine: profiles many datasets + fits BP/VIF regressions per test
    "test_diagnostics",
    # epidemiology: ROC/DeLong + calibration GLM fits per test
    "test_epidemiology",
    # SPC: gage R&R fits two-way ANOVA (OLS + anova_lm) per test
    "test_spc",
    # reliability: censored Weibull/lognormal MLE optimization + Crow-AMSAA per test
    "test_reliability",
    # bayesian MCMC: PyMC NUTS sampling per test (regression / logistic / hierarchical)
    "test_bayesian_mcmc",
    # MICE multiple imputation: runs M chained-equation imputations + per-set OLS
    "test_missing_data",
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.path.stem in SLOW_MODULES:
            item.add_marker(pytest.mark.slow)


@pytest.fixture(autouse=True)
def _isolate_trend_artifacts(tmp_path_factory, monkeypatch):
    """Keep the live-trend cache/snapshot (`~/.researchforge/…`, written by
    `discover --live`) out of the test suite: point them at an isolated empty dir so
    scores are deterministic regardless of whether the dev has run a live fetch.
    Trend tests opt back in by writing to these (now tmp) paths themselves."""
    import researchforge.catalog.trends as trends
    from researchforge.recommender.scoring import _trend_snapshot

    d = tmp_path_factory.mktemp("rf_trends")
    monkeypatch.setattr(trends, "_CACHE_DIR", d)
    monkeypatch.setattr(trends, "_CACHE_FILE", d / "trend_cache.json")
    monkeypatch.setattr(trends, "_SNAPSHOT_FILE", d / "trend_snapshot.json")
    _trend_snapshot.cache_clear()
    yield
    _trend_snapshot.cache_clear()
