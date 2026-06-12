from researchforge.catalog import Catalog


def test_catalog_loads_core_entries():
    ids = {e.id for e in Catalog.load().all()}
    assert {
        "descriptive_stats",
        "correlation",
        "ols_regression",
        "panel_fixed_effects",
        "did",
        "iv_regression",
    } <= ids


def test_did_entry_shape():
    did = Catalog.load().by_id("did")
    assert did is not None
    assert did.preconditions.is_panel is True
    assert did.preconditions.requires_treatment is True
    assert did.executor_ref == "empirical-analysis-python"
    assert did.biases
