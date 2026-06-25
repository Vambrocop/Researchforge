"""Tests for the self-evolution discovery engine (catalog.discover)."""

from __future__ import annotations

from researchforge.catalog.discover import (
    SEED,
    discover_candidates,
    score_candidate,
)


def test_score_candidate_dimensions() -> None:
    m = score_candidate(
        {"id": "x", "method": "X", "family": "causal", "domain": "economics", "goal": "explain"}
    )
    assert 0 <= m.priority <= 100
    # momentum (live trend) joins the offline rubric dims; 0 when no live signal
    assert set(m.breakdown) == {"novelty", "publishability", "popularity", "momentum"}
    assert m.breakdown["momentum"] == 0  # offline spec carries no live momentum
    # causal family is highly publishable
    assert m.breakdown["publishability"] >= 80


def test_discover_ranks_and_dedupes_against_catalog() -> None:
    # one id already in the live catalog (ols_regression) + one genuinely new
    def fetch():
        return [
            {"id": "ols_regression", "method": "OLS", "family": "statistics"},
            {"id": "brand_new_xyz", "method": "Brand New", "family": "causal"},
        ]

    found = discover_candidates(fetch_fn=fetch, persist=False)
    ids = {m.id for m in found}
    assert "ols_regression" not in ids  # deduped (already in catalog)
    assert "brand_new_xyz" in ids


def test_discover_seed_is_ranked_desc_and_excludes_existing() -> None:
    # invariants that hold for ANY queue state — including a fully-drained queue
    # (when every SEED method has been implemented, discover correctly returns []).
    found = discover_candidates(persist=False)
    priorities = [m.priority for m in found]
    assert priorities == sorted(priorities, reverse=True)  # ranked high-first
    # seed ids should not collide with the live catalog
    from researchforge.catalog.registry import Catalog

    live = {e.id for e in Catalog.load().all()}
    assert all(m.id not in live for m in found)
    # every seed entry is well-formed
    assert {s["id"] for s in SEED} >= ids_of(found)


def ids_of(found) -> set:
    return {m.id for m in found}
