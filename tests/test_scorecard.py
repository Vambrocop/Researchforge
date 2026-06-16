"""Tests for the project self-assessment scorecard."""

from __future__ import annotations

from researchforge.quality import compute_scorecard


def test_scorecard_dimensions_in_range() -> None:
    sc = compute_scorecard()
    expected = {"completeness", "correctness", "rigor", "honesty", "design",
                "novelty", "performance", "usability"}
    assert set(sc.dimensions) == expected
    for v in sc.dimensions.values():
        assert 0 <= v <= 100
    assert 0 <= sc.overall <= 100
    # each dimension has a justification note
    assert set(sc.notes) == expected


def test_scorecard_coverage_tracks_catalog() -> None:
    sc = compute_scorecard()
    # a large catalog (60+ methods) -> high completeness; deferred-log present
    assert sc.metrics["n_methods"] >= 40
    assert sc.dimensions["completeness"] >= 80
    assert sc.metrics["has_deferred_log"] == 1.0
    # table renders all dimensions
    assert "完整性" in sc.table() and "可用性" in sc.table()
