import pytest

from researchforge.catalog import (
    AnalysisEntry,
    Catalog,
    CatalogCandidate,
    add_candidate,
    load_candidates,
    promote_candidate,
)


def _entry(eid: str) -> AnalysisEntry:
    return AnalysisEntry(
        id=eid, method=eid.title(), domain="x", family="ml", goal="explore",
        preconditions={"min_rows": 1},
    )


def test_pending_candidate_not_live(tmp_path):
    add_candidate(
        CatalogCandidate(entry=_entry("foo_method"), status="pending", source="CRAN"),
        directory=tmp_path / "candidates",
    )
    cands = load_candidates(tmp_path / "candidates")

    assert any(c.entry.id == "foo_method" for c in cands)
    assert Catalog.load().by_id("foo_method") is None  # pending != live


def test_promote_refuses_pending(tmp_path):
    add_candidate(CatalogCandidate(entry=_entry("bar"), status="pending"), directory=tmp_path / "c")
    with pytest.raises(ValueError):
        promote_candidate("bar", candidates_dir=tmp_path / "c", promoted_file=tmp_path / "p.yaml")


def test_promote_ready_adds_to_catalog(tmp_path):
    add_candidate(CatalogCandidate(entry=_entry("baz"), status="ready"), directory=tmp_path / "c")
    promoted = tmp_path / "entries" / "promoted.yaml"

    entry = promote_candidate("baz", candidates_dir=tmp_path / "c", promoted_file=promoted)

    assert entry.id == "baz"
    assert Catalog.load(directory=promoted.parent).by_id("baz") is not None  # now live
