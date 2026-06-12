"""Catalog layer: the knowledge base of analyses the engine can recommend."""

from researchforge.catalog.candidates import (
    CatalogCandidate,
    add_candidate,
    load_candidates,
    promote_candidate,
)
from researchforge.catalog.registry import Catalog
from researchforge.catalog.schema import AnalysisEntry, Precondition

__all__ = [
    "Catalog",
    "AnalysisEntry",
    "Precondition",
    "CatalogCandidate",
    "load_candidates",
    "add_candidate",
    "promote_candidate",
]
