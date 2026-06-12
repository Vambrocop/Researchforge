"""Ingestion layer: register skills / papers from the inbox into the catalog."""

from researchforge.ingestion.ingest import IngestedItem, ingest_inbox

__all__ = ["IngestedItem", "ingest_inbox"]
