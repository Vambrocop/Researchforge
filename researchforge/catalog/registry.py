"""Load and query the analysis catalog (YAML entry files)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from researchforge.catalog.schema import AnalysisEntry

_ENTRIES_DIR = Path(__file__).resolve().parent / "entries"


class Catalog:
    def __init__(self, entries: list[AnalysisEntry]):
        self.entries = entries

    @classmethod
    def load(cls, directory: str | Path | None = None) -> "Catalog":
        directory = Path(directory) if directory else _ENTRIES_DIR
        entries: list[AnalysisEntry] = []
        for f in sorted(directory.glob("*.yaml")):
            data = yaml.safe_load(f.read_text(encoding="utf-8")) or []
            for item in data:
                entries.append(AnalysisEntry(**item))
        return cls(entries)

    def all(self) -> list[AnalysisEntry]:
        return list(self.entries)

    def by_id(self, entry_id: str) -> Optional[AnalysisEntry]:
        for e in self.entries:
            if e.id == entry_id:
                return e
        return None
