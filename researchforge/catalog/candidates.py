"""Catalog candidate queue — the seed of self-growth.

Discovered / drafted analyses land here as *candidates* (not live). Only a
candidate marked `ready` (preconditions + a wired, tested executor + benchmark
coverage — verified through the quality gate) may be promoted into the live
catalog. This enforces, at the data layer, the rule that an analysis is never
recommended until it can actually run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

from researchforge.catalog.schema import AnalysisEntry

_DIR = Path(__file__).resolve().parent
_CANDIDATES_DIR = _DIR / "candidate_queue"
_PROMOTED_FILE = _DIR / "entries" / "promoted.yaml"


class CatalogCandidate(BaseModel):
    entry: AnalysisEntry
    source: str = ""  # CRAN / rOpenSci / GitHub / paper / manual ...
    status: str = "pending"  # pending | ready | rejected
    notes: str = Field(default="")  # what's missing, e.g. "executor not wired"


def load_candidates(directory: str | Path | None = None) -> list[CatalogCandidate]:
    directory = Path(directory) if directory else _CANDIDATES_DIR
    out: list[CatalogCandidate] = []
    if not directory.exists():
        return out
    for f in sorted(directory.glob("*.yaml")):
        data = yaml.safe_load(f.read_text(encoding="utf-8")) or []
        for item in data:
            out.append(CatalogCandidate(**item))
    return out


def add_candidate(
    candidate: CatalogCandidate, directory: str | Path | None = None
) -> Path:
    directory = Path(directory) if directory else _CANDIDATES_DIR
    directory.mkdir(parents=True, exist_ok=True)
    f = directory / "inbox.yaml"
    existing = yaml.safe_load(f.read_text(encoding="utf-8")) if f.exists() else []
    existing = (existing or []) + [candidate.model_dump()]
    f.write_text(yaml.safe_dump(existing, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return f


def promote_candidate(
    candidate_id: str,
    candidates_dir: str | Path | None = None,
    promoted_file: str | Path | None = None,
) -> AnalysisEntry:
    """Promote a `ready` candidate into the live catalog. Refuses anything not
    `ready` — that is the quality gate."""
    match: Optional[CatalogCandidate] = next(
        (c for c in load_candidates(candidates_dir) if c.entry.id == candidate_id), None
    )
    if match is None:
        raise ValueError(f"未找到候选 {candidate_id}")
    if match.status != "ready":
        raise ValueError(
            f"候选 {candidate_id} 状态为 {match.status}，未过质量门（需 ready：前提+执行器+测试就绪），拒绝上线"
        )

    pf = Path(promoted_file) if promoted_file else _PROMOTED_FILE
    pf.parent.mkdir(parents=True, exist_ok=True)
    live = (yaml.safe_load(pf.read_text(encoding="utf-8")) if pf.exists() else []) or []
    if any(e.get("id") == candidate_id for e in live):
        raise ValueError(f"候选 {candidate_id} 已在正式 catalog 中")
    live.append(match.entry.model_dump())
    pf.write_text(yaml.safe_dump(live, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return match.entry
