"""Ingestion: pull skills / papers from skills_inbox into a registry manifest.

The full LLM enrichment (skill/paper -> structured AnalysisEntry) is the
agent-driven step; here we record what arrived and archive the originals so the
process is deterministic and testable.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml
from pydantic import BaseModel

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_INBOX = _ROOT / "skills_inbox"
_DEFAULT_MANIFEST = _ROOT / "docs" / "analysis-catalog" / "ingested.json"


class IngestedItem(BaseModel):
    name: str
    description: str = ""
    kind: str  # skill | markdown | paper
    source: str


def _parse_frontmatter(text: str) -> dict:
    if text.lstrip().startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                return yaml.safe_load(parts[1]) or {}
            except Exception:
                return {}
    return {}


def _append_manifest(manifest: Path, items: list[IngestedItem]) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    existing: list = []
    if manifest.exists():
        try:
            existing = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            existing = []
    existing.extend(i.model_dump() for i in items)
    manifest.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")


def ingest_inbox(
    inbox: str | Path = _DEFAULT_INBOX,
    manifest: str | Path = _DEFAULT_MANIFEST,
    archive: bool = True,
) -> list[IngestedItem]:
    inbox = Path(inbox)
    items: list[IngestedItem] = []
    if not inbox.exists():
        return items

    processed = inbox / "_processed"
    if archive:
        processed.mkdir(parents=True, exist_ok=True)

    for sub in sorted(p for p in inbox.iterdir() if p.is_dir() and not p.name.startswith("_")):
        skill_md = sub / "SKILL.md"
        if skill_md.exists():
            meta = _parse_frontmatter(skill_md.read_text(encoding="utf-8", errors="ignore"))
            items.append(
                IngestedItem(
                    kind="skill",
                    name=str(meta.get("name", sub.name)),
                    description=str(meta.get("description", "")),
                    source=sub.name,
                )
            )
            if archive:
                shutil.move(str(sub), str(processed / sub.name))

    for md in sorted(inbox.glob("*.md")):
        if md.name.lower() == "readme.md":
            continue
        meta = _parse_frontmatter(md.read_text(encoding="utf-8", errors="ignore"))
        items.append(
            IngestedItem(
                kind="markdown",
                name=str(meta.get("name", md.stem)),
                description=str(meta.get("description", "")),
                source=md.name,
            )
        )
        if archive:
            shutil.move(str(md), str(processed / md.name))

    for pdf in sorted(inbox.glob("*.pdf")):
        items.append(IngestedItem(kind="paper", name=pdf.stem, source=pdf.name))
        if archive:
            shutil.move(str(pdf), str(processed / pdf.name))

    if items:
        _append_manifest(Path(manifest), items)
    return items
