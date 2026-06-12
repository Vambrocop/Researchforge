"""Novelty lens — a hook for "has the literature done this? where's the gap?".

Real scans run via deep-research / paper-search agents, injected as `search_fn`.
The default is an honest offline stub so the pipeline never blocks.
"""

from __future__ import annotations

from typing import Callable, Optional

from pydantic import BaseModel, Field


class NoveltyHint(BaseModel):
    query: str
    status: str  # scanned | not_run
    note: str = ""
    sources: list[str] = Field(default_factory=list)


def novelty_hint(
    query: str, search_fn: Optional[Callable[[str], list[str]]] = None
) -> NoveltyHint:
    if search_fn is None:
        return NoveltyHint(
            query=query,
            status="not_run",
            note="未运行文献扫描（离线）。接入 deep-research / paper-search 后将给出"
            "“文献做过没、白地在哪”。",
        )
    sources = list(search_fn(query))
    note = f"找到 {len(sources)} 条相关文献" if sources else "未找到直接相关文献——可能是白地"
    return NoveltyHint(query=query, status="scanned", note=note, sources=sources)
