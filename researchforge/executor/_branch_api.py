"""Branch dispatch API for the executor.

`run_analysis` (in run.py) used to be one ~5500-line function with a 66-branch
``if/elif entry.id == ...`` chain — so large that merely reading the file blew the
context window. We decompose it: each branch becomes a handler ``def _branch_<id>(ctx)``
living in ``branches/<family>.py`` and registered here.

A handler unpacks ``ctx`` back into the SAME local names the monolith used
(df/fp/entry/cfg/d/files/summary/estimates/code) and then runs the original branch
body **verbatim**. Bodies only mutate summary/estimates/files/code (they never
rebind them), so the move is behaviour-preserving.

This module imports nothing from run.py, so run.py and branches.* can both import
it without a cycle. run.py imports the branches package at its END (once all helpers
and run_analysis are defined), which populates the registry via @register.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # avoid importing pandas / heavy deps at module load
    import pandas as pd

    from researchforge.catalog.schema import AnalysisEntry
    from researchforge.profiler.fingerprint import DataFingerprint


@dataclass
class Ctx:
    """The shared run_analysis state handed to every branch handler."""

    df: "pd.DataFrame"
    fp: "DataFingerprint"
    entry: "AnalysisEntry"
    cfg: dict
    d: Path
    files: list[str]
    summary: list[str]
    estimates: dict[str, float]
    code: list[str]


BRANCH_REGISTRY: dict[str, Callable[["Ctx"], None]] = {}


def register(*ids: str) -> Callable[[Callable[["Ctx"], None]], Callable[["Ctx"], None]]:
    """Decorator registering a handler for one or more analysis ids."""

    def deco(fn: Callable[["Ctx"], None]) -> Callable[["Ctx"], None]:
        for i in ids:
            if i in BRANCH_REGISTRY:
                raise ValueError(f"duplicate branch handler for {i!r}")
            BRANCH_REGISTRY[i] = fn
        return fn

    return deco
