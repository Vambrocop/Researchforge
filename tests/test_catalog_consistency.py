"""Consistency gate: every catalog entry must have an executor branch, and ids
are unique. Catches dangling catalog entries before they ship (the panel_qca
near-miss). Loaded conventions: see CLAUDE.md."""

from __future__ import annotations

import glob
import re
from pathlib import Path

import yaml

from researchforge.executor._branch_api import BRANCH_REGISTRY
from researchforge.executor.run import _REGRESSION

REPO = Path(__file__).resolve().parents[1]


def _catalog_ids() -> list[str]:
    ids: list[str] = []
    for f in glob.glob(str(REPO / "researchforge" / "catalog" / "entries" / "*.yaml")):
        for e in yaml.safe_load(open(f, encoding="utf-8")) or []:
            ids.append(e["entry"]["id"] if "entry" in e else e["id"])
    return ids


def test_every_catalog_id_has_executor_branch() -> None:
    src = (REPO / "researchforge" / "executor" / "run.py").read_text(encoding="utf-8")
    # a catalog id is "handled" if it has an elif branch in run.py OR a registered
    # handler in executor/branches/*.py (the monolith is being split into the registry).
    handled = set(_REGRESSION) | set(BRANCH_REGISTRY) | set(re.findall(r'entry\.id == "([^"]+)"', src))
    missing = sorted(set(_catalog_ids()) - handled)
    assert missing == [], f"catalog entries with no executor branch: {missing}"


def test_no_duplicate_catalog_ids() -> None:
    ids = _catalog_ids()
    dups = sorted({i for i in ids if ids.count(i) > 1})
    assert dups == [], f"duplicate catalog ids: {dups}"


def test_catalog_nonempty() -> None:
    assert len(_catalog_ids()) >= 40  # sanity: the melting-pot has many methods
