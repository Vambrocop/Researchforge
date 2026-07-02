"""Branch handlers, organised by method family.

Importing any submodule (or sub-package module) runs its ``@register`` decorators,
populating ``BRANCH_REGISTRY``. We **auto-discover** every module under this package
so that adding a new analysis is just dropping a file in the right place — no edits
here, no merge conflicts. ``run.py`` imports this package at its end to wire it up.

Layout convention (see CLAUDE.md「引擎架构」): a family is a single ``<family>.py``
until it nears the size guardrail, then it is promoted to a ``<family>/`` package with
one module per analysis (``<family>/<id>.py``). walk_packages recurses, so both forms
work without touching this file.
"""

from __future__ import annotations

import importlib
import pkgutil

# Modules that failed to import during auto-discovery: (module_name, repr(exc)).
# One broken family must not take down the other ~294 analyses — surfaced (not
# swallowed) via `cli status` so it stays visible until fixed.
IMPORT_ERRORS: list[tuple[str, str]] = []

# Recursively import every submodule/sub-package so its @register decorators run.
for _info in pkgutil.walk_packages(__path__, prefix=__name__ + "."):
    try:
        importlib.import_module(_info.name)
    except Exception as exc:  # noqa: BLE001 - quarantine, don't abort the whole registry
        IMPORT_ERRORS.append((_info.name, repr(exc)))
        continue
