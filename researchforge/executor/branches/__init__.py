"""Branch handlers, split by method family.

Importing each family submodule runs its ``@register`` decorators, populating
``BRANCH_REGISTRY``. run.py imports this package at its end to wire everything up.
Add a family submodule import here as it is migrated out of the run.py monolith.
"""

from __future__ import annotations

# Migrated families (each import registers its handlers):
from . import soil  # noqa: F401
