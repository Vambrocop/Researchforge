"""Experimental-design family (split from the former branches/experimental_design.py
monolith) — design-aware analyses that force explicit block/treatment/plot roles
instead of treating a field trial as a flat table.

One module per @register handler; pkgutil.walk_packages (branches/__init__.py)
recurses and imports each, running its @register decorator. Drop a new
experimental_design/<id>.py here to add a method — no edits to this file needed.
Shared role-hint constants and helpers live in ``_shared.py``.
"""
