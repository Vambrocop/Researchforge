"""Causal-inference family (split from the former branches/causal.py monolith).

One module per @register handler; pkgutil.walk_packages (branches/__init__.py)
recurses and imports each, running its @register decorator. Drop a new
causal/<id>.py here to add a method — no edits to this file needed.
"""
