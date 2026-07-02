"""Network-science family (split from the former branches/network_science.py monolith).

One module per @register handler + a shared _common.py (the edge-list resolver and
graph builder every method uses). pkgutil.walk_packages (branches/__init__.py) recurses
and imports each module, running its @register decorator. Drop a new
network_science/<id>.py here to add a method — no edits to this file needed.
"""
