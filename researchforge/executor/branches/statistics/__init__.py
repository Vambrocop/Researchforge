"""Statistics family (split from the former branches/statistics.py monolith, migrated
from the run.py monolith originally).

One module per @register handler (or a small cluster of closely-related handlers).
pkgutil.walk_packages (branches/__init__.py) recurses and imports each module, running
its @register decorator. Drop a new statistics/<id>.py here to add a method — no edits
to this file needed.

Each handler unpacks ctx into the same local names run_analysis used and runs the
original branch body verbatim. See executor/_branch_api.py.
"""
