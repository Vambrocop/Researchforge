"""CI guard — the codebase stays clean under the ruff CORRECTNESS ruleset.

The ruleset lives in pyproject.toml `[tool.ruff]`: pyflakes (F: unused imports /
undefined names / redefinitions) + bugbear (B: likely bugs), with the engine's
intentional conventions encoded as ignores (uniform ctx-unpack F841; run.py re-export
hub F401; FastAPI File()/Depends() defaults). This is a REAL-BUG guard, not a style
police — so unused imports, undefined names, and likely-bug patterns can't regress.

Skips cleanly when ruff isn't installed (it's in the `dev` extra: pip install -e .[dev]).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(importlib.util.find_spec("ruff") is None,
                    reason="ruff not installed (pip install -e '.[dev]')")
def test_ruff_correctness_clean() -> None:
    """`ruff check` must report ZERO violations under the pyproject ruleset."""
    res = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "researchforge", "tests"],
        cwd=_REPO, capture_output=True, text=True,
    )
    assert res.returncode == 0, (
        "ruff found correctness issues (fix them, or — if it's an intentional "
        "convention — encode it in pyproject [tool.ruff]):\n"
        + (res.stdout or "") + (res.stderr or "")
    )
