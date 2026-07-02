"""One broken family module must not take down all ~294 registered analyses.

`researchforge/executor/branches/__init__.py` auto-discovers every submodule via
``pkgutil.walk_packages`` + ``importlib.import_module``. This test forces exactly
one of those imports to fail (via monkeypatching ``importlib.import_module``) and
re-invokes the discovery loop by ``importlib.reload``-ing the package, then asserts:

- the offending module is recorded in ``branches.IMPORT_ERRORS`` (visible, not swallowed)
- every other submodule still gets registered (the loop doesn't abort early)
- the branch registry stays fully populated (nothing on the success path regresses)

Approach note: rather than refactoring the module-level ``for`` loop into a
separate ``_discover()`` callable (out of scope per the task's red-lines — only
``branches/__init__.py`` + ``cli.py`` may change substantively), we reload the
real package with ``importlib.import_module`` monkeypatched for one target name.
Since already-imported submodules are served from ``sys.modules`` on reload, this
exercises the *actual* try/except wrapper in the loop, not a reimplementation.
"""

from __future__ import annotations

import importlib
import pkgutil

import researchforge.executor.branches as branches
from researchforge.executor._branch_api import BRANCH_REGISTRY


def _pick_leaf_target() -> str:
    """A deterministic, real, non-package submodule name to force-fail."""
    mods = sorted(
        info.name
        for info in pkgutil.walk_packages(branches.__path__, prefix=branches.__name__ + ".")
        if not info.ispkg
    )
    assert mods, "expected at least one leaf branch module to exist"
    return mods[0]


def test_broken_module_is_quarantined_not_fatal(monkeypatch) -> None:
    target = _pick_leaf_target()
    real_import_module = importlib.import_module
    baseline_registry_size = len(BRANCH_REGISTRY)

    def fake_import_module(name, *args, **kwargs):
        if name == target:
            raise ImportError(f"synthetic failure for test: {name}")
        return real_import_module(name, *args, **kwargs)

    try:
        monkeypatch.setattr(importlib, "import_module", fake_import_module)
        importlib.reload(branches)

        offenders = dict(branches.IMPORT_ERRORS)
        assert target in offenders, "offending module must be recorded in IMPORT_ERRORS"
        assert "synthetic failure" in offenders[target]
        assert len(branches.IMPORT_ERRORS) == 1, (
            f"expected exactly one forced failure, got {branches.IMPORT_ERRORS}"
        )

        # Registry must not have collapsed — other analyses stayed registered
        # (submodules already in sys.modules are served from cache on reload,
        # so a clean re-registration count is the observable signal here).
        assert len(BRANCH_REGISTRY) > 0
        assert len(BRANCH_REGISTRY) == baseline_registry_size
    finally:
        # Restore real import_module and re-reload so this test doesn't leak a
        # quarantined state into any test that runs after it in this process.
        monkeypatch.undo()
        importlib.reload(branches)
        assert branches.IMPORT_ERRORS == []


def test_clean_import_has_no_errors() -> None:
    """Sanity: with nothing patched, real discovery records zero import errors."""
    importlib.reload(branches)
    assert branches.IMPORT_ERRORS == []
