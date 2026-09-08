"""Guardrails on the test-selection lists themselves.

Every SLOW_MODULES entry must name a real test file — otherwise its tests silently
run in the fast loop (`-m "not slow"`). Catches the test_getis_ord_gi vs
test_getis_ord typo class (Codex review, 2026-06-16).

And a structural gate must never be parked in SLOW_MODULES: that is exactly how
`test_config_params_complete` sat red on origin/main for two waves — it cost 9s, so
it was filed as slow, so the fast loop stopped running it (2026-09-08)."""

from __future__ import annotations

from pathlib import Path

from conftest import GATE_MODULES, SLOW_MODULES


def test_slow_modules_are_real_test_files() -> None:
    actual = {p.stem for p in Path(__file__).parent.glob("test_*.py")}
    missing = sorted(SLOW_MODULES - actual)
    assert not missing, f"SLOW_MODULES names with no matching test file (typo?): {missing}"


def test_gate_modules_are_real_test_files() -> None:
    actual = {p.stem for p in Path(__file__).parent.glob("test_*.py")}
    missing = sorted(GATE_MODULES - actual)
    assert not missing, f"GATE_MODULES names with no matching test file (typo?): {missing}"


def test_gates_are_never_parked_in_the_slow_bucket() -> None:
    """A gate in SLOW_MODULES is skipped by the fast loop — i.e. it stops being a gate."""
    parked = sorted(GATE_MODULES & SLOW_MODULES)
    assert not parked, (
        "structural gates parked in SLOW_MODULES (the fast loop would skip them, which is "
        f"how a gate goes red unnoticed): {parked}. Remove them from SLOW_MODULES."
    )
