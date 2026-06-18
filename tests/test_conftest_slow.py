"""Guardrail: every SLOW_MODULES entry must name a real test file — otherwise its
tests silently run in the fast loop (`-m "not slow"`). Catches the test_getis_ord_gi
vs test_getis_ord typo class (Codex review, 2026-06-16)."""

from __future__ import annotations

from pathlib import Path

from conftest import SLOW_MODULES


def test_slow_modules_are_real_test_files() -> None:
    actual = {p.stem for p in Path(__file__).parent.glob("test_*.py")}
    missing = sorted(SLOW_MODULES - actual)
    assert not missing, f"SLOW_MODULES names with no matching test file (typo?): {missing}"
