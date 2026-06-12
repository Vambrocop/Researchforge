"""Scan a local data folder for datasets (data stays out of the repo)."""

from __future__ import annotations

from pathlib import Path

from researchforge.profiler.fingerprint import DataFingerprint
from researchforge.profiler.profile import profile_dataset

_DATA_SUFFIXES = {".csv", ".xlsx", ".xls"}


def scan_folder(directory: str | Path) -> list[Path]:
    base = Path(directory)
    return sorted(
        p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in _DATA_SUFFIXES
    )


def profile_folder(directory: str | Path) -> dict[str, DataFingerprint]:
    return {str(p): profile_dataset(p) for p in scan_folder(directory)}
