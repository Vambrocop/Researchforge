"""Guardrail: every catalog entry file must be valid YAML and a valid AnalysisEntry.

Lesson (deferred-log 1c): one malformed YAML scalar (an unquoted description with a
colon-space ": " sequence) once cascaded into ~51 failing tests, because the whole
catalog fails to load. This test loads EACH file individually so a broken file fails
fast and names itself, instead of an opaque cross-suite cascade.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from researchforge.catalog.schema import AnalysisEntry

_ENTRIES_DIR = Path(__file__).resolve().parents[1] / "researchforge" / "catalog" / "entries"


def _entry_files() -> list[Path]:
    return sorted(_ENTRIES_DIR.glob("*.yaml"))


def test_entries_dir_nonempty() -> None:
    assert _entry_files(), f"no catalog YAML files found under {_ENTRIES_DIR}"


@pytest.mark.parametrize("yaml_file", _entry_files(), ids=lambda p: p.name)
def test_catalog_file_is_valid_yaml_and_schema(yaml_file: Path) -> None:
    # 1) parses as YAML (catches the colon-space / indentation ScannerError class)
    try:
        data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
    except yaml.YAMLError as err:  # pragma: no cover - the message is the point
        pytest.fail(f"{yaml_file.name}: invalid YAML — {err}")

    if data is None:
        return  # an empty / comment-only file is allowed

    assert isinstance(data, list), f"{yaml_file.name}: top level must be a list of entries"
    # 2) every item constructs as an AnalysisEntry (catches schema / missing-field errors)
    for i, item in enumerate(data):
        assert isinstance(item, dict), f"{yaml_file.name}[{i}]: entry must be a mapping"
        try:
            AnalysisEntry(**item)
        except Exception as err:  # noqa: BLE001 - surface which entry + why
            pytest.fail(f"{yaml_file.name}[{i}] (id={item.get('id', '?')}): {err}")


def test_no_duplicate_ids_across_catalog() -> None:
    seen: dict[str, str] = {}
    dups: list[str] = []
    for f in _entry_files():
        data = yaml.safe_load(f.read_text(encoding="utf-8")) or []
        for item in data:
            if not isinstance(item, dict):
                continue
            eid = item.get("id")
            if eid in seen:
                dups.append(f"{eid} (in {seen[eid]} and {f.name})")
            elif eid:
                seen[eid] = f.name
    assert not dups, "duplicate catalog entry ids: " + "; ".join(dups)
