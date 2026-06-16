"""Guardrail: no source module may grow back into a monolith.

run.py was once 7935 lines with a ~5500-line run_analysis — so large that merely
reading the file blew the context window ("prompt too long"). After decomposing into
branches/*.py + _helpers/*.py, every module sits well under this cap. Keeping it that
way forces new code into the right family module instead of one mega-file.
See CLAUDE.md「引擎架构 & 读码纪律」.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LIMIT = 1500  # lines; also the threshold the scorecard rewards for "modular" design


def test_no_source_module_exceeds_line_limit() -> None:
    offenders = []
    for p in (REPO / "researchforge").rglob("*.py"):
        if "__pycache__" in str(p):
            continue
        n = len(p.read_text(encoding="utf-8").splitlines())
        if n > LIMIT:
            offenders.append(f"{p.relative_to(REPO)} ({n} lines)")
    assert not offenders, (
        f"module(s) over {LIMIT} lines — split into smaller modules "
        f"(executor/branches/<family>.py or executor/_helpers/): {offenders}"
    )
