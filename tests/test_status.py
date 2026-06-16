"""Smoke test for the `status` front-door command (health + next-up + improve)."""

from __future__ import annotations

from researchforge.cli import main


def test_status_runs(capsys) -> None:
    rc = main(["status"])
    assert rc == 0
    out = capsys.readouterr().out
    for section in ("状态速览", "总分", "下一波", "需改进"):
        assert section in out
