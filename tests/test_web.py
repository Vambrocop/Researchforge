"""Tests for researchforge.web — pure service functions (no HTTP server needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from researchforge.synth import make_panel
from researchforge.web.service import analyze_path, run_for_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_panel(tmp_path: Path) -> Path:
    csv = tmp_path / "panel.csv"
    make_panel(n_units=6, n_periods=6, treated=True, seed=42).to_csv(csv, index=False)
    return csv


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_analyze_fingerprint_and_recommendations(tmp_path):
    csv = _write_panel(tmp_path)
    result = analyze_path(csv)

    fp = result["fingerprint"]
    assert fp["is_panel"] is True, "panel data must be detected"
    assert fp["n_rows"] > 0
    assert fp["n_cols"] > 0
    assert isinstance(fp["columns"], list)
    assert all("name" in c and "kind" in c for c in fp["columns"])

    recs = result["recommendations"]
    assert len(recs) > 0, "at least one recommendation expected"

    ids = [r["id"] for r in recs]
    assert "did" in ids, "'did' must appear in recommendations for panel data"

    # Every rec must carry the required keys
    required = {"id", "method", "family", "light", "score", "feasible", "note", "biases"}
    for r in recs:
        assert required <= r.keys(), f"recommendation missing keys: {required - r.keys()}"
        assert r["light"] in {"green", "yellow", "red"}


def test_run_did_returns_report(tmp_path):
    csv = _write_panel(tmp_path)
    result = run_for_path(csv, "did", output_root=str(tmp_path / "outputs"))

    assert "summary" in result, "result must have 'summary'"
    assert "files" in result and result["files"], "result must list output files"
    assert "report.md" in result["files"], "report.md must be in files list"
    assert result["report"], "report text must be non-empty"
    assert "output_dir" in result
    assert "estimates" in result


def test_run_unknown_analysis_returns_error(tmp_path):
    csv = _write_panel(tmp_path)
    result = run_for_path(csv, "does_not_exist_xyz", output_root=str(tmp_path / "outputs"))
    assert "error" in result
    assert result["error"] == "unknown analysis"


def test_app_imports_without_error():
    """The FastAPI app must construct cleanly at import time."""
    import researchforge.web.app as web_app  # noqa: F401

    assert hasattr(web_app, "app"), "module must expose 'app'"
