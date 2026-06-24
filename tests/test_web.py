"""Tests for researchforge.web — pure service functions (no HTTP server needed)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from researchforge.synth import make_panel
from researchforge.web.service import analyze_path, clean_path, run_for_path


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


def test_run_endpoint_forwards_config(monkeypatch):
    """POST /api/run must forward the optional `config` override to the service
    (needed for design-driven methods: rdd running/cutoff, synthetic_control, …)."""
    from fastapi.testclient import TestClient

    import researchforge.web.app as web_app

    captured = {}

    def _fake_run(path, analysis_id, output_root="outputs", config=None):
        captured["config"] = config
        captured["analysis_id"] = analysis_id
        return {"summary": "ok", "output_dir": "outputs/x", "files": [], "report": "", "estimates": {}}

    monkeypatch.setattr("researchforge.web.service.run_for_path", _fake_run)

    fid = "testcfgid_passthrough"
    upload = web_app._WEB_UPLOADS / f"{fid}.csv"
    upload.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    try:
        client = TestClient(web_app.app)
        resp = client.post(
            "/api/run",
            json={"file_id": fid, "analysis_id": "rdd", "config": {"running": "a", "cutoff": 2}},
        )
        assert resp.status_code == 200, resp.text
        assert captured["config"] == {"running": "a", "cutoff": 2}, "config must reach the service"
        assert captured["analysis_id"] == "rdd"
    finally:
        upload.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Helpers for messy data
# ---------------------------------------------------------------------------
def _write_messy_csv(tmp_path: Path) -> Path:
    """Create a CSV with a duplicate row, a missing value, and a constant column."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 3],          # row 3 is a duplicate of row 2
            "value": [10.0, None, 30.0, 30.0],  # one missing value
            "constant": ["x", "x", "x", "x"],  # constant column
        }
    )
    csv = tmp_path / "messy.csv"
    df.to_csv(csv, index=False)
    return csv


# ---------------------------------------------------------------------------
# New tests
# ---------------------------------------------------------------------------
def test_clean_path(tmp_path):
    csv = _write_messy_csv(tmp_path)
    cleaned_out = tmp_path / "cleaned.csv"

    result = clean_path(csv, cleaned_out)

    # plan must have at least one step
    assert isinstance(result["plan"], list), "plan must be a list"
    assert len(result["plan"]) >= 1, "plan must have at least one cleaning step"

    # log must have been returned
    assert isinstance(result["log"], list), "log must be a list"

    # cleaned file must have been written and be readable
    assert cleaned_out.exists(), "cleaned CSV must be written to disk"
    cleaned_df = pd.read_csv(cleaned_out)
    assert len(cleaned_df) > 0, "cleaned file must have rows"

    # duplicate rows must have been dropped
    assert cleaned_df.duplicated().sum() == 0, "cleaned data must have no duplicate rows"


def test_analyze_includes_issues(tmp_path):
    csv = _write_messy_csv(tmp_path)
    result = analyze_path(csv)

    fp = result["fingerprint"]
    assert "issues" in fp, "fingerprint must include 'issues' key"
    assert isinstance(fp["issues"], list), "issues must be a list"
    assert len(fp["issues"]) >= 1, "messy data must produce at least one issue"

    # Each issue must have the required keys
    for iss in fp["issues"]:
        for key in ("kind", "column", "severity", "detail"):
            assert key in iss, f"issue missing key '{key}': {iss}"


# ---------------------------------------------------------------------------
# Download endpoint tests
# ---------------------------------------------------------------------------
def test_download_zip(tmp_path):
    """GET /api/download/<run_name> returns a valid zip; bad names return 400/404."""
    from fastapi.testclient import TestClient

    # -- create a real run so we have an outputs/<dir> to zip --
    csv = _write_panel(tmp_path)
    run_out = tmp_path / "outputs"
    result = run_for_path(csv, "did", output_root=str(run_out))
    assert "output_dir" in result, "run must succeed"

    run_dir = Path(result["output_dir"])
    run_name = run_dir.name  # just the basename

    # Patch _OUTPUTS_DIR inside the app to point at our tmp outputs root
    import researchforge.web.app as web_app
    original_outputs = web_app._OUTPUTS_DIR
    web_app._OUTPUTS_DIR = run_out

    try:
        client = TestClient(web_app.app, raise_server_exceptions=True)

        # -- happy path: real run dir --
        resp = client.get(f"/api/download/{run_name}")
        assert resp.status_code == 200, f"expected 200, got {resp.status_code}: {resp.text}"
        assert "zip" in resp.headers.get("content-type", ""), (
            f"expected zip content-type, got {resp.headers.get('content-type')}"
        )
        assert len(resp.content) > 0, "zip body must be non-empty"

        # Verify the body is actually a valid zip
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        names_in_zip = zf.namelist()
        assert len(names_in_zip) > 0, "zip must contain at least one file"
        # All entries should be plain filenames (no path separators)
        for n in names_in_zip:
            assert "/" not in n and "\\" not in n, f"arcname must be filename only: {n}"

        # -- traversal names must be rejected with 400 --
        for bad_name in ["..", "../x", "foo/bar", "foo\\bar"]:
            r = client.get(f"/api/download/{bad_name}")
            assert r.status_code in (400, 404), (
                f"traversal name {bad_name!r} should return 400 or 404, got {r.status_code}"
            )

        # -- non-existent run dir must return 404 --
        r = client.get("/api/download/does_not_exist_xyz_12345")
        assert r.status_code == 404, f"missing run should return 404, got {r.status_code}"

    finally:
        web_app._OUTPUTS_DIR = original_outputs


def test_zip_run_dir_helper(tmp_path):
    """_zip_run_dir helper can be called directly and builds a valid zip."""
    import researchforge.web.app as web_app

    # Create a fake run dir under tmp
    run_out = tmp_path / "outputs"
    csv = _write_panel(tmp_path)
    result = run_for_path(csv, "did", output_root=str(run_out))
    assert "output_dir" in result

    run_dir = Path(result["output_dir"])
    run_name = run_dir.name

    original_outputs = web_app._OUTPUTS_DIR
    web_app._OUTPUTS_DIR = run_out
    try:
        zip_path = web_app._zip_run_dir(run_name)
        assert zip_path.exists(), "zip file must be created"
        assert zip_path.stat().st_size > 0, "zip must be non-empty"

        with zipfile.ZipFile(zip_path) as zf:
            assert len(zf.namelist()) > 0
    finally:
        web_app._OUTPUTS_DIR = original_outputs
        if zip_path.exists():
            zip_path.unlink()


# ---------------------------------------------------------------------------
# Config-form data contract (end-to-end through the HTTP layer)
#
# The frontend (static/index.html) renders a per-analysis config form from the
# `params` list each recommendation carries in /api/analyze, using
# fingerprint.columns to populate column selectors, then POSTs the assembled
# {file_id, analysis_id, config} to /api/run. These tests lock that contract at
# the HTTP boundary so a field rename / dropped key would fail loudly.
# ---------------------------------------------------------------------------
def _numeric_csv_bytes() -> bytes:
    """Three named numeric columns — enough rows/cols to make correlation_matrix
    feasible (min_numeric_cols=2, min_rows=3) and to give us stable column names."""
    df = pd.DataFrame(
        {
            "alpha": range(20),
            "beta": [i * 1.3 + 2 for i in range(20)],
            "gamma": [i * 0.7 for i in range(20)],
        }
    )
    return df.to_csv(index=False).encode("utf-8")


def _analyze_via_http(client) -> dict:
    """POST a numeric CSV to /api/analyze and return the parsed JSON body."""
    resp = client.post(
        "/api/analyze",
        files={"file": ("d.csv", _numeric_csv_bytes(), "text/csv")},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_analyze_response_carries_params_for_form():
    """/api/analyze recommendations must each carry a `params` list, and a known
    entry must expose the ParamSpec fields the form reads (name/type/required/
    description/choices/default). This is the single source of truth the form
    renders from — if `params` drops, the form silently renders nothing."""
    from fastapi.testclient import TestClient

    import researchforge.web.app as web_app

    client = TestClient(web_app.app)
    data = _analyze_via_http(client)

    # fingerprint.columns is what the form's column/columns selectors are built from
    cols = data["fingerprint"]["columns"]
    assert isinstance(cols, list) and cols, "fingerprint.columns must be non-empty"
    assert all("name" in c and "kind" in c for c in cols), (
        "each column must carry name+kind (the form reads c.name and c.kind)"
    )
    colnames = {c["name"] for c in cols}
    assert {"alpha", "beta", "gamma"} <= colnames

    recs = data["recommendations"]
    assert recs, "expected at least one recommendation"

    # EVERY recommendation must carry a `params` key that is a list (the form does
    # `rec.params || []`; a missing key would still work but a renamed key is a bug)
    for r in recs:
        assert "params" in r, f"recommendation {r.get('id')!r} missing 'params'"
        assert isinstance(r["params"], list)

    # A known entry that declares params must round-trip the full ParamSpec shape.
    by_id = {r["id"]: r for r in recs}
    assert "correlation_matrix" in by_id, (
        "correlation_matrix should be recommended for a 3-numeric-column dataset"
    )
    cm_params = {p["name"]: p for p in by_id["correlation_matrix"]["params"]}
    assert {"method", "columns"} <= cm_params.keys()

    # Each param dict must carry exactly the fields the form's paramFieldHtml reads.
    for p in by_id["correlation_matrix"]["params"]:
        for field in ("name", "type", "required", "description", "choices", "default"):
            assert field in p, f"param {p.get('name')!r} missing form field {field!r}"

    # `method` is a choice param -> the form renders a <select> from p.choices
    method = cm_params["method"]
    assert method["type"] == "choice"
    assert set(method["choices"]) == {"pearson", "spearman"}

    # `columns` is a columns param -> the form renders checkboxes from fingerprint.columns
    assert cm_params["columns"]["type"] == "columns"


def test_run_with_valid_config_has_no_param_warning():
    """A config the form could legitimately assemble (a real column subset + a
    valid choice) must run cleanly with NO '配置参数提示' warning in the summary.
    This proves a well-formed form submission is accepted as-is."""
    from fastapi.testclient import TestClient

    import researchforge.web.app as web_app

    client = TestClient(web_app.app)
    data = _analyze_via_http(client)
    file_id = data["file_id"]

    # valid choice + a real column subset (exactly what the form would POST)
    resp = client.post(
        "/api/run",
        json={
            "file_id": file_id,
            "analysis_id": "correlation_matrix",
            "config": {"method": "spearman", "columns": ["alpha", "beta"]},
        },
    )
    assert resp.status_code == 200, resp.text
    summary = resp.json()["summary"]
    assert "配置参数提示" not in summary, (
        f"valid config must not raise a config warning; got: {summary[:200]}"
    )


def test_run_with_bad_config_key_surfaces_warning():
    """A typo'd key (the realistic failure the spec is meant to catch) must still
    run (non-blocking) but the returned summary must carry '配置参数提示' and name
    the unknown key — proving the form's submissions are validated server-side."""
    from fastapi.testclient import TestClient

    import researchforge.web.app as web_app

    client = TestClient(web_app.app)
    data = _analyze_via_http(client)
    file_id = data["file_id"]

    resp = client.post(
        "/api/run",
        json={
            "file_id": file_id,
            "analysis_id": "correlation_matrix",
            "config": {"method": "pearson", "methdo": "spearman"},  # typo'd key
        },
    )
    # non-blocking: the run still succeeds (200), the warning is in the summary
    assert resp.status_code == 200, resp.text
    summary = resp.json()["summary"]
    assert "配置参数提示" in summary, "unknown key must surface a config warning"
    assert "methdo" in summary, "the warning must name the offending key"


def test_run_with_bad_choice_value_surfaces_warning():
    """A value outside a choice param's allowed set (e.g. a stale option) must
    also surface a warning while still running on the auto default."""
    from fastapi.testclient import TestClient

    import researchforge.web.app as web_app

    client = TestClient(web_app.app)
    data = _analyze_via_http(client)
    file_id = data["file_id"]

    resp = client.post(
        "/api/run",
        json={
            "file_id": file_id,
            "analysis_id": "correlation_matrix",
            "config": {"method": "kendall"},  # not in {pearson, spearman}
        },
    )
    assert resp.status_code == 200, resp.text
    summary = resp.json()["summary"]
    assert "配置参数提示" in summary
    assert "method" in summary
