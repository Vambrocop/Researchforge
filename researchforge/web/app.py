"""FastAPI web application for ResearchForge.

Endpoints:
  GET  /                                     -> serve index.html
  POST /api/analyze                          -> upload CSV, profile + recommend
  POST /api/run                              -> run a chosen analysis
  GET  /api/runs                             -> list previous runs (newest-first)
  GET  /api/runs/{run_name}/file/{filename}  -> serve one artifact (traversal-safe)
  GET  /api/download/{run_name}              -> zip and download an outputs/<run_name> dir
Static mounts:
  /static   -> researchforge/web/static/
  /outputs  -> repo-level outputs/ dir
"""

from __future__ import annotations

import tempfile
import uuid
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_WEB_UPLOADS = Path("web_uploads")
_WEB_UPLOADS.mkdir(exist_ok=True)

_OUTPUTS_DIR = Path("outputs")
_OUTPUTS_DIR.mkdir(exist_ok=True)

_STATIC_DIR = Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# In-memory file-id registry (also persisted to disk as <id>.csv)
# ---------------------------------------------------------------------------
_files: dict[str, Path] = {}


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="ResearchForge Web UI")

app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
app.mount("/outputs", StaticFiles(directory=str(_OUTPUTS_DIR)), name="outputs")


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------
class RunRequest(BaseModel):
    file_id: str
    analysis_id: str
    # optional user overrides for the engine's substantive defaults (column roles,
    # anchors, params) — needed for design-driven methods (rdd running/cutoff,
    # synthetic_control treated_unit/time, …). The service already accepts this.
    config: dict | None = None


class FileRequest(BaseModel):
    file_id: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.post("/api/analyze")
async def api_analyze(file: UploadFile = File(...)) -> JSONResponse:
    """Save uploaded CSV and return fingerprint + recommendations."""
    file_id = uuid.uuid4().hex
    dest = _WEB_UPLOADS / f"{file_id}.csv"
    dest.write_bytes(await file.read())
    _files[file_id] = dest

    from researchforge.web.service import analyze_path

    result = analyze_path(dest)
    return JSONResponse({"file_id": file_id, **result})


@app.post("/api/clean")
def api_clean(body: FileRequest) -> JSONResponse:
    """Run data cleaning on a previously uploaded file and save the cleaned version."""
    path = _WEB_UPLOADS / f"{body.file_id}.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="file_id not found")

    new_id = uuid.uuid4().hex
    cleaned_out = _WEB_UPLOADS / f"{new_id}.csv"

    from researchforge.web.service import clean_path

    result = clean_path(path, cleaned_out)
    _files[new_id] = cleaned_out
    return JSONResponse({**result, "cleaned_file_id": new_id})


@app.post("/api/reanalyze")
def api_reanalyze(body: FileRequest) -> JSONResponse:
    """Re-run profiling and recommendations on a previously uploaded/cleaned file."""
    path = _WEB_UPLOADS / f"{body.file_id}.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="file_id not found")

    from researchforge.web.service import analyze_path

    result = analyze_path(path)
    return JSONResponse(result)


@app.post("/api/run")
def api_run(body: RunRequest) -> JSONResponse:
    """Run the requested analysis on a previously uploaded file."""
    dest = _WEB_UPLOADS / f"{body.file_id}.csv"
    if not dest.exists():
        raise HTTPException(status_code=404, detail="file_id not found")

    from researchforge.web.service import run_for_path

    result = run_for_path(
        dest, body.analysis_id, output_root=str(_OUTPUTS_DIR), config=body.config
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Run history + artifact browsing
# ---------------------------------------------------------------------------
def _safe_run_dir(run_name: str) -> Path:
    """Resolve outputs/<run_name>/ and confirm it stays inside the outputs root.

    Raises HTTPException(400) on any path-traversal attempt, HTTPException(404)
    if the directory does not exist. Returns the resolved run directory.
    """
    # Reject any name that contains path separators or dot-dot up front.
    if "/" in run_name or "\\" in run_name or ".." in run_name:
        raise HTTPException(status_code=400, detail="invalid run_name")

    run_dir = (_OUTPUTS_DIR / run_name).resolve()
    # Confirm the resolved path is still inside _OUTPUTS_DIR (defence in depth).
    try:
        run_dir.relative_to(_OUTPUTS_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid run_name") from None

    if not run_dir.exists() or not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="run not found")
    return run_dir


def _parse_run_name(run_name: str) -> tuple[str, str]:
    """Split a run-dir basename into (analysis_id, timestamp).

    Run dirs are named `<timestamp>_<analysis_id>` (e.g.
    `20260624-143052-123_correlation_matrix`). The analysis id can itself
    contain underscores, so we split on the FIRST underscore only. If the name
    does not match the convention, fall back to (run_name, "").
    """
    ts, sep, aid = run_name.partition("_")
    if not sep:
        return run_name, ""
    return aid, ts


def list_runs() -> list[dict]:
    """Scan the outputs root and return previous runs, newest-first.

    Each entry: {run_name, analysis_id, timestamp, mtime, files:[...], n_files}.
    Robust to a missing/empty outputs dir (returns []).
    """
    root = _OUTPUTS_DIR
    if not root.exists() or not root.is_dir():
        return []

    runs: list[dict] = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        analysis_id, timestamp = _parse_run_name(d.name)
        files = sorted(f.name for f in d.iterdir() if f.is_file())
        try:
            mtime = d.stat().st_mtime
        except OSError:
            mtime = 0.0
        runs.append(
            {
                "run_name": d.name,
                "analysis_id": analysis_id,
                "timestamp": timestamp,
                "mtime": mtime,
                "files": files,
                "n_files": len(files),
            }
        )

    # Newest-first: prefer the dir-name timestamp (lexicographically sortable
    # given the %Y%m%d-%H%M%S-%f format), fall back to filesystem mtime.
    runs.sort(key=lambda r: (r["timestamp"], r["mtime"]), reverse=True)
    return runs


@app.get("/api/runs")
def api_runs() -> JSONResponse:
    """List previous runs (newest-first). Returns [] if outputs is missing/empty."""
    return JSONResponse(list_runs())


@app.get("/api/runs/{run_name}/file/{filename}")
def api_run_file(run_name: str, filename: str) -> FileResponse:
    """Serve a single artifact from outputs/<run_name>/<filename> for inline preview.

    Traversal-safe: both run_name and filename are validated against separators
    and dot-dot, and the fully-resolved target path is asserted to stay within
    the run directory before anything is served.
    """
    run_dir = _safe_run_dir(run_name)

    # Reject any filename that contains path separators or dot-dot.
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="invalid filename")

    target = (run_dir / filename).resolve()
    # Confirm the resolved file is still inside this run dir (defence in depth).
    try:
        target.relative_to(run_dir)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid filename") from None

    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")

    return FileResponse(str(target), filename=filename)


# ---------------------------------------------------------------------------
# Download helper — factored out so tests can import it directly
# ---------------------------------------------------------------------------
def _zip_run_dir(run_name: str) -> Path:
    """Build a zip of outputs/<run_name>/ and return the path to the temp zip.

    Raises HTTPException(400) if run_name looks like a path traversal.
    Raises HTTPException(404) if the directory does not exist.
    The caller is responsible for deleting the temp file when done.
    """
    run_dir = _safe_run_dir(run_name)

    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    tmp.close()
    zip_path = Path(tmp.name)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(run_dir.iterdir()):
            if file.is_file():
                zf.write(file, arcname=file.name)

    return zip_path


@app.get("/api/download/{run_name}")
def api_download(run_name: str) -> FileResponse:
    """Zip outputs/<run_name>/ and return as an attachment."""
    zip_path = _zip_run_dir(run_name)
    return FileResponse(
        str(zip_path),
        media_type="application/zip",
        filename=f"{run_name}.zip",
    )
