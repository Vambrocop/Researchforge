"""FastAPI web application for ResearchForge.

Endpoints:
  GET  /                 -> serve index.html
  POST /api/analyze      -> upload CSV, profile + recommend
  POST /api/run          -> run a chosen analysis
Static mounts:
  /static   -> researchforge/web/static/
  /outputs  -> repo-level outputs/ dir
"""

from __future__ import annotations

import uuid
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

    result = run_for_path(dest, body.analysis_id, output_root=str(_OUTPUTS_DIR))
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return JSONResponse(result)
