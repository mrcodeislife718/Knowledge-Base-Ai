from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .pipeline import DEFAULT_MODEL, ingest, latest_manifest, load_chunks
from .store import VectorStore
from .validation import validate_run

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
WORKDIR = Path(".kbai-ui")
UPLOAD_DIR = WORKDIR / "uploads"

app = FastAPI(
    title="Knowledge-Base AI",
    version="0.2.0",
    description="Employer-facing UI and API for the auditable document-to-vector pipeline.",
)
app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")

_state_lock = threading.Lock()
_run_state: dict[str, Any] = {
    "status": "idle",
    "stage": "Ready",
    "message": "Upload a document or inspect the latest run.",
    "run_id": None,
    "error": None,
}


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


def _set_state(**changes: Any) -> None:
    with _state_lock:
        _run_state.update(changes)


def _state_snapshot() -> dict[str, Any]:
    with _state_lock:
        return dict(_run_state)


def _safe_filename(name: str | None) -> str:
    raw = Path(name or "upload.pdf").name
    cleaned = "".join(char for char in raw if char.isalnum() or char in "._- ").strip()
    return cleaned or "upload.pdf"


def _read_json(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    target = Path(path)
    if not target.is_file():
        return None
    return json.loads(target.read_text(encoding="utf-8"))


def _latest_or_none() -> dict[str, Any] | None:
    try:
        return latest_manifest(WORKDIR)
    except FileNotFoundError:
        return None


def _ingest_background(source: Path, title: str | None, author: str | None, force_ocr: bool) -> None:
    _set_state(status="running", stage="Ingesting", message="Processing document…", error=None)
    try:
        manifest = ingest(
            source,
            workdir=WORKDIR,
            title=title or None,
            author=author or None,
            force_ocr=force_ocr,
        )
        _set_state(
            status="completed",
            stage="Ready",
            message="Ingestion completed. Search and validation are available.",
            run_id=manifest.run_id,
            error=None,
        )
    except Exception as exc:
        _set_state(
            status="failed",
            stage="Failed",
            message="Ingestion failed. Inspect the error and structured log.",
            error=f"{type(exc).__name__}: {exc}",
        )


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "knowledge-base-ai"}


@app.get("/api/status")
def status() -> dict[str, Any]:
    return {"runtime": _state_snapshot(), "manifest": _latest_or_none()}


@app.get("/api/manifest")
def manifest() -> dict[str, Any]:
    current = _latest_or_none()
    if current is None:
        raise HTTPException(status_code=404, detail="No ingestion run is available yet.")
    return current


@app.get("/api/knowledge")
def knowledge() -> dict[str, Any]:
    current = _latest_or_none()
    if current is None:
        raise HTTPException(status_code=404, detail="No ingestion run is available yet.")
    chunks = load_chunks(WORKDIR, current["run_id"])
    return {
        "manifest": current,
        "inventory": _read_json(current.get("inventory_path")),
        "knowledge_tree": _read_json(current.get("knowledge_tree_path")),
        "chunks": chunks[:200],
        "chunk_count": len(chunks),
    }


@app.post("/api/ingest", status_code=202)
def start_ingestion(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(default=""),
    author: str = Form(default=""),
    force_ocr: bool = Form(default=False),
) -> dict[str, Any]:
    if _state_snapshot()["status"] in {"queued", "running"}:
        raise HTTPException(status_code=409, detail="An ingestion run is already in progress.")

    filename = _safe_filename(file.filename)
    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        raise HTTPException(status_code=400, detail="Upload a supported PDF or page image.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    destination = UPLOAD_DIR / filename
    with destination.open("wb") as output:
        shutil.copyfileobj(file.file, output)
    file.file.close()

    _set_state(status="queued", stage="Queued", message=f"{filename} is queued for ingestion.", error=None)
    background_tasks.add_task(_ingest_background, destination, title, author, force_ocr)
    return {"accepted": True, "filename": filename}


@app.post("/api/search")
def search(request: SearchRequest) -> dict[str, Any]:
    current = _latest_or_none()
    if current is None:
        raise HTTPException(status_code=404, detail="Ingest a document before searching.")
    if current.get("status") != "completed":
        raise HTTPException(status_code=409, detail="The latest ingestion did not complete successfully.")

    store = VectorStore(
        WORKDIR / "chroma",
        current["collection_name"],
        current.get("embedding_model") or DEFAULT_MODEL,
    )
    raw = store.query(request.query, top_k=request.top_k)
    ids = (raw.get("ids") or [[]])[0]
    documents = (raw.get("documents") or [[]])[0]
    metadatas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]

    results = []
    for index, chunk_id in enumerate(ids):
        metadata = metadatas[index] or {}
        results.append(
            {
                "chunk_id": chunk_id,
                "text": documents[index],
                "distance": distances[index],
                "chapter": metadata.get("chapter"),
                "semantic_label": metadata.get("semantic_label"),
                "page_start": metadata.get("page_start"),
                "page_end": metadata.get("page_end"),
                "source_path": metadata.get("source_path"),
                "source_sha256": metadata.get("source_sha256"),
                "chunk_sha256": metadata.get("chunk_sha256"),
                "extraction_methods": metadata.get("extraction_methods"),
                "embedding_model": metadata.get("embedding_model"),
            }
        )
    return {"query": request.query, "results": results}


@app.post("/api/validate")
def validate() -> dict[str, Any]:
    try:
        return validate_run(WORKDIR)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def main() -> None:
    """Launch the local employer-facing web surface."""
    uvicorn.run("knowledge_base_ai.web:app", host="127.0.0.1", port=8000, reload=False)
