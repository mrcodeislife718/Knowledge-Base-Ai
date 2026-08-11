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
    version="0.3.0",
    description="Employer-facing UI and API for auditable document intelligence.",
)
app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")

_state_lock = threading.Lock()
_run_state: dict[str, Any] = {
    "status": "idle", "stage": "Ready",
    "message": "Upload a document or inspect the latest run.", "run_id": None, "error": None,
}

class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)

class CompareRequest(BaseModel):
    query_a: str = Field(min_length=2, max_length=2000)
    query_b: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)


def _set_state(**changes: Any) -> None:
    with _state_lock: _run_state.update(changes)

def _state_snapshot() -> dict[str, Any]:
    with _state_lock: return dict(_run_state)

def _safe_filename(name: str | None) -> str:
    raw = Path(name or "upload.pdf").name
    cleaned = "".join(c for c in raw if c.isalnum() or c in "._- ").strip()
    return cleaned or "upload.pdf"

def _read_json(path: str | Path | None) -> dict[str, Any] | None:
    if not path: return None
    target = Path(path)
    if not target.is_file(): return None
    return json.loads(target.read_text(encoding="utf-8"))

def _latest_or_none() -> dict[str, Any] | None:
    try: return latest_manifest(WORKDIR)
    except FileNotFoundError: return None

def _store(current: dict[str, Any]) -> VectorStore:
    return VectorStore(WORKDIR / "chroma", current["collection_name"], current.get("embedding_model") or DEFAULT_MODEL)

def _search_results(current: dict[str, Any], query: str, top_k: int) -> list[dict[str, Any]]:
    raw = _store(current).query(query, top_k=top_k)
    ids=(raw.get("ids") or [[]])[0]; docs=(raw.get("documents") or [[]])[0]
    metas=(raw.get("metadatas") or [[]])[0]; distances=(raw.get("distances") or [[]])[0]
    results=[]
    for i, chunk_id in enumerate(ids):
        m=metas[i] or {}; distance=float(distances[i])
        results.append({"chunk_id":chunk_id,"text":docs[i],"distance":distance,
            "relevance":max(0.0,min(1.0,1.0-distance)),"chapter":m.get("chapter"),
            "semantic_label":m.get("semantic_label"),"page_start":m.get("page_start"),
            "page_end":m.get("page_end"),"source_path":m.get("source_path"),
            "source_sha256":m.get("source_sha256"),"chunk_sha256":m.get("chunk_sha256"),
            "extraction_methods":m.get("extraction_methods"),"embedding_model":m.get("embedding_model")})
    return results

def _ingest_background(source: Path, title: str | None, author: str | None, force_ocr: bool) -> None:
    _set_state(status="running", stage="Ingesting", message="Processing document…", error=None)
    try:
        manifest=ingest(source,workdir=WORKDIR,title=title or None,author=author or None,force_ocr=force_ocr)
        _set_state(status="completed",stage="Ready",message="Ingestion completed. Search and validation are available.",run_id=manifest.run_id,error=None)
    except Exception as exc:
        _set_state(status="failed",stage="Failed",message="Ingestion failed. Inspect the error and structured log.",error=f"{type(exc).__name__}: {exc}")

@app.get("/", include_in_schema=False)
def index() -> FileResponse: return FileResponse(STATIC_DIR / "index.html")

@app.get("/api/health")
def health() -> dict[str,str]: return {"status":"ok","service":"knowledge-base-ai","version":"0.3.0"}

@app.get("/api/status")
def status() -> dict[str,Any]: return {"runtime":_state_snapshot(),"manifest":_latest_or_none()}

@app.get("/api/manifest")
def manifest() -> dict[str,Any]:
    current=_latest_or_none()
    if current is None: raise HTTPException(404,"No ingestion run is available yet.")
    return current

@app.get("/api/runs")
def runs() -> dict[str,Any]:
    root=WORKDIR/"manifests"
    if not root.exists(): return {"runs":[]}
    items=[]
    for path in sorted(root.glob("*.json"),key=lambda p:p.stat().st_mtime,reverse=True)[:20]:
        m=json.loads(path.read_text(encoding="utf-8"))
        items.append({k:m.get(k) for k in ("run_id","title","author","status","page_count","chunk_count","chapter_count","duplicate_page_count","low_quality_page_count","started_at","completed_at")})
    return {"runs":items}

@app.get("/api/knowledge")
def knowledge() -> dict[str,Any]:
    current=_latest_or_none()
    if current is None: raise HTTPException(404,"No ingestion run is available yet.")
    chunks=load_chunks(WORKDIR,current["run_id"])
    return {"manifest":current,"inventory":_read_json(current.get("inventory_path")),"knowledge_tree":_read_json(current.get("knowledge_tree_path")),"chunks":chunks[:200],"chunk_count":len(chunks)}

@app.get("/api/analytics")
def analytics() -> dict[str,Any]:
    current=_latest_or_none()
    if current is None: raise HTTPException(404,"No ingestion run is available yet.")
    chunks=load_chunks(WORKDIR,current["run_id"])
    pages_path=WORKDIR/"pages"/f"{current['run_id']}.jsonl"
    pages=[json.loads(x) for x in pages_path.read_text(encoding="utf-8").splitlines() if x.strip()] if pages_path.exists() else []
    labels={}; chapters={}; methods={}
    for c in chunks:
        labels[c.get("semantic_label") or "unlabeled"]=labels.get(c.get("semantic_label") or "unlabeled",0)+1
        chapters[c.get("chapter") or "Unassigned"]=chapters.get(c.get("chapter") or "Unassigned",0)+1
        for method in str(c.get("extraction_methods") or "unknown").split(","):
            methods[method.strip()]=methods.get(method.strip(),0)+1
    quality=[float(p.get("quality_score",0)) for p in pages]
    avg=sum(quality)/len(quality) if quality else 0
    return {"quality":{"average":avg,"readable":sum(q>=.72 for q in quality),"total":len(quality),"low_quality":sum(q<.72 for q in quality)},
        "labels":labels,"chapters":chapters,"extraction_methods":methods,
        "dedupe_rate":(current.get("duplicate_page_count",0)/current.get("page_count",1)) if current.get("page_count") else 0,
        "avg_chunk_chars":round(sum(len(c.get("text","")) for c in chunks)/len(chunks)) if chunks else 0}

@app.post("/api/ingest", status_code=202)
def start_ingestion(background_tasks:BackgroundTasks,file:UploadFile=File(...),title:str=Form(default=""),author:str=Form(default=""),force_ocr:bool=Form(default=False))->dict[str,Any]:
    if _state_snapshot()["status"] in {"queued","running"}: raise HTTPException(409,"An ingestion run is already in progress.")
    filename=_safe_filename(file.filename); suffix=Path(filename).suffix.lower()
    if suffix not in {".pdf",".png",".jpg",".jpeg",".tif",".tiff",".bmp"}: raise HTTPException(400,"Upload a supported PDF or page image.")
    UPLOAD_DIR.mkdir(parents=True,exist_ok=True); destination=UPLOAD_DIR/filename
    with destination.open("wb") as output: shutil.copyfileobj(file.file,output)
    file.file.close(); _set_state(status="queued",stage="Queued",message=f"{filename} is queued for ingestion.",error=None)
    background_tasks.add_task(_ingest_background,destination,title,author,force_ocr)
    return {"accepted":True,"filename":filename}

@app.post("/api/search")
def search(request:SearchRequest)->dict[str,Any]:
    current=_latest_or_none()
    if current is None: raise HTTPException(404,"Ingest a document before searching.")
    if current.get("status")!="completed": raise HTTPException(409,"The latest ingestion did not complete successfully.")
    return {"query":request.query,"results":_search_results(current,request.query,request.top_k)}

@app.post("/api/compare")
def compare(request:CompareRequest)->dict[str,Any]:
    current=_latest_or_none()
    if current is None: raise HTTPException(404,"Ingest a document before comparing concepts.")
    a=_search_results(current,request.query_a,request.top_k); b=_search_results(current,request.query_b,request.top_k)
    a_ids={x["chunk_id"] for x in a}; b_ids={x["chunk_id"] for x in b}; overlap=a_ids&b_ids
    return {"query_a":request.query_a,"query_b":request.query_b,"results_a":a,"results_b":b,
        "shared_chunks":list(overlap),"overlap_ratio":len(overlap)/max(1,len(a_ids|b_ids))}

@app.post("/api/validate")
def validate()->dict[str,Any]:
    try: return validate_run(WORKDIR)
    except FileNotFoundError as exc: raise HTTPException(404,str(exc)) from exc

def main()->None:
    uvicorn.run("knowledge_base_ai.web:app",host="127.0.0.1",port=8000,reload=False)
