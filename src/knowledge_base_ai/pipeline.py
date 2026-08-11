from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .catalog import write_catalog_artifacts
from .document_io import document_metadata, extract_pages, source_sha256
from .logging_utils import configure_logging, log_event
from .models import RunManifest
from .store import VectorStore
from .text_ops import assign_chapters, deduplicate_pages, semantic_chunks

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION = "knowledge-base-ai"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def ingest(
    source: Path,
    workdir: Path = Path(".kbai"),
    title: str | None = None,
    author: str | None = None,
    force_ocr: bool = False,
    model_name: str = DEFAULT_MODEL,
    collection_name: str = DEFAULT_COLLECTION,
    target_chars: int = 1200,
    overlap_chars: int = 180,
    verbose: bool = False,
) -> RunManifest:
    """Execute one isolated, auditable document-ingestion run."""
    logger = configure_logging(workdir, verbose)
    run_id = uuid.uuid4().hex[:12]
    effective_collection = f"{collection_name}-{run_id}"
    metadata = document_metadata(source)
    manifest = RunManifest(
        run_id=run_id,
        started_at=_now(),
        completed_at=None,
        status="running",
        source_path=str(source),
        source_sha256=source_sha256(source),
        title=title or metadata.get("title") or source.stem,
        author=author or metadata.get("author") or "Unknown",
        document_metadata=metadata,
        embedding_model=model_name,
        collection_name=effective_collection,
        force_ocr=force_ocr,
    )
    log_event(
        logger,
        "run_started",
        "Starting ingestion",
        run_id=run_id,
        source=str(source),
        collection=effective_collection,
    )

    try:
        pages = extract_pages(source, force_ocr=force_ocr)
        manifest.page_count = len(pages)
        manifest.low_quality_page_count = sum(page.quality_score < 0.72 for page in pages)
        log_event(
            logger,
            "extraction_complete",
            "Page extraction and OCR quality scoring complete",
            pages=len(pages),
            low_quality_pages=manifest.low_quality_page_count,
        )

        unique_pages, duplicate_count = deduplicate_pages(pages)
        manifest.unique_page_count = len(unique_pages)
        manifest.duplicate_page_count = duplicate_count
        log_event(logger, "dedupe_complete", "Deduplication complete", duplicates=duplicate_count)

        assign_chapters(unique_pages)
        chunks = semantic_chunks(
            unique_pages,
            target_chars=target_chars,
            overlap_chars=overlap_chars,
        )
        manifest.chunk_count = len(chunks)
        manifest.chapter_count = len({chunk.chapter for chunk in chunks})
        log_event(
            logger,
            "chunking_complete",
            "Chapter detection, labeling and semantic chunking complete",
            chapters=manifest.chapter_count,
            chunks=manifest.chunk_count,
        )

        _write_jsonl(
            workdir / "pages" / f"{run_id}.jsonl",
            [page.to_dict() for page in pages],
        )
        _write_jsonl(
            workdir / "chunks" / f"{run_id}.jsonl",
            [chunk.to_dict() for chunk in chunks],
        )

        inventory_path, tree_path = write_catalog_artifacts(workdir, manifest, pages, chunks)
        manifest.inventory_path = str(inventory_path)
        manifest.knowledge_tree_path = str(tree_path)
        log_event(logger, "catalog_complete", "Inventory and knowledge tree written")

        store = VectorStore(workdir / "chroma", effective_collection, model_name)
        store.upsert_chunks(chunks)
        log_event(
            logger,
            "vector_ingest_complete",
            "Embeddings stored in isolated Chroma collection",
            count=store.count(),
            collection=effective_collection,
        )

        manifest.status = "completed"
        manifest.completed_at = _now()
    except Exception as exc:
        manifest.status = "failed"
        manifest.completed_at = _now()
        manifest.errors.append(f"{type(exc).__name__}: {exc}")
        logger.exception(
            "Ingestion failed",
            extra={"event": "run_failed", "data": {"run_id": run_id}},
        )
        _save_manifest(workdir, manifest)
        raise

    _save_manifest(workdir, manifest)
    log_event(
        logger,
        "run_completed",
        "Ingestion completed",
        run_id=run_id,
        chunks=manifest.chunk_count,
    )
    return manifest


def _save_manifest(workdir: Path, manifest: RunManifest) -> Path:
    path = workdir / "manifests" / f"{manifest.run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def latest_manifest(workdir: Path = Path(".kbai")) -> dict:
    manifest_dir = workdir / "manifests"
    if not manifest_dir.exists():
        raise FileNotFoundError("No ingestion manifests found. Run `kbai ingest` first.")
    manifests = sorted(manifest_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not manifests:
        raise FileNotFoundError("No ingestion manifests found. Run `kbai ingest` first.")
    return json.loads(manifests[-1].read_text(encoding="utf-8"))


def load_chunks(workdir: Path, run_id: str) -> list[dict]:
    path = workdir / "chunks" / f"{run_id}.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
