from __future__ import annotations

import json
from pathlib import Path

from .pipeline import latest_manifest, load_chunks
from .store import VectorStore


def validate_run(workdir: Path = Path(".kbai")) -> dict:
    manifest = latest_manifest(workdir)
    chunks = load_chunks(workdir, manifest["run_id"])
    checks: dict[str, dict] = {}

    def add(name: str, passed: bool, detail: str) -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}

    add("manifest_completed", manifest.get("status") == "completed", f"status={manifest.get('status')}")
    add("pages_present", manifest.get("page_count", 0) > 0, f"pages={manifest.get('page_count', 0)}")
    add("unique_pages_present", manifest.get("unique_page_count", 0) > 0, f"unique_pages={manifest.get('unique_page_count', 0)}")
    add("chunks_present", bool(chunks), f"chunks={len(chunks)}")

    ids = [chunk.get("chunk_id") for chunk in chunks]
    add("chunk_ids_unique", len(ids) == len(set(ids)), f"unique={len(set(ids))}/{len(ids)}")

    lengths = [len(chunk.get("text", "")) for chunk in chunks]
    bad_lengths = sum(length < 80 or length > 1800 for length in lengths)
    add("chunk_lengths_reasonable", bad_lengths == 0, f"outside_80_1800={bad_lengths}")

    required = {
        "chunk_id", "chapter", "page_start", "page_end", "source_path", "source_sha256",
        "chunk_sha256", "pipeline_version", "extraction_methods",
    }
    incomplete = [chunk.get("chunk_id") for chunk in chunks if not required.issubset(chunk) or not chunk.get("source_sha256")]
    add("provenance_complete", not incomplete, f"incomplete={len(incomplete)}")

    chapters = {chunk.get("chapter") for chunk in chunks if chunk.get("chapter")}
    add("chapter_coverage", bool(chapters), f"chapters={len(chapters)}")

    page_count = int(manifest.get("page_count", 0))
    dup_count = int(manifest.get("duplicate_page_count", 0))
    unique_count = int(manifest.get("unique_page_count", 0))
    add("dedupe_accounting", page_count == dup_count + unique_count, f"{page_count}={unique_count}+{dup_count}")

    store = VectorStore(workdir / "chroma", manifest["collection_name"], manifest["embedding_model"])
    vector_count = store.count()
    add("vector_store_count", vector_count >= len(chunks), f"chroma={vector_count}, current_run_chunks={len(chunks)}")

    retrieval_ok = False
    retrieval_detail = "not run"
    if chunks:
        probe = chunks[0]["text"][:180]
        result = store.query(probe, top_k=1)
        retrieved = (result.get("ids") or [[]])[0]
        retrieval_ok = bool(retrieved)
        retrieval_detail = f"hits={len(retrieved)}"
    add("retrieval_smoke_test", retrieval_ok, retrieval_detail)

    passed = sum(1 for item in checks.values() if item["passed"])
    report = {
        "run_id": manifest["run_id"],
        "passed": passed == len(checks),
        "score": passed / len(checks) if checks else 0.0,
        "checks": checks,
    }
    output = workdir / "validation" / f"{manifest['run_id']}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
