from __future__ import annotations

import json
import math
from pathlib import Path

from .pipeline import latest_manifest, load_chunks
from .store import VectorStore


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_run(workdir: Path = Path(".kbai")) -> dict:
    manifest = latest_manifest(workdir)
    chunks = load_chunks(workdir, manifest["run_id"])
    pages = _jsonl(workdir / "pages" / f"{manifest['run_id']}.jsonl")
    checks: dict[str, dict] = {}

    def add(name: str, passed: bool, detail: str) -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}

    add("manifest_completed", manifest.get("status") == "completed", f"status={manifest.get('status')}")
    add("pages_present", bool(pages), f"pages={len(pages)}")
    add("unique_pages_present", manifest.get("unique_page_count", 0) > 0, f"unique_pages={manifest.get('unique_page_count', 0)}")
    add("chunks_present", bool(chunks), f"chunks={len(chunks)}")

    ids = [chunk.get("chunk_id") for chunk in chunks]
    add("chunk_ids_unique", len(ids) == len(set(ids)), f"unique={len(set(ids))}/{len(ids)}")

    lengths = [len(chunk.get("text", "")) for chunk in chunks]
    bad_lengths = sum(length < 40 or length > 2200 for length in lengths)
    add("chunk_lengths_reasonable", bad_lengths == 0, f"outside_40_2200={bad_lengths}")

    required = {
        "chunk_id", "chapter", "semantic_label", "page_start", "page_end", "source_path", "source_sha256",
        "chunk_sha256", "pipeline_version", "extraction_methods",
    }
    incomplete = [chunk.get("chunk_id") for chunk in chunks if not required.issubset(chunk) or not chunk.get("source_sha256")]
    add("provenance_complete", not incomplete, f"incomplete={len(incomplete)}")

    chapters = {chunk.get("chapter") for chunk in chunks if chunk.get("chapter")}
    labels = {chunk.get("semantic_label") for chunk in chunks if chunk.get("semantic_label")}
    add("chapter_coverage", bool(chapters), f"chapters={len(chapters)}")
    add("semantic_labels_present", bool(labels), f"labels={sorted(labels)}")

    page_count = int(manifest.get("page_count", 0))
    dup_count = int(manifest.get("duplicate_page_count", 0))
    unique_count = int(manifest.get("unique_page_count", 0))
    add("dedupe_accounting", page_count == dup_count + unique_count, f"{page_count}={unique_count}+{dup_count}")

    quality_scores = [float(page.get("quality_score", 0.0)) for page in pages]
    readable = sum(score >= 0.72 for score in quality_scores)
    readability_ratio = readable / len(quality_scores) if quality_scores else 0.0
    add("ocr_readability", readability_ratio >= 0.80, f"readable_pages={readable}/{len(quality_scores)} ({readability_ratio:.1%})")

    inventory = Path(manifest.get("inventory_path") or "")
    tree = Path(manifest.get("knowledge_tree_path") or "")
    add("inventory_written", inventory.is_file(), str(inventory))
    add("knowledge_tree_written", tree.is_file(), str(tree))

    store = VectorStore(workdir / "chroma", manifest["collection_name"], manifest["embedding_model"])
    vector_count = store.count()
    add("vector_store_count", vector_count >= len(chunks), f"chroma={vector_count}, current_run_chunks={len(chunks)}")

    sample_texts = [chunk["text"] for chunk in chunks[: min(16, len(chunks))]]
    if sample_texts:
        vectors = store.embed(sample_texts)
        finite = all(all(math.isfinite(value) for value in vector) for vector in vectors)
        norms = [math.sqrt(sum(value * value for value in vector)) for vector in vectors]
        normalized = all(0.98 <= norm <= 1.02 for norm in norms)
        distinct = len({tuple(round(value, 5) for value in vector[:24]) for vector in vectors}) > 1 or len(vectors) == 1
        add("embedding_quality", finite and normalized and distinct, f"finite={finite} normalized={normalized} distinct={distinct}")
    else:
        add("embedding_quality", False, "no chunks to embed")

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
