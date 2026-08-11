from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .models import ChunkRecord, PageRecord, RunManifest


def build_knowledge_tree(manifest: RunManifest, pages: list[PageRecord], chunks: list[ChunkRecord]) -> dict:
    chapters: dict[str, dict] = defaultdict(lambda: {"pages": set(), "chunks": []})
    for page in pages:
        chapters[page.chapter]["pages"].add(page.page_number)
    for chunk in chunks:
        chapters[chunk.chapter]["chunks"].append(
            {
                "chunk_id": chunk.chunk_id,
                "pages": [chunk.page_start, chunk.page_end],
                "semantic_label": chunk.semantic_label,
                "sha256": chunk.chunk_sha256,
            }
        )
    return {
        "document": {
            "title": manifest.title,
            "author": manifest.author,
            "source_sha256": manifest.source_sha256,
            "metadata": manifest.document_metadata,
        },
        "chapters": [
            {
                "title": chapter,
                "pages": sorted(data["pages"]),
                "chunks": data["chunks"],
            }
            for chapter, data in chapters.items()
        ],
    }


def write_catalog_artifacts(
    workdir: Path,
    manifest: RunManifest,
    pages: list[PageRecord],
    chunks: list[ChunkRecord],
) -> tuple[Path, Path]:
    inventory = {
        "run_id": manifest.run_id,
        "document": {
            "title": manifest.title,
            "creator": manifest.author,
            "type": "Text",
            "format": "application/pdf" if Path(manifest.source_path).suffix.lower() == ".pdf" else "image-sequence",
            "identifier": manifest.source_sha256,
            "source": manifest.source_path,
            "language": manifest.document_metadata.get("language", "eng"),
            "rights": manifest.document_metadata.get("rights", "public-domain-source; verify source record"),
        },
        "counts": {
            "pages": manifest.page_count,
            "unique_pages": manifest.unique_page_count,
            "duplicate_pages": manifest.duplicate_page_count,
            "chapters": manifest.chapter_count,
            "chunks": manifest.chunk_count,
            "low_quality_pages": manifest.low_quality_page_count,
        },
        "page_quality": [
            {
                "page": page.page_number,
                "quality_score": page.quality_score,
                "quality_flags": page.quality_flags,
                "label": page.document_label,
                "extraction_method": page.extraction_method,
                "duplicate_of": page.duplicate_of,
            }
            for page in pages
        ],
    }
    tree = build_knowledge_tree(manifest, pages, chunks)

    inventory_path = workdir / "inventory" / f"{manifest.run_id}.json"
    tree_path = workdir / "knowledge_trees" / f"{manifest.run_id}.json"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    tree_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8")
    tree_path.write_text(json.dumps(tree, indent=2, ensure_ascii=False), encoding="utf-8")
    return inventory_path, tree_path
