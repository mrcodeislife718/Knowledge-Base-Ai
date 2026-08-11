from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PageRecord:
    page_number: int
    text: str
    raw_text: str
    extraction_method: str
    source_path: str
    source_sha256: str
    page_sha256: str
    duplicate_of: int | None = None
    chapter: str = "Front Matter"
    document_label: str = "unclassified"
    quality_score: float = 0.0
    quality_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChunkRecord:
    chunk_id: str
    text: str
    chapter: str
    page_start: int
    page_end: int
    source_path: str
    source_sha256: str
    extraction_methods: list[str]
    chunk_sha256: str
    pipeline_version: str
    semantic_label: str = "narrative"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunManifest:
    run_id: str
    started_at: str
    completed_at: str | None
    status: str
    source_path: str
    source_sha256: str
    title: str
    author: str
    document_metadata: dict[str, Any]
    embedding_model: str
    collection_name: str
    force_ocr: bool
    page_count: int = 0
    unique_page_count: int = 0
    duplicate_page_count: int = 0
    chunk_count: int = 0
    chapter_count: int = 0
    low_quality_page_count: int = 0
    knowledge_tree_path: str | None = None
    inventory_path: str | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
