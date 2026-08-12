"""Deterministic text normalization, structure detection, dedupe, and chunking."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

from . import PIPELINE_VERSION
from .models import ChunkRecord, PageRecord
from .quality import classify_chunk, classify_page

_CHAPTER_RE = re.compile(
    r"^(chapter|book|part)\s+([ivxlcdm]+|\d+)(?:\s*[:.\-–—]?\s*(.*))?$",
    re.IGNORECASE,
)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")
_MIN_PARAGRAPH_CHARS = 20
_MIN_CHUNK_CHARS = 80


def sha256_text(text: str) -> str:
    """Return a stable UTF-8 SHA-256 content fingerprint."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_text(text: str) -> str:
    """Normalize OCR/PDF artifacts while preserving paragraph boundaries."""
    text = text.replace("\u00ad", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    out: list[str] = []
    blank = False
    for line in lines:
        if not line:
            if out and not blank:
                out.append("")
            blank = True
            continue
        out.append(line)
        blank = False
    return "\n".join(out).strip()


def deduplicate_pages(pages: list[PageRecord]) -> tuple[list[PageRecord], int]:
    """Remove exact normalized page duplicates while retaining duplicate lineage."""
    seen: dict[str, int] = {}
    unique: list[PageRecord] = []
    duplicates = 0
    for page in pages:
        fingerprint = sha256_text(re.sub(r"\s+", " ", page.text).strip().lower())
        page.page_sha256 = fingerprint
        if fingerprint and fingerprint in seen:
            page.duplicate_of = seen[fingerprint]
            duplicates += 1
            continue
        seen[fingerprint] = page.page_number
        unique.append(page)
    return unique, duplicates


def _heading(line: str) -> str | None:
    """Recognize explicit structural headings without treating running headers as chapters."""
    candidate = re.sub(r"\s+", " ", line).strip()
    match = _CHAPTER_RE.match(candidate)
    if not match:
        return None
    suffix = (match.group(3) or "").strip(" .:-–—")
    base = f"{match.group(1).title()} {match.group(2).upper()}"
    return f"{base}: {suffix}" if suffix else base


def assign_chapters(pages: list[PageRecord]) -> list[PageRecord]:
    """Propagate the latest explicit chapter heading across subsequent pages."""
    current = "Front Matter"
    for page in pages:
        for line in page.text.splitlines()[:20]:
            found = _heading(line)
            if found:
                current = found
                break
        page.chapter = current
        page.document_label = classify_page(page.text, current)
    return pages


def _hard_wrap(text: str, target_chars: int) -> list[str]:
    """Split text that contains a sentence longer than the target size."""
    if len(text) <= target_chars:
        return [text]

    pieces: list[str] = []
    current = ""
    for word in text.split():
        if len(word) > target_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(
                word[index : index + target_chars]
                for index in range(0, len(word), target_chars)
            )
            continue
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > target_chars:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _split_large_paragraph(paragraph: str, target_chars: int) -> list[str]:
    """Prefer sentence boundaries and fall back to word boundaries when necessary."""
    if len(paragraph) <= target_chars:
        return [paragraph]

    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_RE.split(paragraph):
        for fragment in _hard_wrap(sentence, target_chars):
            candidate = f"{current} {fragment}".strip()
            if current and len(candidate) > target_chars:
                pieces.append(current)
                current = fragment
            else:
                current = candidate
    if current:
        pieces.append(current)
    return pieces


def _packed_units(
    units: list[tuple[str, int, str]],
    target_chars: int,
    overlap_chars: int,
) -> list[list[tuple[str, int, str]]]:
    """Pack semantic units into bounded groups and merge tiny trailing fragments."""
    groups: list[list[tuple[str, int, str]]] = []
    current: list[tuple[str, int, str]] = []
    current_len = 0

    for unit in units:
        text = unit[0]
        if current and current_len + len(text) + 2 > target_chars:
            groups.append(current)
            overlap: list[tuple[str, int, str]] = []
            overlap_len = 0
            for prior in reversed(current):
                if overlap_len + len(prior[0]) > overlap_chars:
                    break
                overlap.insert(0, prior)
                overlap_len += len(prior[0])
            current = overlap
            current_len = sum(len(item[0]) + 2 for item in current)
        current.append(unit)
        current_len += len(text) + 2

    if current:
        groups.append(current)

    if len(groups) > 1:
        tail_length = len("\n\n".join(unit[0] for unit in groups[-1]))
        if tail_length < _MIN_CHUNK_CHARS:
            tail = groups.pop()
            merged = list(groups[-1])
            for unit in tail:
                if unit not in merged:
                    merged.append(unit)
            groups[-1] = merged
    return groups


def semantic_chunks(
    pages: list[PageRecord],
    target_chars: int = 1200,
    overlap_chars: int = 180,
) -> list[ChunkRecord]:
    """Build chapter-bounded, paragraph-aware chunks with bounded overlap."""
    if target_chars < 200:
        raise ValueError("target_chars must be at least 200")
    if overlap_chars < 0 or overlap_chars >= target_chars:
        raise ValueError("overlap_chars must be >= 0 and smaller than target_chars")

    by_chapter: dict[str, list[PageRecord]] = defaultdict(list)
    for page in pages:
        by_chapter[page.chapter].append(page)

    chunks: list[ChunkRecord] = []
    sequence = 0
    for chapter, chapter_pages in by_chapter.items():
        units: list[tuple[str, int, str]] = []
        for page in chapter_pages:
            paragraphs = [
                paragraph.strip()
                for paragraph in re.split(r"\n\s*\n", page.text)
                if len(paragraph.strip()) >= _MIN_PARAGRAPH_CHARS
            ]
            for paragraph in paragraphs:
                for piece in _split_large_paragraph(paragraph, target_chars):
                    units.append((piece, page.page_number, page.extraction_method))

        for group in _packed_units(units, target_chars, overlap_chars):
            text_length = len("\n\n".join(unit[0] for unit in group))
            if text_length < _MIN_CHUNK_CHARS:
                continue
            sequence += 1
            chunks.append(_make_chunk(sequence, chapter, group, chapter_pages[0]))
    return chunks


def _make_chunk(
    sequence: int,
    chapter: str,
    units: list[tuple[str, int, str]],
    source_page: PageRecord,
) -> ChunkRecord:
    text = "\n\n".join(unit[0] for unit in units)
    page_numbers = [unit[1] for unit in units]
    digest = sha256_text(text)
    return ChunkRecord(
        chunk_id=f"chunk-{sequence:05d}-{digest[:10]}",
        text=text,
        chapter=chapter,
        page_start=min(page_numbers),
        page_end=max(page_numbers),
        source_path=source_page.source_path,
        source_sha256=source_page.source_sha256,
        extraction_methods=sorted({unit[2] for unit in units}),
        chunk_sha256=digest,
        pipeline_version=PIPELINE_VERSION,
        semantic_label=classify_chunk(text),
    )
