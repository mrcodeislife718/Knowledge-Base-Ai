from __future__ import annotations

import hashlib
import re
from collections import defaultdict

from . import PIPELINE_VERSION
from .models import ChunkRecord, PageRecord
from .quality import classify_chunk, classify_page

_CHAPTER_RE = re.compile(r"^(chapter|book|part)\s+([ivxlcdm]+|\d+)(?:\s*[:.\-–—]?\s*(.*))?$", re.I)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_text(text: str) -> str:
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
    candidate = re.sub(r"\s+", " ", line).strip()
    match = _CHAPTER_RE.match(candidate)
    if match:
        suffix = (match.group(3) or "").strip()
        base = f"{match.group(1).title()} {match.group(2).upper()}"
        return f"{base}: {suffix}" if suffix else base
    if 3 <= len(candidate) <= 70 and candidate.isupper() and len(candidate.split()) <= 10:
        return candidate.title()
    return None


def assign_chapters(pages: list[PageRecord]) -> list[PageRecord]:
    current = "Front Matter"
    for page in pages:
        for line in page.text.splitlines()[:15]:
            found = _heading(line)
            if found:
                current = found
                break
        page.chapter = current
        page.document_label = classify_page(page.text, current)
    return pages


def _split_large_paragraph(paragraph: str, target_chars: int) -> list[str]:
    if len(paragraph) <= target_chars:
        return [paragraph]
    sentences = _SENTENCE_RE.split(paragraph)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > target_chars:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def semantic_chunks(pages: list[PageRecord], target_chars: int = 1200, overlap_chars: int = 180) -> list[ChunkRecord]:
    by_chapter: dict[str, list[PageRecord]] = defaultdict(list)
    for page in pages:
        by_chapter[page.chapter].append(page)

    chunks: list[ChunkRecord] = []
    sequence = 0
    for chapter, chapter_pages in by_chapter.items():
        units: list[tuple[str, int, str]] = []
        for page in chapter_pages:
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", page.text) if p.strip()]
            for paragraph in paragraphs:
                for piece in _split_large_paragraph(paragraph, target_chars):
                    units.append((piece, page.page_number, page.extraction_method))

        current: list[tuple[str, int, str]] = []
        current_len = 0
        for unit in units:
            text, _, _ = unit
            if current and current_len + len(text) + 2 > target_chars:
                sequence += 1
                chunks.append(_make_chunk(sequence, chapter, current, chapter_pages[0]))
                overlap: list[tuple[str, int, str]] = []
                overlap_len = 0
                for prior in reversed(current):
                    if overlap_len + len(prior[0]) > overlap_chars:
                        break
                    overlap.insert(0, prior)
                    overlap_len += len(prior[0])
                current = overlap
                current_len = sum(len(x[0]) + 2 for x in current)
            current.append(unit)
            current_len += len(text) + 2
        if current:
            sequence += 1
            chunks.append(_make_chunk(sequence, chapter, current, chapter_pages[0]))
    return chunks


def _make_chunk(sequence: int, chapter: str, units: list[tuple[str, int, str]], source_page: PageRecord) -> ChunkRecord:
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
