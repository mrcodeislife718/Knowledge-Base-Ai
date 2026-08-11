from __future__ import annotations

import hashlib
from pathlib import Path

import pymupdf

from .models import PageRecord
from .text_ops import clean_text, sha256_text

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_sha256(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    digest = hashlib.sha256()
    for item in sorted(p for p in path.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES):
        digest.update(item.name.encode())
        digest.update(sha256_file(item).encode())
    return digest.hexdigest()


def document_metadata(path: Path) -> dict[str, str]:
    if path.is_file() and path.suffix.lower() == ".pdf":
        with pymupdf.open(path) as doc:
            return {str(k): str(v) for k, v in (doc.metadata or {}).items() if v}
    return {"format": "image-directory", "name": path.name}


def _extract_page(page: pymupdf.Page, force_ocr: bool) -> tuple[str, str]:
    native = page.get_text("text", sort=True).strip()
    useful_chars = sum(ch.isalnum() for ch in native)
    if not force_ocr and useful_chars >= 40:
        return native, "native-text"
    try:
        textpage = page.get_textpage_ocr(language="eng", dpi=300, full=True)
        return page.get_text("text", textpage=textpage, sort=True).strip(), "tesseract-ocr"
    except Exception as exc:
        if native:
            return native, f"native-text-ocr-fallback:{type(exc).__name__}"
        raise RuntimeError(
            "OCR failed. Ensure Tesseract OCR and English language data are installed."
        ) from exc


def extract_pages(path: Path, force_ocr: bool = False) -> list[PageRecord]:
    if not path.exists():
        raise FileNotFoundError(path)
    src_hash = source_sha256(path)
    pages: list[PageRecord] = []

    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise ValueError("Input file must be a PDF; pass a directory for page images.")
        with pymupdf.open(path) as doc:
            for index, page in enumerate(doc, start=1):
                raw, method = _extract_page(page, force_ocr)
                cleaned = clean_text(raw)
                pages.append(PageRecord(index, cleaned, raw, method, str(path), src_hash, sha256_text(cleaned)))
        return pages

    images = sorted(p for p in path.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"No supported page images found in {path}")
    for index, image in enumerate(images, start=1):
        with pymupdf.open(image) as doc:
            raw, method = _extract_page(doc[0], True)
        cleaned = clean_text(raw)
        pages.append(PageRecord(index, cleaned, raw, method, str(image), src_hash, sha256_text(cleaned)))
    return pages
