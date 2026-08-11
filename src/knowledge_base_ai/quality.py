"""Explainable OCR/readability scoring and lightweight semantic labeling.

These heuristics are deliberately deterministic: reviewers can inspect why a
page was flagged instead of relying on an opaque model for the proof project.
"""

from __future__ import annotations

import re


def score_text_quality(text: str) -> tuple[float, list[str]]:
    """Return a readability score in [0, 1] plus actionable quality flags."""
    stripped = text.strip()
    if not stripped:
        return 0.0, ["empty_text"]

    chars = len(stripped)
    alnum = sum(ch.isalnum() for ch in stripped)
    printable = sum(ch.isprintable() or ch in "\n\t" for ch in stripped)
    replacement = stripped.count("�")
    suspicious_tokens = len(
        re.findall(
            r"(?:[^\w\s]{4,}|\w*\d\w*[A-Za-z]\w*|[A-Za-z]\w*\d\w*)",
            stripped,
        )
    )
    words = re.findall(r"[A-Za-z']+", stripped)
    one_char_words = sum(
        len(word) == 1 and word.lower() not in {"a", "i"}
        for word in words
    )

    printable_ratio = printable / chars
    alnum_ratio = alnum / chars
    replacement_penalty = min(0.35, replacement / max(chars, 1) * 10)
    suspicious_penalty = min(0.25, suspicious_tokens / max(len(words), 1))
    singleton_penalty = min(0.2, one_char_words / max(len(words), 1))

    score = 0.45 * printable_ratio + 0.45 * min(1.0, alnum_ratio / 0.65) + 0.10
    score -= replacement_penalty + suspicious_penalty + singleton_penalty
    score = max(0.0, min(1.0, score))

    flags: list[str] = []
    if chars < 40:
        flags.append("sparse_text")
    if printable_ratio < 0.98:
        flags.append("nonprintable_characters")
    if replacement:
        flags.append("replacement_characters")
    if suspicious_tokens > max(3, len(words) * 0.08):
        flags.append("possible_ocr_noise")
    if score < 0.72:
        flags.append("low_quality")
    return round(score, 4), flags


def classify_page(text: str, chapter: str) -> str:
    """Assign a coarse, explainable structural label to a page."""
    sample = text[:1200].lower()
    if chapter == "Front Matter":
        if "contents" in sample or "table of contents" in sample:
            return "table-of-contents"
        return "front-matter"
    if re.search(r"\bchapter\s+(?:[ivxlcdm]+|\d+)\b", sample, re.I):
        return "chapter-opening"
    return "chapter-body"


def classify_chunk(text: str) -> str:
    """Assign a simple semantic label useful for downstream filtering."""
    sample = text.strip().lower()
    if not sample:
        return "empty"
    dialogue_marks = text.count('"') + text.count("“") + text.count("”")
    if dialogue_marks >= 6:
        return "dialogue-heavy-narrative"
    if re.search(r"\bchapter\s+(?:[ivxlcdm]+|\d+)\b", sample[:180], re.I):
        return "chapter-opening"
    if len(re.findall(r"\b(?:said|asked|replied|cried)\b", sample)) >= 4:
        return "dialogue-heavy-narrative"
    return "narrative"
