from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from urllib.parse import urlparse


_INJECTION_PATTERNS = [
    re.compile(r"ignore (all|any|the) previous instructions", re.I),
    re.compile(r"system prompt", re.I),
    re.compile(r"developer message", re.I),
    re.compile(r"do not follow .* instructions", re.I),
    re.compile(r"reveal .* prompt", re.I),
    re.compile(r"execute .* tool", re.I),
]
_SPAM_PATTERNS = [
    re.compile(r"\b(best|top)\s+\d+\b", re.I),
    re.compile(r"click here", re.I),
    re.compile(r"limited time", re.I),
    re.compile(r"guaranteed", re.I),
]


@dataclass
class IntegrityAssessment:
    accepted: bool
    risk_score: float
    flags: list[str] = field(default_factory=list)
    content_fingerprint: str = ""
    source_family: str = ""


class IngestionIntegrityGate:
    """Deterministic pre-knowledge safety gate for hostile or low-integrity text."""

    def assess(self, text: str, source_path: str, known_fingerprints: set[str] | None = None) -> IntegrityAssessment:
        normalized = " ".join(text.split())
        fingerprint = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        flags: list[str] = []
        risk = 0.0
        if any(pattern.search(normalized) for pattern in _INJECTION_PATTERNS):
            flags.append("prompt-injection-pattern")
            risk += 0.55
        spam_hits = sum(bool(pattern.search(normalized)) for pattern in _SPAM_PATTERNS)
        if spam_hits >= 2:
            flags.append("seo-or-promotional-pattern")
            risk += min(0.3, spam_hits * 0.08)
        if known_fingerprints and fingerprint in known_fingerprints:
            flags.append("exact-content-duplicate")
            risk += 0.18
        if len(normalized) > 0 and len(set(normalized.lower().split())) / max(1, len(normalized.split())) < 0.12:
            flags.append("extreme-token-repetition")
            risk += 0.22
        family = source_family(source_path)
        return IntegrityAssessment(
            accepted=risk < 0.65,
            risk_score=min(1.0, risk),
            flags=flags,
            content_fingerprint=fingerprint,
            source_family=family,
        )


def source_family(source_path: str) -> str:
    """Map syndication-like URLs/files to an evidentiary family identifier.

    This prevents multiple copies from automatically counting as independent sources.
    A future citation parser can refine ancestry while preserving this interface.
    """
    parsed = urlparse(source_path)
    if parsed.scheme and parsed.netloc:
        host = parsed.netloc.lower().removeprefix("www.")
        return host
    stem = source_path.rsplit("/", 1)[-1].lower()
    stem = re.sub(r"[-_]?copy\d*", "", stem)
    stem = re.sub(r"[-_]?repost\d*", "", stem)
    return stem


def independent_source_count(source_paths: list[str], ancestry: dict[str, str] | None = None) -> int:
    ancestry = ancestry or {}
    roots: set[str] = set()
    for source in source_paths:
        current = source
        seen: set[str] = set()
        while current in ancestry and current not in seen:
            seen.add(current)
            current = ancestry[current]
        roots.add(source_family(current))
    return len(roots)
