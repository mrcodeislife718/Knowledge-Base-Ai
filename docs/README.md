# Knowledge-Base AI Documentation

This directory contains the engineering documentation for the Knowledge-Base AI proof project.

## Start here

- [Architecture](ARCHITECTURE.md) — component boundaries, data flow, and deployment decisions.
- [Operations](OPERATIONS.md) — install, run, troubleshoot, and reproduce the demo.
- [Quality & Validation](QUALITY_AND_VALIDATION.md) — OCR quality policy, dedupe, provenance, embedding checks, and release gates.
- [Employer Walkthrough](EMPLOYER_WALKTHROUGH.md) — the fastest way to evaluate the project for the target knowledge-base engineering role.
- [Screenshot Manifest](SCREENSHOT_MANIFEST.md) — exact image filenames and README placement for the static visual demo.

## Project objective

Knowledge-Base AI demonstrates a complete auditable document-intelligence workflow:

`PDF/images → OCR-aware extraction → cleanup → metadata → quality scoring → deduplication → chapter/semantic structure → embeddings → Chroma → retrieval → validation`

The project is intentionally designed as production-quality proof code: it is small enough to evaluate quickly, but it preserves the engineering practices required for a real knowledge-base ingestion system—provenance, deterministic artifacts, isolation between runs, validation gates, structured logs, error handling, and reproducibility.
