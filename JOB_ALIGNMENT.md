# Job Alignment — Knowledge-Base AI

Knowledge-Base AI was built as a focused proof for software engineering work that turns scanned and textual archives into curated, searchable knowledge bases for downstream AI systems.

## Responsibility mapping

| Role responsibility | Project evidence |
|---|---|
| OCR and digitization workflows | Native PDF extraction with automatic Tesseract fallback, forced-OCR mode, page-level extraction method, readability scoring, and raw/clean inspection |
| Text cleanup | Deterministic normalization for soft hyphens, line endings, whitespace, and paragraph preservation |
| Metadata extraction | PDF metadata capture plus explicit title/author overrides and run-level source identity |
| Labeling and semantic classification | Explainable page and chunk labels persisted in generated artifacts and vector metadata |
| Dataset curation | Quality scoring, low-quality page quarantine from retrieval, exact deduplication, duplicate lineage, and deterministic validation |
| Book segmentation | Conservative chapter detection plus paragraph/sentence-aware semantic chunks that do not silently cross chapter boundaries |
| Knowledge trees | Generated document → chapter → semantic-chunk hierarchy |
| Vector database operations | Explicit SentenceTransformer embeddings, run-isolated Chroma collections, bounded batch writes, vector-count validation |
| Embedding quality verification | Finite-value, normalization, distinctness, vector-count, and retrieval smoke checks |
| Troubleshooting and observability | Structured JSONL logs, failed manifests, stage telemetry, explicit CLI exit codes, browser-visible errors |
| Structured inventories | Machine-readable document/page inventory, page QA records, chunks, manifests, validation reports |
| Provenance | SHA-256 source/page/chunk fingerprints, page ranges, chapter, extraction method, model and pipeline version |
| Internal tooling / CLI | `kbai ingest`, `kbai query`, `kbai validate`, `kbai manifest`, `kbai demo`, and `kbai-web` |
| Reproducible engineering | Unit tests, Ruff, Docker support, CI, and a clean-machine full-demo GitHub Actions workflow |

## Extra-credit engineering

The proof also includes an employer-facing operator console with live telemetry, corpus analytics, OCR before/after inspection, knowledge-map visualization, semantic retrieval with inspectable provenance, concept comparison, artifact exports, and a one-click public-domain demonstration.

Retrieval uses dense semantic candidates followed by deterministic lexical reranking. This improves precision for concrete names and question terms without introducing another hosted model or network dependency.

Low-quality pages remain visible in page-level audit artifacts and the Inspector, but they are excluded from semantic chunking and vector retrieval. This preserves evidence while preventing known-bad text from silently degrading the knowledge base.

## Production boundary

This repository demonstrates production-oriented engineering practices in a self-contained proof. Local persistent Chroma is intentionally used for a reproducible demo; a production deployment should move vector storage behind a server-backed service and add durable object storage, queue-backed workers, authentication/authorization, human-review workflows, richer OCR/layout models, configurable OCR languages and symbol dictionaries, and formal retrieval evaluation datasets.

## Evaluation shortcut

The fastest evaluation is:

```bash
kbai demo
```

or:

```bash
kbai-web
```

The CLI demo fails if mandatory quality gates fail. The browser surface exposes the same pipeline and validation functions rather than a second demo-only implementation.
