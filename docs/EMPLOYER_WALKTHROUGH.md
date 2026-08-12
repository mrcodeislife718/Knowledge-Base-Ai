# Employer Walkthrough

## What this project proves

Knowledge-Base AI was built as a role-specific proof for knowledge-base engineering, document digitization, dataset curation, and vector-store operations.

The complete path is:

`PDF/images → OCR-aware extraction → cleanup → metadata → quality scoring → deduplication → chapter detection → semantic classification → chunking → knowledge tree → embeddings → Chroma → retrieval → validation`

## Fastest evaluation path

### Option A — no setup

Review the README screenshot gallery. It shows the operator console, ingestion workflow, knowledge inventory, retrieval, analytics, OCR inspector, telemetry, and validation surfaces.

### Option B — visual local demo

```bash
kbai-web
```

Open `http://127.0.0.1:8000`, then use Demo Mode.

### Option C — automated proof

```bash
kbai demo
```

A successful run ends with `DEMO PASSED` only after validation succeeds.

### Option D — clean-machine proof

Run the GitHub Actions `full-demo` workflow. It installs the runtime and Tesseract on a fresh Ubuntu runner, executes the real public-domain book ingestion, and uploads the machine-readable artifacts.

## Recommended interview story

1. **Start with the problem:** large archival/document collections need repeatable digitization, curation, segmentation, metadata, and vector ingestion before they are trustworthy inputs for customized LLM systems.
2. **Show the pipeline:** explain that OCR is quality-gated rather than blindly applied, low-quality pages remain auditable, duplicates retain lineage, and semantic units preserve page/chapter/source provenance.
3. **Show the operator console:** demonstrate that the work is not hidden inside a notebook. An operator can ingest, inspect, retrieve, compare, validate, and export evidence.
4. **Show retrieval provenance:** every result exposes the source hash, chunk hash, extraction method, pages, chapter, semantic label, and embedding model.
5. **Show validation:** emphasize that vector count parity, embedding quality, provenance completeness, dedupe accounting, readability, catalog artifacts, and live retrieval are tested before a run is considered successful.
6. **Close with the production boundary:** local Chroma keeps the proof self-contained; the interfaces are intentionally structured so object storage, distributed workers, human-review queues, server-backed vector storage, and stronger OCR/reranking can be substituted without rewriting the whole pipeline.

## Mapping to the target role

**Semi-automated OCR, metadata extraction, cleaning, labeling, classification** → `document_io.py`, `quality.py`, `text_ops.py`, pipeline artifacts, and Inspector.

**Dataset curation, duplicate removal, quality verification** → deterministic quality scoring, SHA-256 dedupe lineage, low-quality exclusion policy, validation report.

**Books segmented into chapters/pages/logical units** → conservative chapter recognition plus page/chapter-bounded semantic chunks.

**Semantic units and knowledge trees** → semantic chunk JSONL plus document → chapter → chunk hierarchy.

**Vector databases / embedding quality / ingestion troubleshooting** → explicit SentenceTransformer embeddings, run-isolated Chroma collections, validation gates, structured telemetry.

**Structured inventories and documentation** → inventory artifacts, manifests, knowledge trees, `docs/`, README, Demo Guide.

**Scanning/OCR quality, alignment/readability/version consistency** → OCR quality scores, extraction-method lineage, source hashes, page hashes, pipeline versioning, operator inspection.

## One-sentence explanation

Knowledge-Base AI is an auditable document-intelligence pipeline that turns scanned source material into curated, provenance-preserving semantic data and validated vector retrieval infrastructure for downstream AI systems.
