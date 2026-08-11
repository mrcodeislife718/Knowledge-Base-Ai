# Knowledge-Base AI — Employer Demo Guide

This guide is the fastest way to evaluate the project as an engineering proof.

## What this proves

Knowledge-Base AI turns a public-domain scanned book into a searchable, validated vector knowledge base:

`PDF/images → OCR → cleanup → metadata → quality scoring → deduplication → chapter detection → classification → semantic chunking → knowledge tree → embeddings → Chroma → retrieval → validation`

The proof is designed around the same responsibilities expected in document digitization, dataset curation, LLM knowledge-base preparation, and vector-store operations.

## Fastest demo

After installation:

```bash
kbai demo
```

A successful run ends with:

```text
score=100%
DEMO PASSED
```

The command downloads the public-domain 1895 *Alice's Adventures in Wonderland* scan, executes the full OCR-aware ingestion pipeline, validates the resulting dataset and embeddings, then performs semantic retrieval.

## Clean-machine proof

GitHub Actions contains a `full-demo` workflow that:

1. checks out the repository on Ubuntu,
2. installs Python and Tesseract OCR,
3. installs Knowledge-Base AI from the repository,
4. downloads the real public-domain scan,
5. runs `kbai demo`,
6. fails if any mandatory quality gate fails,
7. uploads the manifest, inventory, knowledge tree, validation report, and logs.

This is the recommended way to confirm reproducibility without relying on the developer's machine.

## What to inspect after the run

Generated data is isolated under `.kbai-demo/`:

```text
.kbai-demo/
├── chroma/                 persisted vector store
├── manifests/              run identity, configuration, hashes, counts, errors
├── pages/                  page text, OCR method, quality score, duplicate lineage
├── chunks/                 semantic chunks with page/chapter/source provenance
├── inventory/              document and page QA inventory
├── knowledge_trees/        document → chapter → chunk hierarchy
├── validation/             deterministic validation report
└── kbai.log                structured operational events
```

## Suggested interview walkthrough

### 1. Start with the CLI

```bash
kbai --help
kbai demo
```

Explain that the CLI is the operational surface for a repeatable ingestion workflow rather than a notebook-only prototype.

### 2. Show OCR and scan quality handling

The loader attempts native extraction first, scores readability, and invokes Tesseract when page text is sparse or low quality. The OCR result is also scored, and the pipeline keeps the better extraction. `--force-ocr` proves the pure scanned-document path.

### 3. Show curation and provenance

Open a page record and a chunk record. Point out:

- source SHA-256
- page checksum
- chunk checksum
- extraction method
- page range
- chapter
- semantic label
- pipeline version

Duplicate pages are removed from downstream chunking while retaining `duplicate_of` lineage.

### 4. Show structured segmentation

The pipeline detects conservative chapter boundaries, assigns page labels, creates semantic chunks without silently crossing chapter boundaries, and writes a document → chapter → chunk knowledge tree.

### 5. Show vector ingestion

SentenceTransformer embeddings are generated explicitly. The embedding model name is persisted as provenance. Each ingestion uses a run-isolated Chroma collection to prevent stale vectors from contaminating later runs.

### 6. Show quality validation

```bash
kbai validate
```

The validation suite checks ingestion completion, page coverage, dedupe accounting, chunk bounds, provenance, semantic labels, OCR readability, generated inventories, Chroma count parity, embedding normalization/distinctness, and live retrieval.

### 7. Show retrieval with provenance

```bash
kbai query "What happens after Alice sees the White Rabbit?" --top-k 5
```

Explain that retrieval results are not anonymous text blobs: they retain source, page, chapter, checksum, extraction, and model provenance.

## Engineering decisions worth discussing

**Why local Chroma for the proof?** It makes the employer demo self-contained. The code treats this as a deployment boundary rather than pretending a local client is the final production architecture.

**Why explicit embeddings?** The exact model is controlled by the application and recorded in provenance instead of being hidden behind implicit vector-store behavior.

**Why deterministic heuristics for the proof?** OCR quality scoring, chapter detection, and labels remain explainable and testable. They form stable interfaces that can later be replaced by learned or LLM-based classifiers.

**Why artifacts plus logs?** A knowledge-base pipeline should be inspectable after failure. Run manifests, intermediate records, validation reports, and structured logs make troubleshooting reproducible.

## Production extension path

The proof is intentionally small. A production deployment would naturally add layout-aware OCR, image preprocessing/deskew, configurable language and symbol dictionaries, near-duplicate detection, human-review queues, object storage, distributed workers, evaluation datasets, reranking, API serving, authentication, and a server-backed vector database.

## One-sentence project explanation

> Knowledge-Base AI is an auditable document-ingestion pipeline that converts scanned books into curated, provenance-preserving semantic data and validated vector retrieval infrastructure for downstream LLM systems.
