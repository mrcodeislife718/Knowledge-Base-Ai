# Knowledge-Base AI

A small, employer-facing proof project that turns a public-domain book into a searchable, validated vector knowledge base.

**Pipeline:** PDF/images → text extraction/OCR → cleanup → metadata extraction → deduplication → chapter detection → semantic chunking → embeddings → Chroma → retrieval → quality validation.

The project is intentionally local-first and auditable. Every chunk carries source, page, chapter, checksum, extraction method, and pipeline-version provenance.

## Why this exists

This repository demonstrates the core work involved in building high-quality datasets and knowledge bases for LLM systems: document digitization, text normalization, semantic segmentation, vector ingestion, retrieval, QA, logging, failure isolation, and reproducibility.

## Demo source

The suggested proof source is the Library of Congress 1895 edition of **Alice's Adventures in Wonderland** by Lewis Carroll. The Library of Congress marks the digitized book collection as public domain and provides PDF/images.

Source record: `https://www.loc.gov/item/02020394/`

You can download the PDF from that record and run the pipeline against it. The CLI also includes a `demo-source` command that resolves the Library of Congress JSON record and prints candidate PDF URLs when available.

## Install

System requirement for OCR fallback:

```bash
sudo apt-get install tesseract-ocr
```

Create an environment and install:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev]"
```

## CLI

```bash
kbai --help
```

### 1. Inspect the public-domain demo source

```bash
kbai demo-source
```

### 2. Ingest a PDF

```bash
kbai ingest ./data/alice.pdf --title "Alice's Adventures in Wonderland" --author "Lewis Carroll"
```

For an image-only scan, OCR is invoked automatically when extracted page text is too sparse. Force OCR for every PDF page with:

```bash
kbai ingest ./data/alice.pdf --force-ocr
```

A directory of `.png`, `.jpg`, `.jpeg`, `.tif`, or `.tiff` page images is also accepted:

```bash
kbai ingest ./data/alice-pages/
```

### 3. Query

```bash
kbai query "Why does Alice follow the rabbit?" --top-k 5
```

Each result includes score/distance, page, chapter, chunk id, extraction method, source hash, and text.

### 4. Validate

```bash
kbai validate
```

Validation checks include:

- non-empty pages and chunks
- OCR/extraction coverage
- duplicate-rate accounting
- chunk length bounds
- unique chunk IDs
- provenance completeness
- vector-store count parity
- chapter coverage
- retrieval smoke tests

### 5. View a run manifest

```bash
kbai manifest
```

## Output layout

```text
.kbai/
├── chroma/                 # persisted Chroma DB
├── manifests/              # reproducible run manifests
├── pages/                  # extracted/cleaned page JSONL
├── chunks/                 # semantic chunks JSONL
├── validation/             # validation reports
└── kbai.log                # structured JSONL logs
```

## Architecture

```text
Input PDF / image directory
        │
        ▼
Document loader
        │
        ├── native PDF text when usable
        └── Tesseract OCR fallback / forced OCR
        │
        ▼
Normalization + cleanup
        │
        ▼
Page fingerprints + deduplication
        │
        ▼
Chapter detection
        │
        ▼
Paragraph-aware semantic chunking
        │
        ▼
SentenceTransformer embeddings
        │
        ▼
Persistent Chroma collection
        │
        ├── retrieval CLI
        └── validation report
```

## Design choices

**OCR:** PyMuPDF extracts native text first and can invoke Tesseract OCR for raster pages. This avoids paying the OCR cost when a PDF already contains a usable text layer.

**Deduplication:** normalized page text is SHA-256 fingerprinted. Exact duplicates are removed while the duplicate relationship remains visible in the run manifest.

**Chapter detection:** chapter headings are detected with conservative heading patterns such as `CHAPTER I`, `CHAPTER 1`, and standalone title-like headings. Page provenance is never discarded when chapter boundaries are inferred.

**Semantic chunking:** paragraphs are packed toward a target character size with overlap. Large paragraphs are sentence-split. Chunks never silently cross chapter boundaries.

**Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` is the default local model and can be replaced with `--model`.

**Vector store:** Chroma uses a local `PersistentClient`; embeddings are computed explicitly so the exact embedding model is part of provenance.

**Observability:** each stage emits structured JSON logging and a manifest records source hash, configuration, counts, failures, timestamps, and model name.

## Example employer demo

```bash
kbai ingest data/alice.pdf --title "Alice's Adventures in Wonderland" --author "Lewis Carroll"
kbai validate
kbai query "What changes in Alice's size?" --top-k 5
kbai manifest
```

That four-command sequence demonstrates ingestion, OCR fallback, dataset curation, vectorization, retrieval, validation, and traceability.

## Test

```bash
pytest -q
ruff check .
```

## Scope

This is deliberately a proof project rather than a production SaaS. Production extensions would include layout-aware OCR, fuzzy/near-duplicate detection, batch workers, object storage, evaluation datasets, reranking, API serving, authentication, and remote vector infrastructure.
