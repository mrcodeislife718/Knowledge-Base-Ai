# Knowledge-Base AI

An employer-facing proof project that turns a public-domain scanned book into a searchable, validated vector knowledge base.

**Pipeline:** PDF/images → native extraction/OCR → cleanup → metadata → quality scoring → deduplication → chapter detection → labeling/classification → semantic chunking → knowledge tree → embeddings → Chroma → retrieval → validation.

The system is local-first and auditable. Page and chunk artifacts preserve source hashes, page ranges, chapters, extraction methods, quality signals, semantic labels, pipeline version, and content checksums.

## One-command proof

After installation, run:

```bash
kbai demo
```

The command performs the full demonstration automatically:

1. downloads the public-domain 1895 *Alice's Adventures in Wonderland* scan,
2. extracts usable PDF text and automatically sends low-quality/sparse pages through Tesseract OCR,
3. cleans and fingerprints text,
4. removes exact duplicate pages while recording duplicate lineage,
5. detects chapter boundaries,
6. labels pages and semantic chunks,
7. builds a document → chapter → chunk knowledge tree,
8. computes local SentenceTransformer embeddings,
9. upserts chunks and provenance into persistent Chroma,
10. runs deterministic quality gates,
11. executes a real semantic retrieval query,
12. prints `DEMO PASSED` only when validation succeeds.

Use `kbai demo --force-ocr` to force Tesseract across the entire scan. This is intentionally slower and proves the pure scan/OCR path.

## Why this project exists

It was built specifically to demonstrate software engineering for knowledge-base creation and customized LLM data workflows: digitization, OCR, metadata extraction, dataset curation, labeling, semantic classification, book segmentation, vector ingestion, embedding verification, retrieval, provenance, structured inventories, logging, and error handling.

## Public-domain source

The reproducible demo uses the Library of Congress 1895 edition of **Alice's Adventures in Wonderland** by Lewis Carroll, mirrored as a direct PDF by Internet Archive.

```text
Library of Congress record:
https://www.loc.gov/item/02020394/

Demo PDF mirror:
https://archive.org/download/alicesadventures00carr_17/alicesadventures00carr_17.pdf
```

The source collection is marked public domain by the Library of Congress.

## Install

Python 3.11+ is required. Tesseract is required whenever OCR is needed or `--force-ocr` is used.

Ubuntu / Debian / Linux Mint:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng
```

Create the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev]"
```

Confirm the CLI:

```bash
kbai --help
```

## CLI

Inspect the fixed public-domain demo source:

```bash
kbai demo-source
```

Run the complete proof:

```bash
kbai demo
```

Ingest any PDF:

```bash
kbai ingest ./data/book.pdf --title "Book Title" --author "Author"
```

Force OCR:

```bash
kbai ingest ./data/book.pdf --force-ocr
```

A directory of `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, or `.bmp` page images is also accepted:

```bash
kbai ingest ./data/scanned-pages/
```

Retrieve with provenance:

```bash
kbai query "Why does Alice follow the White Rabbit?" --top-k 5
```

Validate the latest run:

```bash
kbai validate
```

Inspect the latest run manifest:

```bash
kbai manifest
```

## Quality gates

`kbai validate` fails with a non-zero exit code when the proof does not meet its checks. Validation currently covers:

- completed run manifest
- extracted page presence
- unique-page accounting
- non-empty semantic chunks
- unique chunk IDs
- reasonable chunk size bounds
- complete source/chapter/page/checksum provenance
- chapter coverage
- semantic labels
- duplicate accounting consistency
- OCR/readability coverage
- inventory creation
- knowledge-tree creation
- Chroma/vector count parity
- finite, normalized, non-collapsed embeddings
- live retrieval smoke test

## Generated artifacts

```text
.kbai/
├── chroma/                 # persistent vector database
├── manifests/              # run configuration, counts, source hashes, errors
├── pages/                  # cleaned page-level JSONL + OCR quality/provenance
├── chunks/                 # semantic chunk JSONL
├── inventory/              # catalog + page QA inventory
├── knowledge_trees/        # document → chapter → semantic chunk hierarchy
├── validation/             # machine-readable QA reports
└── kbai.log                # structured JSONL operational log
```

The isolated one-command demo uses `.kbai-demo/` instead.

## Architecture

```text
Scanned PDF / page images
          │
          ▼
Document loader
          │
          ├── native text + quality score
          └── automatic / forced Tesseract OCR
          │
          ▼
Cleanup + normalization
          │
          ▼
SHA-256 fingerprints + dedupe lineage
          │
          ▼
Chapter detection + page classification
          │
          ▼
Paragraph/sentence-aware semantic chunking
          │
          ├── structured inventory
          └── document/chapter/chunk knowledge tree
          │
          ▼
SentenceTransformer embeddings
          │
          ▼
Persistent Chroma collection
          │
          ├── semantic retrieval + provenance
          └── deterministic validation
```

## Engineering choices

**OCR and scan QA:** native PDF text is scored first. Sparse or low-quality text triggers Tesseract OCR automatically. The OCR candidate is scored too; if OCR makes a text-bearing page worse, the native extraction is retained and the decision is recorded. `--force-ocr` bypasses that optimization.

**Cleanup:** soft hyphenation and line-break artifacts are normalized without destroying paragraph boundaries used downstream.

**Deduplication:** normalized page text is SHA-256 fingerprinted. Duplicate pages are excluded from chunking but retain `duplicate_of` lineage in page artifacts.

**Segmentation:** conservative chapter-heading recognition establishes chapter boundaries. Paragraphs are packed toward a target size, oversized paragraphs are sentence-split, overlap is bounded, and chunks do not silently cross chapter boundaries.

**Classification:** pages receive labels such as front matter, chapter opening, and chapter body. Chunks receive semantic labels such as narrative and dialogue-heavy narrative. The classifiers are deliberately explainable for this proof and can be replaced by an LLM or learned classifier later.

**Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` is the default local embedding model. The exact model name is stored with vectors and in the run manifest.

**Chroma:** embeddings are computed explicitly and upserted with text plus page, chapter, label, checksum, extraction, model, and source provenance.

**Observability:** all major stages emit structured JSON logs. Failed ingestion writes a failed manifest before propagating the exception.

## Employer demo sequence

The shortest demonstration is simply:

```bash
kbai demo
```

For a walkthrough where each responsibility is shown separately:

```bash
kbai ingest data/book.pdf --title "Alice's Adventures in Wonderland" --author "Lewis Carroll"
kbai validate
kbai query "What happens after Alice sees the White Rabbit?" --top-k 5
kbai manifest
```

## Tests

```bash
pytest -q
ruff check .
```

GitHub Actions also compiles the source and runs the unit test suite on pushes and pull requests.

## Scope

This is deliberately a small proof rather than a production SaaS. Natural production extensions include layout-aware OCR, deskew/image preprocessing, fuzzy and semantic near-duplicate detection, configurable OCR language packs and symbol dictionaries, human-review queues, distributed batch workers, object storage, evaluation datasets, reranking, remote vector stores, and API serving.
