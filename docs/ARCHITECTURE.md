# Architecture

## System goal

Knowledge-Base AI converts scanned or text-bearing documents into curated, provenance-preserving semantic data and a validated vector knowledge base suitable for downstream AI/LLM systems.

## Data flow

```text
PDF / page images
      |
      v
Document loader
      |
      +-- native text extraction
      +-- OCR fallback / forced OCR
      |
      v
Cleanup + readability scoring
      |
      v
Page fingerprints + duplicate lineage
      |
      v
Chapter detection + page classification
      |
      v
Semantic chunking
      |
      +-- page/chapter/source provenance
      +-- semantic labels
      |
      v
Inventory + knowledge tree
      |
      v
SentenceTransformer embeddings
      |
      v
Run-isolated Chroma collection
      |
      +-- CLI retrieval
      +-- FastAPI/browser retrieval
      +-- validation
```

## Module boundaries

- `document_io.py`: source hashing, metadata extraction, native PDF extraction, OCR fallback, page-level quality decisions.
- `quality.py`: deterministic readability scoring and explainable structural/semantic labels.
- `text_ops.py`: cleanup, deduplication, chapter assignment, paragraph/sentence-aware chunking.
- `catalog.py`: structured inventory and document → chapter → chunk knowledge tree artifacts.
- `store.py`: explicit embedding generation, bounded Chroma writes, retrieval.
- `pipeline.py`: orchestration, run isolation, manifests, logging, artifact persistence.
- `validation.py`: release gates across data quality, provenance, embeddings, vector counts, and retrieval.
- `cli.py`: repeatable operator CLI.
- `web.py`: thin FastAPI layer over the same pipeline and validation logic.
- `static/`: local employer-facing operator console.

## Run isolation

Every ingestion receives a unique run ID and a run-specific Chroma collection. This prevents stale vectors from previous documents or failed runs from contaminating retrieval or validation.

## Provenance model

The pipeline records source SHA-256, page checksum, chunk checksum, source path, extraction method, page range, chapter, semantic label, embedding model, and pipeline version. Duplicate page lineage is preserved through `duplicate_of` even when a duplicate is excluded from downstream chunking.

## Retrieval design

Dense SentenceTransformer similarity is the base retriever. The current release adds a lightweight lexical overlap rerank to improve precision for concrete names and terms while keeping the system fully local and explainable.

## Quality-aware corpus policy

Every page remains in the page artifact inventory for auditability. Pages that remain below the readability threshold after extraction/OCR handling are retained for inspection but excluded from the semantic retrieval corpus so obvious extraction noise does not pollute embeddings.

## Production boundary

The proof uses local persistent Chroma because that makes employer evaluation reproducible and self-contained. A production deployment would normally place vector storage behind a service, move ingestion to worker processes, use object storage for source/intermediate artifacts, and add authentication, queueing, retry policies, monitoring, and human-review workflows.
