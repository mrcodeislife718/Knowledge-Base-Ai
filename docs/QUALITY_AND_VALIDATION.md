# Quality & Validation

## Why validation is first-class

Knowledge-base ingestion is only useful when downstream systems can trust the text, structure, provenance, and vectors being produced. Knowledge-Base AI therefore treats validation as a release gate rather than an optional report.

## OCR/readability policy

Each PDF page is first extracted natively. The cleaned text receives a deterministic readability score and flags for sparse text, replacement characters, non-printable characters, likely OCR noise, and other signals.

If native extraction is sparse or below the quality threshold, the pipeline attempts Tesseract OCR. The OCR candidate is scored as well. When OCR is worse than usable native text, the native candidate is retained and the decision is recorded.

Pages that remain low quality are preserved in the page-level inventory for audit and operator review. They are excluded from semantic chunking in the quality-aware retrieval corpus so obvious extraction noise does not pollute embeddings.

## Deduplication

Normalized page text is SHA-256 fingerprinted. Exact duplicates are removed from downstream semantic chunking while their page records retain `duplicate_of` lineage. Validation checks that:

`page_count = unique_page_count + duplicate_page_count`

## Structural validation

The pipeline verifies that:

- pages exist,
- semantic chunks exist,
- chunk IDs are unique,
- chunk sizes stay inside configured safety bounds,
- chapter coverage exists,
- semantic labels are present,
- structured inventory artifacts were written,
- the knowledge tree was written.

## Provenance validation

Each chunk must preserve the fields required to reconstruct its origin and processing lineage, including:

- chunk ID,
- chapter,
- semantic label,
- page range,
- source path,
- source SHA-256,
- chunk SHA-256,
- pipeline version,
- extraction method(s).

## Embedding validation

Knowledge-Base AI verifies a sample of generated embeddings for:

- finite numeric values,
- expected normalized vector magnitude,
- non-collapsed/distinct representations.

The exact embedding model name is stored in the run manifest and vector metadata.

## Vector-store validation

The validator checks Chroma record counts against the current run's semantic chunk count. Every ingestion uses a unique collection name to prevent stale vectors from older runs from satisfying the check accidentally.

## Retrieval validation

Validation includes a live retrieval smoke test against the freshly populated Chroma collection. This proves the system can not only generate embeddings but also retrieve stored evidence.

The browser/CLI retrieval layer additionally uses a lightweight hybrid rerank: dense similarity supplies semantic candidates, then lexical overlap gives a modest precision boost to exact names and concrete terms.

## Release behavior

`kbai validate` returns a non-zero exit status when required checks fail. `kbai demo` prints `DEMO PASSED` only after the complete ingestion and validation path succeeds.

Validation reports are persisted under `validation/` as machine-readable JSON and can also be downloaded from the browser Inspector surface.
