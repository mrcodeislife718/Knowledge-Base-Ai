# Operations Runbook

This runbook covers local operation, validation, artifacts, and common failure recovery for Knowledge-Base AI.

## Prerequisites

- Python 3.11+
- Tesseract OCR with English language data
- Internet access on the first embedding-model download

Ubuntu / Debian / Linux Mint:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv tesseract-ocr tesseract-ocr-eng
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev]"
```

## Pre-demo health check

```bash
pytest -q
ruff check .
kbai --help
```

Expected: tests pass, Ruff prints `All checks passed!`, and the CLI command list renders.

## Browser demo

```bash
kbai-web
```

Open `http://127.0.0.1:8000`.

Use **Demo Mode** for the fixed public-domain source or upload a PDF/page image through **Ingest**.

## CLI demo

```bash
kbai demo
```

A successful run ends with `DEMO PASSED`. The first execution may take longer because the embedding model is downloaded and cached.

## Run lifecycle

1. Establish source identity and SHA-256.
2. Extract pages and score native/OCR candidates.
3. Clean text and retain page-level quality evidence.
4. Fingerprint exact duplicates and preserve duplicate lineage.
5. Assign chapter structure and labels.
6. Quarantine low-quality pages from retrieval while preserving them in audit artifacts.
7. Produce semantic chunks.
8. Write inventory and knowledge-tree artifacts.
9. Compute embeddings and insert them into a run-isolated Chroma collection.
10. Mark the manifest completed.
11. Execute deterministic validation and retrieval checks.

## Output directories

- `.kbai/` — normal CLI runs
- `.kbai-demo/` — CLI public-domain demo
- `.kbai-ui/` — browser/operator-console runs

Each run preserves page records, chunks, a manifest, inventory, knowledge tree, Chroma data, validation reports, and structured logs.

## Troubleshooting

### `python3 -m venv` fails with `ensurepip is not available`

```bash
sudo apt-get install -y python3-venv
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
```

### OCR fails

```bash
tesseract --version
tesseract --list-langs
```

The language list should include `eng`.

### First semantic search is slow

The SentenceTransformer may be loading from disk for the first query. This is expected in the local proof. Subsequent requests benefit from model and OS caches.

### Hugging Face rate-limit warning

The public embedding model can be downloaded without authentication. A Hugging Face token is optional and only improves download limits/speed.

### Empty UI panels before ingestion

Knowledge, analytics, pages, and validation are run-scoped. Before a run exists, their APIs may return `404` to represent `no run yet`; this is not a server failure.

### Failed ingestion

Inspect:

```text
<workdir>/kbai.log
<workdir>/manifests/<run-id>.json
```

A failed run writes its manifest before the exception is propagated.

### Reset generated data

Stop the web server first, then remove only the relevant generated directory:

```bash
rm -rf .kbai-ui
# or
rm -rf .kbai-demo
```

Source code and the virtual environment are unaffected.

## Handoff checklist

Before presenting or handing off the repository:

```bash
git pull origin main
source .venv/bin/activate
pip install -e . --no-deps
pytest -q
ruff check .
```

For a clean-machine proof, use the repository's `full-demo` GitHub Actions workflow rather than relying on local state.
