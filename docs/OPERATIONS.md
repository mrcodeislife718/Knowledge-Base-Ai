# Operations

## Requirements

- Python 3.11+
- Tesseract OCR with English language data
- Internet access for the first embedding-model download and the one-command public-domain demo

Ubuntu / Debian / Linux Mint:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv tesseract-ocr tesseract-ocr-eng
```

## Local install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -e ".[dev]"
```

Verify:

```bash
pytest -q
ruff check .
kbai --help
```

## Browser operator console

```bash
kbai-web
```

Open `http://127.0.0.1:8000`.

The web surface uses `.kbai-ui/` so UI runs do not overwrite CLI/demo artifacts.

## One-command reproducible demo

```bash
kbai demo
```

The command downloads the fixed public-domain Alice scan, runs the complete ingestion pipeline, validates it, and executes retrieval. Success ends with `DEMO PASSED`.

## CLI operations

Ingest a PDF:

```bash
kbai ingest ./data/book.pdf --title "Book Title" --author "Author"
```

Force OCR:

```bash
kbai ingest ./data/book.pdf --force-ocr
```

Ingest scanned page images:

```bash
kbai ingest ./data/scanned-pages/
```

Search:

```bash
kbai query "Why does Alice follow the White Rabbit?" --top-k 5
```

Validate:

```bash
kbai validate
```

Inspect the latest manifest:

```bash
kbai manifest
```

## Generated artifacts

```text
.kbai/
├── chroma/
├── manifests/
├── pages/
├── chunks/
├── inventory/
├── knowledge_trees/
├── validation/
└── kbai.log
```

`.kbai-demo/` is used by the automated CLI demo and `.kbai-ui/` is used by the browser interface.

## Troubleshooting

### `python3 -m venv` fails with `ensurepip is not available`

Install the matching venv package, for example:

```bash
sudo apt install python3.12-venv
```

Then recreate `.venv`.

### OCR fails

Confirm:

```bash
tesseract --version
```

and install `tesseract-ocr-eng` if English language data is missing.

### First search/demo is slow

The first run may download the local SentenceTransformer model. Later runs reuse the local Hugging Face cache.

### UI endpoints return 404 before ingestion

This is expected for corpus-dependent routes such as knowledge, analytics, pages, or validation when no run exists yet. Run an ingestion or Demo Mode first.

### Stop the server

Use `Ctrl+C` in the terminal running `kbai-web`.

## Clean handoff procedure

Before sharing the repository:

```bash
git pull origin main
pip install -e . --no-deps
pytest -q
ruff check .
```

Then launch `kbai-web` or use the GitHub Actions `full-demo` workflow for clean-machine proof.
