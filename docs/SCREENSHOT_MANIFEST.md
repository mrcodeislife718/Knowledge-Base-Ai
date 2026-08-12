# Screenshot Manifest

Add screenshots under `docs/images/` using these exact filenames so the README gallery renders without further edits.

| Order | Filename | Surface | What it proves |
|---|---|---|---|
| 01 | `01-overview.png` | Overview | Complete operator console, run metrics, pipeline stages, recent runs |
| 02 | `02-ingest.png` | Ingest | Upload workflow, metadata overrides, force-OCR control, runtime status |
| 03 | `03-knowledge-base.png` | Knowledge Base | Structured chunk inventory and document → chapter → chunk map |
| 04 | `04-search.png` | Search | Semantic retrieval with ranked evidence and provenance |
| 05 | `05-search-provenance.png` | Search detail | Source hash, chunk hash, extraction method, pages, model provenance |
| 06 | `06-analytics.png` | Analytics | OCR readability, low-quality counts, dedupe rate, semantic/chapter distribution |
| 07 | `07-inspector.png` | Inspector | Raw vs cleaned text, page-level quality, extraction method, evidence exports |
| 08 | `08-telemetry-running.png` | Telemetry | Live processing state while the demo is running |
| 09 | `09-telemetry-complete.png` | Telemetry | Completed pipeline events and operational evidence |
| 10 | `10-validation.png` | Validation | Deterministic release gates and 100% validation score |
| 11 | `11-overview-complete.png` | Overview | Completed 354-page demo with populated metrics |
| 12 | `12-knowledge-base-populated.png` | Knowledge Base | Real Alice corpus populated after ingestion |

## Recommended image selection

If you do not want every screenshot in the README, prioritize 01, 03, 05, 07, 09, and 10. Those six tell the strongest end-to-end story.

## GitHub upload path

Create this folder in the repository:

```text
docs/images/
```

Then upload each screenshot with the exact filename above.

## README gallery markdown

The root README is prepared to reference these paths:

```text
docs/images/01-overview.png
docs/images/02-ingest.png
docs/images/03-knowledge-base.png
docs/images/04-search.png
docs/images/05-search-provenance.png
docs/images/06-analytics.png
docs/images/07-inspector.png
docs/images/08-telemetry-running.png
docs/images/09-telemetry-complete.png
docs/images/10-validation.png
docs/images/11-overview-complete.png
docs/images/12-knowledge-base-populated.png
```

Keep images at their original readable resolution; GitHub will scale them automatically in the rendered README.
