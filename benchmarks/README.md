# Benchmark Suite

Run the controlled reference suite with:

```bash
PYTHONPATH=src python scripts/run-benchmarks.py
```

The suite intentionally separates three evidence classes:

- **reference architectural proxies** for fast deterministic regression checks;
- **Knowledge-Base-AI actual retrieval behavior** through `AdaptiveRetriever`;
- **future official external adapters** for pinned GraphRAG, RAPTOR, LightRAG, and conventional RAG implementations.

Do not collapse these classes in reports. Proxy results are not official competitor results.

A release may claim comparative superiority only after the official external-adapter stage is populated, repeated, normalized, and statistically evaluated under the claim gate in `docs/BENCHMARK_PROOF_LAYER.md`.
