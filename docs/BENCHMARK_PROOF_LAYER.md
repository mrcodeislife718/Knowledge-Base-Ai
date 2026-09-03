# Level 5 / Level 6 Proof Layer

This branch adds a comparative benchmark harness whose purpose is evidence, not marketing.

## What it measures

The harness reports contradiction resolution, multi-hop evidence coverage, stale-knowledge handling, source-independence robustness, question resolution rate, measured latency, estimated cost per question, and calibration through Brier score.

## Comparators

The in-repository comparators are deterministic architectural references:

- `conventional-rag-reference`: flat lexical retrieval proxy for ordinary top-k RAG.
- `graph-rag-style-reference`: seed retrieval plus graph-neighbor expansion.
- `raptor-style-reference`: topic-level hierarchical selection followed by local retrieval.
- `reflection-rag-reference`: flat retrieval with a persisted reflection signal.
- `knowledge-base-ai-level5-6`: the repository's actual `AdaptiveRetriever`, using typed claims, graph relations, temporal scoring, contradiction retrieval, evidence budgets, confidence, and independent source origins.

These reference adapters are **not** official executions of Microsoft GraphRAG, RAPTOR, LightRAG, or any third-party package. Results from them must never be cited as official competitor benchmark results. They exist to make architectural differences reproducible under one controlled corpus and metric contract.

The next external-validation step is to add adapters that execute pinned official competitor versions against the exact same benchmark cases, hardware envelope, model choices, token budgets, and cost accounting rules.

## Reference cases

The built-in suite includes five controlled cases:

1. contradictory controlled-trial claims where both support and counterevidence must be surfaced;
2. a three-hop dependency chain where all required evidence must be recovered;
3. an old and current policy statement where the 2026 fact must rank first;
4. syndicated duplicates plus an independent replication where evidentiary origin, not URL count, matters;
5. a reflection case where derived synthesis can assist retrieval without becoming raw evidence.

## Metrics

`resolution_rate` requires all evidence expected by a case plus its special condition. Contradiction cases require both sides. Temporal cases require the current claim to rank first. Source-independence cases require the configured number of independent origins.

`contradiction_resolution` measures whether expected support and counterevidence are both surfaced.

`multihop_accuracy` is evidence recall over the required multi-hop chain.

`stale_knowledge_accuracy` measures whether the expected current claim ranks first.

`source_independence_robustness` counts unique evidentiary origins rather than documents or domains.

`mean_latency_ms` is wall-clock adapter execution time for the benchmark process.

`mean_cost_per_question` is an explicit deterministic accounting estimate for reference adapters. Official package adapters must replace this with observed token/API/compute cost.

`mean_brier_score` evaluates confidence calibration against whether the case was actually resolved. Lower is better.

## Reproducibility

Run:

```bash
PYTHONPATH=src python scripts/run-benchmarks.py
```

This writes machine-readable `benchmark-results/benchmark-results.json` and human-readable `benchmark-results/benchmark-results.md`.

## Claim gate

No superiority claim is allowed merely because Knowledge-Base-AI wins the built-in reference suite. A technical-superiority claim requires, at minimum:

- pinned official competitor implementations or published-reproduction adapters;
- identical corpus and query sets;
- identical model and embedding constraints where applicable;
- repeated runs with variance reported;
- cost and latency normalized to the same hardware/API environment;
- held-out and adversarial cases not tuned against the implementation;
- statistical significance or an explicitly justified practical-effect threshold;
- failure-case disclosure.

Until those conditions are met, benchmark output is architectural evidence and regression proof, not a claim of state-of-the-art superiority.
