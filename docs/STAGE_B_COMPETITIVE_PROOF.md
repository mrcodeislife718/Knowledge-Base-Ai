# Stage B Competitive Proof

Stage B exists to prevent architectural confidence from becoming an unsupported superiority claim.

## Pinned competitor sources

The repository records exact upstream commits in `configs/competitive_systems.json` for Microsoft GraphRAG, HKUDS LightRAG, RAPTOR, and LlamaIndex. `scripts/bootstrap-competitors.py` clones those repositories and verifies `git rev-parse HEAD` against the recorded SHA before a run is accepted.

A result is invalid if the executed checkout does not match the pin.

## Normalized experiment contract

Every system receives the same `BenchmarkCase` payload: query, documents, expected evidence IDs, temporal/current-answer metadata, contradiction pair, source-independence requirement, and result budget. Competitor bridges return a JSON object containing retrieved evidence IDs, confidence, latency, estimated cost, independent-origin count, and provenance.

The scoring layer, not the competitor bridge, computes correctness. This prevents each system from grading itself.

## Held-out and adversarial suite

`deterministic_heldout_cases()` generates a repeatable suite that is not ordered to favor any retriever. The default is 12 cases in each of five categories, for 60 cases per trial:

- contradiction resolution with explicit counterevidence and spam pressure;
- four-node multi-hop chains plus distractors;
- stale/high-confidence historical claims versus current lower-confidence claims;
- syndicated duplicates versus genuinely independent replication;
- prompt-injection and SEO-style poisoned distractors versus verified measurements.

Change the seed to create another held-out suite without changing the metric contract.

## Repeated trials and statistics

Stage B randomizes case order on every trial and records per-case observations. Reports include resolution rate, bootstrap 95% confidence interval, contradiction resolution, multi-hop accuracy, stale-knowledge accuracy, source-independence robustness, mean and P95 latency, mean cost per question, and Brier calibration.

Pairwise comparisons are paired by `(case_id, trial)` and use bootstrap resampling. A single aggregate leaderboard number is not accepted as proof.

## Claim gate

`technical_superiority_gate()` intentionally sets a high bar. For every named baseline, Knowledge-Base-AI must be non-inferior on every required quality metric, decisively better on total resolution, and decisively better on at least one additional quality dimension. The gate still does not prove universal superiority; it only permits a scoped claim for the exact benchmark envelope.

Every published claim must name:

- exact competitor commits;
- corpus/suite hash and seed;
- trial count;
- model and embedding providers/versions;
- hardware;
- token/result/evidence budgets;
- latency and cost accounting method;
- confidence intervals;
- failed categories and regressions.

## Running

Bootstrap exact upstream sources:

```bash
python scripts/bootstrap-competitors.py
```

Run the native Knowledge-Base-AI held-out suite without making a competitive claim:

```bash
PYTHONPATH=src python scripts/run-competitive-proof.py --trials 5
```

Once the official competitor bridge environments are installed and configured, execute all external systems:

```bash
PYTHONPATH=src python scripts/run-competitive-proof.py --external --trials 10
```

Outputs are written under `benchmark-results/stage-b/`. `superiority-gate.json` is produced only when external competitors were actually executed.

## Important execution boundary

Pinning the upstream repositories and providing the normalized execution protocol does not itself constitute a benchmark result. GraphRAG, LightRAG, RAPTOR, and LlamaIndex require their own dependencies and, depending on configuration, model/embedding credentials. A run must execute their native implementations through the configured bridges before their names may appear in a competitive result. Missing credentials, failed indexing, timeouts, or bridge failures invalidate the affected run rather than silently falling back to a proxy.
