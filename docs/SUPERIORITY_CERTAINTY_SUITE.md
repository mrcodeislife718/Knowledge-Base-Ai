# Superiority Certainty Suite

The goal is not to manufacture a win. The goal is to make a technical-superiority claim impossible unless the evidence is strong enough to survive adversarial review.

## What this adds beyond Stage B

Stage B pinned the real competitor repositories, required native execution, and added repeated held-out trials. This layer closes the remaining ways a result could still be misleading.

### Frozen preregistration

Before execution, the benchmark freezes required baselines, metrics, minimum sample size, minimum trials, alpha, superiority/non-inferiority margins, domains, categories and claim policy. Changing the rules after seeing results creates a new manifest and invalidates the earlier claim path.

### Resource/model parity

Every system is fingerprinted against the same model, embedding model, reranker policy, context/output budgets, concurrency, cache/retry policy, hardware class, provider and temperature. A mismatch blocks the claim.

### Contamination and leakage audit

The held-out suite is checked for duplicate queries, exact leakage into tuning material and excessive lexical overlap. Contaminated runs cannot issue a superiority certificate.

### Statistics that resist cherry-picking

Pairwise evidence is evaluated with paired bootstrap confidence intervals, Holm-Bonferroni correction across metrics, a power check, leave-one-category-out robustness, leave-one-trial-out robustness and per-seed win stability. A result that depends on one seed, one category or one favorable metric fails.

### Negative controls

The proof system includes shuffled-label and impossible-case controls. If the benchmark can "succeed" on evidence-free cases or appears suspiciously predictive after labels are randomized, the proof is considered broken.

### Ablation evidence

Each claimed architectural mechanism can be disabled and measured. This distinguishes real causal contribution from features that merely coexist with good results.

### Scale and success-too-well testing

Index size, indexing time/cost, query latency, peak memory and success rate can be tracked as corpus size increases. A system that wins only at toy scale is not allowed to generalize that claim to larger workloads.

### Blind human adjudication

Answer correctness, evidence faithfulness, completeness and harmful hallucination rate can be scored by judges who see anonymous system labels. Inter-judge agreement is recorded. This catches cases where automatic retrieval metrics look good while the actual answer is wrong or misleading.

### Independent replication

Public superiority claims require at least one independent reproduction matching the frozen manifest fingerprint and artifact hash. The replication cannot silently change the benchmark.

### Reproducibility bundle

Every proof run can freeze environment metadata, code/config/data file hashes, command line and the manifest fingerprint into one content-addressed bundle.

## Claim semantics

Passing the suite permits only a scoped claim such as:

> Under the preregistered, pinned, resource-matched benchmark represented by manifest <fingerprint>, Knowledge-Base-AI demonstrated statistically robust technical superiority over the named baselines on the measured held-out workloads.

The suite intentionally forbids the stronger statement that Knowledge-Base-AI is universally superior. No finite benchmark can establish that.

## Required execution path

1. Freeze the preregistration and resource envelope.
2. Bootstrap and verify every pinned competitor checkout.
3. Build all indexes from scratch and record indexing cost/time/size.
4. Run at least the preregistered number of held-out cases and trials.
5. Run negative controls, contamination checks, ablations and scale points.
6. Run blind answer/evidence adjudication.
7. Generate the claim certificate.
8. Reproduce the frozen run independently before using the result publicly.

`python scripts/audit-superiority-readiness.py` verifies that the required proof machinery and pinned baselines are present before expensive native execution begins.
