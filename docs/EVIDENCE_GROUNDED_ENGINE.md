# Evidence-Grounded Adaptive Knowledge Engine

Knowledge-Base AI now treats document ingestion as compilation into a typed, auditable knowledge representation rather than as chunking plus vector storage.

## Epistemic contract

The system separates knowledge by layer:

- **L0 — Raw evidence:** immutable source-derived passages.
- **L1 — Extracted claims:** atomic statements linked to L0 evidence.
- **L2 — Relationships:** support, contradiction, qualification, duplication, dependency, temporal and causal edges.
- **L3 — Derived inference:** reserved for explicit machine/system inference.
- **L4 — Reflection:** selective synthesis generated only when novelty, importance, contradiction, uncertainty and expected future value justify it.
- **L5 — Consolidated theory:** reserved for higher-order summaries that must maintain reverse provenance to claims and evidence.

A derived layer may reference L0. It may never overwrite L0 or silently become raw evidence.

## Implemented architecture

### Claim-centric Knowledge IR

`models.py` defines typed evidence, claims, entities, relations, reflections, knowledge gaps, confidence breakdowns, evidence budgets, query plans and retrieval telemetry. Every claim carries source lineage, evidence IDs, temporal validity, confidence, epistemic status and invalidation state.

### Knowledge compiler

`knowledge_engine.py` compiles accepted chunks into:

1. evidence records,
2. atomic claims,
3. entities,
4. support/contradiction/qualification/duplicate relationships,
5. typed causal relationships,
6. calibrated confidence,
7. selective reflections,
8. explicit unresolved knowledge gaps.

The compiler is deterministic and model-independent by design. An LLM extractor can later replace a stage without weakening provenance or epistemic separation.

### Contradiction-first reasoning

Contradictions are first-class graph edges. Contested claims are not flattened into one answer. Their state is marked `supported-but-contested` or `contradicted`, and retrieval can explicitly boost counterevidence.

### Temporal truth

Claims support `valid_from`, `valid_until`, `observed_at`, `published_at`, `superseded_by`, freshness and decay metadata. Query planning activates temporal retrieval for timeline and changing-truth questions.

### Continual cross-run learning

`continual.py` compares new claims to prior run claim IR and creates cross-run support, contradiction, qualification and duplicate edges. Historical artifacts stay immutable; only the new run records its relationship to prior knowledge.

### Adaptive multi-representation retrieval

`retrieval_engine.py` provides an adaptive planner and retriever across:

- lexical retrieval,
- pluggable dense/vector scoring,
- claim graph traversal,
- entity/relationship traversal,
- temporal retrieval,
- contradiction retrieval,
- hierarchical strategy hooks.

The vector scorer is injected through an interface, preventing model or vendor lock-in.

### Evidence-budgeted reasoning

Every query can carry limits for latency, tokens, compute, confidence target, source diversity and maximum results. Retrieval stops when resource limits are reached, confidence/diversity targets are satisfied, or marginal evidence gain falls below marginal cost.

### Selective reflection

Reflection priority is computed from:

`novelty × importance × contradiction × uncertainty × expected_future_value`

Low-value duplicate knowledge does not automatically create another reflection. Contested or novel knowledge receives higher priority.

### Active knowledge-gap discovery

Contradicted and unresolved causal claims generate explicit `KnowledgeGap` records with missing-evidence requirements. The repository supports deterministic gap closure once confidence and contradiction conditions are satisfied.

### Belief revision and dependency invalidation

`KnowledgeRepository.revise_belief()` invalidates a rejected claim and transitively invalidates claims that depend on it. This prevents append-only accumulation from preserving conclusions whose foundations no longer hold.

### Causal and counterfactual structure

The relation model distinguishes `causes`, `enables`, `inhibits`, `requires`, `precedes`, `predicts`, `mediates`, `moderates`, and `correlated_with`. `counterfactual_trace()` traverses only causal-like edges and always reports that graph reachability is structural evidence, not causal proof.

### Confidence calibration

Confidence is computed from explicit components rather than an ungrounded model percentage:

- source reliability,
- source independence,
- support strength,
- contradiction penalty,
- recency,
- directness,
- replication,
- retrieval coverage,
- agreement,
- reasoning-depth penalty.

The output exposes the full breakdown so empirical calibration can later compare predicted confidence with observed correctness.

### Adversarial ingestion

`integrity.py` evaluates chunks before they are allowed into derived knowledge. It detects prompt-injection patterns, extreme repetition, duplicate content and promotional/SEO patterns. Rejected chunks remain in raw run artifacts for auditability but are excluded from Knowledge IR compilation.

### Source independence

Source-family and ancestry helpers prevent copies, reposts and syndications from automatically counting as independent confirmation. This interface is intentionally separable so richer citation ancestry can replace the deterministic baseline.

### Provenance-preserving compression

The current Knowledge IR creates the required reverse lineage from reflections and claims back to evidence and passages. Higher-order L5 summaries must preserve those same reverse pointers. This is the contract for future recursive topic/domain/global compression.

### Knowledge garbage collection

`KnowledgeRepository.garbage_collect()` merges near-duplicate claim evidence and identifies invalidated claims for archival instead of allowing indefinite knowledge bloat.

### Self-evaluating retrieval

`KnowledgeRepository.record_telemetry()` records query strategies, retrieved IDs, cited IDs, unused evidence, latency, cost and later correctness/correction signals. This creates the data needed to learn retrieval policy without requiring immediate model fine-tuning.

### Knowledge compiler architecture

The ingestion path is now conceptually:

```text
SOURCE
  -> parse
  -> normalize
  -> quality gate
  -> chunk/passages
  -> adversarial integrity gate
  -> EvidenceIR (L0)
  -> ClaimIR (L1)
  -> EntityIR / RelationIR (L2)
  -> contradiction + novelty analysis
  -> confidence
  -> selective ReflectionIR (L4)
  -> KnowledgeGap IR
  -> link to prior runs
  -> persist immutable run artifacts
  -> vector index
```

## Why this is structurally different from ordinary RAG

Ordinary RAG primarily answers: **Which stored text is similar to this query?**

This architecture can additionally answer:

- What exact evidence supports this claim?
- What contradicts it?
- Which sources are truly independent?
- When was it valid?
- What changed?
- Which conclusions depend on a now-invalid claim?
- Is a relationship causal, correlational or unresolved?
- What evidence is missing?
- What retrieval strategy is appropriate for this question?
- Is additional retrieval worth its latency/token/compute cost?
- How did this conclusion evolve across ingestion runs?

That is the technical target: a system that becomes more structured, calibrated, efficient and resistant to contamination as it ingests knowledge, rather than merely larger.
