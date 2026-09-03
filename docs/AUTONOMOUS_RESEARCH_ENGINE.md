# Level 6 — Autonomous Research Engine

This branch extends the Evidence-Grounded Adaptive Knowledge Engine into a bounded autonomous research loop.

## Core contract

Level 5 answers: **Given the evidence already present, what should be believed and why?**

Level 6 adds: **What evidence is missing, where should we look for it, what would falsify the current belief, and which dependent beliefs must be revalidated when the answer changes?**

The loop is:

```text
OPEN KNOWLEDGE GAP
        ↓
RESEARCH PLANNER
        ↓
SUPPORT + FALSIFICATION QUERIES
        ↓
AUTHORIZED RESEARCH PROVIDER
        ↓
SOURCE INDEPENDENCE / QUALITY SCREEN
        ↓
EVIDENCE ASSESSMENT
        ↓
COMPETING HYPOTHESES
        ↓
HYPOTHESIS TEST
        ↓
REVISION PROPOSAL
        ↓
DEPENDENCY REVALIDATION
        ↓
UNRESOLVED GAP / STOP CONDITION
```

## Implemented mechanisms

### Active evidence acquisition

`ResearchPlanner` converts open `KnowledgeGap` records into prioritized acquisition tasks. Missing evidence requirements become explicit search targets rather than passive metadata. Contested claims automatically generate counterevidence queries so the system seeks potential falsification instead of only confirmation.

### Provider-independent acquisition

`ResearchProvider` is a protocol. A provider can wrap web search, a scholarly database, an internal corpus, an enterprise data source, or another authorized source. The epistemic engine is not coupled to one search vendor. `StaticResearchProvider` provides deterministic offline/replay behavior and testability.

Provider output is candidate evidence, never truth.

### Evidence adjudication

Every acquired source is scored for:

- relevance,
- directness,
- source reliability,
- source independence,
- recency,
- stance relative to the claim.

Low-reliability material is rejected under the research budget. Source fingerprints prevent exact evidence/origin duplication from inflating evidence mass.

### Competing hypotheses and falsification

Contradictory claim groups are converted into explicit `Hypothesis` objects. Evidence is aggregated separately as support and contradiction. The engine emits a provisional posterior and one of:

- `provisionally-supported`,
- `provisionally-falsified`,
- `unresolved`.

This is deliberately provisional: a hypothesis result is derived inference and does not overwrite L0 evidence.

### Belief revision proposals

The engine calculates a proposed confidence/status change from newly acquired evidence. It emits a `RevisionProposal` rather than mutating source-derived claims in place. This preserves the evidence/inference boundary and makes review, rollback, replay and governance possible.

### Dependency-aware revalidation

A revision proposal includes the complete transitive set of claims that depend on the target claim. If foundational claim A changes and B depends on A while C depends on B, both B and C are surfaced for revalidation.

### Continuous revalidation

`due_for_revalidation()` identifies stale, low-confidence, contradicted and contested claims. `revalidation_gaps()` turns those claims back into active research work seeking current independent replication and counterevidence.

This closes the loop:

```text
knowledge → aging/contradiction → research gap → acquisition → evidence → revision → dependent revalidation
```

### Explicit research budgets

Autonomy is bounded by:

- query count,
- source count,
- wall-clock latency,
- confidence target,
- source-diversity target,
- minimum accepted source reliability.

The engine records why research stopped: tasks exhausted, query/source/latency budget exhausted, or confidence/diversity target met.

## What this does not claim

This branch does **not** claim autonomous scientific discovery. It does not invent experiments, run laboratory procedures, infer causal mechanisms from observational text, or claim benchmark superiority over GraphRAG/RAPTOR/LightRAG.

It implements the missing control loop needed to move from an adaptive knowledge engine toward an autonomous research engine: active gap resolution, falsification seeking, hypothesis adjudication, bounded acquisition, belief revision proposals and continuous dependency-aware revalidation.

## Next evidence gate

Level 6 should be considered demonstrated only after comparative evaluation shows that the research loop improves measurable outcomes such as:

- contradiction resolution accuracy,
- missing-evidence recall,
- source independence calibration,
- stale-claim detection,
- downstream invalidation/revalidation correctness,
- research cost per resolved gap,
- time to confidence target,
- robustness to poisoned/duplicated sources.

Until then, the code establishes the architecture and testable mechanisms; it does not establish superiority.
