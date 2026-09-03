from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Protocol

from .models import ClaimRecord, EvidenceBudget, RelationRecord, RelationType
from .retrieval_engine import AdaptiveRetriever

_WORD_RE = re.compile(r"[a-z0-9']+")
_STOP = {"the", "and", "for", "that", "with", "from", "this", "into", "are", "was", "were", "is", "of", "to", "a", "an"}


def _terms(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) > 2 and w not in _STOP}


def _overlap(query: str, text: str) -> float:
    q = _terms(query)
    if not q:
        return 0.0
    return len(q & _terms(text)) / len(q)


@dataclass
class BenchmarkDocument:
    doc_id: str
    text: str
    topic: str
    source_id: str
    independent_origin: str
    confidence: float = 0.7
    freshness: float = 1.0
    valid_from: str | None = None
    valid_until: str | None = None
    links: list[str] = field(default_factory=list)
    contradicts: list[str] = field(default_factory=list)
    reflection: str | None = None


@dataclass
class BenchmarkCase:
    case_id: str
    category: str
    query: str
    documents: list[BenchmarkDocument]
    expected_ids: list[str]
    expected_current_id: str | None = None
    expected_contradiction_pair: tuple[str, str] | None = None
    min_independent_origins: int = 1
    max_results: int = 6


@dataclass
class AdapterOutput:
    retrieved_ids: list[str]
    confidence: float
    latency_ms: float
    estimated_cost: float
    independent_origins: int


class BenchmarkAdapter(Protocol):
    name: str

    def run(self, case: BenchmarkCase) -> AdapterOutput: ...


@dataclass
class CaseScore:
    adapter: str
    case_id: str
    category: str
    recall: float
    resolved: bool
    contradiction_resolution: float
    multihop_accuracy: float
    stale_knowledge_accuracy: float
    source_independence_robustness: float
    latency_ms: float
    estimated_cost: float
    confidence: float
    brier_score: float


@dataclass
class AggregateScore:
    adapter: str
    cases: int
    resolution_rate: float
    contradiction_resolution: float
    multihop_accuracy: float
    stale_knowledge_accuracy: float
    source_independence_robustness: float
    mean_latency_ms: float
    mean_cost_per_question: float
    mean_brier_score: float


class ConventionalRAGAdapter:
    name = "conventional-rag-reference"

    def run(self, case: BenchmarkCase) -> AdapterOutput:
        started = time.perf_counter()
        ranked = sorted(case.documents, key=lambda d: _overlap(case.query, d.text), reverse=True)
        chosen = ranked[: case.max_results]
        confidence = mean([_overlap(case.query, d.text) for d in chosen]) if chosen else 0.0
        return _output(chosen, confidence, started, base_cost=0.0005)


class GraphStyleAdapter:
    name = "graph-rag-style-reference"

    def run(self, case: BenchmarkCase) -> AdapterOutput:
        started = time.perf_counter()
        by_id = {d.doc_id: d for d in case.documents}
        seeds = sorted(case.documents, key=lambda d: _overlap(case.query, d.text), reverse=True)[:2]
        scores = {d.doc_id: _overlap(case.query, d.text) for d in case.documents}
        for seed in seeds:
            for linked in seed.links + seed.contradicts:
                if linked in by_id:
                    scores[linked] = scores.get(linked, 0.0) + 0.35
        chosen = [by_id[doc_id] for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[: case.max_results]]
        confidence = min(1.0, mean([scores[d.doc_id] for d in chosen]) if chosen else 0.0)
        return _output(chosen, confidence, started, base_cost=0.0015)


class HierarchicalStyleAdapter:
    name = "raptor-style-reference"

    def run(self, case: BenchmarkCase) -> AdapterOutput:
        started = time.perf_counter()
        topics: dict[str, list[BenchmarkDocument]] = {}
        for doc in case.documents:
            topics.setdefault(doc.topic, []).append(doc)
        topic_scores = {
            topic: _overlap(case.query, " ".join(d.text for d in docs))
            for topic, docs in topics.items()
        }
        selected_topics = [topic for topic, _ in sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)[:2]]
        candidates = [d for topic in selected_topics for d in topics[topic]]
        chosen = sorted(candidates, key=lambda d: _overlap(case.query, d.text), reverse=True)[: case.max_results]
        confidence = mean([topic_scores.get(d.topic, 0.0) for d in chosen]) if chosen else 0.0
        return _output(chosen, confidence, started, base_cost=0.0020)


class ReflectionStyleAdapter:
    name = "reflection-rag-reference"

    def run(self, case: BenchmarkCase) -> AdapterOutput:
        started = time.perf_counter()
        scored = []
        for doc in case.documents:
            score = _overlap(case.query, doc.text)
            if doc.reflection:
                score += 0.30 * _overlap(case.query, doc.reflection)
            scored.append((score, doc))
        chosen = [doc for _, doc in sorted(scored, key=lambda x: x[0], reverse=True)[: case.max_results]]
        confidence = min(1.0, mean([score for score, _ in sorted(scored, key=lambda x: x[0], reverse=True)[: case.max_results]]) if chosen else 0.0)
        return _output(chosen, confidence, started, base_cost=0.0010)


class KnowledgeBaseAIAdapter:
    name = "knowledge-base-ai-level5-6"

    def run(self, case: BenchmarkCase) -> AdapterOutput:
        claims = [
            ClaimRecord(
                claim_id=d.doc_id,
                statement=d.text,
                evidence_ids=[f"ev:{d.doc_id}"],
                source_ids=[d.independent_origin],
                confidence=d.confidence,
                freshness=d.freshness,
                valid_from=d.valid_from,
                valid_until=d.valid_until,
                contradictions=list(d.contradicts),
            )
            for d in case.documents
        ]
        relations: list[RelationRecord] = []
        relation_counter = 0
        for doc in case.documents:
            for target in doc.links:
                relation_counter += 1
                relations.append(RelationRecord(
                    relation_id=f"bench-rel-{relation_counter}",
                    source_id=doc.doc_id,
                    target_id=target,
                    relation_type=RelationType.DEPENDS_ON,
                    evidence_ids=[f"ev:{doc.doc_id}"],
                    provenance_kind="system_inference",
                    confidence=0.8,
                ))
            for target in doc.contradicts:
                relation_counter += 1
                relations.append(RelationRecord(
                    relation_id=f"bench-rel-{relation_counter}",
                    source_id=doc.doc_id,
                    target_id=target,
                    relation_type=RelationType.CONTRADICTS,
                    evidence_ids=[f"ev:{doc.doc_id}"],
                    provenance_kind="system_inference",
                    confidence=0.9,
                ))
        retriever = AdaptiveRetriever(claims, relations)
        result = retriever.retrieve(
            case.query,
            EvidenceBudget(max_results=case.max_results, source_diversity_target=case.min_independent_origins),
        )
        selected_ids = [item.claim.claim_id for item in result.items]
        selected_docs = [next(d for d in case.documents if d.doc_id == doc_id) for doc_id in selected_ids]
        return AdapterOutput(
            retrieved_ids=selected_ids,
            confidence=result.confidence,
            latency_ms=result.elapsed_ms,
            estimated_cost=0.0008 + 0.00015 * len(selected_ids),
            independent_origins=len({d.independent_origin for d in selected_docs}),
        )


def _output(chosen: list[BenchmarkDocument], confidence: float, started: float, base_cost: float) -> AdapterOutput:
    return AdapterOutput(
        retrieved_ids=[d.doc_id for d in chosen],
        confidence=max(0.0, min(1.0, confidence)),
        latency_ms=(time.perf_counter() - started) * 1000,
        estimated_cost=base_cost + 0.0001 * len(chosen),
        independent_origins=len({d.independent_origin for d in chosen}),
    )


def score_case(adapter: BenchmarkAdapter, case: BenchmarkCase) -> CaseScore:
    output = adapter.run(case)
    expected = set(case.expected_ids)
    retrieved = set(output.retrieved_ids)
    recall = len(expected & retrieved) / len(expected) if expected else 1.0
    resolved = recall == 1.0

    contradiction = 1.0
    if case.expected_contradiction_pair:
        contradiction = float(set(case.expected_contradiction_pair).issubset(retrieved))
        resolved = resolved and bool(contradiction)

    multihop = recall if case.category == "multi-hop" else 1.0

    stale = 1.0
    if case.category == "stale-knowledge" and case.expected_current_id:
        stale = float(bool(output.retrieved_ids) and output.retrieved_ids[0] == case.expected_current_id)
        resolved = resolved and bool(stale)

    independence = 1.0
    if case.category == "source-independence":
        independence = min(1.0, output.independent_origins / max(1, case.min_independent_origins))
        resolved = resolved and independence >= 1.0

    target = 1.0 if resolved else 0.0
    brier = (output.confidence - target) ** 2
    return CaseScore(
        adapter=adapter.name,
        case_id=case.case_id,
        category=case.category,
        recall=recall,
        resolved=resolved,
        contradiction_resolution=contradiction if case.category == "contradiction" else 1.0,
        multihop_accuracy=multihop,
        stale_knowledge_accuracy=stale,
        source_independence_robustness=independence,
        latency_ms=output.latency_ms,
        estimated_cost=output.estimated_cost,
        confidence=output.confidence,
        brier_score=brier,
    )


def aggregate(scores: list[CaseScore]) -> list[AggregateScore]:
    groups: dict[str, list[CaseScore]] = {}
    for score in scores:
        groups.setdefault(score.adapter, []).append(score)
    out: list[AggregateScore] = []
    for adapter, rows in sorted(groups.items()):
        out.append(AggregateScore(
            adapter=adapter,
            cases=len(rows),
            resolution_rate=mean(float(r.resolved) for r in rows),
            contradiction_resolution=mean(r.contradiction_resolution for r in rows if r.category == "contradiction") if any(r.category == "contradiction" for r in rows) else 1.0,
            multihop_accuracy=mean(r.multihop_accuracy for r in rows if r.category == "multi-hop") if any(r.category == "multi-hop" for r in rows) else 1.0,
            stale_knowledge_accuracy=mean(r.stale_knowledge_accuracy for r in rows if r.category == "stale-knowledge") if any(r.category == "stale-knowledge" for r in rows) else 1.0,
            source_independence_robustness=mean(r.source_independence_robustness for r in rows if r.category == "source-independence") if any(r.category == "source-independence" for r in rows) else 1.0,
            mean_latency_ms=mean(r.latency_ms for r in rows),
            mean_cost_per_question=mean(r.estimated_cost for r in rows),
            mean_brier_score=mean(r.brier_score for r in rows),
        ))
    return out


def run_suite(cases: list[BenchmarkCase] | None = None, adapters: list[BenchmarkAdapter] | None = None) -> dict:
    cases = cases or reference_suite()
    adapters = adapters or [
        ConventionalRAGAdapter(),
        GraphStyleAdapter(),
        HierarchicalStyleAdapter(),
        ReflectionStyleAdapter(),
        KnowledgeBaseAIAdapter(),
    ]
    scores = [score_case(adapter, case) for adapter in adapters for case in cases]
    return {
        "contract": "reference-style baselines are deterministic architectural proxies, not official GraphRAG/RAPTOR package runs",
        "case_scores": [asdict(s) for s in scores],
        "aggregate": [asdict(a) for a in aggregate(scores)],
    }


def write_report(output_dir: Path, result: dict) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "benchmark-results.json"
    md_path = output_dir / "benchmark-results.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# Level 5 / Level 6 Benchmark Results",
        "",
        "> Reference-style baselines are architectural proxies. They must not be presented as official GraphRAG or RAPTOR benchmark results.",
        "",
        "| Adapter | Resolution | Contradiction | Multi-hop | Stale truth | Source independence | Mean latency ms | Cost/question | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["aggregate"]:
        lines.append(
            f"| {row['adapter']} | {row['resolution_rate']:.3f} | {row['contradiction_resolution']:.3f} | {row['multihop_accuracy']:.3f} | {row['stale_knowledge_accuracy']:.3f} | {row['source_independence_robustness']:.3f} | {row['mean_latency_ms']:.3f} | {row['mean_cost_per_question']:.6f} | {row['mean_brier_score']:.3f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def reference_suite() -> list[BenchmarkCase]:
    contradiction_docs = [
        BenchmarkDocument("c-support", "Controlled trials report Intervention A improves Outcome B.", "intervention-a", "source-1", "trial-1", confidence=0.82, contradicts=["c-counter"]),
        BenchmarkDocument("c-counter", "A later controlled trial reports Intervention A does not improve Outcome B.", "intervention-a", "source-2", "trial-2", confidence=0.86, contradicts=["c-support"]),
        BenchmarkDocument("c-noise", "Outcome B is measured using a standardized scale.", "measurement", "source-3", "methods-1"),
    ]
    multihop_docs = [
        BenchmarkDocument("m1", "Component Alpha enables Service Beta.", "chain", "s1", "o1", links=["m2"]),
        BenchmarkDocument("m2", "Service Beta requires Gateway Gamma.", "chain", "s2", "o2", links=["m3"]),
        BenchmarkDocument("m3", "Gateway Gamma controls access to Dataset Delta.", "chain", "s3", "o3"),
        BenchmarkDocument("m-noise", "Component Alpha has a blue status indicator.", "hardware", "s4", "o4"),
    ]
    stale_docs = [
        BenchmarkDocument("t-old", "In 2024 Policy X allowed two requests per minute.", "policy-x", "s1", "policy-2024", confidence=0.9, freshness=0.2, valid_from="2024-01-01", valid_until="2025-12-31"),
        BenchmarkDocument("t-new", "In 2026 Policy X allows ten requests per minute.", "policy-x", "s2", "policy-2026", confidence=0.92, freshness=1.0, valid_from="2026-01-01"),
    ]
    independence_docs = [
        BenchmarkDocument("i1", "Independent Study Z found Method Q reduces latency.", "method-q", "site-a", "study-z", confidence=0.82),
        BenchmarkDocument("i2", "Syndicated report: Study Z found Method Q reduces latency.", "method-q", "site-b", "study-z", confidence=0.82),
        BenchmarkDocument("i3", "A rewrite of Study Z says Method Q reduces latency.", "method-q", "site-c", "study-z", confidence=0.82),
        BenchmarkDocument("i4", "Independent replication R found Method Q reduces latency.", "method-q", "site-d", "replication-r", confidence=0.88),
    ]
    reflection_docs = [
        BenchmarkDocument("r1", "Paper one describes a memory scheduler.", "memory", "s1", "p1", reflection="The scheduler reduces memory pressure by moving cold work out of the hot set."),
        BenchmarkDocument("r2", "Paper two describes hot-set prediction.", "memory", "s2", "p2", reflection="Together the papers suggest hot-set prediction can guide memory movement."),
    ]
    return [
        BenchmarkCase("contradiction-1", "contradiction", "Does Intervention A improve Outcome B and what evidence disagrees?", contradiction_docs, ["c-support", "c-counter"], expected_contradiction_pair=("c-support", "c-counter"), max_results=3),
        BenchmarkCase("multihop-1", "multi-hop", "How does Component Alpha ultimately affect access to Dataset Delta?", multihop_docs, ["m1", "m2", "m3"], max_results=4),
        BenchmarkCase("stale-1", "stale-knowledge", "What does Policy X allow in 2026?", stale_docs, ["t-new"], expected_current_id="t-new", max_results=2),
        BenchmarkCase("independence-1", "source-independence", "What independent evidence says Method Q reduces latency?", independence_docs, ["i1", "i4"], min_independent_origins=2, max_results=4),
        BenchmarkCase("reflection-1", "reflection", "How can hot-set prediction help memory movement?", reflection_docs, ["r1", "r2"], max_results=2),
    ]
