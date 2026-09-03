from __future__ import annotations

import time
from pathlib import Path

from knowledge_base_ai.knowledge_engine import (
    KnowledgeCompiler,
    KnowledgeRepository,
    calibrate_confidence,
    evidence_budget_reached,
    plan_query,
)
from knowledge_base_ai.models import ClaimRecord, EpistemicStatus, EvidenceBudget


def _chunk(chunk_id: str, text: str, source: str = "source-a") -> dict:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "chapter": "Test",
        "page_start": 1,
        "page_end": 1,
        "source_path": f"{source}.txt",
        "source_sha256": source,
        "extraction_methods": ["text"],
        "chunk_sha256": chunk_id,
        "pipeline_version": "test",
        "semantic_label": "narrative",
    }


def test_compile_produces_typed_evidence_claims_entities_and_relations() -> None:
    compiler = KnowledgeCompiler(reflection_threshold=0.0)
    compiled = compiler.compile_chunks(
        [
            _chunk("one", "GraphRAG uses Microsoft Knowledge Graph components. GraphRAG improves structured retrieval across documents."),
            _chunk("two", "GraphRAG does not improve structured retrieval across documents.", "source-b"),
        ]
    )
    assert len(compiled["evidence"]) == 2
    assert len(compiled["claims"]) >= 3
    assert compiled["entities"]
    assert compiled["relations"]
    assert compiled["reflections"]


def test_query_planner_selects_temporal_and_graph_strategies() -> None:
    plan = plan_query("Why did the evidence change after 2025 and what caused it?")
    assert plan.requires_temporal
    assert plan.requires_multi_hop
    assert "temporal" in plan.strategies
    assert "claim_graph" in plan.strategies


def test_confidence_penalizes_contradictions() -> None:
    clean = calibrate_confidence(
        source_reliability=0.9,
        independent_sources=3,
        support_count=3,
        contradiction_count=0,
        recency=1.0,
        directness=0.9,
        replication=0.9,
        retrieval_coverage=0.9,
        agreement=0.9,
        reasoning_depth=1,
    )
    contested = calibrate_confidence(
        source_reliability=0.9,
        independent_sources=3,
        support_count=3,
        contradiction_count=2,
        recency=1.0,
        directness=0.9,
        replication=0.9,
        retrieval_coverage=0.9,
        agreement=0.5,
        reasoning_depth=1,
    )
    assert clean.calibrated_score > contested.calibrated_score


def test_belief_revision_invalidates_dependents(tmp_path: Path) -> None:
    a = ClaimRecord("a", "A is true.", ["e1"], ["s1"], confidence=0.8)
    b = ClaimRecord("b", "B depends on A.", ["e2"], ["s2"], depends_on=["a"], confidence=0.8)
    c = ClaimRecord("c", "C depends on B.", ["e3"], ["s3"], depends_on=["b"], confidence=0.8)
    invalidated = KnowledgeRepository(tmp_path).revise_belief([a, b, c], "a", "new counterevidence")
    assert invalidated == ["a", "b", "c"]
    assert all(claim.epistemic_status == EpistemicStatus.INVALIDATED for claim in (a, b, c))


def test_evidence_budget_stops_on_target_or_cost() -> None:
    budget = EvidenceBudget(confidence_target=0.8, source_diversity_target=2)
    assert evidence_budget_reached(
        time.perf_counter(),
        tokens_used=100,
        compute_used=0.1,
        independent_sources=2,
        confidence=0.85,
        budget=budget,
        marginal_gain=0.2,
        marginal_cost=0.1,
    )
