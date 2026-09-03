from knowledge_base_ai.autonomous_research import (
    AutonomousResearchEngine,
    ResearchBudget,
    ResearchSource,
    StaticResearchProvider,
)
from knowledge_base_ai.models import ClaimRecord, EpistemicStatus, KnowledgeGap


def _claim(claim_id: str, statement: str, confidence: float = 0.55, **kwargs):
    return ClaimRecord(
        claim_id=claim_id,
        statement=statement,
        evidence_ids=[f"ev-{claim_id}"],
        source_ids=[f"source-{claim_id}"],
        confidence=confidence,
        **kwargs,
    )


def test_research_engine_seeks_gap_evidence_and_proposes_revision():
    claim = _claim("c1", "Technique X reduces retrieval latency under condition Y", confidence=0.45)
    gap = KnowledgeGap(
        gap_id="g1",
        question="Does Technique X reduce retrieval latency under condition Y?",
        importance=0.9,
        evidence_missing=["independent benchmark"],
        related_claims=["c1"],
        status="open",
    )
    sources = [
        ResearchSource(
            source_id="s1",
            title="Independent benchmark of Technique X",
            text="Technique X reduces retrieval latency under condition Y in an independent benchmark.",
            locator="https://example.test/s1",
            publisher="lab-a",
            reliability=0.9,
            independent_origin="study-a",
        ),
        ResearchSource(
            source_id="s2",
            title="Replication benchmark Technique X",
            text="A replication found Technique X reduces retrieval latency under condition Y.",
            locator="https://example.test/s2",
            publisher="lab-b",
            reliability=0.85,
            independent_origin="study-b",
        ),
    ]
    engine = AutonomousResearchEngine()
    result = engine.run(
        [gap],
        [claim],
        StaticResearchProvider(sources),
        ResearchBudget(max_queries=4, max_sources=8, confidence_target=0.5, source_diversity_target=2),
    )

    assert result.queries_executed >= 1
    assert len(result.acquired_sources) == 2
    assert result.assessments
    assert result.revisions
    assert result.revisions[0].proposed_confidence > claim.confidence


def test_counterevidence_keeps_contested_claim_from_becoming_unquestioned_truth():
    claim = _claim(
        "c1",
        "Intervention A causes outcome B",
        confidence=0.7,
        contradictions=["c2"],
        epistemic_status=EpistemicStatus.SUPPORTED_BUT_CONTESTED,
    )
    counter = _claim("c2", "Intervention A does not cause outcome B", confidence=0.65, contradictions=["c1"])
    gap = KnowledgeGap(
        gap_id="g1",
        question="Does Intervention A cause outcome B?",
        importance=1.0,
        evidence_missing=["independent adjudicating evidence"],
        related_claims=["c1", "c2"],
        status="open",
    )
    provider = StaticResearchProvider(
        [
            ResearchSource(
                source_id="s1",
                title="Controlled study of Intervention A and outcome B",
                text="The controlled study found Intervention A does not cause outcome B.",
                locator="https://example.test/counter",
                publisher="independent-lab",
                reliability=0.95,
                independent_origin="trial-1",
            )
        ]
    )
    result = AutonomousResearchEngine().run([gap], [claim, counter], provider)
    proposal = next(p for p in result.revisions if p.claim_id == "c1")

    assert proposal.contradicting_sources == ["s1"]
    assert proposal.proposed_status in {"contradicted", "supported-but-contested", "unverified"}


def test_dependency_revalidation_propagates_downstream():
    root = _claim("a", "A is true", confidence=0.4)
    child = _claim("b", "B depends on A", confidence=0.7, depends_on=["a"])
    grandchild = _claim("c", "C depends on B", confidence=0.7, depends_on=["b"])
    gap = KnowledgeGap(
        gap_id="g",
        question="Is A true?",
        importance=0.9,
        evidence_missing=["independent evidence"],
        related_claims=["a"],
        status="open",
    )
    provider = StaticResearchProvider(
        [ResearchSource("s", "Evidence about A", "Independent evidence says A is true.", "memory://s", reliability=0.9)]
    )
    result = AutonomousResearchEngine().run([gap], [root, child, grandchild], provider)
    proposal = next(p for p in result.revisions if p.claim_id == "a")

    assert proposal.downstream_claim_ids == ["b", "c"]


def test_revalidation_generates_active_research_gaps_for_stale_or_contested_claims():
    stale = _claim("stale", "Old claim", confidence=0.9, freshness=0.2)
    contested = _claim(
        "contested",
        "Contested claim",
        confidence=0.8,
        contradictions=["other"],
        epistemic_status=EpistemicStatus.SUPPORTED_BUT_CONTESTED,
    )
    healthy = _claim("healthy", "Healthy claim", confidence=0.95, freshness=0.95)
    gaps = AutonomousResearchEngine().revalidation_gaps([stale, contested, healthy])

    ids = {claim_id for gap in gaps for claim_id in gap.related_claims}
    assert "stale" in ids
    assert "contested" in ids
    assert "healthy" not in ids


def test_budget_stops_research_loop():
    topics = ["alpha", "bravo", "charlie"]
    gaps = [
        KnowledgeGap(
            gap_id=f"g{i}",
            question=f"Question about topic {topic}?",
            importance=1.0 - i * 0.1,
            evidence_missing=["independent evidence"],
            related_claims=[f"c{i}"],
            status="open",
        )
        for i, topic in enumerate(topics)
    ]
    claims = [_claim(f"c{i}", f"Topic {topic} claim") for i, topic in enumerate(topics)]
    provider = StaticResearchProvider(
        [ResearchSource("s1", "Topic alpha evidence", "Topic alpha evidence for claims.", "memory://s1", reliability=0.9)]
    )
    result = AutonomousResearchEngine().run(
        gaps,
        claims,
        provider,
        ResearchBudget(max_queries=1, max_sources=10, source_diversity_target=5),
    )

    assert result.queries_executed == 1
    assert result.stopped_reason == "query_budget_exhausted"
