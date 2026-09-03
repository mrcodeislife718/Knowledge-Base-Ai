from __future__ import annotations

import hashlib
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Protocol, Sequence

from .models import ClaimRecord, EpistemicStatus, KnowledgeGap

_WORD_RE = re.compile(r"[a-z0-9']+")
_NEGATIONS = {"not", "no", "never", "none", "without", "fails", "failed", "cannot", "can't"}
_STOPWORDS = {
    "the", "and", "for", "that", "with", "from", "this", "into", "were", "was", "are", "is",
    "has", "have", "had", "but", "you", "your", "their", "they", "them", "than", "then",
    "when", "where", "what", "which", "who", "why", "how", "can", "could", "would", "should",
    "may", "might", "will", "also", "its", "our", "out", "about", "over", "under", "between",
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _id(prefix: str, *parts: str) -> str:
    payload = "\x1f".join(parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:16]}"


def _terms(text: str) -> set[str]:
    return {t for t in _WORD_RE.findall(text.lower()) if len(t) > 2 and t not in _STOPWORDS}


def _overlap(a: str, b: str) -> float:
    left, right = _terms(a), _terms(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _negated(text: str) -> bool:
    words = _terms(text)
    return bool(words & _NEGATIONS) or "n't" in text.lower()


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass
class ResearchBudget:
    max_queries: int = 8
    max_sources: int = 24
    latency_ms: int = 10_000
    confidence_target: float = 0.82
    source_diversity_target: int = 3
    min_source_reliability: float = 0.45


@dataclass
class ResearchSource:
    source_id: str
    title: str
    text: str
    locator: str
    publisher: str = "unknown"
    published_at: str | None = None
    reliability: float = 0.5
    independent_origin: str | None = None
    evidence_kind: str = "unknown"
    metadata: dict = field(default_factory=dict)

    def origin(self) -> str:
        return self.independent_origin or self.publisher or self.locator


@dataclass
class ResearchTask:
    task_id: str
    gap_id: str
    query: str
    purpose: str
    priority: float
    target_claim_ids: list[str] = field(default_factory=list)
    required_evidence_kind: str | None = None


@dataclass
class EvidenceAssessment:
    source_id: str
    claim_id: str
    stance: str
    relevance: float
    directness: float
    reliability: float
    independence: float
    recency: float
    score: float
    rationale: str


@dataclass
class Hypothesis:
    hypothesis_id: str
    statement: str
    claim_ids: list[str]
    kind: str
    prior: float = 0.5


@dataclass
class HypothesisResult:
    hypothesis_id: str
    support_score: float
    contradiction_score: float
    independent_origins: int
    posterior: float
    status: str
    source_ids: list[str]


@dataclass
class RevisionProposal:
    claim_id: str
    prior_confidence: float
    proposed_confidence: float
    prior_status: str
    proposed_status: str
    supporting_sources: list[str]
    contradicting_sources: list[str]
    rationale: str
    downstream_claim_ids: list[str] = field(default_factory=list)


@dataclass
class ResearchRun:
    run_id: str
    started_at: str
    completed_at: str | None
    tasks: list[ResearchTask]
    acquired_sources: list[ResearchSource]
    assessments: list[EvidenceAssessment]
    hypotheses: list[Hypothesis]
    hypothesis_results: list[HypothesisResult]
    revisions: list[RevisionProposal]
    unresolved_gap_ids: list[str]
    queries_executed: int
    stopped_reason: str

    def to_dict(self) -> dict:
        return asdict(self)


class ResearchProvider(Protocol):
    """Search/acquisition boundary for Level-6 autonomous research.

    Providers can wrap a web search API, scholarly index, internal corpus, database,
    or other authorized evidence source. The research engine never treats provider
    output as truth; every result is independently assessed before it can influence
    a revision proposal.
    """

    def search(self, query: str, max_results: int = 5) -> Sequence[ResearchSource]: ...


class StaticResearchProvider:
    """Deterministic provider useful for tests, offline corpora and replay."""

    def __init__(self, sources: Sequence[ResearchSource]):
        self.sources = list(sources)

    def search(self, query: str, max_results: int = 5) -> Sequence[ResearchSource]:
        ranked = sorted(
            self.sources,
            key=lambda source: _overlap(query, f"{source.title} {source.text}"),
            reverse=True,
        )
        return [s for s in ranked if _overlap(query, f"{s.title} {s.text}") > 0][:max_results]


class ResearchPlanner:
    def plan(self, gaps: Sequence[KnowledgeGap], claims: Sequence[ClaimRecord]) -> list[ResearchTask]:
        by_id = {claim.claim_id: claim for claim in claims}
        tasks: list[ResearchTask] = []
        for gap in sorted(gaps, key=lambda g: g.importance, reverse=True):
            if gap.status != "open":
                continue
            claim_text = " ".join(by_id[c].statement for c in gap.related_claims if c in by_id)
            needs = gap.evidence_missing or ["independent evidence"]
            for index, need in enumerate(needs):
                query = f"{gap.question} {need} {claim_text}".strip()
                tasks.append(
                    ResearchTask(
                        task_id=_id("rt", gap.gap_id, str(index), query),
                        gap_id=gap.gap_id,
                        query=query,
                        purpose=f"Resolve knowledge gap with {need}",
                        priority=gap.importance,
                        target_claim_ids=[c for c in gap.related_claims if c in by_id],
                        required_evidence_kind=need,
                    )
                )
            for claim_id in gap.related_claims:
                claim = by_id.get(claim_id)
                if not claim or not claim.contradictions:
                    continue
                tasks.append(
                    ResearchTask(
                        task_id=_id("rt", gap.gap_id, claim_id, "counterevidence"),
                        gap_id=gap.gap_id,
                        query=f"independent counterevidence {claim.statement}",
                        purpose="Actively seek falsifying or adjudicating evidence",
                        priority=min(1.0, gap.importance + 0.1),
                        target_claim_ids=[claim_id],
                        required_evidence_kind="counterevidence",
                    )
                )
        deduped: dict[str, ResearchTask] = {}
        for task in tasks:
            key = " ".join(sorted(_terms(task.query)))
            if key not in deduped or task.priority > deduped[key].priority:
                deduped[key] = task
        return sorted(deduped.values(), key=lambda t: t.priority, reverse=True)


class AutonomousResearchEngine:
    """Bounded Level-6 research loop over the Level-5 epistemic substrate.

    The engine plans evidence acquisition from open gaps, actively seeks both support
    and falsification, deduplicates source ancestry, tests competing hypotheses,
    proposes belief revisions, identifies downstream claims requiring revalidation,
    and stops under explicit cost/confidence limits. It proposes changes rather than
    mutating L0/L1 knowledge in place, preserving the evidence/inference boundary.
    """

    def __init__(self, planner: ResearchPlanner | None = None):
        self.planner = planner or ResearchPlanner()

    def run(
        self,
        gaps: Sequence[KnowledgeGap],
        claims: Sequence[ClaimRecord],
        provider: ResearchProvider,
        budget: ResearchBudget | None = None,
    ) -> ResearchRun:
        budget = budget or ResearchBudget()
        started = time.perf_counter()
        tasks = self.planner.plan(gaps, claims)
        claim_by_id = {claim.claim_id: claim for claim in claims}
        acquired: dict[str, ResearchSource] = {}
        assessments: list[EvidenceAssessment] = []
        queries = 0
        stopped_reason = "tasks_exhausted"

        for task in tasks:
            elapsed_ms = (time.perf_counter() - started) * 1000
            if queries >= budget.max_queries:
                stopped_reason = "query_budget_exhausted"
                break
            if len(acquired) >= budget.max_sources:
                stopped_reason = "source_budget_exhausted"
                break
            if elapsed_ms >= budget.latency_ms:
                stopped_reason = "latency_budget_exhausted"
                break

            remaining = max(1, min(6, budget.max_sources - len(acquired)))
            results = provider.search(task.query, max_results=remaining)
            queries += 1
            for source in results:
                if source.reliability < budget.min_source_reliability:
                    continue
                fingerprint = self._source_fingerprint(source)
                if fingerprint in acquired:
                    continue
                acquired[fingerprint] = source
                for claim_id in task.target_claim_ids:
                    claim = claim_by_id.get(claim_id)
                    if claim:
                        assessments.append(self.assess(source, claim, list(acquired.values())))

            if self._targets_satisfied(task.target_claim_ids, assessments, budget):
                stopped_reason = "confidence_and_diversity_target_met"
                break

        sources = list(acquired.values())
        hypotheses = self.build_hypotheses(claims)
        hypothesis_results = [self.test_hypothesis(h, claims, assessments) for h in hypotheses]
        revisions = self.propose_revisions(claims, assessments)
        unresolved = self._unresolved_gaps(gaps, revisions)
        return ResearchRun(
            run_id=_id("research", _now(), str(len(tasks))),
            started_at=_now(),
            completed_at=_now(),
            tasks=tasks,
            acquired_sources=sources,
            assessments=assessments,
            hypotheses=hypotheses,
            hypothesis_results=hypothesis_results,
            revisions=revisions,
            unresolved_gap_ids=unresolved,
            queries_executed=queries,
            stopped_reason=stopped_reason,
        )

    def assess(
        self,
        source: ResearchSource,
        claim: ClaimRecord,
        acquired_sources: Sequence[ResearchSource],
    ) -> EvidenceAssessment:
        relevance = _overlap(source.text, claim.statement)
        same_polarity = _negated(source.text) == _negated(claim.statement)
        stance = "support" if same_polarity else "contradict"
        directness = _clamp(0.35 + relevance * 0.65)
        origin_counts = sum(1 for s in acquired_sources if s.origin() == source.origin())
        independence = 1.0 if origin_counts <= 1 else 1.0 / origin_counts
        recency = self._recency(source.published_at)
        score = _clamp(
            0.30 * relevance
            + 0.24 * directness
            + 0.24 * source.reliability
            + 0.14 * independence
            + 0.08 * recency
        )
        return EvidenceAssessment(
            source_id=source.source_id,
            claim_id=claim.claim_id,
            stance=stance,
            relevance=relevance,
            directness=directness,
            reliability=source.reliability,
            independence=independence,
            recency=recency,
            score=score,
            rationale=(
                f"{stance}; relevance={relevance:.2f}; reliability={source.reliability:.2f}; "
                f"independence={independence:.2f}; recency={recency:.2f}"
            ),
        )

    def build_hypotheses(self, claims: Sequence[ClaimRecord]) -> list[Hypothesis]:
        by_id = {claim.claim_id: claim for claim in claims}
        hypotheses: list[Hypothesis] = []
        seen: set[tuple[str, ...]] = set()
        for claim in claims:
            group = tuple(sorted({claim.claim_id, *claim.contradictions}))
            if len(group) > 1 and group not in seen:
                seen.add(group)
                hypotheses.append(
                    Hypothesis(
                        hypothesis_id=_id("hyp", *group),
                        statement=claim.statement,
                        claim_ids=list(group),
                        kind="competing-claims",
                        prior=max(0.05, min(0.95, claim.confidence or 0.5)),
                    )
                )
        return hypotheses

    def test_hypothesis(
        self,
        hypothesis: Hypothesis,
        claims: Sequence[ClaimRecord],
        assessments: Sequence[EvidenceAssessment],
    ) -> HypothesisResult:
        relevant = [a for a in assessments if a.claim_id in hypothesis.claim_ids]
        support = sum(a.score for a in relevant if a.stance == "support")
        contradict = sum(a.score for a in relevant if a.stance == "contradict")
        evidence_mass = support + contradict
        likelihood = support / evidence_mass if evidence_mass else hypothesis.prior
        posterior = _clamp(0.35 * hypothesis.prior + 0.65 * likelihood)
        origins = len({a.source_id for a in relevant})
        if posterior >= 0.78 and support > contradict:
            status = "provisionally-supported"
        elif posterior <= 0.32 and contradict > support:
            status = "provisionally-falsified"
        else:
            status = "unresolved"
        return HypothesisResult(
            hypothesis_id=hypothesis.hypothesis_id,
            support_score=support,
            contradiction_score=contradict,
            independent_origins=origins,
            posterior=posterior,
            status=status,
            source_ids=sorted({a.source_id for a in relevant}),
        )

    def propose_revisions(
        self,
        claims: Sequence[ClaimRecord],
        assessments: Sequence[EvidenceAssessment],
    ) -> list[RevisionProposal]:
        by_claim: dict[str, list[EvidenceAssessment]] = {}
        for assessment in assessments:
            by_claim.setdefault(assessment.claim_id, []).append(assessment)
        dependents: dict[str, list[str]] = {}
        for claim in claims:
            for dependency in claim.depends_on:
                dependents.setdefault(dependency, []).append(claim.claim_id)

        proposals: list[RevisionProposal] = []
        for claim in claims:
            evidence = by_claim.get(claim.claim_id, [])
            if not evidence:
                continue
            support = sum(a.score for a in evidence if a.stance == "support")
            contradict = sum(a.score for a in evidence if a.stance == "contradict")
            total = support + contradict
            evidence_confidence = support / total if total else claim.confidence
            diversity = len({a.source_id for a in evidence})
            diversity_factor = min(1.0, diversity / 3)
            proposed = _clamp(0.45 * claim.confidence + 0.55 * evidence_confidence * (0.7 + 0.3 * diversity_factor))
            if contradict > support and contradict >= 0.9:
                status = EpistemicStatus.CONTRADICTED.value
            elif support > 0 and contradict > 0:
                status = EpistemicStatus.SUPPORTED_BUT_CONTESTED.value
            elif support >= 0.9:
                status = EpistemicStatus.SUPPORTED.value
            else:
                status = EpistemicStatus.UNVERIFIED.value
            prior_status = claim.epistemic_status.value if hasattr(claim.epistemic_status, "value") else str(claim.epistemic_status)
            proposals.append(
                RevisionProposal(
                    claim_id=claim.claim_id,
                    prior_confidence=claim.confidence,
                    proposed_confidence=proposed,
                    prior_status=prior_status,
                    proposed_status=status,
                    supporting_sources=sorted({a.source_id for a in evidence if a.stance == "support"}),
                    contradicting_sources=sorted({a.source_id for a in evidence if a.stance == "contradict"}),
                    rationale=(
                        f"New independent evidence mass: support={support:.2f}, contradiction={contradict:.2f}; "
                        f"source diversity={diversity}. Proposal does not overwrite source-derived claims."
                    ),
                    downstream_claim_ids=self._transitive_dependents(claim.claim_id, dependents),
                )
            )
        return proposals

    def due_for_revalidation(
        self,
        claims: Sequence[ClaimRecord],
        freshness_threshold: float = 0.55,
        confidence_threshold: float = 0.7,
    ) -> list[ClaimRecord]:
        return sorted(
            [
                claim
                for claim in claims
                if claim.freshness <= freshness_threshold
                or claim.confidence <= confidence_threshold
                or bool(claim.contradictions)
                or claim.epistemic_status in {EpistemicStatus.CONTRADICTED, EpistemicStatus.SUPPORTED_BUT_CONTESTED}
            ],
            key=lambda c: (c.freshness, c.confidence),
        )

    def revalidation_gaps(self, claims: Sequence[ClaimRecord]) -> list[KnowledgeGap]:
        gaps: list[KnowledgeGap] = []
        for claim in self.due_for_revalidation(claims):
            gaps.append(
                KnowledgeGap(
                    gap_id=_id("gap", "revalidate", claim.claim_id),
                    question=f"What current independent evidence supports or falsifies: {claim.statement}",
                    importance=_clamp(0.55 + (1.0 - claim.freshness) * 0.25 + (1.0 - claim.confidence) * 0.2),
                    evidence_missing=["current independent replication", "counterevidence"],
                    related_claims=[claim.claim_id, *claim.contradictions],
                    status="open",
                    created_at=_now(),
                )
            )
        return gaps

    def _targets_satisfied(
        self,
        target_claim_ids: Sequence[str],
        assessments: Sequence[EvidenceAssessment],
        budget: ResearchBudget,
    ) -> bool:
        if not target_claim_ids:
            return False
        for claim_id in target_claim_ids:
            evidence = [a for a in assessments if a.claim_id == claim_id]
            if not evidence:
                return False
            origins = len({a.source_id for a in evidence})
            if origins < budget.source_diversity_target:
                return False
            confidence = sum(a.score for a in evidence) / len(evidence)
            if confidence < budget.confidence_target:
                return False
        return True

    def _unresolved_gaps(
        self,
        gaps: Sequence[KnowledgeGap],
        revisions: Sequence[RevisionProposal],
    ) -> list[str]:
        revised = {r.claim_id for r in revisions if r.proposed_confidence >= 0.72}
        return [
            gap.gap_id
            for gap in gaps
            if gap.status == "open" and not (set(gap.related_claims) & revised)
        ]

    def _transitive_dependents(self, claim_id: str, dependents: dict[str, list[str]]) -> list[str]:
        seen: set[str] = set()
        stack = list(dependents.get(claim_id, []))
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(dependents.get(current, []))
        return sorted(seen)

    def _source_fingerprint(self, source: ResearchSource) -> str:
        normalized = " ".join(sorted(_terms(source.text)))
        return _id("src", source.origin(), normalized)

    def _recency(self, published_at: str | None) -> float:
        if not published_at:
            return 0.5
        try:
            stamp = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=UTC)
            age_days = max(0.0, (datetime.now(UTC) - stamp).total_seconds() / 86400)
            return _clamp(1.0 / (1.0 + age_days / 730.0))
        except ValueError:
            return 0.5
