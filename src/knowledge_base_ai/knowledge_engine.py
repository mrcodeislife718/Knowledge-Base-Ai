from __future__ import annotations

import hashlib
import json
import math
import re
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from .models import (
    ClaimRecord,
    ConfidenceBreakdown,
    EntityRecord,
    EpistemicStatus,
    EvidenceBudget,
    EvidenceRecord,
    KnowledgeGap,
    ProvenanceKind,
    QueryPlan,
    ReflectionRecord,
    RelationRecord,
    RelationType,
    RetrievalTelemetry,
)

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9']+")
_ENTITY_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9&'\-]+(?:\s+[A-Z][A-Za-z0-9&'\-]+){0,4})\b")
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_NEGATIONS = {"not", "no", "never", "none", "without", "fails", "failed", "cannot", "can't", "doesn't", "didn't"}
_STOPWORDS = {
    "the", "and", "for", "that", "with", "from", "this", "into", "were", "was", "are", "is",
    "has", "have", "had", "but", "not", "you", "your", "their", "they", "them", "than", "then",
    "when", "where", "what", "which", "who", "why", "how", "can", "could", "would", "should",
    "may", "might", "will", "shall", "also", "its", "our", "out", "about", "over", "under", "between",
}

_CAUSAL_PATTERNS: list[tuple[re.Pattern[str], RelationType]] = [
    (re.compile(r"\bcauses?\b|\bleads? to\b|\bresults? in\b", re.I), RelationType.CAUSES),
    (re.compile(r"\benables?\b|\ballows?\b", re.I), RelationType.ENABLES),
    (re.compile(r"\binhibits?\b|\bsuppresses?\b|\bprevents?\b", re.I), RelationType.INHIBITS),
    (re.compile(r"\brequires?\b|\bdepends? on\b", re.I), RelationType.REQUIRES),
    (re.compile(r"\bprecedes?\b|\bbefore\b", re.I), RelationType.PRECEDES),
    (re.compile(r"\bpredicts?\b|\bforecast(?:s|ed)?\b", re.I), RelationType.PREDICTS),
    (re.compile(r"\bmediates?\b", re.I), RelationType.MEDIATES),
    (re.compile(r"\bmoderates?\b", re.I), RelationType.MODERATES),
    (re.compile(r"\bcorrelat(?:e|es|ed|ion)\b|\bassociated with\b", re.I), RelationType.CORRELATED_WITH),
]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _hash(prefix: str, *parts: str) -> str:
    payload = "\x1f".join(parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:16]}"


def _terms(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) > 2 and w not in _STOPWORDS}


def _jaccard(a: str, b: str) -> float:
    ta, tb = _terms(a), _terms(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _contains_negation(text: str) -> bool:
    return bool(_terms(text) & _NEGATIONS) or any(token in text.lower() for token in ("n't", " no "))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _mean(values: Iterable[float], default: float = 0.0) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else default


class KnowledgeCompiler:
    """Compile passages into provenance-preserving, typed knowledge IR.

    The implementation is deliberately deterministic and model-independent. LLMs can
    later replace individual extraction/reflection stages without weakening the layer
    boundaries, provenance rules, contradiction handling, or validation contract.
    """

    def __init__(self, reflection_threshold: float = 0.34):
        self.reflection_threshold = reflection_threshold

    def compile_chunks(self, chunks: list[dict]) -> dict[str, list]:
        evidence: list[EvidenceRecord] = []
        claims: list[ClaimRecord] = []
        entities: dict[str, EntityRecord] = {}

        for chunk in chunks:
            evidence_id = _hash("ev", chunk["source_sha256"], chunk["chunk_id"])
            ev = EvidenceRecord(
                evidence_id=evidence_id,
                text=chunk["text"],
                source_path=chunk["source_path"],
                source_sha256=chunk["source_sha256"],
                passage_id=chunk["chunk_id"],
                page_start=int(chunk["page_start"]),
                page_end=int(chunk["page_end"]),
                independent_origin=chunk["source_sha256"],
                content_sha256=hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest(),
            )
            evidence.append(ev)
            for sentence in self._atomic_sentences(chunk["text"]):
                claim_id = _hash("cl", evidence_id, sentence)
                entity_ids: list[str] = []
                for name in self._extract_entities(sentence):
                    eid = _hash("en", name.lower())
                    entity_ids.append(eid)
                    entity = entities.setdefault(eid, EntityRecord(entity_id=eid, canonical_name=name))
                    if evidence_id not in entity.evidence_ids:
                        entity.evidence_ids.append(evidence_id)
                year = self._extract_year(sentence)
                claims.append(
                    ClaimRecord(
                        claim_id=claim_id,
                        statement=sentence,
                        evidence_ids=[evidence_id],
                        source_ids=[chunk["source_sha256"]],
                        entity_ids=entity_ids,
                        valid_from=f"{year}-01-01T00:00:00+00:00" if year else None,
                        published_at=f"{year}-01-01T00:00:00+00:00" if year else None,
                    )
                )

        relations = self._relate_claims(claims, evidence)
        self._attach_relation_edges(claims, relations)
        self._score_claims(claims, evidence)
        reflections = self._reflect(claims)
        gaps = self._discover_gaps(claims, relations)
        return {
            "evidence": evidence,
            "claims": claims,
            "entities": list(entities.values()),
            "relations": relations,
            "reflections": reflections,
            "gaps": gaps,
        }

    def _atomic_sentences(self, text: str) -> list[str]:
        raw = [s.strip() for s in _SENTENCE_RE.split(re.sub(r"\s+", " ", text.strip()))]
        return [s for s in raw if len(_terms(s)) >= 4 and 20 <= len(s) <= 700]

    def _extract_entities(self, sentence: str) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for match in _ENTITY_RE.findall(sentence):
            candidate = match.strip()
            if candidate.lower() in {"The", "This", "That"}:
                continue
            key = candidate.lower()
            if key not in seen:
                seen.add(key)
                out.append(candidate)
        return out[:12]

    def _extract_year(self, sentence: str) -> int | None:
        match = _YEAR_RE.search(sentence)
        return int(match.group(1)) if match else None

    def _relate_claims(self, claims: list[ClaimRecord], evidence: list[EvidenceRecord]) -> list[RelationRecord]:
        relations: list[RelationRecord] = []
        for index, left in enumerate(claims):
            for right in claims[index + 1 :]:
                similarity = _jaccard(left.statement, right.statement)
                if similarity < 0.28:
                    continue
                if similarity >= 0.88:
                    relation_type = RelationType.DUPLICATES
                    confidence = similarity
                elif _contains_negation(left.statement) != _contains_negation(right.statement) and similarity >= 0.48:
                    relation_type = RelationType.CONTRADICTS
                    confidence = min(0.95, similarity + 0.15)
                elif similarity >= 0.62:
                    relation_type = RelationType.SUPPORTS
                    confidence = similarity
                else:
                    relation_type = RelationType.QUALIFIES
                    confidence = similarity
                relations.append(self._relation(left, right, relation_type, confidence))

        for claim in claims:
            for pattern, relation_type in _CAUSAL_PATTERNS:
                if not pattern.search(claim.statement):
                    continue
                entity_ids = claim.entity_ids
                if len(entity_ids) >= 2:
                    relations.append(
                        RelationRecord(
                            relation_id=_hash("rel", entity_ids[0], entity_ids[1], relation_type.value, claim.claim_id),
                            source_id=entity_ids[0],
                            target_id=entity_ids[1],
                            relation_type=relation_type,
                            evidence_ids=claim.evidence_ids,
                            provenance_kind=ProvenanceKind.SOURCE_ASSERTION,
                            confidence=0.62,
                        )
                    )
                break
        return self._dedupe_relations(relations)

    def _relation(self, left: ClaimRecord, right: ClaimRecord, kind: RelationType, confidence: float) -> RelationRecord:
        return RelationRecord(
            relation_id=_hash("rel", left.claim_id, right.claim_id, kind.value),
            source_id=left.claim_id,
            target_id=right.claim_id,
            relation_type=kind,
            evidence_ids=sorted(set(left.evidence_ids + right.evidence_ids)),
            provenance_kind=ProvenanceKind.SYSTEM_INFERENCE,
            confidence=_clamp(confidence),
        )

    def _dedupe_relations(self, relations: list[RelationRecord]) -> list[RelationRecord]:
        best: dict[tuple[str, str, RelationType], RelationRecord] = {}
        for relation in relations:
            key = (relation.source_id, relation.target_id, relation.relation_type)
            current = best.get(key)
            if current is None or relation.confidence > current.confidence:
                best[key] = relation
        return list(best.values())

    def _attach_relation_edges(self, claims: list[ClaimRecord], relations: list[RelationRecord]) -> None:
        by_id = {claim.claim_id: claim for claim in claims}
        for rel in relations:
            source = by_id.get(rel.source_id)
            target = by_id.get(rel.target_id)
            if not source or not target:
                continue
            if rel.relation_type == RelationType.SUPPORTS:
                target.support.append(source.claim_id)
            elif rel.relation_type == RelationType.CONTRADICTS:
                source.contradictions.append(target.claim_id)
                target.contradictions.append(source.claim_id)
            elif rel.relation_type == RelationType.QUALIFIES:
                target.qualifies.append(source.claim_id)
            elif rel.relation_type in {RelationType.DEPENDS_ON, RelationType.REQUIRES}:
                source.depends_on.append(target.claim_id)

    def _score_claims(self, claims: list[ClaimRecord], evidence: list[EvidenceRecord]) -> None:
        evidence_by_id = {ev.evidence_id: ev for ev in evidence}
        for claim in claims:
            breakdown = calibrate_confidence(
                source_reliability=_mean((evidence_by_id[e].source_reliability for e in claim.evidence_ids), 0.5),
                independent_sources=len({evidence_by_id[e].independent_origin for e in claim.evidence_ids}),
                support_count=len(claim.support),
                contradiction_count=len(claim.contradictions),
                recency=claim.freshness,
                directness=0.9,
                replication=max(0.4, min(1.0, len(claim.source_ids) / 3)),
                retrieval_coverage=1.0,
                agreement=1.0 if not claim.contradictions else 0.55,
                reasoning_depth=0,
            )
            claim.confidence = breakdown.calibrated_score
            if claim.contradictions and claim.support:
                claim.epistemic_status = EpistemicStatus.SUPPORTED_BUT_CONTESTED
            elif claim.contradictions and not claim.support:
                claim.epistemic_status = EpistemicStatus.CONTRADICTED
            elif claim.confidence >= 0.55:
                claim.epistemic_status = EpistemicStatus.SUPPORTED

    def _reflect(self, claims: list[ClaimRecord]) -> list[ReflectionRecord]:
        reflections: list[ReflectionRecord] = []
        for claim in claims:
            peers = [other for other in claims if other.claim_id != claim.claim_id]
            max_similarity = max((_jaccard(claim.statement, p.statement) for p in peers), default=0.0)
            novelty = 1.0 - max_similarity
            contradiction = min(1.0, len(claim.contradictions) / 2)
            uncertainty = 1.0 - claim.confidence
            importance = min(1.0, 0.35 + 0.08 * len(claim.entity_ids) + 0.12 * len(claim.support))
            expected_future_value = min(1.0, 0.45 + 0.1 * len(claim.entity_ids))
            priority = novelty * importance * max(0.2, contradiction) * max(0.2, uncertainty) * expected_future_value
            if priority < self.reflection_threshold:
                continue
            statement = self._reflection_statement(claim, novelty, contradiction)
            reflections.append(
                ReflectionRecord(
                    reflection_id=_hash("rf", claim.claim_id, statement),
                    statement=statement,
                    claim_ids=[claim.claim_id] + claim.support + claim.contradictions,
                    evidence_ids=claim.evidence_ids,
                    novelty=novelty,
                    importance=importance,
                    contradiction=contradiction,
                    uncertainty=uncertainty,
                    expected_future_value=expected_future_value,
                    priority=priority,
                    confidence=claim.confidence,
                    verification_status="verified-structure-only",
                    created_at=_now(),
                )
            )
        return reflections

    def _reflection_statement(self, claim: ClaimRecord, novelty: float, contradiction: float) -> str:
        if contradiction > 0:
            return f"Contested knowledge: {claim.statement} Counterevidence exists and should be resolved before consolidation."
        if novelty > 0.72:
            return f"Novel knowledge: {claim.statement} This claim introduces information weakly represented elsewhere in the corpus."
        return f"Connected knowledge: {claim.statement} This claim reinforces or qualifies existing corpus knowledge."

    def _discover_gaps(self, claims: list[ClaimRecord], relations: list[RelationRecord]) -> list[KnowledgeGap]:
        gaps: list[KnowledgeGap] = []
        for claim in claims:
            causal_language = any(pattern.search(claim.statement) for pattern, _ in _CAUSAL_PATTERNS)
            if claim.contradictions:
                gaps.append(
                    KnowledgeGap(
                        gap_id=_hash("gap", "contradiction", claim.claim_id),
                        question=f"Which evidence best resolves the contradiction around: {claim.statement}",
                        importance=min(1.0, 0.65 + 0.05 * len(claim.contradictions)),
                        evidence_missing=["independent adjudicating evidence"],
                        related_claims=[claim.claim_id] + claim.contradictions,
                        created_at=_now(),
                    )
                )
            elif causal_language and not any(
                r.source_id in claim.entity_ids and r.relation_type == RelationType.CAUSES for r in relations
            ):
                gaps.append(
                    KnowledgeGap(
                        gap_id=_hash("gap", "causal", claim.claim_id),
                        question=f"Is the asserted relationship causal, correlational, or conditional: {claim.statement}",
                        importance=0.72,
                        evidence_missing=["direct causal evidence", "counterfactual or controlled evidence"],
                        related_claims=[claim.claim_id],
                        created_at=_now(),
                    )
                )
        return gaps


def calibrate_confidence(
    *,
    source_reliability: float,
    independent_sources: int,
    support_count: int,
    contradiction_count: int,
    recency: float,
    directness: float,
    replication: float,
    retrieval_coverage: float,
    agreement: float,
    reasoning_depth: int,
) -> ConfidenceBreakdown:
    independence = min(1.0, math.log2(max(1, independent_sources) + 1) / 2)
    support = min(1.0, 0.45 + 0.15 * support_count)
    contradiction_penalty = min(0.65, 0.16 * contradiction_count)
    depth_penalty = min(0.35, 0.06 * reasoning_depth)
    raw = (
        0.18 * _clamp(source_reliability)
        + 0.12 * independence
        + 0.14 * support
        + 0.10 * _clamp(recency)
        + 0.12 * _clamp(directness)
        + 0.09 * _clamp(replication)
        + 0.09 * _clamp(retrieval_coverage)
        + 0.16 * _clamp(agreement)
        - contradiction_penalty
        - depth_penalty
    )
    score = _clamp(raw)
    return ConfidenceBreakdown(
        source_reliability=_clamp(source_reliability),
        source_independence=independence,
        support_strength=support,
        contradiction_penalty=contradiction_penalty,
        evidence_recency=_clamp(recency),
        directness=_clamp(directness),
        replication=_clamp(replication),
        retrieval_coverage=_clamp(retrieval_coverage),
        agreement=_clamp(agreement),
        reasoning_depth_penalty=depth_penalty,
        calibrated_score=score,
    )


def plan_query(query: str, budget: EvidenceBudget | None = None) -> QueryPlan:
    q = query.lower()
    temporal = any(token in q for token in ("when", "timeline", "changed", "current", "latest", "before", "after", "year"))
    contradiction = any(token in q for token in ("contradict", "disagree", "counterevidence", "consensus", "conflict"))
    causal = any(token in q for token in ("cause", "why", "effect", "because", "counterfactual", "if not"))
    comparison = any(token in q for token in ("compare", "versus", " vs ", "difference", "better"))
    global_synthesis = any(token in q for token in ("overall", "across", "summarize", "synthesis", "whole corpus"))
    if temporal:
        intent = "timeline"
    elif contradiction:
        intent = "contradiction_resolution"
    elif causal:
        intent = "causal_reasoning"
    elif comparison:
        intent = "comparison"
    elif global_synthesis:
        intent = "global_synthesis"
    else:
        intent = "lookup"
    strategies = ["lexical", "vector"]
    if causal or contradiction or comparison:
        strategies.extend(["claim_graph", "entity_graph"])
    if temporal:
        strategies.append("temporal")
    if global_synthesis:
        strategies.append("hierarchical")
    return QueryPlan(
        query=query,
        intent=intent,
        strategies=list(dict.fromkeys(strategies)),
        requires_multi_hop=causal or comparison or contradiction,
        requires_temporal=temporal,
        requires_contradiction_resolution=contradiction,
        requires_source_verification=True,
        graph_depth=3 if causal else 2 if contradiction or comparison else 1,
        budget=budget or EvidenceBudget(),
    )


class KnowledgeRepository:
    """Filesystem-backed Knowledge IR with revision, gap closure, GC and telemetry."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def write_run(self, run_id: str, compiled: dict[str, list]) -> Path:
        run_root = self.root / run_id
        run_root.mkdir(parents=True, exist_ok=True)
        for key, records in compiled.items():
            path = run_root / f"{key}.jsonl"
            with path.open("w", encoding="utf-8") as fh:
                for record in records:
                    payload = record.to_dict() if hasattr(record, "to_dict") else asdict(record)
                    fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        manifest = {
            "run_id": run_id,
            "compiled_at": _now(),
            "counts": {key: len(value) for key, value in compiled.items()},
            "epistemic_contract": "derived layers may reference but never overwrite L0 evidence",
        }
        (run_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return run_root

    def revise_belief(self, claims: list[ClaimRecord], invalid_claim_id: str, reason: str) -> list[str]:
        by_id = {claim.claim_id: claim for claim in claims}
        if invalid_claim_id not in by_id:
            return []
        invalidated: list[str] = []
        queue = [invalid_claim_id]
        while queue:
            current_id = queue.pop(0)
            current = by_id.get(current_id)
            if not current or current.epistemic_status == EpistemicStatus.INVALIDATED:
                continue
            current.epistemic_status = EpistemicStatus.INVALIDATED
            current.invalidation_reason = reason if current_id == invalid_claim_id else f"dependency invalidated: {current_id}"
            invalidated.append(current_id)
            queue.extend(c.claim_id for c in claims if current_id in c.depends_on)
        return invalidated

    def close_gaps(self, gaps: list[KnowledgeGap], claims: list[ClaimRecord]) -> list[str]:
        closed: list[str] = []
        for gap in gaps:
            if gap.status != "open":
                continue
            relevant = [claim for claim in claims if claim.claim_id in gap.related_claims]
            if relevant and all(c.confidence >= 0.8 and not c.contradictions for c in relevant):
                gap.status = "closed"
                gap.closed_at = _now()
                gap.closed_by = [c.claim_id for c in relevant]
                closed.append(gap.gap_id)
        return closed

    def garbage_collect(self, claims: list[ClaimRecord], duplicate_threshold: float = 0.93) -> dict[str, list[str]]:
        merged: list[str] = []
        archived: list[str] = []
        for index, left in enumerate(claims):
            if left.epistemic_status == EpistemicStatus.INVALIDATED:
                archived.append(left.claim_id)
                continue
            for right in claims[index + 1 :]:
                if right.claim_id in merged or right.epistemic_status == EpistemicStatus.INVALIDATED:
                    continue
                if _jaccard(left.statement, right.statement) >= duplicate_threshold:
                    left.evidence_ids = sorted(set(left.evidence_ids + right.evidence_ids))
                    left.source_ids = sorted(set(left.source_ids + right.source_ids))
                    left.support = sorted(set(left.support + right.support))
                    merged.append(right.claim_id)
        return {"merged": merged, "archived": archived}

    def record_telemetry(
        self,
        query: str,
        plan: QueryPlan,
        retrieved_ids: list[str],
        cited_ids: list[str],
        latency_ms: float,
        estimated_cost: float = 0.0,
    ) -> RetrievalTelemetry:
        record = RetrievalTelemetry(
            query_id=uuid.uuid4().hex[:16],
            query=query,
            strategies=plan.strategies,
            retrieved_ids=retrieved_ids,
            cited_ids=cited_ids,
            unused_ids=[rid for rid in retrieved_ids if rid not in cited_ids],
            missed_ids=[],
            latency_ms=latency_ms,
            estimated_cost=estimated_cost,
            created_at=_now(),
        )
        path = self.root / "telemetry.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        return record


def evidence_budget_reached(
    started_at: float,
    tokens_used: int,
    compute_used: float,
    independent_sources: int,
    confidence: float,
    budget: EvidenceBudget,
    marginal_gain: float,
    marginal_cost: float,
) -> bool:
    latency_ms = (time.perf_counter() - started_at) * 1000
    if latency_ms >= budget.latency_ms or tokens_used >= budget.token_budget or compute_used >= budget.compute_units:
        return True
    if confidence >= budget.confidence_target and independent_sources >= budget.source_diversity_target:
        return True
    return marginal_gain <= marginal_cost
