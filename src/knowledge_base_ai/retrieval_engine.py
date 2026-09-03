from __future__ import annotations

import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Callable

from .knowledge_engine import evidence_budget_reached, plan_query
from .models import ClaimRecord, EvidenceBudget, QueryPlan, RelationRecord, RelationType

_WORD_RE = re.compile(r"[a-z0-9']+")
_STOP = {"the", "and", "for", "that", "with", "from", "this", "into", "are", "was", "were", "is", "of", "to", "a", "an"}


def _terms(text: str) -> set[str]:
    return {x for x in _WORD_RE.findall(text.lower()) if len(x) > 2 and x not in _STOP}


def _lexical_score(query: str, text: str) -> float:
    q, t = _terms(query), _terms(text)
    if not q or not t:
        return 0.0
    return len(q & t) / len(q)


def _temporal_score(claim: ClaimRecord, query: str) -> float:
    years = re.findall(r"\b(?:19|20)\d{2}\b", query)
    if not years:
        return claim.freshness
    haystack = " ".join(filter(None, [claim.valid_from, claim.valid_until, claim.published_at, claim.observed_at]))
    return 1.0 if any(year in haystack for year in years) else 0.0


@dataclass
class RetrievedKnowledge:
    claim: ClaimRecord
    score: float
    reasons: list[str] = field(default_factory=list)
    graph_distance: int | None = None


@dataclass
class RetrievalResult:
    plan: QueryPlan
    items: list[RetrievedKnowledge]
    stopped_by_budget: bool
    independent_sources: int
    confidence: float
    elapsed_ms: float


class AdaptiveRetriever:
    """Query claims through lexical, vector, graph, temporal and contradiction views.

    A vector scorer is optional and injected as a callback so the knowledge layer is
    not locked to Chroma, one embedding model, or any vendor. The deterministic
    lexical/graph/temporal paths remain available when no vector backend exists.
    """

    def __init__(
        self,
        claims: list[ClaimRecord],
        relations: list[RelationRecord],
        vector_score: Callable[[str, list[ClaimRecord]], dict[str, float]] | None = None,
    ) -> None:
        self.claims = claims
        self.by_id = {claim.claim_id: claim for claim in claims}
        self.relations = relations
        self.vector_score = vector_score
        self.adjacency: dict[str, list[tuple[str, RelationType]]] = defaultdict(list)
        for relation in relations:
            self.adjacency[relation.source_id].append((relation.target_id, relation.relation_type))
            self.adjacency[relation.target_id].append((relation.source_id, relation.relation_type))

    def retrieve(self, query: str, budget: EvidenceBudget | None = None) -> RetrievalResult:
        plan = plan_query(query, budget)
        started = time.perf_counter()
        scores: dict[str, float] = defaultdict(float)
        reasons: dict[str, list[str]] = defaultdict(list)

        for claim in self.claims:
            lex = _lexical_score(query, claim.statement)
            if lex > 0:
                scores[claim.claim_id] += 0.38 * lex
                reasons[claim.claim_id].append(f"lexical={lex:.3f}")
            if plan.requires_temporal:
                temp = _temporal_score(claim, query)
                if temp > 0:
                    scores[claim.claim_id] += 0.16 * temp
                    reasons[claim.claim_id].append(f"temporal={temp:.3f}")
            scores[claim.claim_id] += 0.18 * claim.confidence

        if self.vector_score is not None and "vector" in plan.strategies:
            for claim_id, value in self.vector_score(query, self.claims).items():
                if claim_id in self.by_id:
                    scores[claim_id] += 0.28 * max(0.0, min(1.0, value))
                    reasons[claim_id].append(f"vector={value:.3f}")

        seed_ids = [cid for cid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:6]]
        graph_distance: dict[str, int] = {}
        if any(strategy.endswith("graph") for strategy in plan.strategies):
            self._expand_graph(seed_ids, plan.graph_depth, scores, reasons, graph_distance)

        if plan.requires_contradiction_resolution:
            self._boost_contradictions(seed_ids, scores, reasons)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected: list[RetrievedKnowledge] = []
        independent: set[str] = set()
        confidence_values: list[float] = []
        stopped = False
        token_estimate = 0

        for claim_id, score in ranked:
            claim = self.by_id[claim_id]
            token_estimate += max(1, len(claim.statement) // 4)
            independent.update(claim.source_ids)
            confidence_values.append(claim.confidence)
            selected.append(
                RetrievedKnowledge(
                    claim=claim,
                    score=score,
                    reasons=reasons[claim_id],
                    graph_distance=graph_distance.get(claim_id),
                )
            )
            confidence = sum(confidence_values) / len(confidence_values)
            marginal_gain = max(0.0, score - (ranked[len(selected)][1] if len(selected) < len(ranked) else 0.0))
            marginal_cost = 0.015 + token_estimate / max(1, plan.budget.token_budget) * 0.02
            selected_ids = {item.claim.claim_id for item in selected}
            unresolved_counterevidence = (
                plan.requires_contradiction_resolution
                and any(
                    contradiction_id in self.by_id and contradiction_id not in selected_ids
                    for item in selected
                    for contradiction_id in item.claim.contradictions
                )
            )
            budget_reached = evidence_budget_reached(
                started,
                tokens_used=token_estimate,
                compute_used=min(1.0, len(selected) / max(1, plan.budget.max_results)),
                independent_sources=len(independent),
                confidence=confidence,
                budget=plan.budget,
                marginal_gain=marginal_gain,
                marginal_cost=marginal_cost,
            )
            if len(selected) >= plan.budget.max_results or (budget_reached and not unresolved_counterevidence):
                stopped = True
                break

        elapsed = (time.perf_counter() - started) * 1000
        final_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        return RetrievalResult(
            plan=plan,
            items=selected,
            stopped_by_budget=stopped,
            independent_sources=len(independent),
            confidence=final_confidence,
            elapsed_ms=elapsed,
        )

    def _expand_graph(
        self,
        seeds: list[str],
        max_depth: int,
        scores: dict[str, float],
        reasons: dict[str, list[str]],
        distances: dict[str, int],
    ) -> None:
        queue = deque((seed, 0) for seed in seeds)
        visited = set(seeds)
        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for neighbor, relation_type in self.adjacency.get(current, []):
                if neighbor not in self.by_id:
                    continue
                next_depth = depth + 1
                weight = 0.12 / next_depth
                if relation_type == RelationType.CONTRADICTS:
                    weight += 0.04
                scores[neighbor] += weight
                reasons[neighbor].append(f"graph:{relation_type.value}@{next_depth}")
                distances[neighbor] = min(distances.get(neighbor, next_depth), next_depth)
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, next_depth))

    def _boost_contradictions(
        self,
        seeds: list[str],
        scores: dict[str, float],
        reasons: dict[str, list[str]],
    ) -> None:
        for seed in seeds:
            claim = self.by_id.get(seed)
            if not claim:
                continue
            for contradiction_id in claim.contradictions:
                if contradiction_id in self.by_id:
                    scores[contradiction_id] += 0.22
                    reasons[contradiction_id].append("counterevidence")


def counterfactual_trace(
    cause_id: str,
    effect_id: str,
    relations: list[RelationRecord],
    max_depth: int = 5,
) -> dict:
    """Return explicit causal paths and assumptions for a counterfactual question.

    This is structural counterfactual support, not a claim that graph reachability
    proves causation. Only causal-like relation types are traversed.
    """
    causal = {
        RelationType.CAUSES,
        RelationType.ENABLES,
        RelationType.INHIBITS,
        RelationType.REQUIRES,
        RelationType.MEDIATES,
        RelationType.MODERATES,
        RelationType.PRECEDES,
    }
    adjacency: dict[str, list[tuple[str, RelationRecord]]] = defaultdict(list)
    for relation in relations:
        if relation.relation_type in causal:
            adjacency[relation.source_id].append((relation.target_id, relation))

    queue = deque([(cause_id, [cause_id], [])])
    paths: list[dict] = []
    while queue:
        node, path, assumptions = queue.popleft()
        if len(path) - 1 >= max_depth:
            continue
        for target, relation in adjacency.get(node, []):
            if target in path:
                continue
            new_path = path + [target]
            new_assumptions = assumptions + relation.assumptions
            if target == effect_id:
                paths.append(
                    {
                        "path": new_path,
                        "relation_ids": [r.relation_id for r in relations if r.source_id in new_path and r.target_id in new_path],
                        "assumptions": sorted(set(new_assumptions)),
                        "warning": "Structural path only; intervention data is required for causal certainty.",
                    }
                )
            else:
                queue.append((target, new_path, new_assumptions))
    return {
        "cause_id": cause_id,
        "effect_id": effect_id,
        "paths": paths,
        "counterfactual_supported": bool(paths),
        "generated_at": datetime.now(UTC).isoformat(),
    }
