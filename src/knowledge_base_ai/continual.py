from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .models import ClaimRecord, EpistemicStatus, ProvenanceKind, RelationRecord, RelationType

_WORD_RE = re.compile(r"[a-z0-9']+")
_NEG = {"not", "no", "never", "without", "cannot", "fails", "failed"}
_STOP = {"the", "and", "for", "that", "with", "from", "this", "into", "are", "was", "were", "is", "of", "to", "a", "an"}


def _terms(text: str) -> set[str]:
    return {x for x in _WORD_RE.findall(text.lower()) if len(x) > 2 and x not in _STOP}


def _similarity(left: str, right: str) -> float:
    a, b = _terms(left), _terms(right)
    return len(a & b) / len(a | b) if a and b else 0.0


def _negated(text: str) -> bool:
    terms = _terms(text)
    return bool(terms & _NEG) or "n't" in text.lower()


def _relation_id(left: str, right: str, relation: RelationType) -> str:
    digest = hashlib.sha256(f"{left}|{right}|{relation.value}".encode()).hexdigest()[:16]
    return f"rel_{digest}"


def load_prior_claims(root: Path) -> list[dict]:
    """Load claim IR from completed prior runs without treating reflections as evidence."""
    if not root.exists():
        return []
    claims: list[dict] = []
    for path in sorted(root.glob("*/claims.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                claims.append(json.loads(line))
    return claims


def connect_to_prior_knowledge(compiled: dict[str, list], prior_claims: list[dict]) -> dict[str, int]:
    """Link incoming claims to historical claims and update epistemic state.

    Historical claims remain immutable artifacts. New claims receive support,
    contradiction, qualification and duplicate edges pointing back to them.
    """
    if not prior_claims:
        return {"prior_claims_considered": 0, "cross_run_relations": 0}

    added: list[RelationRecord] = []
    for claim in compiled["claims"]:
        best_matches = sorted(
            ((prior, _similarity(claim.statement, prior.get("statement", ""))) for prior in prior_claims),
            key=lambda item: item[1],
            reverse=True,
        )[:8]
        for prior, similarity in best_matches:
            if similarity < 0.34:
                continue
            prior_id = prior.get("claim_id")
            if not prior_id or prior_id == claim.claim_id:
                continue
            if similarity >= 0.90:
                relation_type = RelationType.DUPLICATES
            elif _negated(claim.statement) != _negated(prior.get("statement", "")) and similarity >= 0.48:
                relation_type = RelationType.CONTRADICTS
                claim.contradictions.append(prior_id)
                claim.epistemic_status = EpistemicStatus.SUPPORTED_BUT_CONTESTED
            elif similarity >= 0.64:
                relation_type = RelationType.SUPPORTS
                claim.support.append(prior_id)
            else:
                relation_type = RelationType.QUALIFIES
                claim.qualifies.append(prior_id)
            added.append(
                RelationRecord(
                    relation_id=_relation_id(claim.claim_id, prior_id, relation_type),
                    source_id=claim.claim_id,
                    target_id=prior_id,
                    relation_type=relation_type,
                    evidence_ids=claim.evidence_ids + list(prior.get("evidence_ids", [])),
                    provenance_kind=ProvenanceKind.SYSTEM_INFERENCE,
                    confidence=min(0.96, similarity + (0.12 if relation_type == RelationType.CONTRADICTS else 0.0)),
                )
            )

    known_ids = {relation.relation_id for relation in compiled["relations"]}
    compiled["relations"].extend(relation for relation in added if relation.relation_id not in known_ids)
    return {
        "prior_claims_considered": len(prior_claims),
        "cross_run_relations": len(added),
    }
