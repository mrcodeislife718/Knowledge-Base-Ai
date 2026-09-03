from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


@dataclass(frozen=True)
class AnswerContract:
    answer: str
    retrieved_ids: tuple[str, ...]
    cited_ids: tuple[str, ...]
    confidence: float
    abstained: bool
    latency_ms: float
    estimated_cost: float
    independent_origins: int
    provenance: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AnswerTarget:
    case_id: str
    expected_ids: tuple[str, ...]
    forbidden_ids: tuple[str, ...] = ()
    answer_must_contain: tuple[str, ...] = ()
    answer_must_not_contain: tuple[str, ...] = ()
    should_abstain: bool = False


@dataclass(frozen=True)
class AnswerScore:
    case_id: str
    evidence_recall: float
    evidence_precision: float
    citation_faithfulness: float
    answer_constraint_accuracy: float
    abstention_accuracy: float
    hallucination_penalty: float
    resolved: bool


REQUIRED_OUTPUT_FIELDS = {
    "answer",
    "retrieved_ids",
    "cited_ids",
    "confidence",
    "abstained",
    "latency_ms",
    "estimated_cost",
    "independent_origins",
}


def validate_answer_output(payload: Mapping[str, object]) -> list[str]:
    errors = [f"missing required output field: {name}" for name in sorted(REQUIRED_OUTPUT_FIELDS - set(payload))]
    if errors:
        return errors
    if not isinstance(payload.get("answer"), str):
        errors.append("answer must be a string")
    for name in ("retrieved_ids", "cited_ids"):
        value = payload.get(name)
        if not isinstance(value, (list, tuple)) or not all(isinstance(x, str) for x in value):
            errors.append(f"{name} must be a list of strings")
    confidence = payload.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
        errors.append("confidence must be in [0,1]")
    if not isinstance(payload.get("abstained"), bool):
        errors.append("abstained must be boolean")
    for name in ("latency_ms", "estimated_cost"):
        value = payload.get(name)
        if not isinstance(value, (int, float)) or float(value) < 0:
            errors.append(f"{name} must be non-negative")
    if not isinstance(payload.get("independent_origins"), int) or int(payload.get("independent_origins", -1)) < 0:
        errors.append("independent_origins must be a non-negative integer")
    return errors


def parse_answer_output(payload: Mapping[str, object]) -> AnswerContract:
    errors = validate_answer_output(payload)
    if errors:
        raise ValueError("; ".join(errors))
    return AnswerContract(
        answer=str(payload["answer"]),
        retrieved_ids=tuple(payload["retrieved_ids"]),
        cited_ids=tuple(payload["cited_ids"]),
        confidence=float(payload["confidence"]),
        abstained=bool(payload["abstained"]),
        latency_ms=float(payload["latency_ms"]),
        estimated_cost=float(payload["estimated_cost"]),
        independent_origins=int(payload["independent_origins"]),
        provenance={str(k): str(v) for k, v in dict(payload.get("provenance", {})).items()},
    )


def score_answer(output: AnswerContract, target: AnswerTarget) -> AnswerScore:
    expected = set(target.expected_ids)
    forbidden = set(target.forbidden_ids)
    retrieved = set(output.retrieved_ids)
    cited = set(output.cited_ids)
    evidence_recall = len(retrieved & expected) / len(expected) if expected else 1.0
    evidence_precision = len(retrieved & expected) / len(retrieved) if retrieved else (1.0 if not expected else 0.0)
    citation_faithfulness = len(cited & expected) / len(cited) if cited else (1.0 if target.should_abstain else 0.0)
    normalized = output.answer.lower()
    must = all(token.lower() in normalized for token in target.answer_must_contain)
    must_not = all(token.lower() not in normalized for token in target.answer_must_not_contain)
    answer_constraints = float(must and must_not)
    abstention = float(output.abstained == target.should_abstain)
    unsupported = (retrieved | cited) & forbidden
    hallucination_penalty = min(1.0, len(unsupported) / max(1, len(forbidden))) if forbidden else 0.0
    if target.should_abstain:
        resolved = output.abstained and hallucination_penalty == 0.0
    else:
        resolved = (
            not output.abstained
            and evidence_recall == 1.0
            and citation_faithfulness == 1.0
            and answer_constraints == 1.0
            and hallucination_penalty == 0.0
        )
    return AnswerScore(
        case_id=target.case_id,
        evidence_recall=evidence_recall,
        evidence_precision=evidence_precision,
        citation_faithfulness=citation_faithfulness,
        answer_constraint_accuracy=answer_constraints,
        abstention_accuracy=abstention,
        hallucination_penalty=hallucination_penalty,
        resolved=resolved,
    )


def macro_average(scores: Sequence[AnswerScore]) -> dict[str, float]:
    if not scores:
        return {name: 0.0 for name in (
            "evidence_recall", "evidence_precision", "citation_faithfulness",
            "answer_constraint_accuracy", "abstention_accuracy", "hallucination_penalty", "resolution_rate"
        )}
    n = len(scores)
    return {
        "evidence_recall": sum(s.evidence_recall for s in scores) / n,
        "evidence_precision": sum(s.evidence_precision for s in scores) / n,
        "citation_faithfulness": sum(s.citation_faithfulness for s in scores) / n,
        "answer_constraint_accuracy": sum(s.answer_constraint_accuracy for s in scores) / n,
        "abstention_accuracy": sum(s.abstention_accuracy for s in scores) / n,
        "hallucination_penalty": sum(s.hallucination_penalty for s in scores) / n,
        "resolution_rate": sum(s.resolved for s in scores) / n,
    }
