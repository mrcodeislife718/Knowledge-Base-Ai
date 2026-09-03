from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class IndexingObservation:
    system: str
    corpus_id: str
    documents: int
    input_bytes: int
    wall_seconds: float
    estimated_cost: float
    index_bytes: int
    peak_memory_mb: float
    failures: int = 0


@dataclass(frozen=True)
class QueryResourceObservation:
    system: str
    case_id: str
    trial: int
    input_tokens: int
    output_tokens: int
    model_calls: int
    embedding_calls: int
    reranker_calls: int
    wall_ms: float
    estimated_cost: float
    peak_memory_mb: float
    cache_hit: bool = False
    retries: int = 0


@dataclass
class ResourceSummary:
    system: str
    indexing_seconds: float
    indexing_cost: float
    index_bytes: int
    indexing_peak_memory_mb: float
    mean_query_ms: float
    p95_query_ms: float
    mean_query_cost: float
    mean_model_calls: float
    mean_input_tokens: float
    mean_output_tokens: float
    mean_peak_memory_mb: float
    retry_rate: float
    cache_hit_rate: float


def _p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))]


def summarize_resources(
    system: str,
    indexing: Sequence[IndexingObservation],
    queries: Sequence[QueryResourceObservation],
) -> ResourceSummary:
    idx = [x for x in indexing if x.system == system]
    q = [x for x in queries if x.system == system]
    return ResourceSummary(
        system=system,
        indexing_seconds=sum(x.wall_seconds for x in idx),
        indexing_cost=sum(x.estimated_cost for x in idx),
        index_bytes=sum(x.index_bytes for x in idx),
        indexing_peak_memory_mb=max((x.peak_memory_mb for x in idx), default=0.0),
        mean_query_ms=statistics.mean(x.wall_ms for x in q) if q else 0.0,
        p95_query_ms=_p95([x.wall_ms for x in q]),
        mean_query_cost=statistics.mean(x.estimated_cost for x in q) if q else 0.0,
        mean_model_calls=statistics.mean(x.model_calls for x in q) if q else 0.0,
        mean_input_tokens=statistics.mean(x.input_tokens for x in q) if q else 0.0,
        mean_output_tokens=statistics.mean(x.output_tokens for x in q) if q else 0.0,
        mean_peak_memory_mb=statistics.mean(x.peak_memory_mb for x in q) if q else 0.0,
        retry_rate=sum(x.retries > 0 for x in q) / len(q) if q else 0.0,
        cache_hit_rate=sum(x.cache_hit for x in q) / len(q) if q else 0.0,
    )


def fairness_violations(
    summaries: Sequence[ResourceSummary],
    *,
    max_token_ratio: float = 1.05,
    max_model_call_ratio: float = 1.05,
) -> list[str]:
    if not summaries:
        return ["no resource summaries"]
    violations: list[str] = []
    positive_input = [s.mean_input_tokens for s in summaries if s.mean_input_tokens > 0]
    positive_calls = [s.mean_model_calls for s in summaries if s.mean_model_calls > 0]
    if positive_input and max(positive_input) / min(positive_input) > max_token_ratio:
        violations.append("input-token budget differs materially across systems")
    if positive_calls and max(positive_calls) / min(positive_calls) > max_model_call_ratio:
        violations.append("model-call budget differs materially across systems")
    if len({round(s.cache_hit_rate, 6) for s in summaries}) > 1:
        violations.append("cache policy/effect differs across systems")
    return violations
