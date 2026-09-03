from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class EpistemicLayer(str, Enum):
    RAW_EVIDENCE = "L0_raw_evidence"
    EXTRACTED_CLAIM = "L1_extracted_claim"
    RELATIONSHIP = "L2_relationship"
    DERIVED_INFERENCE = "L3_derived_inference"
    REFLECTION = "L4_reflection"
    CONSOLIDATED_THEORY = "L5_consolidated_theory"


class RelationType(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    QUALIFIES = "qualifies"
    SUPERSEDES = "supersedes"
    DUPLICATES = "duplicates"
    GENERALIZES = "generalizes"
    SPECIALIZES = "specializes"
    DEPENDS_ON = "depends_on"
    CAUSED_BY = "caused_by"
    CORRELATED_WITH = "correlated_with"
    CAUSES = "causes"
    ENABLES = "enables"
    INHIBITS = "inhibits"
    REQUIRES = "requires"
    PRECEDES = "precedes"
    PREDICTS = "predicts"
    MEDIATES = "mediates"
    MODERATES = "moderates"
    UNKNOWN = "unknown"


class ProvenanceKind(str, Enum):
    SOURCE_ASSERTION = "source_assertion"
    OBSERVATIONAL_EVIDENCE = "observational_evidence"
    EXPERIMENT = "experiment"
    MODEL_INFERENCE = "model_inference"
    SYSTEM_INFERENCE = "system_inference"


class EpistemicStatus(str, Enum):
    UNVERIFIED = "unverified"
    SUPPORTED = "supported"
    SUPPORTED_BUT_CONTESTED = "supported-but-contested"
    CONTRADICTED = "contradicted"
    SUPERSEDED = "superseded"
    INVALIDATED = "invalidated"


@dataclass
class PageRecord:
    page_number: int
    text: str
    raw_text: str
    extraction_method: str
    source_path: str
    source_sha256: str
    page_sha256: str
    duplicate_of: int | None = None
    chapter: str = "Front Matter"
    document_label: str = "unclassified"
    quality_score: float = 0.0
    quality_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChunkRecord:
    chunk_id: str
    text: str
    chapter: str
    page_start: int
    page_end: int
    source_path: str
    source_sha256: str
    extraction_methods: list[str]
    chunk_sha256: str
    pipeline_version: str
    semantic_label: str = "narrative"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceRecord:
    evidence_id: str
    text: str
    source_path: str
    source_sha256: str
    passage_id: str
    page_start: int
    page_end: int
    observed_at: str | None = None
    published_at: str | None = None
    source_reliability: float = 0.5
    independent_origin: str | None = None
    content_sha256: str = ""
    layer: EpistemicLayer = EpistemicLayer.RAW_EVIDENCE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClaimRecord:
    claim_id: str
    statement: str
    evidence_ids: list[str]
    source_ids: list[str]
    entity_ids: list[str] = field(default_factory=list)
    support: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    qualifies: list[str] = field(default_factory=list)
    derived_from: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    confidence: float = 0.0
    epistemic_status: EpistemicStatus = EpistemicStatus.UNVERIFIED
    valid_from: str | None = None
    valid_until: str | None = None
    observed_at: str | None = None
    published_at: str | None = None
    superseded_by: str | None = None
    freshness: float = 1.0
    decay_function: str = "none"
    model_generated: bool = False
    layer: EpistemicLayer = EpistemicLayer.EXTRACTED_CLAIM
    invalidation_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EntityRecord:
    entity_id: str
    canonical_name: str
    aliases: list[str] = field(default_factory=list)
    entity_type: str = "unknown"
    evidence_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RelationRecord:
    relation_id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    evidence_ids: list[str]
    provenance_kind: ProvenanceKind
    confidence: float = 0.0
    assumptions: list[str] = field(default_factory=list)
    valid_from: str | None = None
    valid_until: str | None = None
    layer: EpistemicLayer = EpistemicLayer.RELATIONSHIP

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReflectionRecord:
    reflection_id: str
    statement: str
    claim_ids: list[str]
    evidence_ids: list[str]
    novelty: float
    importance: float
    contradiction: float
    uncertainty: float
    expected_future_value: float
    priority: float
    verification_status: str = "pending"
    confidence: float = 0.0
    assumptions: list[str] = field(default_factory=list)
    created_at: str | None = None
    layer: EpistemicLayer = EpistemicLayer.REFLECTION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class KnowledgeGap:
    gap_id: str
    question: str
    importance: float
    evidence_missing: list[str]
    related_claims: list[str]
    status: str = "open"
    created_at: str | None = None
    closed_at: str | None = None
    closed_by: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConfidenceBreakdown:
    source_reliability: float
    source_independence: float
    support_strength: float
    contradiction_penalty: float
    evidence_recency: float
    directness: float
    replication: float
    retrieval_coverage: float
    agreement: float
    reasoning_depth_penalty: float
    calibrated_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceBudget:
    latency_ms: int = 1500
    token_budget: int = 8000
    compute_units: float = 1.0
    confidence_target: float = 0.8
    source_diversity_target: int = 3
    max_results: int = 12

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QueryPlan:
    query: str
    intent: str
    strategies: list[str]
    requires_multi_hop: bool
    requires_temporal: bool
    requires_contradiction_resolution: bool
    requires_source_verification: bool
    graph_depth: int
    budget: EvidenceBudget

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalTelemetry:
    query_id: str
    query: str
    strategies: list[str]
    retrieved_ids: list[str]
    cited_ids: list[str]
    unused_ids: list[str]
    missed_ids: list[str]
    latency_ms: float
    estimated_cost: float
    answer_correct: bool | None = None
    user_corrected: bool = False
    created_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunManifest:
    run_id: str
    started_at: str
    completed_at: str | None
    status: str
    source_path: str
    source_sha256: str
    title: str
    author: str
    document_metadata: dict[str, Any]
    embedding_model: str
    collection_name: str
    force_ocr: bool
    page_count: int = 0
    unique_page_count: int = 0
    duplicate_page_count: int = 0
    chunk_count: int = 0
    chapter_count: int = 0
    low_quality_page_count: int = 0
    evidence_count: int = 0
    claim_count: int = 0
    entity_count: int = 0
    relation_count: int = 0
    reflection_count: int = 0
    knowledge_gap_count: int = 0
    knowledge_ir_path: str | None = None
    knowledge_tree_path: str | None = None
    inventory_path: str | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
