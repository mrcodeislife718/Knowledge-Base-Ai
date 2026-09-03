from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class IndependentReplication:
    reviewer_id: str
    institution_or_team: str
    manifest_fingerprint: str
    artifact_hash: str
    reproduced: bool
    notes: str = ""


@dataclass(frozen=True)
class BlindJudgment:
    case_id: str
    judge_id: str
    system_label: str
    correct: bool
    evidence_faithful: bool
    complete: bool
    harmful_hallucination: bool = False


@dataclass
class BlindEvaluationReport:
    total: int
    correctness: float
    evidence_faithfulness: float
    completeness: float
    harmful_hallucination_rate: float
    judge_agreement: float


@dataclass(frozen=True)
class AblationResult:
    mechanism: str
    full_score: float
    ablated_score: float
    delta: float
    expected_direction: str = "decrease"


@dataclass
class AblationReport:
    results: list[AblationResult]
    mechanisms_with_measured_contribution: list[str]
    unexplained_mechanisms: list[str]


@dataclass(frozen=True)
class ScalePoint:
    corpus_documents: int
    corpus_bytes: int
    index_seconds: float
    index_cost: float
    index_bytes: int
    query_p50_ms: float
    query_p95_ms: float
    peak_memory_mb: float
    success_rate: float


@dataclass
class ScaleReport:
    points: list[ScalePoint]
    monotonic_resource_growth: bool
    no_catastrophic_success_drop: bool
    max_success_drop: float


@dataclass
class ReproducibilityBundle:
    manifest_fingerprint: str
    files: dict[str, str]
    environment: dict[str, str]
    command: str
    artifact_hash: str


def artifact_hash(paths: Sequence[str | Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(p) for p in paths):
        digest.update(str(path).encode())
        if path.exists() and path.is_file():
            digest.update(path.read_bytes())
        else:
            digest.update(b"<missing>")
    return digest.hexdigest()


def blind_evaluation(judgments: Sequence[BlindJudgment]) -> BlindEvaluationReport:
    if not judgments:
        return BlindEvaluationReport(0, 0.0, 0.0, 0.0, 1.0, 0.0)
    total = len(judgments)
    by_case: dict[str, list[BlindJudgment]] = {}
    for judgment in judgments:
        by_case.setdefault(judgment.case_id, []).append(judgment)
    agreements = []
    for rows in by_case.values():
        if len(rows) < 2:
            continue
        votes = [r.correct for r in rows]
        majority = sum(votes) >= len(votes) / 2
        agreements.extend(float(v == majority) for v in votes)
    return BlindEvaluationReport(
        total=total,
        correctness=sum(j.correct for j in judgments) / total,
        evidence_faithfulness=sum(j.evidence_faithful for j in judgments) / total,
        completeness=sum(j.complete for j in judgments) / total,
        harmful_hallucination_rate=sum(j.harmful_hallucination for j in judgments) / total,
        judge_agreement=sum(agreements) / len(agreements) if agreements else 0.0,
    )


def assess_ablations(results: Sequence[AblationResult], minimum_effect: float = 0.005) -> AblationReport:
    measured = []
    unexplained = []
    for result in results:
        expected = result.delta >= minimum_effect if result.expected_direction == "decrease" else result.delta <= -minimum_effect
        (measured if expected else unexplained).append(result.mechanism)
    return AblationReport(list(results), measured, unexplained)


def assess_scale(points: Sequence[ScalePoint], max_allowed_success_drop: float = 0.05) -> ScaleReport:
    ordered = sorted(points, key=lambda p: p.corpus_documents)
    if not ordered:
        return ScaleReport([], False, False, 1.0)
    resource_growth = all(
        b.index_bytes >= a.index_bytes and b.peak_memory_mb >= 0 and b.query_p95_ms >= 0
        for a, b in zip(ordered, ordered[1:])
    )
    drops = [max(0.0, a.success_rate - b.success_rate) for a, b in zip(ordered, ordered[1:])]
    max_drop = max(drops, default=0.0)
    return ScaleReport(ordered, resource_growth, max_drop <= max_allowed_success_drop, max_drop)


def verify_independent_replications(
    replications: Sequence[IndependentReplication],
    manifest_fingerprint: str,
    minimum_independent: int = 1,
) -> bool:
    valid = {
        (r.reviewer_id, r.institution_or_team)
        for r in replications
        if r.reproduced and r.manifest_fingerprint == manifest_fingerprint and r.artifact_hash
    }
    teams = {team for _, team in valid}
    return len(valid) >= minimum_independent and len(teams) >= minimum_independent


def build_reproducibility_bundle(
    *,
    manifest_fingerprint: str,
    files: Mapping[str, str | Path],
    environment: Mapping[str, str],
    command: str,
) -> ReproducibilityBundle:
    hashes: dict[str, str] = {}
    for label, pathlike in sorted(files.items()):
        path = Path(pathlike)
        hashes[label] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"
    raw = json.dumps(
        {
            "manifest": manifest_fingerprint,
            "files": hashes,
            "environment": dict(sorted(environment.items())),
            "command": command,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return ReproducibilityBundle(
        manifest_fingerprint=manifest_fingerprint,
        files=hashes,
        environment=dict(environment),
        command=command,
        artifact_hash=hashlib.sha256(raw.encode()).hexdigest(),
    )


def write_bundle(bundle: ReproducibilityBundle, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(bundle), indent=2, sort_keys=True), encoding="utf-8")
    return target
