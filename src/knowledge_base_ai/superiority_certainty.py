from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .competitive_proof import RunObservation

QUALITY_METRICS = (
    "resolved",
    "recall",
    "contradiction_resolution",
    "multihop_accuracy",
    "stale_knowledge_accuracy",
    "source_independence_robustness",
)
LOWER_IS_BETTER = ("brier_score", "latency_ms", "estimated_cost")


@dataclass(frozen=True)
class ResourceEnvelope:
    model: str
    embedding_model: str
    reranker: str | None
    max_context_tokens: int
    max_output_tokens: int
    concurrency: int
    cache_policy: str
    retry_policy: str
    hardware_class: str
    provider: str
    temperature: float = 0.0

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class FrozenRunManifest:
    benchmark_commit: str
    dataset_hash: str
    query_hash: str
    random_seeds: tuple[int, ...]
    competitor_commits: Mapping[str, str]
    resource_envelope: ResourceEnvelope
    preregistered_metrics: tuple[str, ...]
    minimum_cases: int
    minimum_trials: int
    alpha: float = 0.05
    superiority_margin: float = 0.01
    noninferiority_margin: float = 0.01

    def fingerprint(self) -> str:
        payload = asdict(self)
        payload["competitor_commits"] = dict(sorted(self.competitor_commits.items()))
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()


@dataclass
class ContaminationReport:
    leaked_case_ids: list[str] = field(default_factory=list)
    duplicate_case_ids: list[str] = field(default_factory=list)
    overlap_ratio: float = 0.0

    @property
    def clean(self) -> bool:
        return not self.leaked_case_ids and not self.duplicate_case_ids and self.overlap_ratio <= 0.02


@dataclass
class ParityReport:
    equal: bool
    differences: dict[str, dict[str, object]]


@dataclass
class RobustnessReport:
    leave_one_category_out_min_delta: float
    leave_one_trial_out_min_delta: float
    seed_win_rate: float
    stable: bool


@dataclass
class NegativeControlReport:
    shuffled_label_accuracy: float
    impossible_case_success_rate: float
    passed: bool


@dataclass
class MultipleComparisonResult:
    adjusted_alpha: float
    passed_metrics: list[str]
    failed_metrics: list[str]


@dataclass
class ClaimCertificate:
    permitted: bool
    scope: str
    certainty_score: float
    blockers: list[str]
    evidence: dict[str, object]
    manifest_fingerprint: str
    claim_text: str


def capture_environment() -> dict[str, str]:
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        git_sha = "unknown"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "git_sha": git_sha,
        "pid": str(os.getpid()),
    }


def hash_records(records: Iterable[object]) -> str:
    normalized = []
    for record in records:
        if hasattr(record, "__dataclass_fields__"):
            normalized.append(asdict(record))
        else:
            normalized.append(record)
    raw = json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def verify_resource_parity(envelopes: Mapping[str, ResourceEnvelope]) -> ParityReport:
    if not envelopes:
        return ParityReport(False, {"missing": {"expected": "at least one system", "actual": None}})
    names = sorted(envelopes)
    reference = envelopes[names[0]]
    differences: dict[str, dict[str, object]] = {}
    for name in names[1:]:
        candidate = envelopes[name]
        for field_name in reference.__dataclass_fields__:
            left = getattr(reference, field_name)
            right = getattr(candidate, field_name)
            if left != right:
                differences[f"{name}.{field_name}"] = {"expected": left, "actual": right}
    return ParityReport(not differences, differences)


def detect_contamination(
    test_cases: Sequence[Mapping[str, object]],
    tuning_corpus: Sequence[str] = (),
    known_training_text: Sequence[str] = (),
) -> ContaminationReport:
    seen: set[str] = set()
    duplicates: list[str] = []
    leaked: list[str] = []
    overlap_scores: list[float] = []
    tuning = "\n".join(tuning_corpus).lower()
    training = "\n".join(known_training_text).lower()
    for case in test_cases:
        case_id = str(case.get("case_id", "unknown"))
        query = str(case.get("query", "")).strip().lower()
        digest = hashlib.sha256(query.encode()).hexdigest()
        if digest in seen:
            duplicates.append(case_id)
        seen.add(digest)
        if query and (query in tuning or query in training):
            leaked.append(case_id)
        q_terms = set(query.split())
        if q_terms and tuning:
            t_terms = set(tuning.split())
            overlap_scores.append(len(q_terms & t_terms) / len(q_terms))
    return ContaminationReport(
        leaked_case_ids=sorted(set(leaked)),
        duplicate_case_ids=sorted(set(duplicates)),
        overlap_ratio=statistics.mean(overlap_scores) if overlap_scores else 0.0,
    )


def _metric_value(obs: RunObservation, metric: str) -> float:
    value = getattr(obs, metric)
    return float(value)


def _paired_deltas(observations: Sequence[RunObservation], challenger: str, baseline: str, metric: str) -> list[float]:
    c = {(r.case_id, r.trial): r for r in observations if r.system == challenger}
    b = {(r.case_id, r.trial): r for r in observations if r.system == baseline}
    keys = sorted(set(c) & set(b))
    sign = -1.0 if metric in LOWER_IS_BETTER else 1.0
    return [sign * (_metric_value(c[k], metric) - _metric_value(b[k], metric)) for k in keys]


def bootstrap_delta_ci(
    observations: Sequence[RunObservation],
    challenger: str,
    baseline: str,
    metric: str,
    *,
    seed: int = 4401,
    iterations: int = 5000,
) -> tuple[float, float, float]:
    deltas = _paired_deltas(observations, challenger, baseline, metric)
    if not deltas:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    means = []
    n = len(deltas)
    for _ in range(iterations):
        means.append(statistics.mean(deltas[rng.randrange(n)] for _ in range(n)))
    means.sort()
    lo = means[max(0, int(0.025 * len(means)))]
    hi = means[min(len(means) - 1, int(0.975 * len(means)))]
    prob = sum(x > 0 for x in means) / len(means)
    return (lo, hi, prob)


def holm_bonferroni(
    p_values: Mapping[str, float], alpha: float = 0.05
) -> MultipleComparisonResult:
    ordered = sorted((max(0.0, min(1.0, p)), metric) for metric, p in p_values.items())
    passed: list[str] = []
    failed: list[str] = []
    m = len(ordered)
    for rank, (p, metric) in enumerate(ordered):
        threshold = alpha / max(1, m - rank)
        if p <= threshold and not failed:
            passed.append(metric)
        else:
            failed.append(metric)
    return MultipleComparisonResult(alpha / max(1, m), passed, failed)


def robustness_analysis(
    observations: Sequence[RunObservation], challenger: str, baseline: str
) -> RobustnessReport:
    categories = sorted({r.category for r in observations})
    trials = sorted({r.trial for r in observations})
    category_deltas = []
    for category in categories:
        subset = [r for r in observations if r.category != category]
        deltas = _paired_deltas(subset, challenger, baseline, "resolved")
        if deltas:
            category_deltas.append(statistics.mean(deltas))
    trial_deltas = []
    for trial in trials:
        subset = [r for r in observations if r.trial != trial]
        deltas = _paired_deltas(subset, challenger, baseline, "resolved")
        if deltas:
            trial_deltas.append(statistics.mean(deltas))
    seed_wins = []
    for trial in trials:
        subset = [r for r in observations if r.trial == trial]
        deltas = _paired_deltas(subset, challenger, baseline, "resolved")
        if deltas:
            seed_wins.append(statistics.mean(deltas) > 0)
    min_category = min(category_deltas) if category_deltas else -1.0
    min_trial = min(trial_deltas) if trial_deltas else -1.0
    win_rate = sum(seed_wins) / len(seed_wins) if seed_wins else 0.0
    return RobustnessReport(
        leave_one_category_out_min_delta=min_category,
        leave_one_trial_out_min_delta=min_trial,
        seed_win_rate=win_rate,
        stable=min_category >= 0 and min_trial >= 0 and win_rate >= 0.8,
    )


def negative_controls(
    observations: Sequence[RunObservation], *, seed: int = 99
) -> NegativeControlReport:
    if not observations:
        return NegativeControlReport(1.0, 1.0, False)
    rng = random.Random(seed)
    labels = [float(r.resolved) for r in observations]
    shuffled = labels[:]
    rng.shuffle(shuffled)
    original = labels
    accidental = sum(a == b for a, b in zip(original, shuffled)) / len(labels)
    impossible_rows = [r for r in observations if r.category in {"negative-control", "impossible"}]
    impossible_success = (
        statistics.mean(float(r.resolved) for r in impossible_rows) if impossible_rows else 0.0
    )
    return NegativeControlReport(
        shuffled_label_accuracy=accidental,
        impossible_case_success_rate=impossible_success,
        passed=accidental < 0.8 and impossible_success <= 0.05,
    )


def statistical_power(n_pairs: int, observed_effect: float, alpha: float = 0.05) -> float:
    """Conservative normal approximation for paired binary/continuous mean deltas."""
    if n_pairs <= 1 or observed_effect <= 0:
        return 0.0
    z_alpha = 1.959963984540054
    standardized = abs(observed_effect) * math.sqrt(n_pairs) / 0.5
    # Phi(x) approximation via erf.
    return max(0.0, min(1.0, 0.5 * (1 + math.erf((standardized - z_alpha) / math.sqrt(2)))))


def build_claim_certificate(
    *,
    observations: Sequence[RunObservation],
    challenger: str,
    baselines: Sequence[str],
    manifest: FrozenRunManifest,
    parity: ParityReport,
    contamination: ContaminationReport,
    official_verified: Mapping[str, bool],
    external_execution_verified: Mapping[str, bool],
    negative_control: NegativeControlReport,
    minimum_power: float = 0.8,
) -> ClaimCertificate:
    blockers: list[str] = []
    evidence: dict[str, object] = {}
    if not parity.equal:
        blockers.append("resource/model parity failed")
    if not contamination.clean:
        blockers.append("held-out contamination or duplicate cases detected")
    if not negative_control.passed:
        blockers.append("negative controls failed")

    systems = {r.system for r in observations}
    if challenger not in systems:
        blockers.append(f"missing challenger results: {challenger}")
    for baseline in baselines:
        if baseline not in systems:
            blockers.append(f"missing baseline results: {baseline}")
        if not official_verified.get(baseline, False):
            blockers.append(f"official checkout not verified: {baseline}")
        if not external_execution_verified.get(baseline, False):
            blockers.append(f"native execution not verified: {baseline}")

    trials = {r.trial for r in observations if r.system == challenger}
    cases = {r.case_id for r in observations if r.system == challenger}
    if len(trials) < manifest.minimum_trials:
        blockers.append("insufficient repeated trials")
    if len(cases) < manifest.minimum_cases:
        blockers.append("insufficient held-out cases")

    decisive_baselines = 0
    all_pairwise: dict[str, object] = {}
    powers: list[float] = []
    for baseline in baselines:
        pair: dict[str, object] = {}
        p_values: dict[str, float] = {}
        superior_dimensions = 0
        quality_noninferior = True
        for metric in QUALITY_METRICS + LOWER_IS_BETTER:
            lo, hi, prob = bootstrap_delta_ci(observations, challenger, baseline, metric)
            delta = statistics.mean(_paired_deltas(observations, challenger, baseline, metric)) if _paired_deltas(observations, challenger, baseline, metric) else 0.0
            pair[metric] = {"delta": delta, "ci95": [lo, hi], "probability_better": prob}
            p_values[metric] = max(0.0, min(1.0, 1.0 - prob))
            if metric in QUALITY_METRICS and lo < -manifest.noninferiority_margin:
                quality_noninferior = False
            if lo > manifest.superiority_margin:
                superior_dimensions += 1
        correction = holm_bonferroni(p_values, manifest.alpha)
        resolution = pair["resolved"]
        decisive_resolution = (
            resolution["ci95"][0] > manifest.superiority_margin
            and "resolved" in correction.passed_metrics
        )
        robust = robustness_analysis(observations, challenger, baseline)
        n_pairs = len(_paired_deltas(observations, challenger, baseline, "resolved"))
        power = statistical_power(n_pairs, abs(float(resolution["delta"])), manifest.alpha)
        powers.append(power)
        baseline_pass = decisive_resolution and quality_noninferior and superior_dimensions >= 2 and robust.stable and power >= minimum_power
        if baseline_pass:
            decisive_baselines += 1
        else:
            blockers.append(f"competitive evidence not decisive and robust against {baseline}")
        pair["holm_bonferroni"] = asdict(correction)
        pair["robustness"] = asdict(robust)
        pair["power"] = power
        pair["passes"] = baseline_pass
        all_pairwise[baseline] = pair

    evidence["pairwise"] = all_pairwise
    evidence["parity"] = asdict(parity)
    evidence["contamination"] = asdict(contamination)
    evidence["negative_controls"] = asdict(negative_control)
    evidence["official_verified"] = dict(official_verified)
    evidence["external_execution_verified"] = dict(external_execution_verified)
    evidence["trials"] = len(trials)
    evidence["cases"] = len(cases)
    evidence["power_min"] = min(powers) if powers else 0.0

    permitted = not blockers and decisive_baselines == len(baselines) and bool(baselines)
    certainty_components = [
        float(parity.equal),
        float(contamination.clean),
        float(negative_control.passed),
        decisive_baselines / max(1, len(baselines)),
        min(powers) if powers else 0.0,
        min(1.0, len(trials) / max(1, manifest.minimum_trials)),
        min(1.0, len(cases) / max(1, manifest.minimum_cases)),
    ]
    certainty = statistics.mean(certainty_components)
    if permitted:
        scope = "measured held-out workloads under the frozen resource envelope"
        claim = (
            f"Under the preregistered, pinned, resource-matched benchmark represented by manifest "
            f"{manifest.fingerprint()[:12]}, {challenger} demonstrated statistically robust technical superiority "
            f"over {', '.join(baselines)} on the measured held-out workloads. This claim does not imply universal superiority."
        )
    else:
        scope = "no superiority claim permitted"
        claim = "Insufficient evidence for a technical-superiority claim."
    return ClaimCertificate(
        permitted=permitted,
        scope=scope,
        certainty_score=certainty,
        blockers=sorted(set(blockers)),
        evidence=evidence,
        manifest_fingerprint=manifest.fingerprint(),
        claim_text=claim,
    )


def write_certificate(certificate: ClaimCertificate, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(certificate), indent=2, sort_keys=True), encoding="utf-8")
    return target
