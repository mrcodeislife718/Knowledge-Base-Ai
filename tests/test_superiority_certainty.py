from __future__ import annotations

from knowledge_base_ai.competitive_proof import RunObservation
from knowledge_base_ai.superiority_certainty import (
    FrozenRunManifest,
    ResourceEnvelope,
    build_claim_certificate,
    detect_contamination,
    holm_bonferroni,
    negative_controls,
    robustness_analysis,
    verify_resource_parity,
)


def obs(system: str, case: str, trial: int, resolved: bool, category: str = "contradiction", **kw):
    return RunObservation(
        system=system,
        case_id=case,
        category=category,
        trial=trial,
        resolved=resolved,
        recall=kw.get("recall", 1.0 if resolved else 0.0),
        contradiction_resolution=kw.get("contradiction_resolution", 1.0 if resolved else 0.0),
        multihop_accuracy=kw.get("multihop_accuracy", 1.0),
        stale_knowledge_accuracy=kw.get("stale_knowledge_accuracy", 1.0),
        source_independence_robustness=kw.get("source_independence_robustness", 1.0),
        confidence=kw.get("confidence", 0.9 if resolved else 0.3),
        brier_score=kw.get("brier_score", 0.01 if resolved else 0.49),
        latency_ms=kw.get("latency_ms", 10.0),
        estimated_cost=kw.get("estimated_cost", 0.01),
    )


def envelope(model="same-model"):
    return ResourceEnvelope(
        model=model,
        embedding_model="same-embed",
        reranker=None,
        max_context_tokens=8192,
        max_output_tokens=512,
        concurrency=1,
        cache_policy="cold",
        retry_policy="none",
        hardware_class="cpu-standard",
        provider="local",
    )


def test_resource_parity_detects_drift():
    assert verify_resource_parity({"a": envelope(), "b": envelope()}).equal
    report = verify_resource_parity({"a": envelope(), "b": envelope("different")})
    assert not report.equal
    assert "b.model" in report.differences


def test_contamination_detects_duplicate_and_leak():
    cases = [
        {"case_id": "a", "query": "secret held out question"},
        {"case_id": "b", "query": "secret held out question"},
    ]
    report = detect_contamination(cases, tuning_corpus=["secret held out question"])
    assert not report.clean
    assert report.leaked_case_ids
    assert report.duplicate_case_ids == ["b"]


def test_holm_bonferroni_is_conservative():
    result = holm_bonferroni({"a": 0.001, "b": 0.03, "c": 0.9}, alpha=0.05)
    assert "a" in result.passed_metrics
    assert "c" in result.failed_metrics


def test_robustness_requires_seed_stability():
    rows = []
    for trial in range(5):
        for case in range(10):
            rows.append(obs("kb", f"c{case}", trial, True))
            rows.append(obs("base", f"c{case}", trial, case < 6))
    report = robustness_analysis(rows, "kb", "base")
    assert report.stable
    assert report.seed_win_rate == 1.0


def test_negative_controls_pass_without_impossible_success():
    rows = []
    for i in range(20):
        rows.append(obs("kb", f"normal{i}", 0, bool(i % 2)))
    rows.append(obs("kb", "impossible", 0, False, category="impossible"))
    assert negative_controls(rows).passed


def test_certificate_refuses_unverified_competitor():
    rows = []
    for trial in range(5):
        for case in range(40):
            rows.append(obs("kb", f"c{case}", trial, True, latency_ms=5, estimated_cost=0.005))
            rows.append(obs("graph", f"c{case}", trial, case < 20, latency_ms=20, estimated_cost=0.02))
    manifest = FrozenRunManifest(
        benchmark_commit="abc",
        dataset_hash="d",
        query_hash="q",
        random_seeds=(1, 2, 3, 4, 5),
        competitor_commits={"graph": "sha"},
        resource_envelope=envelope(),
        preregistered_metrics=("resolved",),
        minimum_cases=40,
        minimum_trials=5,
    )
    cert = build_claim_certificate(
        observations=rows,
        challenger="kb",
        baselines=["graph"],
        manifest=manifest,
        parity=verify_resource_parity({"kb": envelope(), "graph": envelope()}),
        contamination=detect_contamination([{"case_id": f"c{i}", "query": f"q{i}"} for i in range(40)]),
        official_verified={"graph": False},
        external_execution_verified={"graph": True},
        negative_control=negative_controls(rows),
        minimum_power=0.1,
    )
    assert not cert.permitted
    assert any("official checkout" in x for x in cert.blockers)
