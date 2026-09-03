from __future__ import annotations

from dataclasses import asdict

from knowledge_base_ai.competitive_proof import (
    RunObservation,
    deterministic_heldout_cases,
    paired_bootstrap,
    score_external_output,
    summarize,
    technical_superiority_gate,
)


def _obs(system: str, case_id: str, resolved: bool, trial: int = 0, quality: float | None = None):
    q = float(resolved) if quality is None else quality
    return RunObservation(
        system=system,
        case_id=case_id,
        category="contradiction",
        trial=trial,
        resolved=resolved,
        recall=q,
        contradiction_resolution=q,
        multihop_accuracy=q,
        stale_knowledge_accuracy=q,
        source_independence_robustness=q,
        confidence=0.8 if resolved else 0.3,
        brier_score=0.04 if resolved else 0.09,
        latency_ms=10.0,
        estimated_cost=0.001,
    )


def test_heldout_suite_is_deterministic_and_covers_required_categories():
    left = deterministic_heldout_cases(seed=9, per_category=2)
    right = deterministic_heldout_cases(seed=9, per_category=2)
    assert [c.case_id for c in left] == [c.case_id for c in right]
    assert {c.category for c in left} == {
        "contradiction",
        "multi-hop",
        "stale-knowledge",
        "source-independence",
        "adversarial",
    }
    assert len(left) == 10


def test_external_scoring_detects_stale_and_independence_requirements():
    cases = deterministic_heldout_cases(seed=12, per_category=1)
    stale = next(c for c in cases if c.category == "stale-knowledge")
    output = {
        "retrieved_ids": [stale.expected_current_id],
        "confidence": 0.9,
        "latency_ms": 5,
        "estimated_cost": 0.01,
        "independent_origins": 1,
    }
    scored = score_external_output(stale, output, "x", 0)
    assert scored.resolved
    assert scored.stale_knowledge_accuracy == 1.0


def test_summary_contains_bootstrap_interval_and_p95():
    rows = [_obs("a", f"c{i}", i % 2 == 0) for i in range(10)]
    summary = summarize(rows)[0]
    assert 0 <= summary.resolution_ci95[0] <= summary.resolution_rate <= summary.resolution_ci95[1] <= 1
    assert summary.latency_p95_ms == 10.0


def test_paired_bootstrap_finds_clear_win():
    rows = []
    for i in range(20):
        rows.append(_obs("challenger", f"c{i}", True))
        rows.append(_obs("baseline", f"c{i}", False))
    result = paired_bootstrap(rows, "challenger", "baseline", "resolved", iterations=500)
    assert result.decisive
    assert result.mean_delta == 1.0
    assert result.probability_challenger_better == 1.0


def test_superiority_gate_refuses_non_decisive_evidence():
    rows = []
    for i in range(12):
        won = i % 2 == 0
        rows.append(_obs("kb", f"c{i}", won, quality=0.8))
        rows.append(_obs("competitor", f"c{i}", not won, quality=0.8))
    gate = technical_superiority_gate(rows, "kb", ["competitor"])
    assert gate["claim_allowed"] is False
    assert gate["comparisons"]
