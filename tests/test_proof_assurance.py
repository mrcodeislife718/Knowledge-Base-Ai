from knowledge_base_ai.proof_assurance import (
    AblationResult,
    BlindJudgment,
    IndependentReplication,
    ScalePoint,
    assess_ablations,
    assess_scale,
    blind_evaluation,
    verify_independent_replications,
)


def test_blind_evaluation_scores_faithfulness():
    rows = [
        BlindJudgment("c1", "j1", "A", True, True, True),
        BlindJudgment("c1", "j2", "A", True, True, False),
        BlindJudgment("c2", "j1", "B", False, False, False, True),
        BlindJudgment("c2", "j2", "B", False, False, False, True),
    ]
    report = blind_evaluation(rows)
    assert report.correctness == 0.5
    assert report.evidence_faithfulness == 0.5
    assert report.harmful_hallucination_rate == 0.5
    assert report.judge_agreement == 1.0


def test_ablations_require_measured_contribution():
    report = assess_ablations([
        AblationResult("contradiction-graph", 0.90, 0.80, 0.10),
        AblationResult("unused-feature", 0.90, 0.899, 0.001),
    ])
    assert "contradiction-graph" in report.mechanisms_with_measured_contribution
    assert "unused-feature" in report.unexplained_mechanisms


def test_scale_report_catches_success_collapse():
    points = [
        ScalePoint(100, 1000, 1, 0, 100, 5, 8, 64, 0.95),
        ScalePoint(1000, 10000, 10, 0, 1000, 8, 12, 128, 0.80),
    ]
    assert not assess_scale(points).no_catastrophic_success_drop


def test_replication_requires_matching_manifest_and_independent_team():
    rows = [IndependentReplication("r1", "team-a", "abc", "hash", True)]
    assert verify_independent_replications(rows, "abc", minimum_independent=1)
    assert not verify_independent_replications(rows, "other", minimum_independent=1)
