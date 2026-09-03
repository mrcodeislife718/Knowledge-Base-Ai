from pathlib import Path

from knowledge_base_ai.benchmarking import (
    KnowledgeBaseAIAdapter,
    aggregate,
    reference_suite,
    run_suite,
    score_case,
    write_report,
)


def test_reference_suite_covers_required_proof_dimensions():
    categories = {case.category for case in reference_suite()}
    assert {"contradiction", "multi-hop", "stale-knowledge", "source-independence", "reflection"} <= categories


def test_knowledge_base_adapter_resolves_reference_contradiction_and_temporal_cases():
    adapter = KnowledgeBaseAIAdapter()
    by_category = {case.category: case for case in reference_suite()}

    contradiction = score_case(adapter, by_category["contradiction"])
    temporal = score_case(adapter, by_category["stale-knowledge"])

    assert contradiction.contradiction_resolution == 1.0
    assert contradiction.resolved
    assert temporal.stale_knowledge_accuracy == 1.0
    assert temporal.resolved


def test_suite_emits_all_comparators_and_calibration_cost_latency_metrics():
    result = run_suite()
    names = {row["adapter"] for row in result["aggregate"]}
    assert {
        "conventional-rag-reference",
        "graph-rag-style-reference",
        "raptor-style-reference",
        "reflection-rag-reference",
        "knowledge-base-ai-level5-6",
    } <= names
    for row in result["aggregate"]:
        assert 0.0 <= row["resolution_rate"] <= 1.0
        assert row["mean_latency_ms"] >= 0.0
        assert row["mean_cost_per_question"] >= 0.0
        assert 0.0 <= row["mean_brier_score"] <= 1.0


def test_aggregate_is_deterministic_in_structure():
    scores = [score_case(KnowledgeBaseAIAdapter(), case) for case in reference_suite()]
    rows = aggregate(scores)
    assert len(rows) == 1
    assert rows[0].cases == len(reference_suite())
    assert rows[0].adapter == "knowledge-base-ai-level5-6"


def test_write_report_creates_machine_and_human_readable_artifacts(tmp_path: Path):
    json_path, md_path = write_report(tmp_path, run_suite())
    assert json_path.exists()
    assert md_path.exists()
    assert "Reference-style baselines" in md_path.read_text(encoding="utf-8")
    assert "mean_brier_score" in json_path.read_text(encoding="utf-8")
