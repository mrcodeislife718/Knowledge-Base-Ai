from knowledge_base_ai.benchmark_accounting import (
    IndexingObservation,
    QueryResourceObservation,
    fairness_violations,
    summarize_resources,
)


def test_resource_summary_tracks_index_and_query_costs():
    idx = [IndexingObservation("a", "c", 10, 1000, 2.5, 0.2, 500, 128)]
    q = [
        QueryResourceObservation("a", "q1", 0, 100, 20, 1, 1, 0, 10, 0.01, 64),
        QueryResourceObservation("a", "q2", 0, 100, 20, 1, 1, 0, 20, 0.01, 65),
    ]
    s = summarize_resources("a", idx, q)
    assert s.indexing_seconds == 2.5
    assert s.mean_query_ms == 15
    assert s.index_bytes == 500


def test_fairness_flags_material_token_or_call_advantage():
    a = summarize_resources("a", [], [QueryResourceObservation("a", "q", 0, 100, 10, 1, 1, 0, 10, 0, 32)])
    b = summarize_resources("b", [], [QueryResourceObservation("b", "q", 0, 150, 10, 2, 1, 0, 10, 0, 32)])
    violations = fairness_violations([a, b])
    assert any("input-token" in v for v in violations)
    assert any("model-call" in v for v in violations)
